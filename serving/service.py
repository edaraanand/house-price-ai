import os
import time

import bentoml
import mlflow
import pandas as pd
from mlflow import MlflowClient
from opentelemetry.trace import Status, StatusCode

from serving.kafka import publish_prediction
from serving.observability import (
    configure_observability,
    logger,
    request_id_ctx,
    reset_request_id,
    set_request_id,
    tracer,
)
from serving.schemas import HousingRequest, HousingResponse


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-mlflow.mlflow.svc.cluster.local:5000",
)

MLFLOW_MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "california_housing_model",
)

MLFLOW_MODEL_ALIAS = os.getenv(
    "MLFLOW_MODEL_ALIAS",
    "production",
)


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Configure OpenTelemetry when the worker imports the service.
configure_observability()


@bentoml.service(
    resources={
        "cpu": "1",
        "memory": "512Mi",
    },
)
class HousingModelService:

    def __init__(self):

        model_uri = (
            f"models:/{MLFLOW_MODEL_NAME}"
            f"@{MLFLOW_MODEL_ALIAS}"
        )

        #
        # Resolve the alias once when the worker starts.
        #
        # This gives us the concrete MLflow model version
        # that this BentoML worker is actually serving.
        #

        client = MlflowClient()

        model_version = client.get_model_version_by_alias(
            MLFLOW_MODEL_NAME,
            MLFLOW_MODEL_ALIAS,
        )

        self.model_info = {
            "name": MLFLOW_MODEL_NAME,
            "alias": MLFLOW_MODEL_ALIAS,
            "version": str(model_version.version),
            "run_id": model_version.run_id,
            "source": model_version.source,
        }

        #
        # Load model
        #

        with tracer.start_as_current_span(
            "housing.model.load"
        ) as span:

            span.set_attribute(
                "mlflow.model.name",
                self.model_info["name"],
            )

            span.set_attribute(
                "mlflow.model.alias",
                self.model_info["alias"],
            )

            span.set_attribute(
                "mlflow.model.version",
                self.model_info["version"],
            )

            span.set_attribute(
                "mlflow.model.run_id",
                self.model_info["run_id"],
            )

            span.set_attribute(
                "mlflow.model.uri",
                model_uri,
            )

            start = time.perf_counter()

            try:

                logger.info(
                    "model_loading",
                    extra={
                        "model_name": self.model_info["name"],
                        "model_alias": self.model_info["alias"],
                        "model_version": self.model_info["version"],
                        "model_run_id": self.model_info["run_id"],
                    },
                )

                self.model = mlflow.pyfunc.load_model(
                    model_uri
                )

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                span.set_attribute(
                    "mlflow.model.load_duration_ms",
                    elapsed_ms,
                )

                span.set_status(
                    Status(StatusCode.OK)
                )

                logger.info(
                    "model_loaded",
                    extra={
                        "model_name": self.model_info["name"],
                        "model_alias": self.model_info["alias"],
                        "model_version": self.model_info["version"],
                        "model_run_id": self.model_info["run_id"],
                        "duration_ms": round(
                            elapsed_ms,
                            2,
                        ),
                    },
                )

            except Exception as ex:

                span.record_exception(ex)

                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        str(ex),
                    )
                )

                logger.exception(
                    "model_load_failed",
                    extra={
                        "model_name": self.model_info["name"],
                        "model_alias": self.model_info["alias"],
                        "model_version": self.model_info["version"],
                        "model_run_id": self.model_info["run_id"],
                    },
                )

                raise

    @bentoml.api(route="/healthz")
    def healthz(self) -> dict:
        return {
            "status": "ok",
            "service": "house-price-serving",
        }

    @bentoml.api
    def predict(
        self,
        input_data: HousingRequest,
    ) -> HousingResponse:

        token = set_request_id()

        try:

            request_id = request_id_ctx.get()

            with tracer.start_as_current_span(
                "housing.predict"
            ) as span:

                request_start = time.perf_counter()

                #
                # Request / model metadata
                #

                span.set_attribute(
                    "housing.request_id",
                    request_id,
                )

                span.set_attribute(
                    "mlflow.model.name",
                    self.model_info["name"],
                )

                span.set_attribute(
                    "mlflow.model.alias",
                    self.model_info["alias"],
                )

                span.set_attribute(
                    "mlflow.model.version",
                    self.model_info["version"],
                )

                span.set_attribute(
                    "mlflow.model.run_id",
                    self.model_info["run_id"],
                )

                try:

                    #
                    # Prepare input
                    #

                    data = pd.DataFrame(
                        [
                            input_data.model_dump()
                        ]
                    )

                    span.set_attribute(
                        "housing.input.columns",
                        len(data.columns),
                    )

                    span.set_attribute(
                        "housing.input.rows",
                        len(data),
                    )

                    logger.info(
                        "prediction_started",
                        extra={
                            "endpoint": "/predict",
                            "request_id": request_id,
                            "model_name": self.model_info["name"],
                            "model_alias": self.model_info["alias"],
                            "model_version": self.model_info["version"],
                            "model_run_id": self.model_info["run_id"],
                        },
                    )

                    #
                    # Model inference
                    #

                    with tracer.start_as_current_span(
                        "housing.model.predict"
                    ) as predict_span:

                        predict_start = time.perf_counter()

                        prediction = self.model.predict(
                            data
                        )

                        predict_duration_ms = (
                            time.perf_counter()
                            - predict_start
                        ) * 1000

                        predict_span.set_attribute(
                            "mlflow.model.version",
                            self.model_info["version"],
                        )

                        predict_span.set_attribute(
                            "housing.prediction.duration_ms",
                            predict_duration_ms,
                        )

                        predict_span.set_attribute(
                            "housing.prediction.count",
                            len(prediction),
                        )

                        predict_span.set_status(
                            Status(StatusCode.OK)
                        )

                    #
                    # Result
                    #

                    result = float(
                        prediction[0]
                    )

                    #
                    # Duration before event publication.
                    #
                    # This represents application processing time
                    # before publishing the event.
                    #

                    request_duration_ms = (
                        time.perf_counter()
                        - request_start
                    ) * 1000

                    #
                    # Prediction event
                    #

                    event = {
                        "event_type": "housing_prediction",
                        "event_version": "1",

                        "timestamp": (
                            pd.Timestamp.utcnow().isoformat()
                        ),

                        "request_id": request_id,

                        "service": "house-price-ai",

                        "model": self.model_info,

                        "features": (
                            input_data.model_dump()
                        ),

                        "prediction": result,

                        # Ground truth can be populated later.
                        "actual": None,

                        "performance": {
                            "prediction_duration_ms": (
                                predict_duration_ms
                            ),
                            "request_duration_ms": (
                                request_duration_ms
                            ),
                        },
                    }

                    #
                    # Publish prediction event
                    #

                    publish_prediction(event)

                    #
                    # Final request duration.
                    #
                    # This includes event publication.
                    #

                    total_duration_ms = (
                        time.perf_counter()
                        - request_start
                    ) * 1000

                    #
                    # Trace
                    #

                    span.set_attribute(
                        "housing.request.duration_ms",
                        total_duration_ms,
                    )

                    span.set_status(
                        Status(StatusCode.OK)
                    )

                    #
                    # Log
                    #

                    logger.info(
                        "prediction_completed",
                        extra={
                            "endpoint": "/predict",
                            "request_id": request_id,

                            "model_name": (
                                self.model_info["name"]
                            ),

                            "model_alias": (
                                self.model_info["alias"]
                            ),

                            "model_version": (
                                self.model_info["version"]
                            ),

                            "model_run_id": (
                                self.model_info["run_id"]
                            ),

                            "duration_ms": round(
                                total_duration_ms,
                                2,
                            ),

                            "prediction_duration_ms": round(
                                predict_duration_ms,
                                2,
                            ),

                            "prediction": result,

                            "status": 200,
                        },
                    )

                    return HousingResponse(
                        prediction=result
                    )

                except Exception as ex:

                    elapsed_ms = (
                        time.perf_counter()
                        - request_start
                    ) * 1000

                    span.record_exception(ex)

                    span.set_status(
                        Status(
                            StatusCode.ERROR,
                            str(ex),
                        )
                    )

                    logger.exception(
                        "prediction_failed",
                        extra={
                            "endpoint": "/predict",
                            "request_id": request_id,

                            "model_name": (
                                self.model_info["name"]
                            ),

                            "model_alias": (
                                self.model_info["alias"]
                            ),

                            "model_version": (
                                self.model_info["version"]
                            ),

                            "model_run_id": (
                                self.model_info["run_id"]
                            ),

                            "duration_ms": round(
                                elapsed_ms,
                                2,
                            ),
                        },
                    )

                    raise

        finally:
            reset_request_id(token)
