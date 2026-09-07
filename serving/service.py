import os
import time

import bentoml
import mlflow
import pandas as pd
from opentelemetry.trace import Status, StatusCode

from observability import (
    configure_observability,
    logger,
    request_id_ctx,
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


mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# Configure OTel when the worker imports the service.
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

        with tracer.start_as_current_span(
            "housing.model.load"
        ) as span:

            span.set_attribute(
                "mlflow.model.name",
                MLFLOW_MODEL_NAME,
            )

            span.set_attribute(
                "mlflow.model.alias",
                MLFLOW_MODEL_ALIAS,
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
                        "model_name": MLFLOW_MODEL_NAME,
                        "model_alias": MLFLOW_MODEL_ALIAS,
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

                logger.info(
                    "model_loaded",
                    extra={
                        "model_name": MLFLOW_MODEL_NAME,
                        "model_alias": MLFLOW_MODEL_ALIAS,
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
                        "model_name": MLFLOW_MODEL_NAME,
                        "model_alias": MLFLOW_MODEL_ALIAS,
                    },
                )

                raise

    @bentoml.api
    def predict(
        self,
        input_data: HousingRequest,
    ) -> HousingResponse:

        request_id = request_id_ctx.get()

        with tracer.start_as_current_span(
            "housing.predict"
        ) as span:

            request_start = time.perf_counter()

            span.set_attribute(
                "housing.request_id",
                request_id,
            )

            span.set_attribute(
                "mlflow.model.name",
                MLFLOW_MODEL_NAME,
            )

            span.set_attribute(
                "mlflow.model.alias",
                MLFLOW_MODEL_ALIAS,
            )

            try:

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
                        "housing.prediction.duration_ms",
                        predict_duration_ms,
                    )

                    predict_span.set_attribute(
                        "housing.prediction.count",
                        len(prediction),
                    )

                #
                # Result
                #

                result = float(
                    prediction[0]
                )

                total_duration_ms = (
                    time.perf_counter()
                    - request_start
                ) * 1000

                span.set_attribute(
                    "housing.prediction",
                    result,
                )

                span.set_attribute(
                    "housing.request.duration_ms",
                    total_duration_ms,
                )

                span.set_status(
                    Status(StatusCode.OK)
                )

                logger.info(
                    "prediction_completed",
                    extra={
                        "endpoint": "/predict",
                        "request_id": request_id,
                        "duration_ms": round(
                            total_duration_ms,
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
                        "duration_ms": round(
                            elapsed_ms,
                            2,
                        ),
                    },
                )

                raise
