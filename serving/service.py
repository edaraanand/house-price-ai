import os

import bentoml
import mlflow
import pandas as pd

from serving.schemas import HousingRequest, HousingResponse

MLFLOW_TRACKING_URI = os.getenv(
"MLFLOW_TRACKING_URI",
"http://mlflow-mlflow.mlflow.svc.cluster.local:5000",
)

MLFLOW_MODEL_NAME = os.getenv(
"MLFLOW_MODEL_NAME",
"xgboost-regression",
)

MLFLOW_MODEL_ALIAS = os.getenv(
"MLFLOW_MODEL_ALIAS",
"production",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@bentoml.service(
    resources={
    "cpu": "1",
    "memory": "512Mi",
    },
)

class HousingModelService:
    def __init__(self):
        model_uri = (
            f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
        )

        # self.model = mlflow.xgboost.load_model(model_uri)
        self.model = mlflow.pyfunc.load_model(model_uri)

    @bentoml.api
    def predict(self, input_data: HousingRequest) -> HousingResponse:

        data = pd.DataFrame(
            [
                request.model_dump()
            ]
        )

        prediction = self.model.predict(data)

        return HousingResponse(
            prediction=float(prediction[0])
        )
