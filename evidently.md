python evaluate.py --candidate-version 2 --baseline-version 1

Evidently: input drift, prediction drift, data quality, feature distributions, missing values, outliers, and model performance when ground-truth labels become available.

                         ┌─────────────────────┐
                         │    BentoML API       │
                         │     /predict         │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
              Prometheus          OTel          Prediction
                    │               │                │
                    ▼               ▼                ▼
                 Grafana         Collector       Evidently
                    │                                │
                    │                    ┌───────────┴──────────┐
                    │                    │                      │
                    │                 Data Drift          Prediction Drift
                    │                    │                      │
                    │                    └──────────┬───────────┘
                    │                               │
                    └───────────────────────────────▼
                                      ML Monitoring Dashboard

My model has these eight features:
MedInc
HouseAge
AveRooms
AveBedrms
Population
AveOccup
Latitude
Longitude

Input data quality
Missing values
Null percentage
Unexpected values
Feature distributions
Outliers
Feature ranges

Feature Drift
MedInc drift
HouseAge drift
AveRooms drift
AveBedrms drift
Population drift
AveOccup drift
Latitude drift
Longitude drift

Prediction drift
prediction distribution
prediction mean
prediction median
prediction p95
prediction p99
prediction range


Option	Kafka → datastore	Kubernetes	Good for Evidently	Complexity
Kafka Connect	Excellent	Yes	⭐⭐⭐⭐⭐	Low
Flink	Excellent	Yes	⭐⭐⭐⭐	High
Logstash	Good	Yes	⭐⭐⭐	Low
Benthos	Excellent	Yes	⭐⭐⭐⭐⭐	Very low
Python consumer	Excellent	Yes	⭐⭐⭐⭐	Low
Debezium	Specialized	Yes	⭐⭐⭐	Medium

Level 1 — prediction monitoring
    This gives Evidently enough information to monitor:

    feature distributions
    feature drift
    prediction distribution
    missing values
    unexpected values
    feature statistics
    prediction drift
    data quality

    {
        "event_type": "housing_prediction",
        "timestamp": "...",
        "request_id": "...",
        "service": "house-price-ai",
        "model": {
            "name": "california_housing_model",
            "alias": "production"
        },
        "features": {
            "...": "..."
        },
        "prediction": 123.45,
        "performance": {
            "prediction_duration_ms": 2.1,
            "request_duration_ms": 8.4
        }
    }
    

Level 2 — actual model performance
    Eventually you want:
        features
        prediction
        actual_target

    {
    "features": {
        "MedInc": 8.3,
        "HouseAge": 25,
        "AveRooms": 5.4,
        "AveBedrms": 1.1,
        "Population": 1200,
        "AveOccup": 2.8,
        "Latitude": 37.8,
        "Longitude": -122.4
    },
    "prediction": 4.21,
    "actual": 4.05
    }

    Then Evidently can monitor:
        MAE
        RMSE
        error distribution
        residuals
        performance degradation
        performance by feature ranges
        performance by segments

Level 3 — automated decisions
Evidently
    |
    +-- feature drift > threshold
    |
    +-- prediction drift > threshold
    |
    +-- MAE > threshold
    |
    +-- data quality failure
             |
             v
          Alert
             |
             v
       GitHub Actions
             |
             v
        retraining
             |
             v
           MLflow
             |
             v
       model validation
             |
             v
       production alias
