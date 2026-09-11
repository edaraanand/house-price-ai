import json
import os

from confluent_kafka import Producer


KAFKA_BOOTSTRAP_SERVERS = os.environ[
    "KAFKA_BOOTSTRAP_SERVERS"
]

KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "housing.predictions",
)

def delivery_report(err, msg):
    if err is not None:
        print(
            f"Kafka delivery failed: {err}"
        )
    else:
        print(
            f"Kafka delivered: "
            f"topic={msg.topic()} "
            f"partition={msg.partition()} "
            f"offset={msg.offset()}"
        )


producer = Producer({
    "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
})


def publish_prediction(event):
    producer.produce(
        KAFKA_TOPIC,
        key=event["request_id"],
        value=json.dumps(event),
        callback=delivery_report,
    )

    producer.flush()

# producer = Producer(
#     {
#         "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,

        # "client.id": "housing-bentoml",
        # "acks": "all",
        # "enable.idempotence": True,
        # "message.timeout.ms": 5000,
#     }
# )


# def _delivery_callback(err, msg):

#     if err is not None:
#         print(f"Kafka delivery failed: {err}")
#         return

#     print(
#         "Kafka delivery succeeded: "
#         f"{msg.topic()} "
#         f"partition={msg.partition()} "
#         f"offset={msg.offset()}"
#     )


# def publish_prediction(event: dict):

#     producer.produce(
#         topic=KAFKA_TOPIC,
#         key=str(event["request_id"]),
#         value=json.dumps(event),
#         # callback=_delivery_callback,
#     )

#     producer.poll(0)
