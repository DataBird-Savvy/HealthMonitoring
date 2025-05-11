import os
import json
import logging
import pandas as pd
from kafka import KafkaConsumer
from dotenv import load_dotenv
from threading import Timer
from collections import deque
from datetime import datetime

load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPICS = ["blood_monitoring", "bp_monitoring"]


buffers = {
    "blood_monitoring": deque(maxlen=250),
    "bp_monitoring": deque(maxlen=250)
}




def save_to_csv():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for topic, buffer in buffers.items():
        if buffer:
            df = pd.DataFrame(list(buffer))

            
            folder_path = os.path.join("artifacts/kafkaConsumerData", topic)
            os.makedirs(folder_path, exist_ok=True)

            
            filename = f"{topic}.csv"
            filepath = os.path.join(folder_path, filename)

            df.to_csv(filepath, index=False)
            logging.info(f"💾 Saved {len(df)} rows to {filepath}")

   
    Timer(120, save_to_csv).start()  # 300 seconds = 5 minutes


save_to_csv()


consumer = KafkaConsumer(
    *TOPICS,
    bootstrap_servers=KAFKA_BROKER,
    group_id="health_monitoring_group",
    auto_offset_reset="latest",
    key_deserializer=lambda k: k.decode('utf-8'),
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)


try:
    logging.info(f"🚀 Consumer started. Listening to topics: {TOPICS}")
    for msg in consumer:
        topic = msg.topic
        data = msg.value
        buffers[topic].append(data)
        logging.info(f"[{topic}] Buffered message from patient {data.get('patient_id')}")

except KeyboardInterrupt:
    logging.warning("Stopped by user.")
finally:
    consumer.close()
    logging.info("📴 Consumer closed.")
