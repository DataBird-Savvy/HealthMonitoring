import os
import time
import json
import pandas as pd
import logging
from kafka import KafkaProducer
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv


load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPICS = {
    "blood": "blood_monitoring",
    "bp": "bp_monitoring",
}

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    key_serializer=lambda k: k.encode('utf-8'),
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


base_path = 'data'


patient_data = {}
for patient_id in os.listdir(base_path):
    patient_folder = os.path.join(base_path, patient_id)
    if os.path.isdir(patient_folder):
        try:
            patient_data[patient_id] = {
                'blood': pd.read_csv(os.path.join(patient_folder, 'blood_monitoring.csv')),
                'bp': pd.read_csv(os.path.join(patient_folder, 'bp_monitoring.csv')),
            }
            logging.info(f"Loaded data for patient: {patient_id}")
        except Exception as e:
            logging.error(f"Error loading data for patient {patient_id}: {e}")


if not patient_data:
    logging.error("No patient data found. Exiting.")
    exit(1)


num_rows = len(next(iter(patient_data.values()))['blood'])


def stream_patient_data(patient_id, records):
    for i in range(num_rows):
        try:
            timestamp = int(time.time() * 1000)

            
            blood_msg = records['blood'].iloc[i].to_dict()
            blood_msg.update({'patient_id': patient_id, 'timestamp': timestamp})
            producer.send(TOPICS['blood'], key=patient_id, value=blood_msg)

            
            bp_msg = records['bp'].iloc[i].to_dict()
            bp_msg.update({'patient_id': patient_id, 'timestamp': timestamp})
            producer.send(TOPICS['bp'], key=patient_id, value=bp_msg)

            logging.info(f"[{patient_id}] Sent row {i}")
            time.sleep(1) 

        except Exception as e:
            logging.error(f"[{patient_id}] Error at row {i}: {e}")


try:
    with ThreadPoolExecutor(max_workers=len(patient_data)) as executor:
        for pid, records in patient_data.items():
            executor.submit(stream_patient_data, pid, records)
    producer.flush()
    logging.info("All patient data streamed.")

except KeyboardInterrupt:
    logging.warning("Interrupted by user.")
finally:
    producer.close()
