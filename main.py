import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from logger import logging
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from src.data_preprocessing import DataPreprocessor

# Load environment variables
load_dotenv()



class PatientHealthMonitor:
    def __init__(self):
        """Initialize paths, load models, and preprocess data."""
        logging.info("Initializing PatientHealthMonitor...")

        self.merged_data_path = os.getenv("MERGED_DATA_PATH")
        self.encoder_path = os.getenv("ENCODER_PATH")
        self.scaler_path = os.getenv("SCALER_PATH")
        self.model_path = os.getenv("MODEL_PATH")

        try:
            preprocessor = DataPreprocessor(self.merged_data_path, self.encoder_path, self.scaler_path)
            self.df_scaled = preprocessor.preprocess()
            logging.info("Data preprocessing completed successfully.")

            self.model = load_model(self.model_path)
            logging.info("Model loaded successfully.")

            self.encoder = joblib.load(self.encoder_path)
            self.scaler = joblib.load(self.scaler_path)
            logging.info("Encoder and Scaler loaded successfully.")

        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            raise e  # Stop execution if model loading fails

        self.all_feature_names = list(self.df_scaled.columns[1:])  # Exclude 'Patient_ID'

        self.critical_thresholds = {
            "Blood Glucose Level (mg/dL)": (70, 140),
            "Blood Oxygen (SpO₂)": (90, 100),
            "Electrocardiogram (ECG/EKG)": (0.5, 1.5),
            "Hydration Levels": (50, 70),
            "Heart Rate (HR)": (50, 110),
            "Respiratory Rate (RR)": (12, 20),
            "Body Temperature": (36.1, 37.8),
            "Hemoglobin": (12, 18),
            "Glucose": (70, 140),
            "Cholesterol": (100, 200),
            "Platelet Count": (150000, 450000),
            "WBC Count": (4000, 11000),
            "RBC Count": (4.7, 6.1),
            "Creatinine": (0.7, 1.3),
            "Urea": (7, 20),
            "Sodium": (135, 145),
            "Potassium": (3.5, 5.1),
            "Calcium": (8.5, 10.5),
            "Systolic_BP": (90, 140),
            "Diastolic_BP": (60, 90),
        }

    def make_predictions(self):
        """Generate predictions for each patient and detect critical alerts."""
        logging.info("Generating predictions...")
        predictions = {}
        alerts = {}

        for patient_id in self.df_scaled['Patient_ID'].unique():
            patient_data = self.df_scaled[self.df_scaled['Patient_ID'] == patient_id].iloc[:, 1:].values  

            if len(patient_data) < 30:
                logging.warning(f"Patient {patient_id} has insufficient data (<30 readings), skipping...")
                continue  

            last_sequence = patient_data[-30:].reshape(1, 30, -1)
            next_row_prediction = self.model.predict(last_sequence)

            if next_row_prediction.shape[1] != self.scaler.scale_.shape[0]:
                logging.warning(f"Feature mismatch for patient {patient_id}, skipping...")
                continue  

            next_row_original = self.scaler.inverse_transform(next_row_prediction)
            predictions[patient_id] = np.round(next_row_original, 2)

            alert_messages = []
            for i, feature in enumerate(self.all_feature_names):
                if feature in self.critical_thresholds:
                    min_val, max_val = self.critical_thresholds[feature]
                    feature_value = predictions[patient_id][0, i]
                    if feature_value < min_val or feature_value > max_val:
                        alert_msg = f"⚠️ {feature} is out of range: {feature_value:.2f}"
                        alert_messages.append(alert_msg)
                        logging.warning(f"ALERT for Patient {patient_id}: {alert_msg}")

            if alert_messages:
                alerts[patient_id] = alert_messages

        return predictions, alerts

    def highlight_abnormal_values(self, val, feature_name):
        """Highlight values that are outside the normal range."""
        if feature_name in self.critical_thresholds:
            min_val, max_val = self.critical_thresholds[feature_name]
            if val < min_val or val > max_val:
                return "background-color: red; color: white;"  # Highlight in red
        return ""

    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        logging.info("Starting Streamlit dashboard...")
        st.set_page_config(page_title="Patient Health Monitoring", layout="wide")
        st.title("🏥 Patient Health Monitoring Dashboard")

        predictions, alerts = self.make_predictions()

        for patient_id, pred_values in predictions.items():
            decoded_patient_id = self.encoder.inverse_transform([int(patient_id)])[0]

            st.subheader(f"🔮 Predicted Readings for Patient {decoded_patient_id}")

            pred_df = pd.DataFrame(pred_values, columns=self.all_feature_names)
            styled_pred_df = pred_df.style.apply(lambda col: col.apply(self.highlight_abnormal_values, args=(col.name,)), axis=0)

            st.dataframe(styled_pred_df.format(precision=2))

            if patient_id in alerts:
                st.error("⚠️ **Critical Alerts Detected!**")
                for alert in alerts[patient_id]:
                    st.write(alert)

            st.write("---")  

        logging.info("Dashboard rendered successfully.")


if __name__ == "__main__":
    monitor = PatientHealthMonitor()
    monitor.run_dashboard()
    time.sleep(600)  # Wait for 10 minutes
    st.rerun()
