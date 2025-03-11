import os
import time
from logger import logging
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from src.data_merging import PatientDataMerger
from src.data_preprocessing import PatientDataPreprocessor
from exception import HealthMonitoringException
import sys



class PatientMonitoringApp:
    def __init__(self, data_file="artifacts/merged_patient_data.csv", model_dir="artifacts/models", seq_length=5, refresh_interval=120, main_folder="data"):
        try:
            self.data_file = data_file
            self.model_dir = model_dir
            self.seq_length = seq_length
            self.refresh_interval = refresh_interval
            self.main_folder = main_folder
            self.df = None
            self.scaler = None
            self.predictions = {}

            logging.info("PatientMonitoringApp initialized.")


            self.alert_thresholds = {
                "Systolic_BP": (90, 120),
                "Diastolic_BP": (60, 80),
                "Heart Rate (HR)": (60, 100),
                "Blood Glucose Level (mg/dL)": (70, 140),
                "Blood Oxygen (SpO₂)": (95, 100),
                "Body Temperature": (36.1, 37.5),
                "Respiratory Rate (RR)": (12, 20),
                "Hemoglobin": (12, 18),
                "Cholesterol": (125, 200),
                "Platelet Count": (150, 450),
                "WBC Count": (4, 11),
                "RBC Count": (4.7, 6.1),
                "Creatinine": (0.7, 1.3),
                "Urea": (7, 25),
                "Sodium": (135, 145),
                "Potassium": (3.5, 5.1),
                "Calcium": (8.5, 10.2),
                "Electrocardiogram (ECG/EKG)": (0.88, 1.02),
                "Hydration Levels": (55, 65),  
            }
            

            st.set_page_config(page_title="Real-Time Patient Monitoring", layout="wide")
        except Exception as e:
            raise HealthMonitoringException("Error initializing PatientMonitoringApp", sys) from e


    def merge_and_load_data(self):
        """Loads and preprocesses patient data."""
        logging.info("Starting data merging and loading process...")
        try:
            merger = PatientDataMerger(self.main_folder)
            merger.run()
            
            if os.path.exists(self.data_file):
                try:
                    df = pd.read_csv(self.data_file)
                    preprocessor = PatientDataPreprocessor()
                    df, self.scaler = preprocessor.preprocess_data(df)
                    self.df = df
                    logging.info(f"Data successfully loaded from {self.data_file}.")
                except Exception as e:
                    logging.error(f"Error loading patient data: {e}")
                    st.error(f"Error loading patient data: {e}")
            else:
                logging.error(f"Merged data file not found: {self.data_file}")
                st.error("Merged data file not found!")
                st.stop()
        except Exception as e:
            raise HealthMonitoringException("Error initializing PatientMonitoringApp", sys) from e


    def create_sequences(self, data):
        """Creates time-series sequences for LSTM model."""
        try:
            X = []
            for i in range(len(data) - self.seq_length):
                X.append(data[i:i + self.seq_length])
            return np.array(X)
        except Exception as e:
            logging.error(f"Error in sequence creation: {e}")
            raise HealthMonitoringException("Error in creating sequences", sys) from e
            

    def predict_next_row(self):
        """Predicts the next patient vitals using LSTM models."""
        logging.info("Starting predictions...")
        self.predictions = {}

        for patient_id, group in self.df.groupby('Patient_ID'):
            model_path = os.path.join(self.model_dir, f"patient_{patient_id}_lstm_model.keras")

            if not os.path.exists(model_path):
                logging.warning(f"Model file not found for patient {patient_id}: {model_path}")
                continue
            
            try:
                model = load_model(model_path)
                data_values = group.drop(columns=['Patient_ID', 'Timestamp']).values

                if len(data_values) < self.seq_length:
                    logging.warning(f"Not enough data to create sequences for patient {patient_id}.")
                    continue

                X = self.create_sequences(data_values)
                last_sequence = X[-1].reshape(1, self.seq_length, X.shape[2])

                next_row_pred = model.predict(last_sequence)
                next_row_original = self.scaler.inverse_transform(next_row_pred)

                self.predictions[patient_id] = np.round(next_row_original, 2)
                logging.info(f"Prediction successful for patient {patient_id}.")

            except Exception as e:
                logging.error(f"Prediction error for patient {patient_id}: {e}")
                raise HealthMonitoringException("Error in prediction process", sys) from e

    def categorize_alert(self, parameter, value):
        
        """Color-code values based on threshold."""
        
        try:
            
            value = round(float(value), 2)
            if parameter in self.alert_thresholds:
                low, high = self.alert_thresholds[parameter]
                if value < low or value > high:
                    return f"<span style='color:red; font-weight:bold;'>{value} </span>", True
                return f"<span style='color:green; font-weight:bold;'>{value} </span>", False
            return f"<span style='color:gray; font-weight:bold;'>{value}</span>", False
        except Exception as e:
            raise HealthMonitoringException("Error initializing PatientMonitoringApp", sys) from e


    def display_results(self):
        """Displays predictions in Streamlit."""
        st.title("Real-Time Patient Health Monitoring")
        st.write(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write("**Predicted Next Health Metrics for Each Patient**")

        if self.predictions:
            pred_data = []
            for patient_id, pred in self.predictions.items():
                critical_alerts = 0
                formatted_values = []

                for col, val in zip(self.df.columns[2:], pred.flatten()):
                    formatted_value, is_critical = self.categorize_alert(col, val)
                    formatted_values.append(formatted_value)
                    if is_critical:
                        critical_alerts += 1

               
                if critical_alerts > 2:
                    patient_id_cell = f"<span style='color:white; background-color:red; padding:5px; font-weight:bold;'>{patient_id}</span>"
                    logging.warning(f"High alert triggered for patient {patient_id}.")
                else:
                    patient_id_cell = f"<span style='color:black; font-weight:bold;'>{patient_id}</span>"
                
                pred_data.append([patient_id_cell] + formatted_values)

            columns = ['Patient_ID'] + list(self.df.columns[2:])
            pred_df = pd.DataFrame(pred_data, columns=columns)

            st.write(pred_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            logging.warning("No predictions available. Check if patient models exist.")
            st.warning("No predictions available. Ensure patient models exist.")

        time.sleep(self.refresh_interval)  # Wait before refreshing
        st.rerun()

    def run(self):
        """Runs the full pipeline."""
        logging.info("Application started.")
        try:
            
            self.merge_and_load_data()
            self.predict_next_row()
            self.display_results()
            logging.info("Application execution completed.")
        except HealthMonitoringException as e:
            logging.error(f"Application error: {e}")
            st.error(str(e))


if __name__ == "__main__":
    try:
        app = PatientMonitoringApp()
        app.run()
    except HealthMonitoringException as e:
        logging.critical(f"Critical application failure: {e}")