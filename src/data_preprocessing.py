import pandas as pd
import numpy as np
import joblib

class DataPreprocessor:
    """Class for preprocessing patient health data."""

    def __init__(self, file_path, encoder_path, scaler_path):
        self.file_path = file_path
        self.encoder_path = encoder_path
        self.scaler_path = scaler_path
        self.df = None  

    def load_data(self):
        """Load CSV data into a DataFrame."""
        self.df = pd.read_csv(self.file_path)
        self.df=self.df.drop(['patient_id_x','patient_id_y','timestamp_x','timestamp_y'],axis=1)
        return self.df

    def process_timestamp(self):
        """Convert Date & Time columns into a Timestamp and drop original columns."""
        if 'Date' in self.df.columns and 'Time' in self.df.columns:
            self.df['Timestamp'] = pd.to_datetime(
                self.df['Date'] + ' ' + self.df['Time'], format='%d-%m-%Y %H.%M.%S',errors="coerce"
            )
            self.df.drop(columns=['Date', 'Time'], inplace=True)

    def split_blood_pressure(self):
        """Split Blood Pressure column into Systolic and Diastolic BP."""
        if 'Blood Pressure' in self.df.columns:
            self.df[['Systolic_BP', 'Diastolic_BP']] = self.df['Blood Pressure'].str.split('/', expand=True).astype(float)
            self.df.drop(columns=['Blood Pressure'], inplace=True)

    def encode_patient_id(self):
        """Encode Patient_ID column."""
        encoder = joblib.load(self.encoder_path)
        self.df['Patient_ID'] = encoder.fit_transform(self.df['Patient_ID'])

    def scale_features(self):
        """Apply feature scaling to numerical data (excluding Patient_ID)."""
        scaler = joblib.load(self.scaler_path)
        self.df = self.df.astype(np.float64)
        self.df_scaled = self.df.copy()
        self.df_scaled.iloc[:, 1:] = scaler.fit_transform(self.df_scaled.iloc[:, 1:])

    def preprocess(self):
        """Perform all preprocessing steps and return the processed DataFrame."""
        self.load_data()
        self.process_timestamp()
        self.split_blood_pressure()
        self.encode_patient_id()
        self.df.drop(columns=['Timestamp'], errors='ignore', inplace=True)
        self.scale_features()
        return self.df_scaled

if __name__=="__main__":
    
    merged_data_path="artifacts/merged_patient_kafka_data.csv"
    
    preprocessor = DataPreprocessor(merged_data_path, "artifacts/encoder.pkl", "artifacts/scaler.pkl")
    df_scaled = preprocessor.preprocess()
    