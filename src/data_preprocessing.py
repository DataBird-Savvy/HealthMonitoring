import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class PatientDataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def preprocess_data(self, df):
        df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%Y %H.%M.%S')
        df = df.drop(columns=['Date', 'Time','Glucose','Heart Rate'])
        
        col_order = ['Timestamp', 'Patient_ID'] + [col for col in df.columns if col not in ['Timestamp', 'Patient_ID']]
        df = df[col_order]

        if "Blood Pressure" in df.columns:
            df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
            df = df.drop(columns=['Blood Pressure'])
        
        df = df.ffill()
        
        def normalize_group(group):
            df_numeric = group.drop(columns=['Patient_ID', 'Timestamp'])
            group[df_numeric.columns] = self.scaler.fit_transform(df_numeric)
            return group

        df = df.groupby('Patient_ID', group_keys=False).apply(normalize_group)
        return df, self.scaler
