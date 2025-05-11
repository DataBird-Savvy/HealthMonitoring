import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import streamlit as st
import os
import json

class EvidentlyMonitor:
    def __init__(self, report_dir="artifacts"):
        
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

    def generate_report(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame, title: str):
       
        
        report = Report([DataDriftPreset(drift_share=0.7)])
        report.run(reference_data=baseline_df, current_data=current_df)

        report_file = os.path.join(self.report_dir, f"{title.lower().replace(' ', '_')}.html")
        report.save_html(report_file)
        snapshot_dict = report.as_dict()  
        print(snapshot_dict)
       
        
      


monitor = EvidentlyMonitor()

reference_data=pd.read_csv("data/merged_patient_data.csv")
current_data=pd.read_csv("artifacts/merged_patient_kafka_data.csv")
current_data[['Systolic', 'Diastolic']] = current_data['Blood Pressure'].str.split('/', expand=True).astype(float)
current_data.drop('Blood Pressure', axis=1, inplace=True)
reference_data[['Systolic', 'Diastolic']] = reference_data['Blood Pressure'].str.split('/', expand=True).astype(float)
reference_data.drop('Blood Pressure', axis=1, inplace=True)

reference_data=reference_data.drop(['Date','Time','Patient_ID'],axis=1)
current_data=current_data.drop(['Date','Time','Patient_ID','timestamp_y','timestamp_x','patient_id_y','patient_id_x'],axis=1)
monitor.generate_report(reference_data,current_data,title="Input Feature Drift")