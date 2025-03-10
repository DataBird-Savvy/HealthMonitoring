import os
import pandas as pd
import logging

class PatientDataMerger:
    def __init__(self, main_folder, output_file="artifacts/merged_patient_data.csv", log_file="merge_logs.log", max_rows=50):
        self.main_folder = main_folder
        self.output_file = output_file
        self.log_file = log_file
        self.max_rows = max_rows
        self.all_patients_data = []
        
        logging.basicConfig(filename=self.log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Patient Data Merger initialized.")
    
    def merge_patient_data(self):
        for patient_folder in os.listdir(self.main_folder):
            patient_path = os.path.join(self.main_folder, patient_folder)
            
            if os.path.isdir(patient_path):  
                blood_file = os.path.join(patient_path, "blood_monitoring.csv")
                bp_file = os.path.join(patient_path, "bp_monitoring.csv")
                lab_file = os.path.join(patient_path, "lab_results.csv")

                try:
                    blood_df = pd.read_csv(blood_file)
                    bp_df = pd.read_csv(bp_file)
                    lab_df = pd.read_csv(lab_file)
                    logging.info(f"Successfully loaded data for {patient_folder}.")
                except Exception as e:
                    
                    logging.error(f"Error reading files for {patient_folder}: {e}")
                    print(f"Error reading files for {patient_folder}: {e}")

                
                merged_df = blood_df.merge(bp_df, on=["Patient_ID", "Date", "Time"], how="outer")
                lab_columns = set(lab_df.columns) - {"Patient_ID", "Date"}
                existing_columns = lab_columns.intersection(merged_df.columns)
                lab_df = lab_df.drop(columns=existing_columns, errors="ignore")
                merged_df = merged_df.merge(lab_df, on=["Patient_ID", "Date"], how="outer")
                
                
                
                # Sort by Timestamp in descending order (most recent first)
                merged_df = merged_df.sort_values(by=["Patient_ID", "Date"], ascending=False)
                merged_df[list(existing_columns)] = merged_df[list(existing_columns)].ffill()
                
                # Keep only the most recent max_rows per patient
                merged_df = merged_df.groupby("Patient_ID").head(self.max_rows)
                
                self.all_patients_data.append(merged_df)
    
    def save_merged_data(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
            logging.info(f"Existing file {self.output_file} removed.")
        
        if self.all_patients_data:
            final_df = pd.concat(self.all_patients_data, ignore_index=True)
            final_df = final_df.drop(columns=["Lab Result"], errors="ignore")
            final_df.to_csv(self.output_file, index=False)
            logging.info(f"Final merged dataset saved at: {self.output_file}")
        else:
            logging.warning("No data available to save.")
    
    def run(self):
        self.merge_patient_data()
        self.save_merged_data()

# Example usage
if __name__ == "__main__":
    main_folder = "D:\\EDrivebackup/brocamp\\DataScienceProjects\\bw2_prj\\HealthMonitoring\\data"
    merger = PatientDataMerger(main_folder)
    merger.run()
