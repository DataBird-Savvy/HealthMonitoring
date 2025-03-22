import os
import pandas as pd

def merge_patient_data(main_folder, output_file):
    all_patients_data = []

    for patient_folder in os.listdir(main_folder):
        patient_path = os.path.join(main_folder, patient_folder)

        if os.path.isdir(patient_path):
            blood_file = os.path.join(patient_path, "blood_monitoring.csv")
            bp_file = os.path.join(patient_path, "bp_monitoring.csv")
            lab_file = os.path.join(patient_path, "lab_results.csv")

            # Read CSV Files
            blood_df = pd.read_csv(blood_file)
            bp_df = pd.read_csv(bp_file)
            lab_df = pd.read_csv(lab_file)

            # Merge Blood & BP
            merged_df = blood_df.merge(bp_df, on=["Patient_ID", "Date", "Time"], how="outer")

            # Drop existing lab columns
            lab_columns = set(lab_df.columns) - {"Patient_ID", "Date"}
            existing_columns = lab_columns.intersection(merged_df.columns)
            lab_df = lab_df.drop(columns=existing_columns, errors="ignore")

            # Merge Lab Data
            merged_df = merged_df.merge(lab_df, on=["Patient_ID", "Date"], how="outer")

            # Sort & Forward Fill Missing Values
            merged_df = merged_df.sort_values(by=["Patient_ID", "Date"], ascending=False)
            merged_df[list(existing_columns)] = merged_df[list(existing_columns)].ffill()

            all_patients_data.append(merged_df)

 
    final_df = pd.concat(all_patients_data, ignore_index=True)
    final_df = final_df.drop("Lab Result", axis=1, errors="ignore")
    final_df = final_df.sort_values(by=['Date', 'Time']).reset_index(drop=True)
    final_df = final_df.ffill()

    
    cols = ['Patient_ID'] + [col for col in final_df.columns if col != 'Patient_ID']
    final_df = final_df[cols]


    final_df = final_df.dropna(axis=0)
    final_df.to_csv(output_file, index=False)
    print(f"✅ Final merged dataset saved at: {output_file}")
