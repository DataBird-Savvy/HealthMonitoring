import os
import pandas as pd
import logging

class PatientDataMerger:
    def __init__(self, kafka_path, lab_path, output_file):
        self.kafka_path = kafka_path
        self.lab_path = lab_path
        self.output_file = output_file

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("merge_log.log", mode='w'),
                logging.StreamHandler()
            ]
        )

    def preprocess_date(self, df):
        try:
            print(df['Date'][0])
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
            logging.info("Converted 'Date' column to datetime.")
        except Exception as e:
            logging.error(f"Error in date preprocessing: {e}")
        return df

    def merge(self):
        try:
            logging.info("Starting merge process...")
            blood_file = os.path.join(self.kafka_path, "blood_monitoring", "blood_monitoring.csv")
            bp_file = os.path.join(self.kafka_path, "bp_monitoring", "bp_monitoring.csv")

            if not os.path.exists(blood_file) or not os.path.exists(bp_file):
                logging.error("Missing blood or BP file.")
                return

            try:
                blood_df = pd.read_csv(blood_file)
                bp_df = pd.read_csv(bp_file)
                logging.info("Loaded blood and BP files.")
            except Exception as e:
                logging.error(f"Error reading blood or BP files: {e}")
                return

            blood_df = self.preprocess_date(blood_df)
            bp_df = self.preprocess_date(bp_df)

            try:
                merged_df = blood_df.merge(bp_df, on=["Patient_ID", "Date", "Time"], how="outer")
                logging.info("Merged blood and BP data.")
            except Exception as e:
                logging.error(f"Error merging blood and BP data: {e}")
                return

            all_patients_data = []

            for file_name in os.listdir(self.lab_path):
                lab_path = os.path.join(self.lab_path, file_name)

                if os.path.isfile(lab_path) and file_name.endswith(".csv"):
                    if not os.path.exists(lab_path):
                        logging.warning(f"Lab file missing for: {self.lab_path}")
                        continue

                    try:
                        lab_df = pd.read_csv(lab_path)
                        logging.info(f"Loaded lab file: {file_name}")
                    except Exception as e:
                        logging.error(f"Error reading lab file {file_name}: {e}")
                        continue

                    lab_df = self.preprocess_date(lab_df)
                    print("lab_df", lab_df.shape)

                    try:
                        lab_columns = set(lab_df.columns) - {"Patient_ID", "Date"}
                        existing_columns = lab_columns.intersection(merged_df.columns)
                        lab_df = lab_df.drop(columns=existing_columns, errors="ignore")

                        current_patient_id = lab_df["Patient_ID"].iloc[0]
                        logging.info(f"Processing patient: {current_patient_id}")
                        print("current_patient_id", current_patient_id)
                        patient_data = merged_df[merged_df["Patient_ID"] == current_patient_id].copy()

                        if patient_data.empty:
                            logging.warning(f"No matching patient data found for: {current_patient_id}")
                            continue

                        lab_df_sorted = lab_df.sort_values("Date")
                        patient_data_sorted = patient_data.sort_values("Date")

                        merged_asof = pd.merge_asof(
                            patient_data_sorted,
                            lab_df_sorted,
                            by="Patient_ID",
                            on="Date",
                            direction="backward"
                        )

                        merged_exact = patient_data_sorted.merge(
                            lab_df_sorted,
                            on=["Patient_ID", "Date"],
                            how="left",
                            suffixes=('', '_lab_exact')
                        )

                        print("merged_exact", merged_exact.shape)

                        for col in lab_columns:
                            merged_asof[col] = merged_exact[col].combine_first(merged_asof[col])

                        merged_asof = merged_asof.sort_values(by=["Patient_ID", "Date"])
                        merged_asof[list(lab_columns)] = merged_asof.groupby("Patient_ID")[list(lab_columns)].ffill()

                        print("merged_asof", merged_asof.shape)

                        all_patients_data.append(merged_asof)
                        logging.info(f"Merged lab data for patient: {current_patient_id}")

                    except Exception as e:
                        logging.error(f"Error processing lab data in {file_name}: {e}")
                        continue

            print("len of data", len(all_patients_data))
            if all_patients_data:
                try:
                    final_df = pd.concat(all_patients_data, ignore_index=True)
                    print("final_df", final_df.shape)
                    final_df = final_df.drop("Lab Result", axis=1, errors="ignore")
                    final_df = final_df.sort_values(by=['Date', 'Time']).reset_index(drop=True)

                    cols = ['Patient_ID'] + [col for col in final_df.columns if col != 'Patient_ID']
                    final_df = final_df[cols]

                    print("final_df", final_df.shape)
                    final_df.to_csv(self.output_file, index=False)
                    logging.info(f"Final merged dataset saved at: {self.output_file}")
                except Exception as e:
                    logging.error(f"Error saving final DataFrame: {e}")
            else:
                logging.warning("No patient data to merge.")
        except Exception as e:
            logging.critical(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    kafka_path = "artifacts\\kafkaConsumerData"
    lab_path = "data\\lab_reports"
    output_file = "artifacts\\merged_patient_kafka_data.csv"
    
    merger = PatientDataMerger(kafka_path, lab_path, output_file)
    merger.merge()
