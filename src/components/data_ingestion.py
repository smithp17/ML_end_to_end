import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Define the path where logs should be saved
LOG_DIR = r"C:\Users\Smith\OneDrive\Desktop\MLend-to-end\logs"
LOG_FILE_PATH = os.path.join(LOG_DIR, "data_ingestion.log")

# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging to output to both console and a log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

# Custom exception class
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = CustomException.get_detailed_error_message(error_message=error_message,
                                                                        error_detail=error_detail)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):
        _, _, exec_tb = error_detail.exc_info()
        line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename
        error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{error_message}]"
        return error_message

    def __str__(self):
        return self.error_message

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', "train.csv")
    test_data_path: str = os.path.join('artifact', "test.csv")
    raw_data_path: str = os.path.join('artifact', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        print("Entered data ingestion method")
        logging.info("Enter the data ingestion method or component")
        try:
            # Step 1: Verify if the CSV file exists
            file_path = 'notebook/data/stud.csv'
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist!")
                logging.error(f"File {file_path} does not exist!")
                return
            else:
                print(f"File {file_path} found!")

            # Step 2: Read the CSV file
            print("Reading CSV file")
            df = pd.read_csv(file_path)
            print(f"DataFrame loaded with shape: {df.shape}")
            logging.info('Exported dataset as dataframe')

            # Step 3: Create directories for artifacts
            print("Creating directories for artifacts")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Step 4: Split the data into training and testing sets
            logging.info("Train test split initiated")
            print("Splitting dataset into train and test")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Step 5: Log completion
            logging.info("Ingestion of data is complete")
            print("Ingestion complete, returning paths")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
