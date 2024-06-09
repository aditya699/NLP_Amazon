
'''
Author - Aditya Bhatt 09-06-2024 7:46 AM
Objective - 
1. Write load data module
'''

import pandas as pd
import os
import logging

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

# Set up logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_ingestion.log')  # Only file handler
    ]
)

class LoadData:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """This function will be used to load data into 3 sets data , data_analyze and data_batch

        Raises:
            FileNotFoundError: Check for File Path

        Returns:
            _type_: pd.DataFrame
        """
        
        
        
        if not os.path.exists(self.file_path):
            logging.error(f"File {self.file_path} does not exist.")
            raise FileNotFoundError(f"File {self.file_path} does not exist.")
        
        try:
            data = pd.read_csv(self.file_path)
            data_analyze=data.drop(['PRODUCT_ID'],axis=1)
            data_batch=data_analyze.sample(20)
            data_batch.to_csv("data_batch.csv",index=False)
            logging.info(f"Data loaded successfully from {self.file_path}.")
            logging.info(f"Data Shape: {data.shape}")
            logging.info(f"Data Preview:\n{data.head()}")
            return data,data_analyze,data_batch
        except Exception as e:
            logging.error(f"An error occurred while reading the file: {e}")
            raise

if __name__ == "__main__":
    data_file_path = './Data/train/train.csv'  # Update this path as necessary
    loader = LoadData(data_file_path)
    data = loader.load_data()
    logging.info("Data ingestion completed successfully.")