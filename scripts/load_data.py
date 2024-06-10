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
        logging.info(f"Initialized LoadData with file path: {self.file_path}")

    def load_data(self):
        """This function will be used to load data into 3 sets data , data_analyze and data_batch

        Raises:
            FileNotFoundError: Check for File Path

        Returns:
            _type_: pd.DataFrame
        """
        logging.info("Starting data load process.")
        
        if not os.path.exists(self.file_path):
            logging.error(f"File {self.file_path} does not exist.")
            raise FileNotFoundError(f"File {self.file_path} does not exist.")
        
        try:
            logging.info(f"Reading data from {self.file_path}")
            data = pd.read_csv(self.file_path)
            logging.info("Data reading successful.")
            
            # Logging shape and preview
            logging.info(f"Original Data Shape: {data.shape}")
            logging.info(f"Original Data Preview:\n{data.head()}")

            # Data preprocessing
            data_analyze = data.drop(['PRODUCT_ID'], axis=1)
            logging.info("Dropped 'PRODUCT_ID' column.")

            data_batch = data_analyze.sample(20)
            data_batch.to_csv("data_batch.csv", index=False)
            logging.info("Sample batch data saved to 'data_batch.csv'.")
            logging.info(f"Data batch preview:\n{data_batch.head()}")

            return data, data_analyze
        
        except Exception as e:
            logging.error(f"An error occurred while reading the file: {e}")
            raise

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """This function will be used to clean the data

        Args:
            data (pd.DataFrame): Pass the dataframe

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logging.info("Starting data cleaning process.")

        total_rows = len(data)
        missing_val_count = data.isnull().sum()
        missing_val_percentage = (missing_val_count / total_rows) * 100

        logging.info("Percentage of Missing Values per Column:")
        for column, percent in missing_val_percentage.items():
            logging.info(f"{column}: {percent:.2f}%")

        data['TITLE'] = data['TITLE'].fillna("No Title")
        data['BULLET_POINTS'] = data['BULLET_POINTS'].fillna("No Bullet Points")
        data['DESCRIPTION'] = data['DESCRIPTION'].fillna("No Description")
        logging.info("Filled missing values for 'TITLE', 'BULLET_POINTS', and 'DESCRIPTION'.")

        missing_val_count = data.isnull().sum()
        missing_val_percentage = (missing_val_count / total_rows) * 100

        logging.info("Percentage of Missing Values per Column After text operations:")
        for column, percent in missing_val_percentage.items():
            logging.info(f"{column}: {percent:.2f}%")

        logging.info("Data cleaning process completed.")
        return data
    
    def combine_text_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """This function will combine TITLE, BULLET_POINTS, and DESCRIPTION into a single column TEXT_SUMMARY

        Args:
            data (pd.DataFrame): DataFrame with text columns

        Returns:
            pd.DataFrame: DataFrame with combined text column
        """
        logging.info("Starting text columns combination process.")
        if 'TITLE' not in data.columns or 'BULLET_POINTS' not in data.columns or 'DESCRIPTION' not in data.columns:
            logging.error("Missing one or more required text columns for combination.")
            raise ValueError("Missing one or more required text columns for combination.")
        
        data['TEXT_SUMMARY'] = data['TITLE'].astype(str) + '. ' + data['BULLET_POINTS'].astype(str) + '. ' + data['DESCRIPTION'].astype(str)
        logging.info("Combined text columns into 'TEXT_SUMMARY'.")
        
        # Logging preview of the new column
        logging.info(f"Combined Text Summary Preview:\n{data[['TEXT_SUMMARY']].head()}")
        logging.info("Text columns combination process completed.")
        
        return data
    
    def push_to_staging(self, data: pd.DataFrame) -> pd.DataFrame:
        """This function will save the DataFrame to a staging area

        Args:
            data (pd.DataFrame): DataFrame to be saved

        Returns:
            pd.DataFrame: The same DataFrame after saving
        """
        staging_dir = "Data/Staging"
        os.makedirs(staging_dir, exist_ok=True)
        staging_path = os.path.join(staging_dir, "train.csv")
        
        logging.info(f"Starting push to staging area: {staging_path}")
        try:
            data.to_csv(staging_path, index=False)
            logging.info(f"Data successfully pushed to staging area at {staging_path}")

            data_batch = data.sample(20)
            data_batch.to_csv("data_staging_batch.csv", index=False)
            logging.info("Sample batch data saved to 'data_staging_batch.csv'.")
        except Exception as e:
            logging.error(f"Failed to push data to staging area: {e}")
            raise
        
        return data

if __name__ == "__main__":
    data_file_path = './Data/train/train.csv'  # Update this path as necessary
    logging.info(f"Starting data ingestion with file: {data_file_path}")
    loader = LoadData(data_file_path)
    
    try:
        data, data_analyze = loader.load_data()
        logging.info("Data ingestion completed successfully.")
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
    
    try:
        data = loader.combine_text_columns(data_analyze)
        logging.info("Text columns combination completed successfully.")
    except Exception as e:
        logging.error(f"Text columns combination failed: {e}")
    
    try:
        data = loader.push_to_staging(data)
        logging.info("Data push to staging area completed successfully.")
    except Exception as e:
        logging.error(f"Data push to staging area failed: {e}")
