"""
Author - Aditya Bhatt 11:50 PM 11-06-2024

Comments:
1. BART is a powerful pre-trained language model that combines the best of both worlds: the contextual 
   understanding of BERT and the language generation capabilities of GPT.

2. Its flexibility and strong performance on various tasks make it a valuable tool for NLP research and development.
"""

import pandas as pd
import logging
from transformers import pipeline
from tqdm import tqdm

class GetSummary:
    def __init__(self, filepath):
        """
        Initialize the GetSummary class with the path to the CSV file.

        Parameters:
        filepath (str): Path to the CSV file containing the data.
        """
        self.filepath = filepath
        self.data = None
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self._setup_logging()
        self.progress_tracker = {
            'load_data': False,
            'summarize_texts': False,
            'save_data': False
        }

    def _setup_logging(self):
        """
        Set up the logging configuration.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _update_progress(self, task):
        """
        Update the progress tracker for the given task.

        Parameters:
        task (str): The name of the task to update progress for.
        """
        self.progress_tracker[task] = True
        self.logger.info(f"Progress update: {task} completed. Current progress: {self.progress_tracker}")

    def load_data(self):
        """
        Load data from the CSV file into a pandas DataFrame.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            self.logger.info(f"Data loaded successfully from {self.filepath}")
            self._update_progress('load_data')
        except Exception as e:
            self.logger.error(f"Failed to load data from {self.filepath}: {e}")
            raise

    def summarize_texts(self):
        """
        Summarize the TEXT_SUMMARY column and add a new column clean_summary.
        """
        if self.data is None:
            self.logger.error("Data not loaded. Please call load_data() first.")
            return

        if 'TEXT_SUMMARY' not in self.data.columns:
            self.logger.error("TEXT_SUMMARY column not found in the data.")
            return

        summaries = []

        # Using tqdm to track progress
        for text in tqdm(self.data['TEXT_SUMMARY'], desc="Summarizing", unit="text"):
            try:
                input_length = len(text.split())
                # Set max_length to half of the input length, with a minimum of 10 and a maximum of 150
                max_length = min(150, max(10, input_length // 2))
                min_length = max(10, max_length // 2)

                summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                self.logger.error(f"Error summarizing text: {e}")
                summaries.append("")

        self.data['clean_summary'] = summaries
        self.logger.info("Summarization completed and clean_summary column added to the DataFrame.")
        self._update_progress('summarize_texts')

    def save_data(self, output_filepath):
        """
        Save the DataFrame with summaries to a new CSV file.

        Parameters:
        output_filepath (str): Path to save the new CSV file.
        """
        try:
            self.data.to_csv(output_filepath, index=False)
            self.logger.info(f"Data with summaries saved successfully to {output_filepath}")
            self._update_progress('save_data')
        except Exception as e:
            self.logger.error(f"Failed to save data to {output_filepath}: {e}")
            raise

    def get_progress(self):
        """
        Return the current progress tracker status.

        Returns:
        dict: A dictionary indicating the completion status of each task.
        """
        return self.progress_tracker

# Example usage
if __name__ == "__main__":
    summarizer = GetSummary('data_staging_batch.csv')  # Replace with your file path
    summarizer.load_data()
    summarizer.summarize_texts()
    summarizer.save_data('output_data_with_summaries.csv')  # Replace with desired output file path
    progress = summarizer.get_progress()
    print("Task progress:", progress)
