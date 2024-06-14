'''
Author - Aditya Bhatt

Objective-
1.This code will be used to move data into curated container post that modelling can start
'''
import pandas as pd
class GetData:
    def __init__(self,file_path):
        self.file_path=file_path

    def get_summary(self):
        data=pd.read_csv(self.file_path)
    
    pass