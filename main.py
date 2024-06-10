from scripts.load_data import LoadData
loader=LoadData("Data/Raw/train/train.csv")
data_main_train,data_train=loader.load_data()
data_cleaned_train=loader.clean_data(data_train)
data_analyze_train=loader.combine_text_columns(data_cleaned_train)
data_analyze_train=loader.push_to_staging(data_analyze_train)
# grouped_data = data.groupby('PRODUCT_TYPE_ID')['PRODUCT_LENGTH'].count()
# print(data.columns)
# print(grouped_data)