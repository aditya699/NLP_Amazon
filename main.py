from scripts.load_data import LoadData
loader=LoadData("Data/train/train.csv")
data_main,data=loader.load_data()
grouped_data = data.groupby('PRODUCT_TYPE_ID')['PRODUCT_LENGTH'].count()
print(data.columns)
print(grouped_data)