from scripts.load_data import LoadData
loader=LoadData("Data/train/train.csv")
data_main,data=loader.load_data()
print(data.columns)