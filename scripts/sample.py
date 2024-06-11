import pandas as pd

# Assuming your dataset is named 'df'
df = pd.read_csv('Data/Staging/train.csv')

# Calculate the proportion of records to sample from each class
class_proportions = df.groupby('PRODUCT_TYPE_ID').size() / len(df)

# Set the total sample size you want
total_sample_size = 100000

# Calculate the number of records to sample from each class based on the proportion
class_sample_sizes = (class_proportions * total_sample_size).astype(int)

# Sample records from each class based on the calculated sample sizes
sampled_df = df.groupby('PRODUCT_TYPE_ID', group_keys=False).apply(lambda x: x.sample(n=class_sample_sizes[x.name]))

# Reset index
sampled_df.reset_index(drop=True, inplace=True)

# Save sampled_df to a CSV file
sampled_df.to_csv('Data/Staging/sampled_train_data.csv', index=False)

# Now sampled_df contains a sample of records from each class, and it's saved to 'sampled_data.csv'
