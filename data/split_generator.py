import pandas as pd
import numpy as np

CSV_PATH = 'data/1024/split.csv'

df = pd.read_csv(CSV_PATH)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n_total = len(df)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.15)

df.loc[:n_train-1, 'split'] = 'train'
df.loc[n_train:n_train+n_val-1, 'split'] = 'val'
df.loc[n_train+n_val:, 'split'] = 'test'

# Ghi đè csv
df.to_csv(CSV_PATH, index=False)

print(f"✅ Split completed: {df['split'].value_counts().to_dict()}")