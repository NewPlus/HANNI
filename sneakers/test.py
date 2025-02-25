import pandas as pd
from datasets import load_dataset

ds = load_dataset("facebook/natural_reasoning")

df = pd.DataFrame(ds["train"])
print(df.head())
