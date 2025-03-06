import pandas as pd
from datasets import load_dataset
from sneakers import data_filtering


class_data_filter = data_filtering.DataFilter().load_file("data/data.csv")

ds = load_dataset("facebook/natural_reasoning")

df = pd.DataFrame(ds["train"])
print(df.head())

