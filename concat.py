import pandas as pd
import numpy as np

import glob

path = r'D:\Cemar_dataset_folds_completa' # use your path
all_files = glob.glob(path + "/*TEST*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv("D:\\concat_test.csv",index=False)