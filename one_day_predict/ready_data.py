import numpy as np
import pandas as pd

kodex200 = pd.read_excel('./data/KODEX200.xlsx',
                        header=0,
                        index_col=0)

print(kodex200.head())

kodex200 = kodex200.sort_values(['일자'],ascending=['True'])
print(kodex200.head())

kodex200 = kodex200.to_numpy()
np.save('./data/kodex200.npy',arr=kodex200)

