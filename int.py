import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
data = pd.Series([1, 3, 2, 4, 5],['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'])
plt.plot(data)
plt.show()