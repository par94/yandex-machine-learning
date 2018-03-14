import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

df_price = pd.read_csv(os.path.join(__location__, 'close_prices.csv'))
df_price['date'] = pd.to_datetime(df_price['date'])
df_price.set_index('date', inplace=True)

pca = PCA(n_components=10)
pca.fit(df_price)

var = 0
n_var = 0
for v in pca.explained_variance_ratio_:
    n_var += 1
    var += v
    if var >= 0.9:
        break
print(n_var)

df_new = pd.DataFrame(pca.transform(df_price))
df_dowj = pd.read_csv(os.path.join(__location__, 'djia_index.csv'))
df_dowj['date'] = pd.to_datetime(df_dowj['date'])
df_dowj.set_index('date', inplace=True)
corr = np.corrcoef(df_new.iloc[:,0],df_dowj.iloc[:,0])
print(corr)

comp0_w = pd.Series(pca.components_[0])
print(comp0_w)
comp0_w_top = comp0_w.sort_values(ascending=False).head(1).index[0]
print(comp0_w_top)
company = df_price.columns[comp0_w_top]
print(company)


