import pandas as pd

data = {'Category': ['Low', 'Medium', 'High', 'Medium', 'Low']}
categories_order = ['Low', 'Medium', 'High']

df = pd.DataFrame(data)
df['Category'] = pd.Categorical(df['Category'], categories=categories_order, ordered=True)

one_hot = pd.get_dummies(df['Category'])
df = pd.concat([df, one_hot], axis=1)

import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

data = np.array(['Low', 'Medium', 'High', 'Medium', 'Low'])
data = data.reshape(-1, 1)
categories_order = [['Low', 'Medium', 'High']]

oridnal_encoder = OrdinalEncoder(categories=categories_order)
ordinal_encoded = oridnal_encoder.fit_transform(data)
print(ordinal_encoded)

one_hot_encoder = OneHotEncoder(categories='auto', sparse_output=True)
one_hot_encoded = one_hot_encoder.fit_transform(data)
print(one_hot_encoded.toarray())



