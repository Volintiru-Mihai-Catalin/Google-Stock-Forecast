import pandas as pd
import numpy as np
import math, quandl
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

forecast_col = 'Adj. Close'
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = preprocessing.scale(np.array(df.drop('label', axis=1)))
y = np.array(df['label'])

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y_lately = y[-forecast_out:]
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

print(accuracy)
for i in range(forecast_out):
	print("The prediction {0} vs the result {1}".format(y_lately[i], int(math.ceil(forecast_set[i]))))
