import pandas as pd
import numpy as np
import math, quandl, datetime
from statistics import mean
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

pd.options.mode.chained_assignment = None
style.use('ggplot')

def calculate_points(X, m_star, b):
	return X[0] * m_star[0][0] + X[1] * m_star[1][0] + X[2] * m_star[2][0] + X[3] * m_star[3][0] + b

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

X = np.array(df.drop('label', axis=1))
y = np.array(df['label'])

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y_lately = y[-forecast_out:]
y = y[:-forecast_out]



B = [
		[mean(X[:, 0] ** 2), mean(X[:, 0] * X[:, 1]), mean(X[:, 0] * X[:, 2]), mean(X[:, 0] * X[:, 3])], 
		[mean(X[:, 1] * X[:, 0]), mean(X[:, 1] ** 2), mean(X[:, 1] * X[:, 2]), mean(X[:, 1] * X[:, 3])],
		[mean(X[:, 2] * X[:, 0]), mean(X[:, 2] * X[:, 1]), mean(X[:, 2] ** 2), mean(X[:, 2] * X[:, 3])],
		[mean(X[:, 3] * X[:, 0]), mean(X[:, 3] * X[:, 1]), mean(X[:, 3] * X[:, 2]), mean(X[:, 3] ** 2)]
	]

Y = [
		[mean(X[:, 0] * y) - mean(X[:, 0]) * mean(y)],
		[mean(X[:, 1] * y) - mean(X[:, 1]) * mean(y)],
		[mean(X[:, 2] * y) - mean(X[:, 2]) * mean(y)],
		[mean(X[:, 3] * y) - mean(X[:, 3]) * mean(y)]
	]

m_star = np.linalg.solve(B, Y)

b = mean(y) - m_star[0] * mean(X[:, 0]) - m_star[1] * mean(X[:, 1]) - m_star[2] * mean(X[:, 2]) - m_star[3] * mean(X[:, 3])

forecast_set = []

for i in range(len(X_lately)):
	z = calculate_points(X_lately[i], m_star, b)
	forecast_set.append(z[0])
	print("The prediction {0} vs the result {1}".format(y_lately[i], int(math.ceil(z[0]))))


df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()