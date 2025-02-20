import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2024-01-01')

data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()


data['Daily_Return'] = data['Close'].pct_change()  # Daily percentage change
data['Volatility'] = data['Daily_Return'].rolling(window=10).std()  # 10-day rolling std dev


data.dropna(inplace=True)


X = data[['Close', 'Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'Daily_Return', 'Volatility']]
y = data['Close'].shift(-1)  
X, y = X.iloc[:-1], y.dropna() 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)


mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R² Score: {r2:.4f}')

# Plot of the predictiona
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label='Actual Price', color='blue')
plt.plot(y_test.index, predictions, label='Predicted Price', color='red', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{ticker} - Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()
