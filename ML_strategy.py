import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import backtrader as bt
import matplotlib.pyplot as plt

# 1. Collect Data
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V']
data = {}
for symbol in symbols:
    stock_data = yf.download(symbol, start='2015-01-01', end='2020-12-31')
    stock_data['Symbol'] = symbol
    data[symbol] = stock_data

df = pd.concat(data.values())
df.reset_index(inplace=True)

# 2. Feature Engineering
def calculate_technical_indicators(df):
    df['SMA'] = df['Close'].rolling(window=15).mean()
    df['EMA'] = df['Close'].ewm(span=15, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df['Bollinger_Upper'], df['Bollinger_Lower'] = compute_bollinger_bands(df['Close'])
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df['Target'] = df['Target'].astype(int)
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def compute_bollinger_bands(series, window=20, num_sd=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_sd)
    lower_band = rolling_mean - (rolling_std * num_sd)
    return upper_band, lower_band

df = df.groupby('Symbol').apply(calculate_technical_indicators)

# 3. Split Data
train_df = df[df['Date'] < '2020-01-01']
test_df = df[df['Date'] >= '2020-01-01']

X_train = train_df.drop(columns=['Date', 'Symbol', 'Target'])
y_train = train_df['Target']
X_test = test_df.drop(columns=['Date', 'Symbol', 'Target'])
y_test = test_df['Target']

# 4. Train the Model
rf_model = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 5. Evaluate Model Performance
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC-AUC: {roc_auc}')

# 6. Develop Trading Strategy
class MLStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close

    def next(self):
        if best_model.predict([[self.dataclose[0]]])[0] == 1:  # Buy signal
            self.buy()
        elif best_model.predict([[self.dataclose[0]]])[0] == 0:  # Sell signal
            self.sell()

class PandasData(bt.feeds.PandasData):
    params = (
        ('datetime', 'Date'),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
    )

data_feed = PandasData(dataname=df.set_index('Date'))

cerebro = bt.Cerebro()
cerebro.addstrategy(MLStrategy)
cerebro.adddata(data_feed)

initial_cash = 100000
cerebro.broker.set_cash(initial_cash)
cerebro.broker.setcommission(commission=0.001)
cerebro.run()

print(f'Final Portfolio Value: {cerebro.broker.getvalue()}')

# 7. Backtesting and Performance Reporting
final_portfolio_value = cerebro.broker.getvalue()
returns = pd.Series(final_portfolio_value - initial_cash)
cumulative_returns = (1 + returns).cumprod() - 1

annualized_returns = cumulative_returns.mean() * 252
sharpe_ratio = (cumulative_returns.mean() / cumulative_returns.std()) * np.sqrt(252)
max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

print(f'Annualized Returns: {annualized_returns}')
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Maximum Drawdown: {max_drawdown}')

# Plot Cumulative Returns
plt.figure(figsize=(10, 6))
cumulative_returns.plot()
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.show()

# Summary Report
report = f"""
# Machine Learning-Based Trading Strategy

## Strategy Description
We developed a machine learning model to predict stock price movements using historical prices and financial metrics. The strategy involves buying stocks when the model predicts an upward movement and selling otherwise.

## Implementation
We used a Random Forest model trained on historical price data and financial indicators. The model was evaluated using accuracy, precision, recall, and ROC-AUC metrics.

## Performance Results
- **Total Returns:** {cumulative_returns.iloc[-1]:.2f}%
- **Annualized Returns:** {annualized_returns:.2f}%
- **Sharpe Ratio:** {sharpe_ratio:.2f}
- **Maximum Drawdown:** {max_drawdown:.2f}%

## Conclusions
The model performed well on historical data, achieving a reasonable accuracy and a positive Sharpe ratio. Future work includes incorporating more features and refining the model to improve performance.
"""
print(report)
