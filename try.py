import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 股票代碼、預測天數和日期範圍
stock_symbol = "2317.TW"
n_days = 30
start_date = "2020-01-01"
end_date = "2024-06-15"

def prepare_data(stock_symbol, start_date, end_date, n_days):
    # 下載股票數據
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # 去除NaN值
    data = data.dropna()

    # 定義特徵和標籤
    X = pd.DataFrame(data['Close'].shift(1).dropna().values, columns=['Close_Prev'])
    y = data['Close'].shift(-1).dropna()

    # 去除最後一行，以保持特徵和標籤數量一致
    X = X[:-1]
    y = y[:-1]

    return X, y

def predict_future_prices(stock_symbol, start_date, end_date, n_days):
    # 數據準備
    X, y = prepare_data(stock_symbol, start_date, end_date, n_days)

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 建立和訓練線性回歸模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 預測未來N天的收盤價
    last_known_price = X.tail(1).iloc[0, 0]
    future_prices = []
    for i in range(n_days):
        next_pred = model.predict([[last_known_price]])[0]
        future_prices.append(next_pred)
        last_known_price = next_pred

    # 輸出預測結果
    return future_prices

# 預測未來30天的鴻海股票的收盤價
predictions = predict_future_prices(stock_symbol, start_date, end_date, n_days)

print("Future 30 Days Predicted Close Prices:")
for i, pred in enumerate(predictions, 1):
    print(f"Day {i}: {pred}")

