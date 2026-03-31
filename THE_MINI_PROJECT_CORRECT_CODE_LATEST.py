# ==============================================================
# 1. IMPORT LIBRARIES
# ==============================================================

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import joblib


# ==============================================================
# 2. LOAD AND MERGE MULTIPLE STOCK FILES
# ==============================================================

folder_path = r"C:\Users\charu\OneDrive\Desktop\Mini Project"

dfs = []
for file in os.listdir(folder_path):
    if file.endswith(".csv") and file != "metadata.csv":
        df = pd.read_csv(os.path.join(folder_path, file))
        df["Stock"] = file.replace(".csv","")
        dfs.append(df)
merged_df = pd.concat(dfs, ignore_index=True)


# ==============================================================
# 3. DATA CLEANING
# ==============================================================

merged_df = merged_df[[
    "Date","Stock","Open","High","Low","Close","Volume"]]

merged_df = merged_df.dropna()
merged_df["Date"] = pd.to_datetime(merged_df["Date"])
merged_df = merged_df.reset_index(drop=True)
merged_df.to_csv("merged_nifty50_cleaned.csv", index=False)
print("Dataset shape:", merged_df.shape)



# ==============================================================
# 4. VISUALIZATION
# ==============================================================

# ---------------------------
# HEATMAP
# ---------------------------

pivot_df = merged_df.pivot_table(
    index="Date",
    columns="Stock",
    values="Close")
plt.figure(figsize=(8,6))
sns.heatmap(pivot_df.corr())
plt.title("Stock Correlation Heatmap")
plt.show()



# ---------------------------
# CANDLESTICK CHART
# ---------------------------

def plot_candle(stock_name):
    df = merged_df[merged_df["Stock"]==stock_name]
    df = df.sort_values("Date")
    df = df.set_index("Date")
    df = df[["Open","High","Low","Close","Volume"]]
    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        volume=True,
        title=stock_name)
plot_candle("ADANIPORTS")
plot_candle("RELIANCE")
plot_candle("ZEEL")



# ---------------------------
# LINE CHART
# ---------------------------

def plot_line(stock):
    df = merged_df[merged_df["Stock"]==stock]
    df = df.sort_values("Date")
    plt.figure()
    plt.plot(df["Date"], df["Close"])
    plt.title(stock + " Price Trend")
    plt.show()
plot_line("BAJFINANCE")
plot_line("HCLTECH")



# ---------------------------
# MOVING AVERAGE
# ---------------------------

stock = "GAIL"
df = merged_df[merged_df["Stock"]==stock]
df = df.sort_values("Date")
df["MA20"] = df["Close"].rolling(20).mean()
plt.figure()
plt.plot(df["Date"], df["Close"], label="Close")
plt.plot(df["Date"], df["MA20"], label="MA20")
plt.legend()
plt.title("Moving Average")
plt.show()



# ---------------------------
# VOLUME ANALYSIS
# ---------------------------

stock = "COALINDIA"
df = merged_df[merged_df["Stock"]==stock]
df = df.sort_values("Date")
plt.figure()
plt.bar(df["Date"], df["Volume"])
plt.title("Trading Volume")
plt.show()


# ---------------------------
# RETURN DISTRIBUTION
# ---------------------------

stock = "INDUSINDBK"
df = merged_df[merged_df["Stock"]==stock]
df = df.sort_values("Date")
df["Return"] = df["Close"].pct_change()
plt.figure()
plt.hist(df["Return"].dropna())
plt.title("Return Distribution")
plt.show()




# ==============================================================
# 5. FEATURE ENGINEERING
# ==============================================================

merged_df = merged_df.sort_values(["Stock","Date"])
merged_df["Return"] = merged_df.groupby("Stock")["Close"].pct_change()
merged_df["MA10"] = merged_df.groupby("Stock")["Close"].transform(
    lambda x: x.rolling(10).mean())
merged_df["MA20"] = merged_df.groupby("Stock")["Close"].transform(
    lambda x: x.rolling(20).mean())
merged_df["EMA10"] = merged_df.groupby("Stock")["Close"].transform(
    lambda x: x.ewm(span=10).mean())
merged_df["Volatility"] = merged_df.groupby("Stock")["Return"].transform(
    lambda x: x.rolling(10).std())
merged_df["Momentum"] = merged_df.groupby("Stock")["Close"].diff()
merged_df["Prev_Close"] = merged_df.groupby("Stock")["Close"].shift(1)
merged_df["Prev_Return"] = merged_df.groupby("Stock")["Return"].shift(1)
merged_df["High_Low_Diff"] = merged_df["High"] - merged_df["Low"]
merged_df["MA_Ratio"] = merged_df["MA10"] / merged_df["MA20"]
merged_df = merged_df.dropna()



# ==============================================================
# 6. SELECT FEATURES
# ==============================================================

features = [
    "Open","High","Low","Volume",
    "Return","MA10","MA20","EMA10",
    "Volatility","Momentum",
    "Prev_Close","Prev_Return",
    "High_Low_Diff","MA_Ratio"]
target = "Close"
X = merged_df[features]
y = merged_df[target]
print("Feature shape:", X.shape)


# ==============================================================
# 7. TRAIN TEST SPLIT
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False)
print("Train shape:", X_train.shape)



# ==============================================================
# 8. RANDOM FOREST MODEL
# ==============================================================

rf_model = RandomForestRegressor(
    n_estimators=80,
    max_depth=12,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)



# ==============================================================
# 9. ARIMA MODEL
# ==============================================================

arima_model = ARIMA(y_train, order=(5,1,0))
arima_result = arima_model.fit()
arima_pred = arima_result.forecast(
    steps=len(y_test))



# ==============================================================
# 10. LSTM MODEL
# ==============================================================

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(
    y.values.reshape(-1,1))
def create_sequences(data, seq_length=10):
    X_seq = []
    y_seq = []
    for i in range(len(data)-seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(data[i+seq_length])
    return np.array(X_seq), np.array(y_seq)
X_lstm, y_lstm = create_sequences(scaled_close)
split = int(0.8 * len(X_lstm))
X_train_lstm = X_lstm[:split]
X_test_lstm = X_lstm[split:]
y_train_lstm = y_lstm[:split]
y_test_lstm = y_lstm[split:]
model = Sequential()
model.add(LSTM(50, return_sequences=True,
               input_shape=(X_train_lstm.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(
    optimizer="adam",
    loss="mse")
model.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=5,
    batch_size=32)
lstm_pred = model.predict(X_test_lstm)
lstm_pred = scaler.inverse_transform(lstm_pred)



# ==============================================================
# 11. METRICS
# ==============================================================

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)
arima_mae = mean_absolute_error(y_test, arima_pred)
arima_rmse = np.sqrt(mean_squared_error(y_test, arima_pred))
arima_r2 = r2_score(y_test, arima_pred)
y_test_lstm_actual = scaler.inverse_transform(y_test_lstm)
lstm_mae = mean_absolute_error(
    y_test_lstm_actual,
    lstm_pred)
lstm_rmse = np.sqrt(mean_squared_error(
    y_test_lstm_actual,
    lstm_pred))
lstm_r2 = r2_score(
    y_test_lstm_actual,
    lstm_pred)



# ==============================================================
# 12. MODEL COMPARISON
# ==============================================================

comparison_df = pd.DataFrame({

    "Model":[
        "Random Forest",
        "ARIMA",
        "LSTM"
    ],

    "MAE":[
        rf_mae,
        arima_mae,
        lstm_mae
    ],

    "RMSE":[
        rf_rmse,
        arima_rmse,
        lstm_rmse
    ],

    "R2":[
        rf_r2,
        arima_r2,
        lstm_r2
    ]

})

comparison_df = comparison_df.round(3)
print(comparison_df)



# ==============================================================
# 13. COMPARISON GRAPHS
# ==============================================================

models = ["RF","ARIMA","LSTM"]
mae_values = [
    rf_mae,
    arima_mae,
    lstm_mae]
rmse_values = [
    rf_rmse,
    arima_rmse,
    lstm_rmse]
plt.figure()
plt.bar(models, mae_values)
plt.title("MAE Comparison")
plt.show()
plt.figure()
plt.bar(models, rmse_values)
plt.title("RMSE Comparison")
plt.show()



# ==============================================================
# 14. ACTUAL VS PREDICTED GRAPH
# ==============================================================

plt.figure(figsize=(12,6))
plt.plot(y_test.values,
         label="Actual")
plt.plot(rf_pred,
         label="RF")
plt.legend()
plt.title("Random Forest Prediction")
plt.show()



# ==============================================================
# 15. FEATURE IMPORTANCE
# ==============================================================

sorted_idx = np.argsort(
    rf_model.feature_importances_)
plt.figure(figsize=(8,6))
plt.barh(
    np.array(features)[sorted_idx],
    rf_model.feature_importances_[sorted_idx])
plt.title("Feature Importance")
plt.show()



# ==============================================================
# 16. CROSS VALIDATION
# ==============================================================

# time series split
tscv = TimeSeriesSplit(n_splits=5)
# R2 score
cv_r2_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=tscv,
    scoring="r2")
# RMSE score
cv_rmse_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=tscv,
    scoring="neg_root_mean_squared_error")
# convert negative values to positive
cv_rmse_scores = -cv_rmse_scores
# create results table
cv_results = pd.DataFrame({
    "Fold": range(1,6),
    "R2 Score": cv_r2_scores,
    "RMSE": cv_rmse_scores})
# round values
cv_results = cv_results.round(4)
print("\nCross Validation Results")
print(cv_results)
# average performance
avg_r2 = np.mean(cv_r2_scores)
avg_rmse = np.mean(cv_rmse_scores)
print("\nAverage R2 Score:", round(avg_r2,4))
print("Average RMSE:", round(avg_rmse,2))



# ==============================================================
# 17. FINAL METRICS AFTER CROSS VALIDATION
# ==============================================================

# train again on training data
rf_model.fit(X_train, y_train)
# predict
rf_cv_pred = rf_model.predict(X_test)
# metrics
rf_cv_mae = mean_absolute_error(y_test, rf_cv_pred)
rf_cv_rmse = np.sqrt(
    mean_squared_error(y_test, rf_cv_pred))
rf_cv_r2 = r2_score(
    y_test,
    rf_cv_pred)
print("\nFINAL RANDOM FOREST METRICS AFTER CV")
print("MAE :", round(rf_cv_mae,3))
print("RMSE :", round(rf_cv_rmse,3))
print("R2 :", round(rf_cv_r2,5))



# ==============================================================
# 18. Pickle file 
# ==============================================================

rf_model = joblib.load("stock_price_detection.pkl")
print("Model loaded successfully")

#to know where it is stored
print(os.getcwd())




