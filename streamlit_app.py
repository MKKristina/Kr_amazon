import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from kr_amazon import evaluate_and_plot

st.title("Прогнозирование цен акций Amazon")

try:
    # Загрузка данных
    data = pd.read_csv('AMZN.csv')
    st.write("Данные загружены успешно.")
    st.dataframe(data.head())

    # Выбор признаков и целевой переменной
    features = ['Open', 'High', 'Low', 'Volume', 'Adj Close']
    target = 'Close'

    # Преобразование колонок в числовой тип и обработка пропущенных значений
    for col in features + [target]:
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        except ValueError as e:
            st.error(f"Ошибка при преобразовании столбца '{col}': {e}")
            st.stop()  # Прекращаем выполнение, если есть ошибка

    data.dropna(subset=features + [target], inplace=True)


    # Разделение данных на тренировочный и тестовый наборы
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Нормализация данных
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    # Преобразование данных для LSTM
    timesteps = 10
    X_train_lstm = []
    y_train_lstm = []
    for i in range(timesteps, len(X_train_scaled)):
        X_train_lstm.append(X_train_scaled[i-timesteps:i])
        y_train_lstm.append(y_train_scaled[i])
    X_train_lstm = np.array(X_train_lstm)
    y_train_lstm = np.array(y_train_lstm)

    X_test_lstm = []
    y_test_lstm = []
    for i in range(timesteps, len(X_test_scaled)):
        X_test_lstm.append(X_test_scaled[i-timesteps:i])
        y_test_lstm.append(y_test_scaled[i])
    X_test_lstm = np.array(X_test_lstm)
    y_test_lstm = np.array(y_test_lstm)

    # Обучение модели LSTM
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, verbose=0)
    predictions_lstm = model_lstm.predict(X_test_lstm)

    # Обучение модели XGBoost 
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model_xgb.fit(X_train_scaled, y_train_scaled.ravel())
    predictions_xgb = model_xgb.predict(X_test_scaled)

    # Прогнозирование и обратное преобразование (изменено для корректного размера)
    y_test_reshaped = y_test_scaled[timesteps:] # Убираем первые timesteps элементов

    # Оценка и визуализация 
    st.subheader("Результаты LSTM")
    rmse_lstm, mae_lstm, r2_lstm, plt_fig_lstm = evaluate_and_plot("LSTM", predictions_lstm, y_test_reshaped, scaler_y)
    if plt_fig_lstm:
        st.pyplot(plt_fig_lstm)
    else:
        st.error("Ошибка при построении графика LSTM.")
    st.write(f"RMSE: {rmse_lstm:.2f}, MAE: {mae_lstm:.2f}, R-squared: {r2_lstm:.2f}")

    st.subheader("Результаты XGBoost")
    rmse_xgb, mae_xgb, r2_xgb, plt_fig_xgb = evaluate_and_plot("XGBoost", predictions_xgb.reshape(-1,1), y_test_scaled, scaler_y)
    if plt_fig_xgb:
        st.pyplot(plt_fig_xgb)
    else:
        st.error("Ошибка при построении графика XGBoost.")
    st.write(f"RMSE: {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}, R-squared: {r2_xgb:.2f}")

except Exception as e:
    st.error(f"Произошла ошибка: {e}")
