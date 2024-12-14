import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_and_plot(model_name, predictions, y_test, scaler_y):
    """Оценивает модель и строит график."""
    try:
        if len(predictions) != len(y_test):
            st.error(f"Несовпадение размеров массивов: predictions ({len(predictions)}) и y_test ({len(y_test)}).")
            return

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        st.write(f'{model_name}:')
        st.write(f'RMSE: {rmse:.2f}')
        st.write(f'MAE: {mae:.2f}')
        st.write(f'R-squared: {r2:.2f}')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_test, label='Фактические значения')
        ax.plot(predictions, label='Прогнозы')
        ax.set_xlabel('Время')
        ax.set_ylabel('Цена закрытия')
        ax.set_title(f'Фактические значения vs. Прогнозы ({model_name})')
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Произошла ошибка при оценке модели: {e}")
