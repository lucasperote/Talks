import streamlit as st
import pandas as pd
from sklearn.externals import joblib

# Carregar o modelo treinado
model = joblib.load('talksmodelo.pkl')

# Título do aplicativo
st.title('Aplicativo de Previsão')

# Widgets para entrada de dados
st.sidebar.header('Parâmetros de Entrada')
r_debt_income = st.sidebar.slider('R_DEBT_INCOME', min_value=0.0, max_value=1.0, value=0.5)
r_utilities_income = st.sidebar.slider('R_UTILITIES_INCOME', min_value=0.0, max_value=1.0, value=0.5)
t_health_12 = st.sidebar.slider('T_HEALTH_12', min_value=0.0, max_value=100.0, value=50.0)

# Preparar dados para fazer a previsão
input_data = pd.DataFrame({
    'R_DEBT_INCOME': [r_debt_income],
    'R_UTILITIES_INCOME': [r_utilities_income],
    'T_HEALTH_12': [t_health_12]
})

# Realizar a previsão usando o modelo carregado
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Mostrar o resultado da previsão
st.subheader('Resultado da Previsão')
if prediction[0] == 0:
    st.write('Não Default')
else:
    st.write('Default')

# Mostrar a probabilidade da previsão
st.subheader('Probabilidade')
st.write(f'Probabilidade de Default: {prediction_proba[0][1]:.2f}')
