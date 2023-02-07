import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', 0.01, 88.89, 3.91)
    ZN = st.sidebar.slider('ZN', 0.00, 100.00, 11.36)
    INDUS = st.sidebar.slider('INDUS', 0.46, 27.74, 11.14)
    CHAS = st.sidebar.slider('CHAS', 0.00, 1.00, 0.07)
    NOX = st.sidebar.slider('NOX', 0.39, 0.87, 0.55)
    RM = st.sidebar.slider('RM', 3.56, 87.8, 6.28)
    AGE = st.sidebar.slider('AGE', 2.90, 100.00, 68.57)
    DIS = st.sidebar.slider('DIS', 1.13, 12.13, 3.80)
    RAD = st.sidebar.slider('RAD', 0.00, 100.00, 38.26)
    TAX = st.sidebar.slider('TAX', 0.36, 86.32, 12.23)
    PTRATIO = st.sidebar.slider('PTRATIO', 1.23, 86.23, 45.26)
    B = st.sidebar.slider('B', 4.60, 90.25, 45.36)
    LSTAT = st.sidebar.slider('LSTAT', 0.00, 100.00, 38.36)
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')
st.set_option('deprecation.showPyplotGlobalUse', False)

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
st.set_option('deprecation.showPyplotGlobalUse', False)