# To run type in terminal:
# from Anaconda Prompt: conda activate tool_app, then code . and then from here the rest
# python -m streamlit run Main.py or C:\Users\sdv.werkstudent\.conda\envs\tool_app\python.exe -m streamlit run Main.py
import streamlit as st
from streamlit import __main__
from PIL import Image
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define image paths relative to the script directory
image1 = os.path.join(script_dir, "DeVoogt_logo.jpg")
image2 = os.path.join(script_dir, "Feadship_logo.jpg")
image3 = os.path.join(script_dir, "DesignAID_logo.png")

st.image(image1, width=800, use_container_width=False)
col1, col2 = st.columns([1, 1])
with col1:
    st.image(image2, use_container_width=True)
with col2:
    st.image(image3, use_container_width=True)

st.title("PredictionAID App")
#st.write(os.getcwd())

st.markdown("""
Use the sidebar to navigate between the different available tools in the PredictionAID App. The App contains three tools with different purposes. These will be explained below:
- **Prediction: OLS Regression (linear)**: This tool allows you to perform Ordinary Least Squares (OLS) regression analysis on your dataset. You can upload your dataset, select dependent and independent variables, and run the regression to predict outcomes. 
- **Prediction: Random Forest AI (nonlinear)**: This tool allows you to perform Random Forest regression analysis on your dataset. Similar to OLS Regression, you can upload your dataset, select dependent and independent variables, and run the regression to predict outcomes using a more complex model that can capture non-linear relationships.
- **Selection: Variable Importances Tool (linear + nonlinear)**: This tool helps you to explore and evaluate the relationships between different variables in your dataset. You can visualize correlations and other statistical properties of your data. The goal is to filter the most important variables for predictive purposes.
- **Correlation Matrix**: This tool provides a visual representation of the correlation between different variables in your dataset. You can filter the matrix to focus on specific variables and understand their relationships better.""")


# To do:
############
# Inleidend tekstje schrijven
# Correlatie matrix filter automatiseren
# RF: decision tree vaste foto inpluggen for runtime
# Summary result comparison tabel Section 6.
# Manual aanpassen hier en daar
# Eventueel confidence intervals toevoegen aan de metrics voor prediction range
# Woensdag: deployen via Sietse/algemene account

# Later nog over nadenken?
############
# kubus volume kan je niet aanpassen ! Hier zitten limieten aan
# navigeren makkelijker maken, onthouden geschiedenis, Variable Tool in subkopje