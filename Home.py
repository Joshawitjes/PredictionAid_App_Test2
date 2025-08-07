# To run type in terminal
# from Anaconda Prompt: conda activate tool_app, then code . and then from here the rest
# python -m streamlit run Home.py or C:\Users\sdv.werkstudent\.conda\envs\tool_app\python.exe -m streamlit run Home.py
import streamlit as st
#from streamlit import __main__
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

st.markdown("""
Use the sidebar to navigate between the different available tools in the PredictionAID App. The App contains several specialized tools for predictive modeling, variable analysis, and correlation exploration:

## 🔍 **Tool for Variable Selection & Correlation Analysis (Investigative)**
This comprehensive tool helps you identify the most important variables for prediction and understand relationships within your data by providing:
- **Correlation Analysis** - Visualize and quantify correlations between variables to detect multicollinearity and key relationships
- **Linear SVM Regression** - Identifies linear relationships between features and target variables
- **Nonlinear SVM Regression (RBF kernel)** - Captures complex, nonlinear patterns in your data  
- **Elastic Net Regression** - Combines Lasso and Ridge penalties for robust variable selection

The tool provides performance comparisons, feature importance rankings, correlation heatmaps, and recommendations for which variables to use in your final predictive models. **Start here** to understand your data and select the best features before moving to the prediction tools.

## 📊 **OLS Regression (Linear)**
Perform Ordinary Least Squares regression analysis for linear relationships. Upload your dataset, select dependent and independent variables, and get:
- Detailed regression coefficients and statistical significance
- Model performance metrics (R², MSE, RMSE)
- Actual vs predicted visualizations
- Individual predictions for new data points

## 🌲 **Random Forest AI (NonLinear)**  
Advanced nonlinear modeling using Random Forest regression. This tool handles complex, non-linear relationships that linear models cannot capture:
- Feature importance rankings based on Random Forest algorithms
- Cross-validation performance metrics
- Robust predictions that work well with complex datasets
- Visual decision tree examples to understand the modeling process

## 💡 **Recommended Workflow**
1. **Start with the Variable Selection (& Correlation Analysis Tool)** to identify the most important features and understand variable relationships in your dataset
2. **Use OLS Regression** if your data shows primarily linear relationships
3. **Use Random Forest** if your data contains complex, nonlinear patterns
4. Compare results between linear and nonlinear approaches to find the best model for your specific use case
""")

