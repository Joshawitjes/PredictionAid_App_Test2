# PredictionAID App

A Streamlit-based application for predictive analysis with multiple tools including Variable Selection, OLS Regression and Random Forest AI.

## Quick Start

# Clone the git repository in some new directory
git clone https://github.com/De-Voogt-Naval-Architects/engineering_spec_predictiontool.git

# Navigate to the correct subfolder with the PredictionAid_App
cd engineering_spec_predictiontool/PredictionAid_App

### To actually run the file: 
```bash

# Create a new environment (Python must be installed)
python -m venv venv

# Activate environment
venv\Scripts\activate # should give (venv)

# Install required packages
pip install -r requirements.txt

# Run streamlit app from within the Tool_App folder
python -m streamlit run Home.py
```

## Requirements
Make sure you have Python installed and the required packages:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- streamlit
- pandas
- scikit-learn
- PIL (Pillow)
- Other dependencies as listed in requirements.txt

## Project Structure
```
PredictionAid_App/
├── pages/               # Streamlit pages
│   ├── 1_Tool_for_Variable_Selection_(Investigative).py
│   ├── 2_OLS_Regression_(Linear).py
│   └── 3_Random_Forest_AI_(NonLinear).py
└── utils/               # Utility functions
    ├── __pycache__/
    └── snowflake_utils.py
├── __init__.py
├── DesignAID_logo.png
├── DeVoogt_logo.jpg
├── example_dataset.png
├── Feadship_logo.jpg
├── Home.py              # Main Streamlit application
├── README.md
├── requirements.txt         # Python dependencies  
└── runtime.txt          
```

## Features

### Page 1 **Variable Selection Tool**
- **Multi-Model Comparison** - Linear SVM, Nonlinear SVM (RBF), and Elastic Net
- **Auto Multicollinearity Removal** - Removes variables with |r| > 0.94
- **Interactive Correlation Matrix** - Visual relationship analysis
- **Smart Recommendations** - Determines linear vs nonlinear data patterns

### Page 2 **OLS Regression**
- **Statistical Analysis** - R², MAPE, p-values, confidence intervals
- **Interactive Predictions** - Range-validated inputs with tooltips
- **Visual Diagnostics** - Actual vs predicted with confidence bands

### Page 3 **Random Forest AI**
- **Ensemble Learning** - 100+ trees with 5-fold cross-validation
- **Feature Importance** - Tree-based variable ranking
- **Bootstrap Intervals** - Uncertainty quantification via tree agreement

## Development

The application uses relative paths so it can be run from any location where the repository is cloned. All image and file paths are resolved relative to the script