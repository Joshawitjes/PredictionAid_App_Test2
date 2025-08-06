import streamlit as st
#from utils.snowflake_utils import get_snowflake_connection

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Add a page configuration for multi-page navigation
st.set_page_config(page_title="OLS Regression (Linear)")

#if st.button("Test Snowflake Connection"):
    #try:
        #conn = get_snowflake_connection()
        #st.success("Connected to Snowflake!")
        #conn.close()
    #except Exception as e:
        #st.error(f"Connection failed: {e}")

########################################################################
# Page 2: OLS Regression
########################################################################

st.markdown("## 📊 OLS Regression (Linear) for Prediction")
st.markdown("Upload your dataset and predict outcomes using multivariable linear regression.")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (Excel or CSV file):", type=["xlsx", "csv"])
if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        table = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        table = pd.read_csv(uploaded_file)

    with st.expander("Preview of the Dataset"):
            st.dataframe(table)
            len_table = len(table)
            st.write(f"Number of rows: {len_table}")

    # Select target variable
    y_column = st.selectbox("**Select your dependent variable (y)**", table.columns)

    # Ensure the selected dependent variable is numeric
    if not pd.api.types.is_numeric_dtype(table[y_column]):
        st.error(f"The selected dependent variable **{y_column}** contains non-numerical values. Please select a numeric variable.")
        st.stop()

    # Select explanatory variables
    x_columns = st.multiselect("**Select your independent variables (X)**", options=table.columns.drop(y_column))

    # Ensure all selected independent variables are numeric
    non_numeric_columns = [col for col in x_columns if not pd.api.types.is_numeric_dtype(table[col])]
    if non_numeric_columns:
        st.error(f"The following selected independent variables contain non-numerical values: **{', '.join(non_numeric_columns)}**. Please select only numeric variables.")
        st.stop()

    # Drop missing values of x and y
    if y_column and x_columns:
        table = table.dropna(subset=[y_column] + x_columns)
        with st.expander("Preview Cleaned Dataset (without missing values)"):
            st.dataframe(table)
            len_table = len(table)
            st.write(f"Number of rows: {len_table}")
        
        # Prepare data
        y = table[y_column]
        x = table[x_columns]
        x = sm.add_constant(x)

        # Run Regression
        model = sm.OLS(y, x)
        results = model.fit(cov_type="HC0")

        # Display regression summary
        st.subheader("1. Regression Summary")
        st.code(results.summary().as_text())

        summary_df = pd.DataFrame({
            "coef": results.params,
            "std err": results.bse,
            "P>|t|": results.pvalues
        })
        summary_df["Significant"] = summary_df["P>|t|"].apply(lambda p: "Yes" if p < 0.05 else "No")
        st.dataframe(summary_df.style.format(precision=3))

##################
        # Model Performance - User-Friendly Version
        st.subheader("2. How Well Does Your Model Perform?")
        
        # Calculate all metrics
        predicted_values = results.predict(x)
        mse = mean_squared_error(y, predicted_values)
        rmse = np.sqrt(mse)
        residuals = y - predicted_values

        # Handle MAPE calculation with zero values
        def calculate_mape_safe(actual, predicted):
            """Calculate MAPE while handling zero values in actual data"""
            # Create a mask for non-zero actual values
            mask = actual != 0
            
            if mask.sum() == 0:  # All actual values are zero
                return float('inf'), 0, len(actual)
            elif mask.sum() < len(actual):  # Some values are zero
                # Calculate MAPE only for non-zero actual values
                mape_subset = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
                return mape_subset, mask.sum(), len(actual) - mask.sum()
            else:  # No zero values
                mape_all = np.mean(np.abs((actual - predicted) / actual)) * 100
                return mape_all, len(actual), 0

        # Calculate MAPE safely
        mape, valid_count, zero_count = calculate_mape_safe(y, predicted_values)
        
        # Calculate SMAPE (always works)
        smape = np.mean(2 * np.abs(predicted_values - y) / (np.abs(y) + np.abs(predicted_values))) * 100
        
        # Determine which error metric to use and display appropriate message
        if zero_count > 0:
            error_metric = smape
            error_label = "SMAPE"
            #if zero_count == 1:
                #st.info(f"ℹ️ Note: Found {zero_count} zero value in target variable. Using SMAPE ({smape:.2f}%) instead of MAPE for accuracy.")
            #else:
                #st.info(f"ℹ️ Note: Found {zero_count} zero values in target variable. Using SMAPE ({smape:.2f}%) instead of MAPE for accuracy.")
        else:
            error_metric = mape
            error_label = "MAPE"

        # Create an intuitive performance summary box
        st.markdown("""
        <div style="background-color:#e8f4fd; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #1976d2;">
        <h4 style="color:#1976d2; margin-top:0;">📈 Model Quality Summary</h4>
        <p style="margin-bottom:5px;">Your model explains <strong>{:.1f}%</strong> of the variation in your data.</p>
        <p style="margin-bottom:0;">On average, predictions are off by <strong>{:.1f}%</strong> from actual values.</p>
        </div>
        """.format(results.rsquared_adj * 100, mape), unsafe_allow_html=True)
        
        # Main performance metrics in an easy-to-understand layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Model Fit Quality
            st.markdown("### 🎯 **Model Fit Quality**")
            if results.rsquared_adj > 0.8:
                fit_quality = "🟢 Excellent"
                fit_desc = "Your model captures the data patterns very well"
            elif results.rsquared_adj > 0.6:
                fit_quality = "🟡 Good"
                fit_desc = "Your model captures most data patterns"
            elif results.rsquared_adj > 0.4:
                fit_quality = "🟠 Fair"
                fit_desc = "Your model captures some data patterns"
            else:
                fit_quality = "🔴 Poor"
                fit_desc = "Your model struggles to capture data patterns"
            
            st.metric("R² Score", f"{results.rsquared_adj:.3f}", f"{results.rsquared_adj * 100:.1f}%")
            st.markdown(f"**{fit_quality}**")
            st.caption(fit_desc)
            
        with col2:
            # Prediction Accuracy
            st.markdown("### 🔍 **Prediction Accuracy**")
            if mape < 10:
                accuracy_quality = "🟢 Very Accurate"
                accuracy_desc = "Predictions are very close to actual values"
            elif mape < 20:
                accuracy_quality = "🟡 Good Accuracy"
                accuracy_desc = "Predictions are reasonably close to actual values"
            elif mape < 30:
                accuracy_quality = "🟠 Moderate Accuracy"
                accuracy_desc = "Predictions have noticeable errors"
            else:
                accuracy_quality = "🔴 Low Accuracy"
                accuracy_desc = "Predictions have significant errors"
            
            st.metric("Average Error", f"{mape:.1f}%", "Lower is better")
            st.markdown(f"**{accuracy_quality}**")
            st.caption(accuracy_desc)
            
        with col3:
            # Overall Model Reliability
            st.markdown("### ⭐ **Overall Reliability**")
            if results.rsquared_adj > 0.7 and mape < 15:
                overall_quality = "🟢 Highly Reliable"
                overall_desc = "Great for making predictions"
            elif results.rsquared_adj > 0.5 and mape < 25:
                overall_quality = "🟡 Reliable"
                overall_desc = "Good for making predictions"
            elif results.rsquared_adj > 0.3 and mape < 35:
                overall_quality = "🟠 Moderately Reliable"
                overall_desc = "Use predictions with caution"
            else:
                overall_quality = "🔴 Not Reliable"
                overall_desc = "Consider improving the model"
            
            st.metric("Model Grade", overall_quality.split(' ', 1)[1], "")
            st.markdown(f"**{overall_quality}**")
            st.caption(overall_desc)
        
        # Detailed metrics in expandable sections
        st.markdown("---")
        st.markdown("### 📋 **Detailed Performance Metrics**")
        
        col4, col5 = st.columns(2)
        
        with col4:
            with st.expander("🎯 What does R² Score mean?", expanded=False):
                st.markdown(f"""
                **Your R² Score: {results.rsquared_adj:.3f} ({results.rsquared_adj * 100:.1f}%)**
                
                Think of R² as a percentage showing how much of the 'story' your model explains:
                - **100%** = Perfect model (explains everything)
                - **80%+** = Very good model
                - **60-80%** = Good model  
                - **40-60%** = Fair model
                - **Below 40%** = Poor model
                
                Your model explains **{results.rsquared_adj * 100:.1f}%** of why your target variable changes.
                """)
            
            with st.expander("🎯 Technical Metrics (Advanced)", expanded=False):
                st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
                st.write(f"**AIC (Model Complexity):** {results.aic:.1f}")
                st.caption("These are technical measures used to compare different models. Lower values generally indicate better performance.")
                
        with col5:
            with st.expander("🔍 What does Average Error mean?", expanded=False):
                st.markdown(f"""
                **Your Average Error: {mape:.1f}%**
                
                This shows how far off your predictions typically are:
                - **Under 10%** = Very accurate predictions
                - **10-20%** = Good accuracy
                - **20-30%** = Moderate accuracy
                - **Over 30%** = Poor accuracy
                
                On average, your model's predictions are off by **{mape:.1f}%** from the actual values.
                """)
            
            with st.expander("⭐ How to improve your model?", expanded=False):
                if results.rsquared_adj < 0.6 or mape > 20:
                    st.markdown("""
                    **Suggestions to improve your model:**
                    - ✅ Try adding more relevant variables
                    - ✅ Check for outliers in your data
                    - ✅ Consider data transformations
                    - ✅ Ensure you have enough data points
                    - ✅ Try the Random Forest tool for non-linear patterns
                    """)
                else:
                    st.markdown("""
                    **Your model is performing well! To make it even better:**
                    - ✅ Collect more data if possible
                    - ✅ Fine-tune variable selection
                    - ✅ Consider interaction terms between variables
                    """)
        
        # Visual performance indicator
        st.markdown("---")
        st.markdown("### 📊 **Performance Visualization**")
        
        # Create a simple performance gauge
        performance_score = (results.rsquared_adj * 0.6) + ((100 - min(mape, 50)) / 100 * 0.4)
        
        # Performance bar
        if performance_score > 0.8:
            bar_color = "#4CAF50"  # Green
            performance_text = "Excellent Performance"
        elif performance_score > 0.6:
            bar_color = "#FF9800"  # Orange
            performance_text = "Good Performance"
        elif performance_score > 0.4:
            bar_color = "#FFC107"  # Yellow
            performance_text = "Fair Performance"
        else:
            bar_color = "#F44336"  # Red
            performance_text = "Needs Improvement"
        
        st.markdown(f"""
        <div style="background-color: #f0f0f0; border-radius: 10px; padding: 10px; margin: 10px 0;">
            <div style="background-color: {bar_color}; width: {performance_score*100:.0f}%; height: 30px; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {performance_text} ({performance_score*100:.0f}/100)
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

######################

        # Plot actual vs predicted values with confidence bands
        st.subheader("3. Actual vs Predicted Values")
        
        # Get prediction intervals for all data points
        pred_summary_all = results.get_prediction(x)
        pred_intervals_all = pred_summary_all.summary_frame(alpha=0.05)  # 95% confidence
        
        # Create figure in aesthetic style
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Main scatter plot
        scatter = ax.scatter(
            y, predicted_values,
            alpha=0.6,
            edgecolor='black',
            linewidth=0.5,
            s=70,
            c=predicted_values,
            cmap='viridis',
            label='Actual vs Predicted'
        )
        
        # Perfect prediction line
        ax.plot(
            [y.min(), y.max()],
            [y.min(), y.max()],
            'r--',
            lw=2,
            label='Perfect Prediction'
        )
        
        # Add confidence bands
        sorted_indices = np.argsort(predicted_values)
        sorted_pred = predicted_values.iloc[sorted_indices]
        sorted_lower = pred_intervals_all['mean_ci_lower'].iloc[sorted_indices]
        sorted_upper = pred_intervals_all['mean_ci_upper'].iloc[sorted_indices]
        
        ax.fill_between(sorted_pred, sorted_lower, sorted_upper, 
                       alpha=0.2, color='blue', label='95% Confidence Band')
        
        ax.set_title("Actual vs Predicted Values with Confidence Intervals", fontsize=16, weight='bold', pad=15)
        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper left")
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Value Intensity')
        st.pyplot(fig)

        # Add predictions and residuals to the dataset dynamically based on selected y_column
        st.subheader("Predictions and Residuals")
        table[f"Predicted_{y_column}_OLS"] = predicted_values
        table[f"Residual_{y_column}_OLS"] = table[y_column] - table[f"Predicted_{y_column}_OLS"]
        table[f"Residual_{y_column}_OLS_%"] = (table[f"Residual_{y_column}_OLS"] / table[y_column]) * 100
        st.dataframe(table[[f"{y_column}", f"Predicted_{y_column}_OLS", f"Residual_{y_column}_OLS", f"Residual_{y_column}_OLS_%"]])

#######################
# Make your own prediction
#######################

        # Input fields for prediction
        st.header("4. Make a Prediction")
        input_values = {}
        for col in x_columns:
            input_values[col] = st.number_input(f"Enter value for {col}:", value=0.0)

        # Confidence level (fixed at 95%)
        confidence_level = 95

        # Prediction logic
        if st.button("Predict"):
            # Create input array for prediction
            input_array = np.array([[1] + [input_values[col] for col in x_columns]])  # Add constant
            
            # Get prediction and prediction intervals
            prediction = results.predict(input_array)[0]
            pred_summary = results.get_prediction(input_array)
            
            # Calculate prediction intervals
            alpha = 1 - (confidence_level / 100)
            pred_intervals = pred_summary.summary_frame(alpha=alpha)
            
            lower_bound = pred_intervals['mean_ci_lower'].iloc[0]
            upper_bound = pred_intervals['mean_ci_upper'].iloc[0]
            pred_lower = pred_intervals['obs_ci_lower'].iloc[0]
            pred_upper = pred_intervals['obs_ci_upper'].iloc[0]
            
            # Display results
            st.success(f"**Predicted {y_column}: {prediction:.2f}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**{confidence_level}% Confidence Interval:**\n{lower_bound:.2f} to {upper_bound:.2f}")
                st.caption("Range where we expect the true mean to fall")
            
            with col2:
                st.warning(f"**{confidence_level}% Prediction Interval:**\n{pred_lower:.2f} to {pred_upper:.2f}")
                st.caption("Range where we expect individual predictions to fall")
            
            # Reliability assessment
            interval_width = pred_upper - pred_lower
            relative_width = (interval_width / abs(prediction)) * 100 if prediction != 0 else float('inf')
            
            if relative_width < 20:
                reliability = "🟢 High reliability"
            elif relative_width < 40:
                reliability = "🟡 Moderate reliability"
            else:
                reliability = "🔴 Low reliability"
            
            st.markdown(f"**Model Reliability:** {reliability}")
            st.caption(f"Prediction interval width: ±{interval_width/2:.2f} ({relative_width:.1f}% of predicted value)")
            
            st.balloons()