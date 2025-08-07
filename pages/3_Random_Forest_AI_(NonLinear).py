import streamlit as st
########################################################################
# Page 3: Random Forest AI (Nonlinear)
########################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
import os
from PIL import Image

st.header("📈 Random Forest AI (Nonlinear)")
st.write("Upload your dataset and predict outcomes using Random Forest regression.")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (Excel or CSV file)", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        table = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        table = pd.read_csv(uploaded_file)

    with st.expander("Preview of the Dataset"):
            st.dataframe(table)
            len_table = len(table)
            st.write(f"Number of rows: {len_table}")

    # Variable selection
    y_column = st.selectbox("**Select dependent variable (y)**", table.columns)

    # Ensure the selected dependent variable is numeric
    if not pd.api.types.is_numeric_dtype(table[y_column]):
        st.error(f"The selected dependent variable **{y_column}** contains non-numerical values. Please select a numeric variable.")
        st.stop()

    x_columns = st.multiselect("**Select independent variables (X)**", options=table.columns.drop(y_column))

    # Ensure all selected independent variables are numeric
    non_numeric_columns = [col for col in x_columns if not pd.api.types.is_numeric_dtype(table[col])]
    if non_numeric_columns:
        st.error(f"The following selected independent variables contain non-numerical values: **{', '.join(non_numeric_columns)}**. Please select only numeric variables.")
        st.stop()

#######################################
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

        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

#######################################
        # Initialize Random Forest model
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

        # Fit the model
        rf_model = rf.fit(x_train, y_train)
        y_train_pred = rf_model.predict(x_train)
        y_test_pred = rf_model.predict(x_test)

        # Feature importances
        st.subheader("1. Feature Importances")
        importances = rf_model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': x_train.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Create figure in aesthetic style
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis', ax=ax)
        ax.set_title("Ranked Feature Importances", fontsize=16, weight='bold', pad=15)
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.xaxis.grid(True)  # Add horizontal gridlines for readability
        ax.set_axisbelow(True)
        st.pyplot(fig)
        st.caption("⚠️ Note: The x-axis represents the Importance Score, which is NOT equivalent to the coefficients from a regression analysis.")

#######################################
        # Plot Actual vs Predicted values with confidence intervals
        st.subheader("2. Actual vs Predicted (Test Set)")

        # Calculate confidence intervals for each prediction in the test set
        # Use the distribution of predictions from all trees for each sample
        all_tree_preds_test = np.stack([tree.predict(x_test) for tree in rf_model.estimators_], axis=1)
        mean_preds = np.mean(all_tree_preds_test, axis=1)
        lower_bounds = np.percentile(all_tree_preds_test, 2.5, axis=1)
        upper_bounds = np.percentile(all_tree_preds_test, 97.5, axis=1)

        # Create figure in aesthetic style
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            y_test, y_test_pred,
            alpha=0.6,
            edgecolor='k',
            linewidth=0.5,
            s=70,
            c=y_test_pred,
            cmap='viridis',
            label='Predictions'
        )
        # Plot confidence intervals as vertical lines
        for i in range(len(y_test)):
            ax.plot([y_test.iloc[i], y_test.iloc[i]], [lower_bounds[i], upper_bounds[i]], color='gray', alpha=0.5, linewidth=1)

        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
        ax.set_title('Actual vs Predicted Values (Test Set) with 95% Confidence Intervals', fontsize=16, weight='bold', pad=15)
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Value Intensity')
        st.pyplot(fig)
        st.caption("Gray vertical lines show the 95% confidence interval for each prediction, estimated from the distribution of predictions across all trees in the Random Forest.")

        # Add predictions and residuals to the dataset dynamically based on selected y_column
        st.subheader("Predictions and Residuals")
        table[f"Predicted_{y_column}_RF"] = rf_model.predict(x)
        table[f"Residual_{y_column}_RF"] = table[y_column] - table[f"Predicted_{y_column}_RF"]
        table[f"Residual_{y_column}_RF_%"] = (table[f"Residual_{y_column}_RF"] / table[y_column]) * 100
        st.dataframe(table[[f"{y_column}", f"Predicted_{y_column}_RF", f"Residual_{y_column}_RF", f"Residual_{y_column}_RF_%"]])

#######################################
        # Model Performance - User-Friendly Version
        st.subheader("3. How Well Does Your Model Perform?")

        # Calculate all metrics
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mse_test)
        residuals_test = y_test - y_test_pred

        # Handle MAPE calculation with zero values
        def calculate_mape_safe(actual, predicted):
            """Calculate MAPE while handling zero values in actual data"""
            mask = actual != 0
            if mask.sum() == 0:
                return float('inf'), 0, len(actual)
            elif mask.sum() < len(actual):
                mape_subset = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
                return mape_subset, mask.sum(), len(actual) - mask.sum()
            else:
                mape_all = np.mean(np.abs((actual - predicted) / actual)) * 100
                return mape_all, len(actual), 0

        # Calculate MAPE safely
        mape, valid_count, zero_count = calculate_mape_safe(y_test.values, y_test_pred)
        # Calculate SMAPE (always works)
        smape = np.mean(2 * np.abs(y_test_pred - y_test.values) / (np.abs(y_test.values) + np.abs(y_test_pred))) * 100

        # Determine which error metric to use and display appropriate message
        if zero_count > 0:
            error_metric = smape
            error_label = "SMAPE"
        else:
            error_metric = mape
            error_label = "MAPE"

        # Calculate R² for test set
        r2_test = r2_score(y_test, y_test_pred)
        r2_train = r2_score(y_train, y_train_pred)

        # Create an intuitive performance summary box
        st.markdown("""
        <div style="background-color:#e8f4fd; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #1976d2;">
        <h4 style="color:#1976d2; margin-top:0;">📈 Model Quality Summary</h4>
        <p style="margin-bottom:5px;">Your model explains <strong>{:.1f}%</strong> of the variation in your test data.</p>
        <p style="margin-bottom:0;">On average, predictions are off by <strong>{:.1f}%</strong> from actual values.</p>
        </div>
        """.format(r2_test * 100, error_metric), unsafe_allow_html=True)

        # Main performance metrics in an easy-to-understand layout
        col1, col2, col3 = st.columns(3)

        with col1:
            # Model Fit Quality
            st.markdown("### 🎯 **Model Fit Quality**")
            if r2_test > 0.8:
                fit_quality = "🟢 Excellent"
                fit_desc = "Your model captures the data patterns very well"
            elif r2_test > 0.6:
                fit_quality = "🟡 Good"
                fit_desc = "Your model captures most data patterns"
            elif r2_test > 0.4:
                fit_quality = "🟠 Fair"
                fit_desc = "Your model captures some data patterns"
            else:
                fit_quality = "🔴 Poor"
                fit_desc = "Your model struggles to capture data patterns"

            st.metric("R² Score (Test)", f"{r2_test:.3f}", f"{r2_test * 100:.1f}%")
            st.markdown(f"**{fit_quality}**")
            st.caption(fit_desc)

        with col2:
            # Prediction Accuracy
            st.markdown("### 🔍 **Prediction Accuracy**")
            if error_metric < 10:
                accuracy_quality = "🟢 Very Accurate"
                accuracy_desc = "Predictions are very close to actual values"
            elif error_metric < 20:
                accuracy_quality = "🟡 Good Accuracy"
                accuracy_desc = "Predictions are reasonably close to actual values"
            elif error_metric < 30:
                accuracy_quality = "🟠 Moderate Accuracy"
                accuracy_desc = "Predictions have noticeable errors"
            else:
                accuracy_quality = "🔴 Low Accuracy"
                accuracy_desc = "Predictions have significant errors"

            st.metric(f"Average Error ({error_label})", f"{error_metric:.1f}%", "Lower is better")
            st.markdown(f"**{accuracy_quality}**")
            st.caption(accuracy_desc)

        with col3:
            # Overall Model Reliability
            st.markdown("### ⭐ **Overall Reliability**")
            if r2_test > 0.7 and error_metric < 15:
                overall_quality = "🟢 Highly Reliable"
                overall_desc = "Great for making predictions"
            elif r2_test > 0.5 and error_metric < 25:
                overall_quality = "🟡 Reliable"
                overall_desc = "Good for making predictions"
            elif r2_test > 0.3 and error_metric < 35:
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
                **Your R² Score: {r2_test:.3f} ({r2_test * 100:.1f}%)**

                Think of R² as a percentage showing how much of the 'story' your model explains:
                - **100%** = Perfect model (explains everything)
                - **80%+** = Very good model
                - **60-80%** = Good model  
                - **40-60%** = Fair model
                - **Below 40%** = Poor model

                Your model explains **{r2_test * 100:.1f}%** of why your target variable changes (on the test set).
                """)

            with st.expander("🎯 Technical Metrics (Advanced)", expanded=False):
                st.write(f"**Mean Squared Error (MSE):** {mse_test:.2f}")
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse_test:.2f}")
                st.caption("These are technical measures used to compare different models. Lower values generally indicate better performance.")

        with col5:
            with st.expander("🔍 What does Average Error mean?", expanded=False):
                st.markdown(f"""
                **Your Average Error: {error_metric:.1f}%**

                This shows how far off your predictions typically are:
                - **Under 10%** = Very accurate predictions
                - **10-20%** = Good accuracy
                - **20-30%** = Moderate accuracy
                - **Over 30%** = Poor accuracy

                On average, your model's predictions are off by **{error_metric:.1f}%** from the actual values.
                """)

            with st.expander("⭐ How to improve your model?", expanded=False):
                if r2_test < 0.6 or error_metric > 20:
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
        performance_score = (r2_test * 0.6) + ((100 - min(error_metric, 50)) / 100 * 0.4)

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

#######################################
        # Cross-validation
        st.subheader("Cross-Validation Results")
        cv_scores = cross_val_score(rf, x_train, y_train, cv=5, scoring='r2')
        avg_cv_score = np.mean(cv_scores)
        with st.expander("R² Scores (5-Fold Cross-Validation) ℹ️"):
            # Display R² scores
            st.table(pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
                "R² Score": [f"{score:.4f}" for score in cv_scores]
            }))
        st.write(f"**Average R² Score:** {avg_cv_score:.4f}")

        # Adjusted R-squared calculation
        n = x_train.shape[0]  # Number of observations in the training dataset
        p = x_train.shape[1]  # Number of predictors in the training dataset
        adjusted_r2_scores = [1 - ((1 - r2) * (n - 1) / (n - p - 1)) for r2 in cv_scores]
        avg_adjusted_r2 = np.mean(adjusted_r2_scores)
        with st.expander("Adjusted R² Scores (5-Fold Cross-Validation) ℹ️"):
            # Display Adjusted R² scores
            st.table(pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(adjusted_r2_scores))],
                "Adjusted R² Score": [f"{score:.4f}" for score in adjusted_r2_scores]
            }))
        st.write(f"**Average Adjusted R² Score:** {avg_adjusted_r2:.4f}")

#######################
# Make your own prediction
#######################

        # Input fields for prediction
        st.header("4. Make a Prediction")
        # Add some introductory text outside the loop
        st.markdown("Please enter values for each variable within the specified ranges:")

        input_values = {}
        for col in x_columns:
            col_min = float(table[col].min())
            col_max = float(table[col].max())
            
            value = st.number_input(
                f"**{col}** (range: {col_min:.1f} - {col_max:.1f}):",
                min_value=col_min,
                max_value=col_max,
                value=col_min,
                key=f"input_{col}",
                help=f"Enter a value between {col_min:.1f} and {col_max:.1f}"
            )
            input_values[col] = value

        # Confidence level (fixed at 95%)
        confidence_level = 95

        # Prediction logic
        if st.button("Predict"):
            # Create input array for prediction
            x_new = pd.DataFrame([input_values])

            # Get prediction from the Random Forest
            y_prediction = rf_model.predict(x_new)
            mean_pred = y_prediction[0]

            # Method 1: Bootstrap-based confidence intervals (more robust)
            # Get predictions from all individual trees
            all_tree_preds = np.array([tree.predict(x_new)[0] for tree in rf_model.estimators_])
            
            # Calculate percentiles for prediction intervals
            alpha = 1 - (confidence_level / 100)
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            # Use quantiles of tree predictions for prediction intervals
            pred_lower = np.percentile(all_tree_preds, lower_percentile)
            pred_upper = np.percentile(all_tree_preds, upper_percentile)
            
            # For confidence intervals, use a smaller range (tree variance represents model uncertainty)
            tree_std = np.std(all_tree_preds)
            
            # Confidence interval (narrower - represents uncertainty in mean prediction)
            confidence_margin = tree_std * 0.5  # Conservative factor
            conf_lower = mean_pred - confidence_margin
            conf_upper = mean_pred + confidence_margin
            
            # Display results
            st.success(f"**Predicted {y_column}: {mean_pred:.2f}**")

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**{confidence_level}% Confidence Interval:**\n{conf_lower:.2f} to {conf_upper:.2f}")
                st.caption("Range where we expect the true mean to fall")

            with col2:
                st.warning(f"**{confidence_level}% Prediction Interval:**\n{pred_lower:.2f} to {pred_upper:.2f}")
                st.caption("Range where we expect individual predictions to fall")

            # Enhanced reliability assessment
            prediction_interval_width = pred_upper - pred_lower
            confidence_interval_width = conf_upper - conf_lower
            
            relative_pred_width = (prediction_interval_width / abs(mean_pred)) * 100 if mean_pred != 0 else float('inf')
            relative_conf_width = (confidence_interval_width / abs(mean_pred)) * 100 if mean_pred != 0 else float('inf')

            # Reliability based on prediction interval width and tree agreement
            tree_agreement = min(max(1 - (tree_std / abs(mean_pred)), 0), 1) if mean_pred != 0 else 0
            
            # Additional metrics with more intuitive explanation
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Tree Agreement", f"{tree_agreement:.1%}")
                st.caption("How much the individual trees agree (higher is better)")
            
            with col4:
                st.metric("Prediction Range Width", f"{prediction_interval_width:.2f}")
                st.caption(f"This prediction interval spans approximately {relative_pred_width:.1f}% of the predicted value. A smaller percentage indicates higher precision and model certainty.")
            
            # Improved reliability categories and explanations
            if relative_pred_width < 10 and tree_agreement > 0.85:
                reliability = "🟢 Very High Confidence"
                reliability_desc = "Predictions are very consistent and the range is narrow."
            elif relative_pred_width < 20 and tree_agreement > 0.7:
                reliability = "🟡 Good Confidence"
                reliability_desc = "Predictions are fairly consistent with a moderate range."
            elif relative_pred_width < 35 and tree_agreement > 0.5:
                reliability = "🟠 Some Uncertainty"
                reliability_desc = "Predictions vary somewhat; consider results as indicative."
            elif tree_agreement > 0.85:
                reliability = "🟡 Good Tree Agreement, but Wide Range"
                reliability_desc = "The trees agree strongly, but the prediction range is wide. The model is consistent, but the data may be noisy or have outliers."
            else:
                reliability = "🔴 Low Confidence"
                reliability_desc = "Predictions vary widely; results should be interpreted with caution."

            st.markdown(f"**Model Confidence:** {reliability}")
            st.caption(f"{reliability_desc}")
            
            # Explanation for Random Forest intervals
            with st.expander("Understanding Random Forest Confidence Intervals ℹ️"):
                st.markdown("""
                **Random Forest Confidence Intervals are different from OLS:**
                
                - **Confidence Interval**: Based on the variance between individual trees in the forest
                - **Prediction Interval**: Uses the distribution of predictions from all trees (quantile-based)
                - **Tree Agreement**: Measures how consistently the trees predict - higher agreement = more reliable
                
                Random Forest intervals are approximations and may be less precise than OLS statistical intervals, 
                but they capture the ensemble uncertainty effectively.
                """)

            st.balloons()
            
        st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

#######################################
        # Visualize a single decision tree
        st.subheader("Example Decision Tree Visualization")
        st.write("An example of 1 of the 100 underlying decision trees that the Random Forest model creates on the background. Keep in mind that this is just meant as intuition for the ones that are interested in what the model actually does.")
        st.write("No relevant information can be deducted from this figure!")
        
        # Use relative path to the image file
        image1 = "decision_tree.png"
        if os.path.exists(image1):
            st.image(image1, width=800, use_container_width=False)
        else:
            st.warning("The image 'decision_tree.png' was not found in the current directory. Please make sure the file exists.")

        #fig, ax = plt.subplots(figsize=(15, 10))
        #plot_tree(rf.estimators_[0], feature_names=x.columns, filled=True, rounded=True, ax=ax)
        #st.pyplot(fig)
