import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#  Load your datasets
df_before = pd.read_csv("merged_PRSA_data.csv")
df_after = pd.read_csv("merged_cleaned_PRSA_data.csv")

# App Layout
st.set_page_config(page_title="Air Quality Analysis", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Data Analysis (EDA)", "Modeling & Prediction"])

#  Data Overview
if page == "Data Overview":

    st.title("Data Overview")

    st.markdown("""
    Welcome to the Data Overview section!  
    This page shows the structure, issues, and cleaning of air quality data collected from Beijing.  
    We aim to ensure the data is accurate and ready for analysis and modeling!
    """)

    st.info(
        "Use the tabs below to view the dataset before and after cleaning. Navigate between Shape, Missing Data, "
        "and Visual Analysis.")

    # Top Level Tabs
    tab1, tab2 = st.tabs(["Data Before Cleaning", "Data After Cleaning"])

    with tab1:
        st.header("Before Cleaning")
        option = st.radio("Select what to view:",
                          ("Shape", "Column Names and Data Types", "Sample of Data", "Missing Values",
                           "Visualize Missing Data"),
                          key="before_cleaning_tab")

        if option == "Shape":
            st.subheader("Dataset Shape")
            st.markdown("Total number of rows (records) and columns (features) **before cleaning**.")
            st.write(df_before.shape)

        elif option == "Column Names and Data Types":
            st.subheader("Columns and Data Types")
            st.markdown("Each column name and the type of data it holds (e.g., integers, floats, text).")
            st.write(df_before.dtypes)

        elif option == "Sample of Data":
            st.subheader("Sample Data")
            st.markdown("First few rows of the dataset to understand its structure.")
            st.dataframe(df_before.head())

        elif option == "Missing Values":
            st.subheader("Missing Values")
            st.markdown("Count of missing (null) values present in each column.")
            st.write(df_before.isnull().sum())

        elif option == "Visualize Missing Data":
            st.subheader("Missing Data Visualization")
            st.markdown("White lines indicate missing data points across the dataset.")
            fig, ax = plt.subplots(figsize=(10, 6))
            msno.matrix(df_before, ax=ax)
            st.pyplot(fig)

    with tab2:
        st.header("After Cleaning")
        option2 = st.radio("Select what to view:",
                           ("Shape", "Missing Values", "Visualize Missing Data"),
                           key="after_cleaning_tab")

        if option2 == "Shape":
            st.subheader("Dataset Shape After Cleaning")
            st.markdown("""
            New number of rows and columns after handling missing and unreliable data.

            Cleaning steps taken:
            - Dropped rows missing critical air pollutant data.
            - Filled missing weather measurements with median values.
            """)
            st.write(df_after.shape)

        elif option2 == "Missing Values":
            st.subheader("Missing Values After Cleaning")
            st.markdown("Confirming that no missing values are left after the cleaning process.")
            st.write(df_after.isnull().sum())

        elif option2 == "Visualize Missing Data":
            st.subheader("Missing Data Visualization After Cleaning")
            st.markdown("A complete matrix confirms successful cleaning (no white gaps).")
            fig, ax = plt.subplots(figsize=(10, 6))
            msno.matrix(df_after, ax=ax)
            st.pyplot(fig)


# Exploratory Data Analysis Page

elif page == "Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)")

    st.markdown("""
    This section provides a detailed exploration of the cleaned air quality dataset to identify patterns, trends, and relationships among key pollutants and weather variables.
    Together, these visualizations support a deeper understanding of the data before modeling.
    """)

    st.info(
        "Use the tabs below to view the different forms of data. You can select more than one on each tab to allow an "
        "easier comparison between data.")

    tab1, tab2, tab3, tab4 = st.tabs(["Histograms", "Boxplots", "Correlation Heatmap", "Scatterplots"])

    # Load Cleaned Dataset
    df = pd.read_csv('merged_cleaned_PRSA_data.csv')

    relevant_columns = [
        'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
        'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM'
    ]

    with tab1:
        st.header("Histograms")

        st.markdown("""Histograms show how values are distributed for each variable. They help us understand if the 
        data is skewed, normal, or has extreme values. This is useful for spotting patterns like pollution spikes or 
        steady conditions. """)

        selected_hist = st.multiselect("Select variables to display histograms:", relevant_columns)

        for col in selected_hist:
            st.subheader(f"Histogram of {col}")
            fig, ax = plt.subplots()
            df[col].hist(bins=30, color='skyblue', edgecolor='black', ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

    with tab2:
        st.header("Boxplots")

        st.markdown("""These boxplots are used to identify outliers and show how values are spread across each 
        variable. The box shows the middle 50% of the data, the line inside is the median, and the dots represent 
        unusual values, or outliers. This helps spot variables with high variability or extreme pollution events. """)

        selected_box = st.multiselect("Select variables to display boxplots:", relevant_columns)

        for col in selected_box:
            st.subheader(f"Boxplot of {col}")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=col, ax=ax)
            ax.set_xlabel(col)
            st.pyplot(fig)

    with tab3:
        st.header("Correlation Heatmap")

        st.markdown("""
        This heatmap shows relationships between different variables.
        Strong positive correlations are in dark red, strong negative correlations in dark blue.
        """)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df[relevant_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

        st.markdown("""
        **Insight:** PM2.5 strongly correlates with PM10 and CO, suggesting similar sources (e.g., traffic, industry).
        Temperature negatively correlates with pollution, likely due to weather patterns affecting dispersion.
        """)

    with tab4:
        st.header("Scatterplots")

        st.markdown("""Scatterplots show the relationship between two variables. Each point represents one record, 
        and the red line shows the trend. These plots help identify correlations or patterns between pollutants and 
        conditions. The chosen scatterplots were based on the highest correlation in the previous heatmap.""")

        scatter_options = {
            "PM2.5 vs PM10": ('PM2.5', 'PM10'),
            "PM2.5 vs CO": ('PM2.5', 'CO'),
            "CO vs NO2": ('CO', 'NO2'),
            "CO vs PM10": ('CO', 'PM10')
        }

        selected_scatters = st.multiselect("Select scatterplots to display:", scatter_options.keys())

        for title, (x, y) in scatter_options.items():
            if title in selected_scatters:
                st.subheader(f"Scatterplot: {title}")
                fig, ax = plt.subplots()
                sns.regplot(x=x, y=y, data=df, scatter_kws={'s':20}, line_kws={'color':'red'}, ax=ax)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                st.pyplot(fig)

                if title == "PM2.5 vs PM10":
                    st.markdown("*Strong positive relationship — both rise together due to similar sources.*")
                elif title == "PM2.5 vs CO":
                    st.markdown("*Moderate positive — CO pollution and PM2.5 often from vehicles/combustion.*")
                elif title == "CO vs NO2":
                    st.markdown("*Moderate positive — suggests CO and NO2 share sources like traffic emissions.*")
                elif title == "CO vs PM10":
                    st.markdown("*Moderate relationship — particulate and CO emissions often overlap.*")


# Modeling & Prediction
elif page == "Modeling & Prediction":
    st.title("Modeling & Prediction")

    # Relevant Features and Target
    features = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    target = 'PM2.5'

    X = df_after[features]
    y = df_after[target]

    # Train/Test Splits
    splits = {
        "80/20": train_test_split(X, y, test_size=0.2, random_state=42),
        "70/30": train_test_split(X, y, test_size=0.3, random_state=42),
        "90/10": train_test_split(X, y, test_size=0.1, random_state=42)
    }


    # Modeling Functions
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return {
            "R² Score": round(r2_score(y_test, preds), 2),
            "MAE": round(mean_absolute_error(y_test, preds), 2),
            "MSE": round(mean_squared_error(y_test, preds), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2)
        }


    # Streamlit Page
    st.markdown("""
    This section focuses on using machine learning to **predict PM2.5 levels** based on other environmental and pollutant features in the dataset.

    It includes three parts:

    - **Model Options:**  
      Outlines the four machine learning models used (Linear Regression, Decision Tree, Random Forest, and K-Nearest Neighbors), including how they work and why they are relevant.

    - **Metrics:**  
      Displays performance results from each model based on different train/test data splits. Metrics include **R²**, **MAE**, and **RMSE**, helping us evaluate model accuracy.

    - **Prediction:**  
      This allows the user to input their own data into the model to get a prediction of **PM2.5**.

    These tools aim to support **air quality forecasting** to help mitigate the effects of pollution on public health and the environment.
    """)

    st.info(
        "Use the tabs below to switch between the model overview, metrics and prediction using a pre-trained model.")

    logic_tab, metrics_tab, prediction_tab = st.tabs([
        "Model Options", "Model Metrics", "Prediction"])


    #  Model Options Tab
    with logic_tab:
        st.header("Model Types")
        model_choice = st.selectbox("Choose a model to learn more about it:",
                                    ["Linear Regression", "Decision Tree", "Random Forest", "KNN"])

        if model_choice == "Linear Regression":
            st.subheader("Linear Regression")
            st.markdown("""
            **How it works:**  
            Linear Regression is a statistical model that predicts a continuous target variable based on the weighted sum of input features.
            It tries to find the best-fitting straight line (or hyperplane) that minimizes the difference between actual and predicted values (using Least Squares).

            **Use Case in Air Pollution:**  
            Helps identify basic linear trends, such as how PM2.5 might increase as CO increases.

            **Pros:**  
            - Simple and fast to train  
            - Easy to interpret  
            - Good baseline model

            **Cons:**  
            - Assumes linear relationships only  
            - Sensitive to outliers  
            - Can underperform with complex or non-linear data
            """)

        elif model_choice == "Decision Tree":
            st.subheader("Decision Tree Regressor")
            st.markdown("""
            **How it works:**  
            Decision Trees split the dataset into smaller subsets based on feature values, using thresholds (e.g., CO < 0.3). 
            At each split, the algorithm chooses the feature that best reduces prediction error (usually by minimizing Mean Squared Error).

            **Use Case in Air Pollution:**  
            Captures how certain pollution patterns emerge from combinations of conditions (e.g., low wind and high CO → high PM2.5).

            **Pros:**  
            - Handles non-linear relationships well  
            - Intuitive and easy to visualize  
            - Requires little data preparation

            **Cons:**  
            - Prone to overfitting on noisy data  
            - Can be unstable (small changes in data → different trees)  
            - Lower accuracy compared to ensemble models
            """)

        elif model_choice == "Random Forest":
            st.subheader("Random Forest Regressor")
            st.markdown("""
            **How it works:**  
            Random Forest builds multiple Decision Trees on random subsets of the data and features, then averages their predictions.
            This reduces variance and helps generalize better to new, unseen data.

            **Use Case in Air Pollution:**  
            Great for capturing complex interactions between features that affect PM2.5 levels across different conditions.

            **Pros:**  
            - High accuracy  
            - Robust to overfitting and outliers  
            - Automatically handles interactions between features

            **Cons:**  
            - Slower to train and predict than simpler models  
            - Less interpretable than a single Decision Tree  
            - May require tuning of hyperparameters

            **Fun Fact:**  
            It's often one of the top-performing models in many real-world prediction tasks!
            """)

        elif model_choice == "KNN":
            st.subheader("K-Nearest Neighbors (KNN)")
            st.markdown("""
            **How it works:**  
            KNN makes predictions by finding the 'K' most similar data points (neighbors) to the new input, based on distance (usually Euclidean).
            The model then averages the PM2.5 values of these neighbors to make a prediction.

            **Use Case in Air Pollution:**  
            Useful when local environmental patterns strongly affect air quality (e.g., same weather + pollution → similar PM2.5).

            **Pros:**  
            - Simple and intuitive  
            - No training time (lazy learning)  
            - Flexible to different types of patterns

            **Cons:**  
            - Slow during prediction (must compare with all data)  
            - Sensitive to feature scaling and irrelevant variables  
            - Doesn’t work well with high-dimensional or noisy data
            """)

    # --- Metrics Tab ---
    with metrics_tab:
        st.header("Model Performance Metrics")
        with st.expander("What do these metrics mean?"):
            st.markdown("""
            To evaluate how well each machine learning model predicts PM2.5 levels, we use the following metrics:
    
            - **R² Score (Coefficient of Determination):**  
              Measures how well the predictions fit the actual data.  
              - Closer to **1** means better performance.
              - A value of 0.9 means 90% of the variance is explained by the model.
    
            - **MAE (Mean Absolute Error):**  
              The average absolute difference between predicted and actual values.  
              - Lower is better. It shows how far off your predictions are, on average.
    
            - **RMSE (Root Mean Squared Error):**  
              Similar to MAE, but gives more weight to large errors.  
              - Lower is better. It's more sensitive to outliers and extreme prediction errors.
    
            #### The best model has:
            - **High R²**
            - **Low MAE**
            - **Low RMSE**
            """)

        split_choice = st.selectbox("Choose a train/test split:", ["80/20", "70/30", "90/10"])

        # --- Predefined results ---
        results_data = {
            "80/20": {
                "Linear Regression": {
                    "R² Score": 0.85,
                    "MAE": 20.52,
                    "RMSE": 31.28
                },
                "Decision Tree": {
                    "R² Score": 0.88,
                    "MAE": 16.73,
                    "RMSE": 28.50
                },
                "Random Forest": {
                    "R² Score": 0.94,
                    "MAE": 12.32,
                    "RMSE": 19.94
                },
                "KNN": {
                    "R² Score": 0.91,
                    "MAE": 15.01,
                    "RMSE": 24.07
                }
            },
            "70/30": {
                "Linear Regression": {
                    "R² Score": 0.85,
                    "MAE": 20.47,
                    "RMSE": 31.16
                },
                "Decision Tree": {
                    "R² Score": 0.87,
                    "MAE": 16.77,
                    "RMSE": 28.70
                },
                "Random Forest": {
                    "R² Score": 0.94,
                    "MAE": 12.61,
                    "RMSE": 20.32
                },
                "KNN": {
                    "R² Score": 0.91,
                    "MAE": 15.36,
                    "RMSE": 24.81
                }
            },
            "90/10": {
                "Linear Regression": {
                    "R² Score": 0.84,
                    "MAE": 20.58,
                    "RMSE": 31.74
                },
                "Decision Tree": {
                    "R² Score": 0.87,
                    "MAE": 16.67,
                    "RMSE": 29.14
                },
                "Random Forest": {
                    "R² Score": 0.93,
                    "MAE": 12.04,
                    "RMSE": 20.01
                },
                "KNN": {
                    "R² Score": 0.90,
                    "MAE": 15.44,
                    "RMSE": 25.18
                }
            }
        }

        selected_results = results_data[split_choice]

        for model_name, metrics in selected_results.items():
            with st.expander(f"{model_name} Results"):
                for metric_name, value in metrics.items():
                    st.write(f"{metric_name}: {value}")

    #Prediction Tab
    with prediction_tab:
        st.header("PM2.5 Prediction")

        import joblib

        # Load pre-trained model and scaler
        scaler = joblib.load("scaler.pkl")
        rf_model = joblib.load("random_forest_model.pkl")

        st.markdown("""
        ### How it Works
        This section allows you to predict the PM2.5 concentration (fine particulate matter in the air)  
        using a **pre-trained Random Forest Regressor** model.

        **Why only Random Forest?**  
        After evaluating multiple models, Random Forest consistently delivered the **highest accuracy**  
        with a good balance of speed and performance. Since the model is pre-trained and loaded instantly,  
        there's no added wait time for predictions.

        Just enter the current environmental conditions below and click **Predict** to get the estimated PM2.5 level.
        """)

        # Input fields
        st.subheader("Enter Environmental Conditions")

        input_cols = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        user_input = {}

        col1, col2 = st.columns(2)

        for i, feature in enumerate(input_cols):
            with col1 if i % 2 == 0 else col2:
                default_value = float(df_after[feature].median())
                user_input[feature] = st.number_input(f"{feature}:", value=default_value)

        # Prediction button
        if st.button("Predict PM2.5"):
            user_df = pd.DataFrame([user_input])
            scaled_input = scaler.transform(user_df)

            prediction = rf_model.predict(user_df)[0]
            st.success(f"Predicted PM2.5 concentration: **{prediction:.2f} μg/m³**")


