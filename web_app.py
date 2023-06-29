import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load('best_random_forest.pkl')

# Define prediction function which takes demographic data, treatment, and feature names as input
def predict(data, treatment, feature_names):
    # Convert data dictionary to a DataFrame
    df = pd.DataFrame([data])

    # Add the treatment column to the DataFrame
    df['SERVICES_AT_ADMISSION'] = treatment
    
    # # Filter rows based on condition
    # df = df[df['REASON_FOR_DISCHARGE'] != 'Transferred to another treatment program or facility']

    # # Create target variable
    # df['SUCCESSFUL_TREATMENT'] = df.apply(lambda row: 1 if row['REASON_FOR_DISCHARGE'] == 'Treatment completed' and row['PRIOR_TREATMENT_EPISODES'] == "No prior treatment episode" else 0, axis=1)
    
    # # Remove unnecessary columns
    # features = df.drop(['REASON_FOR_DISCHARGE', 'PRIOR_TREATMENT_EPISODES', 'SUCCESSFUL_TREATMENT'], axis=1)
    
    # Perform one-hot encoding on the DataFrame
    features_one_hot = pd.get_dummies(df)    

    # Align DataFrame with the training DataFrame's columns
    df_aligned = features_one_hot.reindex(columns=feature_names, fill_value=0)
    # Check for missing columns
    missing_columns = set(feature_names) - set(df_aligned.columns)

    # Temporary fix for missing columns, need to determine why there is a mismatch
    # for column in missing_columns:
    #     df_aligned[column] = 0
    

    # Convert the DataFrame to a numerical array
    X = df_aligned.values

    # Make predictions using the model
    return model.predict_proba(X)[0][1]




def main():
    # Set title of app
    st.title("Treatment Protocol Predictor")

    # Read cleaned DF used to train the model
    df = pd.read_csv('cleaned_data.csv')
    treatments = df['SERVICES_AT_ADMISSION'].unique().tolist()

    # Preprocessing
    df = df[df['REASON_FOR_DISCHARGE'] != 'Transferred to another treatment program or facility']
    df = df.drop(['REASON_FOR_DISCHARGE', 'PRIOR_TREATMENT_EPISODES'], axis=1)
    df['HEALTH_INSURANCE'] = df['HEALTH_INSURANCE'].fillna('None')
    df['ARRESTS_IN_30_DAYS_PRIOR_TO_ADMISSION'] = df['ARRESTS_IN_30_DAYS_PRIOR_TO_ADMISSION'].fillna('None')
    df['ARRESTS_IN_30_DAYS_PRIOR_TO_DISCHARGE'] = df['ARRESTS_IN_30_DAYS_PRIOR_TO_DISCHARGE'].fillna('None')

    # Drop the treatment column - need to fix
    #df = df.drop(columns=['SERVICES_AT_ADMISSION'])
    
    # Get the feature names from the training DataFrame
    feature_names = pd.get_dummies(df).columns
    
    # Create dictionary to hold user inputs
    user_inputs = {}


    # Populate user inputs with unique values from the df
    for column in df.columns:
        # Get unique values for the column
        unique_values = df[column].unique().tolist()

        # Create a user input for the column
        user_inputs[column] = st.selectbox(column, unique_values)


    # Create a button and use it to run predictions for each treatment option
    if st.button("Predict"):
        predictions = []
        for treatment in treatments:
            prediction = predict(user_inputs.copy(), treatment, feature_names) 
            predictions.append((treatment, prediction))

        # Sort treatments by prediction probability
        predictions.sort(key=lambda x: x[1], reverse=True)
        # Display the prediction
        st.write("The treatment protocols, sorted by likelihood of success, are:")
        for treatment, prob in predictions:
            st.write(f"Treatment:{treatment} --- Probability of Success: {prob}")
        
    # link to dashboard
    #st.markdown("[View Detailed Insights on Tableau Dashboard](https://your_tableau_dashboard_link)")

if __name__ == "__main__":
    main()
