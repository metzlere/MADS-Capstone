import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os


# Load trained model
model = joblib.load('best_random_forest.pkl')

# Define prediction function which takes demographic data, treatment, and feature names as input
def predict(data, treatment, feature_names):
    # Convert data dictionary to a DataFrame
    df = pd.DataFrame([data])

    # Add the treatment column to the DataFrame
    df['SERVICES_AT_ADMISSION'] = treatment
    
    # Perform one-hot encoding on the DataFrame
    features_one_hot = pd.get_dummies(df)    

    # for feature in features_one_hot.columns:
    #     st.write(feature)
    

    # Align DataFrame with the training DataFrame's columns
    df_aligned = features_one_hot.reindex(columns=feature_names, fill_value=0)


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
    df = pd.read_csv('cleaned_data_sample.csv')
    treatments = df['SERVICES_AT_ADMISSION'].unique().tolist()

    # Preprocessing
    df = df[df['REASON_FOR_DISCHARGE'] != 'Transferred to another treatment program or facility']
    df = df.drop(['REASON_FOR_DISCHARGE'], axis=1)
    df['ARRESTS_IN_30_DAYS_PRIOR_TO_ADMISSION'] = df['ARRESTS_IN_30_DAYS_PRIOR_TO_ADMISSION'].fillna('None') # some values come over as NaN from the sample df, fill with 'None'
    df['ALCOHOL_OR_DRUG_ABUSE'] = df['ALCOHOL_OR_DRUG_ABUSE'].fillna('None')
    df['HEALTH_INSURANCE'] = df['HEALTH_INSURANCE'].fillna('None')
    df['PRIMARY_SUBSTANCE_ABUSE'] = df['PRIMARY_SUBSTANCE_ABUSE'].fillna('None')


    # Get the feature names from the training DataFrame
    feature_names = pd.get_dummies(df).columns

    # for feature in feature_names:
    #    st.write(feature)
    
    # Create dictionary to hold user inputs
    user_inputs = {}

    # Drop the treatment column 
    df = df.drop(['SERVICES_AT_ADMISSION'], axis=1)

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
            treatment = treatment.replace("/", "").replace(",", "")
            st.write(f"Treatment:{treatment} --- Probability of Success: {prob}")
        
    # Link to Dashboard embedding
    # Link to Github repo

if __name__ == "__main__":
    main()
