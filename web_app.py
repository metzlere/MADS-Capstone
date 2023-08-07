import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import plotly.express as px

# Define main function for the app and navigation
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["About", "Prediction Page", "Visualization Page"])
    if selection == "About":
        render_about_page("landing_page_text.txt")
    elif selection == "Prediction Page":
        render_prediction_page()
    elif selection == "Visualization Page":
        render_visualization_page()


# Define function to render the landing page which displays the project description
def render_about_page(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return st.markdown(data)

# Define function to render the visualization page which displays the value counts for each column
def render_visualization_page():  
    # Load the summary dataframe
    summary_df = pd.read_csv('cleaned_data_counts.csv',  index_col=0)

    # Create a dropdown for column selection
    variable = st.selectbox('Select a variable to plot:', summary_df.index)

    # Get the value counts for the selected variable
    value_counts = summary_df.loc[variable].dropna()

    # Convert the index and values to data frame
    data = pd.DataFrame({variable: value_counts.index, 'Count': value_counts.values.astype(int)})

    # Create a bar plot for the value counts
    fig = px.bar(data, x=variable, y='Count', color=variable, title=f'Value Counts for {variable}',
                 labels={variable: 'Value', 'Count': 'Count'}, hover_data=[variable, 'Count'])
    
    fig.update_layout(autosize=False, width=800, height=600, showlegend=False)

    # Display the plot
    st.plotly_chart(fig)

# Load trained model
model = joblib.load('model_final.pkl')

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


    # Convert the DataFrame to a numerical array
    X = df_aligned.values

    # Make predictions using the model
    return model.predict_proba(X)[0][1]


def render_prediction_page():
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
            st.write(f"{treatment} --- Probability of Success: {prob}")
        
    # Link to Github repo

if __name__ == "__main__":
    main()