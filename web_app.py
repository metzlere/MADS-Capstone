import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import plotly.express as px

# Define main function for the app and navigation
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["About", "Prediction Page", "Visualization Page", "Model Performance", "Bias Analysis"])
    if selection == "About":
        render_about_page("landing_page_text.txt")
    elif selection == "Prediction Page":
        render_prediction_page()
    elif selection == "Visualization Page":
        render_visualization_page()
    elif selection == "Model Performance":
        render_performance_table()
    elif selection == "Bias Analysis":
        render_bias_analysis()

def render_bias_analysis():
    st.markdown("## Bias Analysis")
    st.markdown("### LIME")
    st.write("""
             In order to explore the potential of racial bias in our model, we used the LIME (Local Interpretable Model-Agnostic Explanations) 
             package to generate explanations for our model's predictions. LIME is a model-agnostic method for explaining the predictions of
             a model. It approximates any model with a local linear model which improves interpretability.
             One instance of the LIME analysis is shown below:
             """)
    
    data = [('RACE_Alaskan Native = 0', -0.0075929092),
        ('RACE_Not known = 0', -0.0027402459),
        ('RACE_American Indian = 0', -0.0022644763),
        ('RACE_Two or more races = 0', -0.001571695),
        ('RACE_Asian = 0', -0.0007708449),
        ('RACE_White = 0', -0.000252296),
        ('RACE_Asian or Pacific Islander = 0', 0.0),
        ('RACE_Other single race = 0', 0.0013001929),
        ('RACE_Black or African American = 0', 0.00285716),
        ('RACE_Native Hawaiian or Other Pacific Islander = 0', 0.0042015106)]

    # Creating the dataframe
    df = pd.DataFrame(data, columns=['Race Variable', 'Value'])

    st.table(df)

    st.write("""
            The table above shows the weights of the race features which contributed to a prediction of success
            for a given treatment protocol. For example, if a patient is NOT Alaskan Native (RACE_Alaskan Native <= 0.00),
            the model is less likely to predict a successful treatment outcome with a weight of -0.0076. If the patient is 
            NOT Native Hawaiian or other Pacific Islander, then the model is more likely to predict a successful treatment
            outcome with a weight of 0.0042.
             """)
    
    st.write("""
            It is important to note that the results of LIME are specific to the instance and do not 
            necessarily represent the model as a whole. However, LIME can be useful as a starting point to explore the 
            potential for bias. In this case, the weights are very small relative to other features, but in future work 
            we would like to explore the potential for bias in more detail. Specifically, we would like to
            examine distributions of successful treatment predictions for different racial groups and evaluating 
            model performance (accuracy, precision, recall) by racial group.
             """)
    
    st.write("""
            ### Feature Ablation
            We also performed a feature ablation analysis to determine which features were most important to the model's predictions.
            While this analysis does not directly examine the bias in the model, it can be used to identify features
            which may be contributing to bias. The result is shown below:
             """)
    
    st.write("""
            **Accuracy score after removing race variables: 0.7055**
             """)
    
    st.write("""
            Removing the race variables results in a decrease of model accuracy of about .0045.
            This result does not appear to be signficant. In the context of this analysis, we can be reasonably
            confident that the model is not biased against any particular racial group. However, we are only
            considering a few possible sources of racial bias with a limited number of techniques. It is also important
            to note that racial bias is not the only form of bias which may be present in the model, and other protected
            variables should be reviewed for bias in future work.
             """)

# Define function to render a table of model performance metrics
def render_performance_table():

    st.write("""
             ## Model Performance Baseline (no hyperparameter tuning)
             """)

    data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
            ['Naive Bayes', 0.529, 0.599, 0.663, 0.505],
            ['Logistic Regression', 0.706, 0.638, 0.724, 0.635],
            ['Random Forest', 0.822, 0.669, 0.689, 0.678]]

    # Convert data to a DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])

    st.table(df)

    st.write("""
             The Random Forest Classifier performed significantly better than the other models we considered in a baseline evaluation
             """)
    
    st.write("""
                ## Model Performance with Hyperparameter Tuning and 5 Fold Cross Validation
                """)


    data2 = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
            ['Random Forest', 0.710, 0.639, 0.725, 0.638]]

    # Convert data to a DataFrame
    df2 = pd.DataFrame(data2[1:], columns=data2[0])

    st.table(df2)

    st.write("""
            After initial training, the model was tuned using a grid search and 5 fold cross validation. 
            We found that training the model with 'None' as the max_depth parameter resulted in a model size that was too large to deploy in Streamlit, because the max 
            file size in the linked GitHub repo is 100 MB. The default parameters for the Scikit-Learn Random Forest have a max depth of 'None', and the model trained on the 
            default parameters actually outperformed the fine-tuned model (as seen in the tables above). We were able to reduce the model size by setting the max_depth parameter to 12, and n_estimators to 200, which were the best 
            possible parameters which resulted in a trained model within the 100 MB requirement.
            ## Future Work
            In the future, given additional compute resources and time to work through model size-related issues using alternative means (Git Large File Store is not free for our use case), we would like to train our model on the full dataset (over 6M records).
            Additionally, we would like to perform hyperparameter tuning on a larger grid of parameters.
            We encountered memory issues even on higher memory machines procured in GCP. We would be interested to see how accurate our model can be without any memory concerns or contraints on model sizes.
            """)


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

    st.write("""
             ## Future Work
            We were limited in terms of data visualization due to the size of the dataset and the file size limit of 100 MB for the GitHub repo for integration with the web app in Streamlit.
            In the future, we would like to explore using more powerful data visualization software such as Tableau or Power BI to create visualizations with robust filtering and drill-down capabilities on the full dataset. 
             """)

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

    # Align DataFrame with the training DataFrame's columns
    df_aligned = features_one_hot.reindex(columns=feature_names, fill_value=0)


    # Convert the DataFrame to a numerical array
    X = df_aligned.values

    # Make predictions using the model
    return model.predict_proba(X)[0][1]

# Define function to render the prediction page, which calls the prediction function
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
        st.write("### The treatment protocols, sorted by likelihood of success, are:")

        treatments_list = []
        probabilities_list = []
        for treatment, prob in predictions:
            treatment_cleaned = treatment.replace("/", "")
            treatments_list.append(treatment_cleaned)
            probabilities_list.append(round(prob, 4))
            st.write(f"{treatment_cleaned} --- Probability of Success: {round(prob, 4)}")

        # Make bar chart to highlight magnitude of differences in probabilities
        chart_data = pd.DataFrame({
            'Treatment': treatments_list,
            'Probability of Success': probabilities_list
        })
        st.bar_chart(chart_data.set_index('Treatment'))
        
        st.write("""
                 
                 ---------------------------------------
                 **Explanation of treatments:**

                 • Detoxification, 24-hour service, hospital inpatient: 24 hours per day medical acute care services in hospital setting for
                detoxification of persons with severe medical complications associated with withdrawal.
                                
                • Detoxification, 24-hour service, free-standing residential: 24 hours per day services in non-hospital setting providing for
                safe withdrawal and transition to ongoing treatment.
                                
                • Rehabilitation/Residential – hospital (other than detoxification): 24 hours per day medical care in a hospital facility in
                conjunction with treatment services for alcohol and other drug use and dependency.
                                
                • Rehabilitation/Residential – short term (30 days or fewer): Typically, 30 days or fewer of non-acute care in a setting with
                treatment services for alcohol and other drug use and dependency.
                                
                • Rehabilitation/Residential – long term (more than 30 days): Typically, more than 30 days of non-acute care in a setting
                with treatment services for alcohol and other drug use and dependency; may include transitional living arrangements such
                as halfway houses.
                                
                • Ambulatory - intensive outpatient: At a minimum, treatment lasting two or more hours per day for 3 or more days per
                week.
                                
                • Ambulatory - non-intensive outpatient: Ambulatory treatment services including individual, family and/or group services;
                may include pharmacological therapies.
                                
                • Ambulatory - detoxification: Outpatient treatment services providing for safe withdrawal in an ambulatory setting
                (pharmacological or non-pharmacological)
                """)

if __name__ == "__main__":
    main()