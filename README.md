# Capstone Team - 7 Substance Use Treatment

## Background & Objectives

Substance use in the United States is a growing problem. In 2021 approximately 22% of the population used illicit drugs and 17% of the population met the DSM-V criteria for having substance use disorder [HHS, 2021](https://www.hhs.gov/about/news/2023/01/04/samhsa-announces-national-survey-drug-use-health-results-detailing-mental-illness-substance-use-levels-2021.html).

![vis1](https://github.com/metzlere/MADS-Capstone/assets/37027603/4c263e64-d347-44e0-965d-331ad439c81e) 

<sub> Note: There is evidence supporting a large drop in admissions in 2020 during the [COVID-19 Pandemic](https://www.usnews.com/news/health-news/articles/2022-09-26/big-drop-seen-in-drug-treatment-admissions-during-pandemic#:~:text=Before%202020%2C%20admissions%20to%20treatment,10%2C000%20to%2067%20per%2010%2C000.) <sub> 

<img width="816" alt="vis2" src="https://github.com/metzlere/MADS-Capstone/assets/37027603/540659e0-77c9-42cf-a99b-0568486ed59e">

Substance use treatment availability and admissions vary across the United States. Generally, more rural areas have less accessability to treatment services and minority groups face more barriers to care than their majority counterparts ([Priester et al., 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4695242/)). Additionally, even when they are in treatment, an average of 30% of individuals drop out of substance use treatment ([Lappan et al., 2019](https://pubmed.ncbi.nlm.nih.gov/31454123/)).

Therefore, identifying a treatment program that will help the individual 1) complete the treatment successfully and 2) reduce the possibility of re-admissions would be the ultimate, but challenging goal for providers.

The objective of this project is to develop a tool for healthcare providers to determine the best possible substance use treatment for patients.

We will:

1. Design a predictive model using historical substance use admission data
2. Use the predictice to create an easy-to-use dashboard tool

## Datasets

We utilized Treatment Episode Data Set: Discharges (TEDS-D) from the Substance Abuse Mental Health Services Administration (SAHMA) Data Archive. The TEDS-D contains records on admissions/discharges of people 12 years and older, and includes demographics information and substance use characteristics. We selected to use the TEDS-D datasets from 2016-2020.

Datasets are publically available here: https://www.datafiles.samhsa.gov/dataset/teds-d-2016-ds0001-teds-d-2016-ds0001

# How to Run

## Files to Download

1. Download datasets from [SAHMSA](https://www.datafiles.samhsa.gov/dataset/teds-d-2017-ds0001-teds-d-2017-ds0001). We used 2017-2020.

## Data Preparation and Feature Engineering

1. Run **combine_teds_d_data_files.ipynb** to merge all the datasets together
2. Next, preprocess the datasets with **preprocessing.ipynb**. The resulting cleaned dataframe is saved to cleaned_data.csv

## Predictive Modeling 

1. Run **model_development.ipynb** to train and evaluate model baselines as well as tune the model to the best parameters. This script will save the tuned model output from GridSearchCV to a pickle file, which will be loaded in the web app.

## Web App Development

1. Run **create_summary_df_for_visuals.ipynb** which will take the cleaned_data.csv file and aggregate the data by value counts, saving the output to cleaned_data_counts.csv. This reduces the size of the data which must be uploaded to GitHub and used in the web app. The full cleaned_data.csv file is too large.
2. Run **create_sample_df.ipynb**, which will create a new dataframe that contains all of the unique values for each column in the original cleaned_data.csv file, and save the output to cleaned_data_sample.csv. This is necessary because cleaned_data.csv is too large to upload to GitHub, but the sample dataframe provides all of the necessary information to create the web app prediction page.
3. Run **web_app.py**, which will create a streamlit web app consisting of a landing page, dashboard visualization, prediction page, and evaluation metrics for the model. To test the web app locally, you will need to set up an environment. We deployed to Streamlit Community Cloud, which links to our GitHub repo and pulls all files necessary to run the web app from there.
