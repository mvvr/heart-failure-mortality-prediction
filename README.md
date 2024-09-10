# Heart Failure Mortality Prediction

- Streamlit Link: [heart-failure-mortality-prediction](https://heartfailuremortalityprediction.streamlit.app/)


## Project Overview

Cardiovascular diseases (CVDs) are the leading cause of death globally, accounting for an estimated 17.9 million lives annually. Early detection of individuals at high risk of cardiovascular diseases is crucial for reducing mortality rates. This project aims to develop a predictive model using machine learning to assess heart failure risk based on patient data and to provide a user-friendly interface through a Streamlit web app.

## Project Objective

The main objectives of this project are:
1. Explore the Heart Failure Dataset.
2. Apply a Random Forest Classifier model to predict mortality risk.
3. Create a Streamlit web application to provide an interactive interface for data exploration, prediction, and model performance evaluation.

## Dataset

The dataset consists of 299 patients with heart failure, collected in 2015. Key parameters related to each patient's clinical status are included. The dataset features are as follows:

- **Age**: Age of the patient at the time of heart failure.
- **Anaemia**: Binary value indicating absence (0) or presence (1) of Anaemia.
- **Creatinine Phosphokinase (CPK)**: Level of CPK enzyme in the blood (mcg/L).
- **Diabetes**: Binary value indicating absence (0) or presence (1) of Diabetes.
- **Ejection Fraction (EF)**: Ejection fraction percentage.
- **High Blood Pressure (HBP)**: Binary value indicating absence (0) or presence (1) of hypertension.
- **Platelets (P)**: Number of platelets.
- **Serum Creatinine (SC)**: Level of Serum Creatinine in the blood (mg/dL).
- **Serum Sodium (SS)**: Level of Serum Sodium in the blood (mEq/L).
- **Sex**: Binary value indicating the sex of the patient. 0 for female, 1 for male.
- **Smoking**: Binary value indicating nicotine addiction. 0 for absent, 1 for present.
- **Time**: Follow-up period in days.
- **Death Event**: Binary value indicating whether the patient deceased during the follow-up period (1) or not (0).

## Installation

To run the project, you'll need Python and some specific packages. Follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/mvvr/heart-failure-mortality-prediction.git
   cd heart-failure-mortality-prediction
