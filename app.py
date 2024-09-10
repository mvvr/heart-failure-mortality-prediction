import streamlit as st
import pandas as pd 
import altair as alt

from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load CSS for styling
def load_css():
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

# Function for social media buttons
def st_button(icon, url, label, iconsize):
    button_code = ''
    if icon == 'twitter':
        button_code = f'''
        <p>
        <a href={url} class="btn btn-outline-primary btn-lg btn-block" type="button" aria-pressed="true">
            <svg xmlns="http://www.w3.org/2000/svg" width={iconsize} height={iconsize} fill="currentColor" class="bi bi-twitter" viewBox="0 0 16 16">
                <path d="M5.026 15c6.038 0 9.341-5.003 9.341-9.334 0-.14 0-.282-.006-.422A6.685 6.685 0 0 0 16 3.542a6.658 6.658 0 0 1-1.889.518 3.301 3.301 0 0 0 1.447-1.817 6.533 6.533 0 0 1-2.087.793A3.286 3.286 0 0 0 7.875 6.03a9.325 9.325 0 0 1-6.767-3.429 3.289 3.289 0 0 0 1.018 4.382A3.323 3.323 0 0 1 .64 6.575v.045a3.288 3.288 0 0 0 2.632 3.218 3.203 3.203 0 0 1-.865.115 3.23 3.23 0 0 1-.614-.057 3.283 3.283 0 0 0 3.067 2.277A6.588 6.588 0 0 1 .78 13.58a6.32 6.32 0 0 1-.78-.045A9.344 9.344 0 0 0 5.026 15z"/>
            </svg>
            {label}
        </a>
        </p>'''
    elif icon == 'linkedin':
        button_code = f'''
        <p>
            <a href={url} class="btn btn-outline-primary btn-lg btn-block" type="button" aria-pressed="true">
                <svg xmlns="http://www.w3.org/2000/svg" width={iconsize} height={iconsize} fill="currentColor" class="bi bi-linkedin" viewBox="0 0 16 16">
                    <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
                </svg>
                {label}
            </a>
        </p>''' 
    elif icon == 'GitHub':
        button_code = f'''
        <p>
            <a href={url} class="btn btn-outline-primary btn-lg btn-block" type="button" aria-pressed="true">
                {label}
            </a>
        </p>'''
    return st.markdown(button_code, unsafe_allow_html=True)


# Initial configurations

icon_size = 20


st.set_page_config(page_title='Heart Failure Mortality Prediction')

st.title('Heart Failure Mortality Prediction')
#st.subheader('***Random Forest Classifier***')
st.write('---')

st.sidebar.header('Directory')
app = st.sidebar.selectbox('', ['Home', 'Explore Data', 'Predict Mortality', 'About the Data'])
st.sidebar.write('---')
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

if app == 'Home':
    
    img = Image.open('Heart.jpg')
    st.write("""
    **Introduction**
    
    According to the World Health Organization, cardiovascular diseases (CVDs) are the number one cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Of these deaths, 85% are due to heart attack and stroke. Heart failure is a common event caused by cardiovascular diseases. The early detection of people with cardiovascular diseases or who are at high cardiovascular risk due to the presence of one or more risk factors is paramount to reducing deaths arising from heart failures. As a result, predictive models become indispensable.
    
    **Objective**
    
    The main objective of this project is to explore the Heart Failure Dataset, apply Random Forest Classifier models of machine learning on it, Predict morality using clinical data and create a Streamlit web app for the entire architecture.
    """)

    about_expander = st.expander('Random Forest Classifier',expanded=True)
    with about_expander:
        
        st.write("""
                Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique.
                 It can be used for both Classification and Regression problems in ML. 
                 It is based on the concept of ensemble learning, 
                 which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.
                 As the name suggests, "Random Forest is a classifier that contains a number of decision trees on various subsets 
                 of the given dataset and takes the average to improve the predictive accuracy of that dataset." 
                 Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.
                 The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.
                """) 
        img = Image.open('RanFore.jpg')
        st.image(img)

    about_expander = st.expander('About the Creator of Project',expanded=True)
    with about_expander:
        
        st.write("""
                ### Maddula Vishnu Vardhan Reddy
                ##### Computer Science Engineering at Lovely Professional University 
                Specialised in :
                |Artificial Intelligence|Machine Learning|Deep Learning|Computer Vision|NLP|RPA|

                """)
        st_button('twitter', 'https://twitter.com/vikkymvvr/', 'Follow me on Twitter', icon_size)
        st_button('linkedin', 'https://www.linkedin.com/in/mvvr/', 'Follow me on LinkedIn', icon_size)
        st_button('GitHub', 'https://github.com/mvvr/', 'Follow me on Github', icon_size)

        img = Image.open('Vikkymvvr.jpg')
        st.image(img)                   
        


if app == 'Explore Data':

    st.subheader('**Explore the dataset**')
    col1, col2 = st.columns(2)
    selectbox_options = col1.selectbox('Transform', ['Head','Tail', 
                                                        'Describe','Shape', 
                                                        'DTypes'])
    if selectbox_options == 'Head':
        input_count = col2.number_input('Count', 5, 50, help='min=5, max=50')
        st.write(df.head(input_count))
    elif selectbox_options == 'Tail':
        input_count = col2.number_input('Count', 5, 50, help='min=5, max=50')
        st.write(df.tail(input_count))
    elif selectbox_options == 'Describe':
        st.write(df.describe())
    elif selectbox_options == 'Shape':
        st.write(df.head())
        st.write('Shape: ', df.shape)
    elif selectbox_options == 'DTypes':
        st.write(df.dtypes)
    st.write('---')

    st.sidebar.subheader('Visualization Settings')
    y_axis = st.sidebar.selectbox('Select y-axis', ['age', 'ejection_fraction', 
                                                    'time'])
    x_axis = st.sidebar.selectbox('Select x-axis', ['platelets', 'creatinine_phosphokinase', 
                                                    'serum_creatinine', 'serum_sodium'])
    label = st.sidebar.selectbox('Select label', ['DEATH_EVENT', 'anaemia', 'diabetes', 
                                                    'high_blood_pressure', 'sex', 
                                                    'smoking'])
    st.subheader('**Visualization**')
    st.write("""Customize the x and y axis through the sidebar visualization settings. 
                You can also select binary features as labels which will be in the form 
                of a color.""")
    select_graph = st.sidebar.radio('Select Graph', ('point', 'bar', 'area', 'line'))

    col1, col2, col3 = st.columns([.5,.5,1])
    graph_hgt = col1.slider('Height', 200, 600, 400, step=10)
    graph_wgt = col2.slider('Width',400, 800, 600, step=10)
        
    df = df.loc[(df.creatinine_phosphokinase < 800) & (df.platelets < 500000) & 
                (df.serum_creatinine < 2.2) & (df.age >= 40)]

    chart = alt.Chart(data=df, mark=select_graph).encode(alt.X(x_axis, scale=alt.Scale(zero=False)), 
                                                            alt.Y(y_axis, scale=alt.Scale(zero=False)),color=label).properties(
        height=graph_hgt,width=graph_wgt)
    st.write(chart)
    
    if y_axis == 'age' and x_axis == 'platelets' and label == 'DEATH_EVENT':
        st.write('Majority of deceased patients had platelet count ranging from 150,000 - 300,000 and aged 58 - 75')
    elif y_axis == 'age' and x_axis == 'creatinine_phosphokinase' and label == 'DEATH_EVENT':
        st.write('Majority of deceased patients had creatinine phosphokinase count ranging from 100 - 250 and aged 55 - 70')
    elif y_axis == 'age' and x_axis == 'serum_creatinine' and label == 'DEATH_EVENT':
        st.write('Majority of deceased patients had serum creatinine count ranging from 1.2 - 1.9 and aged 50 - 75')
    elif y_axis == 'age' and x_axis == 'serum_sodium' and label == 'DEATH_EVENT':
        st.write('Majority of deceased patients had serum sodium count ranging from 134 - 140 and aged 55 - 80')
    
    elif y_axis == 'ejection_fraction' and x_axis == 'platelets' and label == 'DEATH_EVENT':
        st.write('Majority of deceased patients had platelet count ranging from 150,000 - 250,000 and ejection fraction count of 10 - 30') 
    elif y_axis == 'ejection_fraction' and x_axis == 'creatinine_phosphokinase' and label == 'DEATH_EVENT':
        st.write('Majority of deceased patients had creatinine phosphokinase count ranging from 50 - 175 and ejection fraction count of 20 - 30') 
    elif y_axis == 'ejection_fraction' and x_axis == 'serum_creatinine' and label == 'DEATH_EVENT':
        st.write('Majority of deceased patients had serum creatinine count ranging from 1.8 - 2 and ejection fraction count of 20 - 40') 
    elif y_axis == 'ejection_fraction' and x_axis == 'serum_sodium' and label == 'DEATH_EVENT':
        st.write('Majority of deceased patients had serum_sodium count ranging from 134 - 138 and ejection fraction count of 20 - 40') 
        
    elif y_axis == 'time' and x_axis == 'platelets' and label == 'DEATH_EVENT':
        st.write('Majority of deceased patients had platelet count ranging from 150,000 - 350,000 and a follow up time of less than 50 days') 
    elif y_axis == 'time' and x_axis == 'creatinine_phosphokinase' and label == 'DEATH_EVENT':
        st.write('Majority of deceased patients had creatinine phosphokinase count ranging from 50 - 250, 550 - 600, and a follow up time of less than 50 days') 
    elif y_axis == 'time' and x_axis == 'serum_creatinine' and label == 'DEATH_EVENT':
        st.write('Majority of deceased patients had serum creatinine count ranging from 0.9 - 1.5 and follow up time of less than 50 days') 
    elif y_axis == 'time' and x_axis == 'serum_sodium' and label == 'DEATH_EVENT':
        st.write('Majority of deceased patients had serum_sodium count ranging from 134 - 140 and follow up time of less than 100 days') 
        

elif app == 'Predict Mortality':
    st.title('Heart Failure Prediction')
    st.sidebar.subheader('User Input Features')

    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

    def user_input_features():
        display = ("Female (0)", "Male (1)")
        options = list(range(len(display)))
        sex = st.sidebar.radio("Sex", options, format_func=lambda x: display[x])

        smoking = st.sidebar.checkbox('Smoking')
        if smoking:
            smoking = 1
        high_blood_pressure = st.sidebar.checkbox('High Blood Presure')
        if high_blood_pressure:
            high_blood_pressure = 1
        diabetes = st.sidebar.checkbox('Diabetic')
        if diabetes:
            diabetes = 1
        anaemia = st.sidebar.checkbox('Anemic')
        if anaemia:
            anaemia = 1
            
        age = st.sidebar.slider('Age', 40, 95, 60)
        ejection_fraction = st.sidebar.slider('Ejection Fraction', 14, 80, 38)
        serum_sodium = st.sidebar.slider('Serum Sodium', 113, 148, 136)
        
        creatinine_phosphokinase = st.sidebar.number_input('Creatinine Phosphokinase', 23, 7861, 581)
        platelets = st.sidebar.number_input('Platelet Count', 25100.00, 850000.00, 263358.00, help='25100 < input < 850000')
        serum_creatinine = st.sidebar.number_input('Serum Creatinine', 0.5, 9.4, 1.3)
        time = st.sidebar.number_input('Follow-up period (Days)', 4, 285, 130)
        data = {'age': age,
                'anaemia': anaemia,
                'creatinine_phosphokinase': creatinine_phosphokinase,
                'diabetes': diabetes,
                'ejection_fraction': ejection_fraction,
                'high_blood_pressure': high_blood_pressure,
                'platelets': platelets,
                'serum_creatinine': serum_creatinine,
                'serum_sodium': serum_sodium,
                'sex': sex,
                'smoking': smoking,
                'time': time
                }
        features = pd.DataFrame(data, index=[0])
        return features

    user_data = user_input_features()
    st.subheader('**User Input parameters**')
    st.write(user_data)
    my_expander = st.expander('Check dataset')
    with my_expander:
        st.write(df)

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    user_result = classifier.predict(user_data)

    st.title('')
    st.subheader('**Conclusion:**')
    pred_button = st.button('Predict')
    if pred_button:
        if user_result[0] == 0:
            st.success('Patient survived  (0)')
        else:
            st.error('Patient deceased  (1)')

elif app == 'Model Performance':
    st.title('Model Performance')
    st.write('---')
    
    # Dummy data (replace with actual model evaluation)
    y_true = df['DEATH_EVENT']
    y_pred = [0] * len(y_true)  # Replace with actual predictions
    
    st.subheader('Performance Metrics')
    st.write(f"**Accuracy:** {accuracy_score(y_true, y_pred):.2f}")
    st.write(f"**Precision:** {precision_score(y_true, y_pred):.2f}")
    st.write(f"**Recall:** {recall_score(y_true, y_pred):.2f}")
    st.write(f"**F1 Score:** {f1_score(y_true, y_pred):.2f}")

    st.subheader('Feature Importance')
    # Dummy data (replace with actual feature importances)
    features = df.columns[:-1]
    importances = [0.1] * len(features)  # Replace with actual feature importances
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
    st.bar_chart(feature_importances.set_index('Feature'))

    st.subheader('Classification Report')
    st.text(classification_report(y_true, y_pred))

# About the Data Page
elif app == 'About the Data':
    st.title('About the Data')
    st.write('---')
    
    st.subheader('Dataset Description')
    st.write("""
        The dataset contains clinical records of heart failure patients. 
        It includes features such as age, anemia, diabetes, and high blood pressure, 
        along with the target variable indicating whether the patient has died.
        """)
    st.write("""
    **The Dataset**
    
    The dataset is composed of 299 patients with heart failure collected in 2015. For every patient, key parameters of their clinical picture were collected, which theoretically and realistically correlate with their status.
    
    **Features/Variables/Columns in the Dataset:**
    
    - **Age**: The age of each patient at the time of the heart failure.
    - **Anaemia**: Binary value indicating the absence (0) or presence (1) of Anaemia.
    - **Creatinine Phosphokinase (CPK)**: The level of the CPK enzyme in the blood (mcg/L).
    - **Diabetes**: Binary value indicating the absence (0) or presence (1) of Diabetes.
    - **Ejection Fraction (EF)**: The ejection fraction percentage.
    - **High Blood Pressure (HBP)**: Binary value indicating the absence (0) or presence (1) of hypertension.
    - **Platelets (P)**: The number of platelets.
    - **Serum Creatinine (SC)**: The level of Serum Creatinine in the blood (mg/dL).
    - **Serum Sodium (SS)**: The level of Serum Sodium in the blood (mEq/L).
    - **Sex**: Binary value indicating the sex of the patient. 0 for female, 1 for male.
    - **Smoking**: Binary value indicating nicotine addiction. 0 for absent, 1 for present.
    - **Time**: Represents the follow-up period in days.
    - **Death Event**: Binary value indicating whether the patient deceased during the follow-up period (1) or not (0).
    """)
    st.subheader('Data Source')
    st.write("The dataset is sourced from [Kaggle Heart Failure Dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction).")

    st.subheader('Data Preprocessing')
    st.write("""
        Data preprocessing steps include handling missing values, encoding categorical variables, 
        and splitting the data into training and testing sets.
        """)

    st.subheader('Additional Resources')
    st.write("""
        - [Random Forest Classifier Documentation](https://scikit-learn.org/stable/modules/ensemble.html#forest)
        - [Heart Failure Prediction Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6409450/)
        """)


