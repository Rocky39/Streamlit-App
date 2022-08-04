import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Creating Container for each Section of the App
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

# Design layout
st.markdown(
    """
    <style>
    .main {
    background-color: #800080;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Cache data (to avoid reloading)
@st.cache
def get_data(filename):
    covid_data = pd.read_csv(filename)

    return covid_data

# Text-Input for each Section
with header:
    st.title('Welcome to my awesome data science project!')
    st.text('In this project I look into the transactions of taxis in NYC. ...')

with dataset:
    st.header('Covid19-Data-Germany-Dataset')
    st.text('In found this dataset on blablabla.com, ...')

    # Import Data
    covid_data = get_data('data/COVID19_data Germany_Deutschland_RKI.csv')
    st.write(covid_data.head())

    # Add Graph
    st.subheader('Covid Cases per 100k')
    cases_per_100k = pd.DataFrame(covid_data['cases_per_100k'].value_counts()).head(50)
    st.bar_chart(cases_per_100k)

with features:
    st.header('The features I created')

    st.markdown('* **first feature:** I created this feature because of this... I calculated it using ths logic... ')
    st.markdown('* **second feature:** I created this feature because of this... I calculated it using ths logic... ')

with model_training:
    st.header('Time to train the model!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance changes!')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100, 200, 300, 'No limits'], index = 0)

    sel_col.text('Here is a list of features in my data:')
    sel_col.write(covid_data.columns)

    input_feature = sel_col.text_input('Which feature should be used as the input feature?','cases_per_100k')

    # Integrate Machine Learning

    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    x = covid_data[[input_feature]]
    y = covid_data[['cases_per_100k']]

    regr.fit(x, y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean squared  error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R squared error of the model is:')
    disp_col.write(r2_score(y, prediction))