import altair as alt
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
    df = pd.read_csv('Data/Sleep_health_and_lifestyle_dataset.csv', index_col='Person ID')
    df[['Blood Pressure_high', 'Blood Pressure_low']] = df['Blood Pressure'].str.split('/', expand=True)
    df.drop(columns=['Blood Pressure'], inplace=True)
    df['Blood Pressure_high'] = pd.to_numeric(df['Blood Pressure_high'], errors='coerce')
    df['Blood Pressure_low'] = pd.to_numeric(df['Blood Pressure_low'], errors='coerce')
    df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Underweight')
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    return df


st.title('Sleep Health and Lifestyle Visual Analysis')

# load the data
df = load_data()

numeric_cols = [
    'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
    'Stress Level', 'Heart Rate', 'Daily Steps',
    'Blood Pressure_high', 'Blood Pressure_low'
]
numeric_option = st.multiselect('Which factors would you like to view? (Numeric Values)', numeric_cols, numeric_cols[0])

categorical_cols = ['BMI Category', 'Sleep Disorder']
st.write("Which factors would you like to view? (Categorical Values)")
selected_categories = []
for category in categorical_cols:
    if st.checkbox(category, value=False): 
        selected_categories.append(category)

# show the data in a table
if st.sidebar.checkbox('Show dataframe'):
    st.write(df)

st.write("###")

filter_by = st.sidebar.radio("Filter by:", ["None", "Gender", "Age", "Occupation"])

if filter_by == "Gender":
    df_gender = df.groupby('Gender')[numeric_option].mean()
    
    df_melted = df_gender.reset_index().melt(id_vars=["Gender"], var_name="Metric", value_name="Value")

    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X("Gender:N", title="Gender"),
        y=alt.Y("Value:Q", title="Average Value"),
        color="Metric:N", 
        xOffset="Metric:N"
    ).properties(width=800, height=500, title="Comparison by Gender")

    st.altair_chart(chart)



elif filter_by == "Age":
    
    num_bins = st.slider("Select number of bins:", min_value=2, max_value=6, value=4)
    min_age = df['Age'].min()
    max_age = df['Age'].max()

    bins = np.linspace(min_age, max_age, num_bins + 1).astype(int)
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]    
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    df_age = df.groupby('Age Group')[numeric_option].mean()
    df_melted = df_age.reset_index().melt(id_vars=["Age Group"], var_name="Metric", value_name="Value")

    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X("Age Group:N", title="Age Group"),
        y=alt.Y("Value:Q", title="Average Value"),
        color="Metric:N", 
        xOffset="Metric:N"
    ).properties(width=800, height=500, title="Comparison by Age Group")

    st.altair_chart(chart)

elif filter_by == "Occupation":

    df_occupation = df.groupby('Occupation')[numeric_option].mean()
    
    df_melted = df_occupation.reset_index().melt(id_vars=["Occupation"], var_name="Metric", value_name="Value")

    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X("Occupation:N", title="Occupation"),
        y=alt.Y("Value:Q", title="Average Value"),
        color="Metric:N", 
        xOffset="Metric:N"
    ).properties(width=800, height=500, title="Comparison by Occupation")

    st.altair_chart(chart)

else:
    st.markdown("### Select a category to view the chart :)")
    st.stop()


