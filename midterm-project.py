import altair as alt
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide")

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
    'Stress Level', 'Heart Rate', 'Daily Steps', 'Blood Pressure_high', 'Blood Pressure_low'
]
categorical_cols = ['BMI Category', 'Sleep Disorder']
filter_cols = ['Gender', 'Age', 'Occupation']
order_dict = {"BMI Category": ["Underweight", "Normal", "Overweight", "Obese"],
                "Sleep Disorder": ["None", "Sleep Apnea", "Insomnia"]}
df_encoded = pd.get_dummies(df, columns=['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder'])
corr_matrix = df_encoded.corr()


# Radio Group
view_option = st.sidebar.radio(
    "Select what you want to view:",
    ["Dataset analysis", "View filtered data", "Check correlation by variable", "Linear regression analysis"]
)


if view_option == "Dataset analysis":
    st.write("## Dataset Analysis")
    if st.checkbox('Show dataframe'):
        st.write(df)

    col1, col2 = st.columns(2) 

    with col1:
        chart1 = alt.Chart(df).mark_bar().encode(
            x="Gender:N", 
            y="count():Q",
            color="Gender:N"
        ).properties(width=300, height=320, title='Gender Distribution') 
        st.altair_chart(chart1)

        chart2 = alt.Chart(df).mark_bar().encode(
            x="Occupation:N", 
            y="count():Q",
            color="Occupation:N"
        ).properties(width=300, height=320, title='Occupation Distribution')
        st.altair_chart(chart2)

        chart3 = alt.Chart(df).mark_bar().encode(
            x="BMI Category:N", 
            y="count():Q",
            color="BMI Category:N"
        ).properties(width=300, height=320, title='BMI Category Distribution')
        st.altair_chart(chart3)

        chart4 = alt.Chart(df).mark_bar().encode(
            x="Sleep Disorder:N", 
            y="count():Q",
            color="Sleep Disorder:N"
        ).properties(width=300, height=320, title='Sleep Disorder Distribution')
        st.altair_chart(chart4)

    with col2:
        fig1 = px.pie(df, names='Gender')
        fig1.update_layout(height=320)
        st.plotly_chart(fig1)

        fig2 = px.pie(df, names='Occupation')
        fig2.update_layout(height=320)
        st.plotly_chart(fig2)

        fig3 = px.pie(df, names='BMI Category')
        fig3.update_layout(height=320)
        st.plotly_chart(fig3)

        fig4 = px.pie(df, names='Sleep Disorder')
        fig4.update_layout(height=320)
        st.plotly_chart(fig4)

elif view_option == "View filtered data":
    st.write("## View Filtered Data")

    numeric_option = st.multiselect('Which factors would you like to view? (Numeric Values)', numeric_cols, numeric_cols[0])
    filter_by_option = st.selectbox('How would you like to group the data?', filter_cols)
    st.write("###")

    if filter_by_option == 'Gender':
        
        df_gender = df.groupby('Gender')[numeric_option].mean()
        
        df_melted = df_gender.reset_index().melt(id_vars=["Gender"], var_name="Metric", value_name="Value")

        st.write(f"#### Comparison by Gender")
        chart = alt.Chart(df_melted).mark_bar().encode(
            x=alt.X("Gender:N", title="Gender", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Value:Q", title="Average Value"),
            color="Metric:N", 
            xOffset="Metric:N"
        ).properties(width=800, height=500)

        st.altair_chart(chart)

        for category in categorical_cols:
            st.write(f"#### Distribution of {category} by Gender")
            
            df_count = df.groupby(["Gender", category]).size().reset_index(name="Count")

            chart = alt.Chart(df_count).mark_bar().encode(
                x=alt.X("Gender:N", title="Gender", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color(f"{category}:N", title=category),
                xOffset=alt.XOffset(f"{category}:N")
            ).properties(width=800, height=500)

            st.altair_chart(chart)
            
    elif filter_by_option == 'Age':
        
        num_bins = st.slider("Select number of bins:", min_value=2, max_value=6, value=4)
        min_age = df['Age'].min()
        max_age = df['Age'].max()

        bins = np.linspace(min_age, max_age, num_bins + 1).astype(int)
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]    
        df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

        df_age = df.groupby('Age Group')[numeric_option].mean()
        df_melted = df_age.reset_index().melt(id_vars=["Age Group"], var_name="Metric", value_name="Value")

        chart = alt.Chart(df_melted).mark_bar().encode(
            x=alt.X("Age Group:N", title="Age Group", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Value:Q", title="Average Value"),
            color="Metric:N", 
            xOffset="Metric:N"
        ).properties(width=800, height=500, title="Comparison by Age Group")

        st.altair_chart(chart)

        for category in categorical_cols:
            st.write(f"#### Distribution of {category} by Age Group")
            
            df_count = df.groupby(["Age Group", category]).size().reset_index(name="Count")

            chart = alt.Chart(df_count).mark_bar().encode(
                x=alt.X("Age Group:N", title="Age Group", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color(f"{category}:N", title=category),
                xOffset=alt.XOffset(f"{category}:N")
            ).properties(width=800, height=500)

            st.altair_chart(chart)

    elif filter_by_option == 'Occupation':
        
        df_occupation = df.groupby('Occupation')[numeric_option].mean()
        
        df_melted = df_occupation.reset_index().melt(id_vars=["Occupation"], var_name="Metric", value_name="Value")

        chart = alt.Chart(df_melted).mark_bar().encode(
            x=alt.X("Occupation:N", title="Occupation"),
            y=alt.Y("Value:Q", title="Average Value"),
            color="Metric:N", 
            xOffset="Metric:N"
        ).properties(width=800, height=500, title="Comparison by Occupation")

        st.altair_chart(chart)

        for category in categorical_cols:
            st.write(f"#### Distribution of {category} by Occupation")
            
            df_count = df.groupby(["Occupation", category]).size().reset_index(name="Count")

            chart = alt.Chart(df_count).mark_bar().encode(
                x=alt.X("Occupation:N", title="Occupation"),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color(f"{category}:N", title=category),
                xOffset=alt.XOffset(f"{category}:N")
            ).properties(width=800, height=500)

            st.altair_chart(chart)
    

elif view_option == "Check correlation by variable":
    st.write("## Check Correlation by Variable")

    st.write("### Correlation Matrix Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.write("### Select correlation threshold")
    threshold = st.slider("", 0.0, 1.0, 0.6, 0.1)
    filtered_corr = corr_matrix[(corr_matrix >= threshold) | (corr_matrix <= -threshold)]

    st.write(f"### Filtered Correlation Matrix (|corr| ≥ {threshold})")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, 
            linewidths=0.5, linecolor='black', mask=filtered_corr.isna())
    st.pyplot(fig)

elif view_option == "Linear regression analysis":

    st.write("## Linear Regression Analysis")
    st.write("Work in progress")

else:

    st.markdown("### Select a category from the left side to view the chart. :)")
    st.stop()
