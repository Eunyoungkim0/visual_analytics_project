# streamlit run midterm-project.py
import os
import altair as alt
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import openai

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score, median_absolute_error
from statsmodels.stats.stattools import durbin_watson

st.set_page_config(layout="wide")
client = openai.OpenAI(
    base_url=os.getenv("GROQ_BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

@st.cache_data
def load_data():
    df = pd.read_csv('Data/Sleep_health_and_lifestyle_dataset.csv', index_col='Person ID')
    df[['Blood Pressure_high', 'Blood Pressure_low']] = df['Blood Pressure'].str.split('/', expand=True)
    df.drop(columns=['Blood Pressure'], inplace=True)
    df['Blood Pressure_high'] = pd.to_numeric(df['Blood Pressure_high'], errors='coerce')
    df['Blood Pressure_low'] = pd.to_numeric(df['Blood Pressure_low'], errors='coerce')
    df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

    occ_counts = df['Occupation'].value_counts()
    low_occ = occ_counts[occ_counts < 5].index
    mask = df['Occupation'].isin(low_occ)
    df.loc[mask, 'Occupation'] = 'Other'

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
order_dict = {"BMI Category": ["Normal", "Overweight", "Obese"],
                "Sleep Disorder": ["None", "Sleep Apnea", "Insomnia"]}
df_encoded = pd.get_dummies(df, columns=['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder'])
corr_matrix = df_encoded.corr()

def categorize_sleep_duration(x, bins):
    return pd.cut([x], bins=bins, labels=[f"{b.left:.1f}-{b.right:.1f}" for b in bins[:-1]], include_lowest=True)[0]

def categorize_physical_activity(x, bins):
    return pd.cut([x], bins=bins, labels=[f"{b.left:.1f}-{b.right:.1f}" for b in bins[:-1]], include_lowest=True)[0]

def categorize_daily_steps(x, bins):
    return pd.cut([x], bins=bins, labels=[f"{b.left:.1f}-{b.right:.1f}" for b in bins[:-1]], include_lowest=True)[0]

def categorize_heart_rate(x, bins):
    return pd.cut([x], bins=bins, labels=[f"{b.left:.1f}-{b.right:.1f}" for b in bins[:-1]], include_lowest=True)[0]


def introduction():
    st.write("## Introduction")

    st.markdown("### 👥 **Team 3**")
    st.markdown("""
    - **Andreas Lambropoulos**  
    - **Eunyoung Kim**  
    - **Rohith Arikatla**
    """)

    st.markdown("### 🎯 **Objective**")
    st.markdown(
        """
        This project helps you explore how your sleep habits, physical activity, stress, and other health factors are connected.  
        With interactive visuals and simple prediction tools, you can better understand your own health and how your lifestyle may be affecting it.
        """
    )

    st.markdown("### 📊 **What You Can Get from This Project**")
    st.markdown(
        """
        Using data like **gender, age, occupation, sleep duration, sleep quality, physical activity, stress, BMI, heart rate, daily steps, sleep disorder**, and **blood pressure**, you can:

        1. 🧮 **Get a quick overview of the data**  
        - Compare different groups (e.g., average sleep time by age or occupation)

        2. 🔍 **Filter and explore specific trends**  
        - Want to see how stress levels vary by gender or age? You can do that with a few clicks.

        3. 🔗 **Check how variables are related**  
        - Discover interesting correlations, like how physical activity may affect sleep quality or stress.

        4. 📈 **Predict health values**  
        - Know a few health metrics? Use linear regression to estimate others.

        5. 🧑‍⚕️ **Input your own data for personalized results**  
        - Enter your info in the input tab and get instant predictions and feedback!
        """
    )

def dataset_analysis():
    st.write("## Dataset Analysis")
    if st.checkbox('Show dataframe'):
        st.write(df)

    col1, col2, col3 = st.columns(3) 

    with col1:

        chart1 = alt.Chart(df).mark_bar().encode(
            x="Gender:N", 
            y="count():Q",
            color=alt.Color("Gender:N", legend=None)
        ).properties(width=400, height=400, title='Gender Distribution') 
        st.altair_chart(chart1)

        chart3 = alt.Chart(df).mark_bar().encode(
            x="Occupation:N", 
            y="count():Q",
            color=alt.Color("Occupation:N", legend=None)
        ).properties(width=400, height=400, title='Occupation Distribution')
        st.altair_chart(chart3)

        chart5 = alt.Chart(df).mark_bar().encode(
            x=alt.X("Stress Level:O", title="Stress Level", sort="ascending"),
            y="count():Q",
            color=alt.Color("Stress Level:N", legend=None)
        ).properties(width=400, height=400, title="Stress Level Distribution")
        st.altair_chart(chart5)

        chart7 = alt.Chart(df).mark_bar().encode(
            x=alt.X("Sleep Disorder:N", sort=order_dict["Sleep Disorder"]), 
            y="count():Q",
            color=alt.Color("Sleep Disorder:N", legend=None)
        ).properties(width=400, height=400, title='Sleep Disorder Distribution')
        st.altair_chart(chart7)
        
    with col2:

        age_bins = [25, 30, 40, 50, 60, 70]
        age_labels = ["20's", "30's", "40's", "50's", "60's'"]
        df["Age Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)
        df["Age Group"] = pd.Categorical(df["Age Group"], categories=age_labels, ordered=True)
        chart2 = alt.Chart(df).mark_bar().encode(
            x=alt.X("Age Group:N", sort=age_labels),
            y="count():Q",
            color=alt.Color("Age Group:N", legend=None)
        ).properties(width=400, height=400, title='Age Group Distribution')
        st.altair_chart(chart2)

        chart4 = alt.Chart(df).mark_bar().encode(
            x=alt.X("BMI Category:N", sort=order_dict["BMI Category"]), 
            y="count():Q",
            color=alt.Color("BMI Category:N", legend=None)
        ).properties(width=400, height=400, title='BMI Category Distribution')
        st.altair_chart(chart4)

        chart6 = alt.Chart(df).mark_bar().encode(
            x=alt.X("Quality of Sleep:O", title="Quality of Sleep", sort="ascending"),
            y="count():Q",
            color=alt.Color("Quality of Sleep:N", legend=None)
        ).properties(width=400, height=400, title="Quality of Sleep Distribution")
        st.altair_chart(chart6)

        sleep_bins = [5, 6, 7, 8, 9]
        sleep_labels = ["5-6", "6-7", "7-8", "8-9"]
        df["Sleep Duration Group"] = pd.cut(df["Sleep Duration"], bins=sleep_bins, labels=sleep_labels, right=False)
        df_group = df.groupby("Sleep Duration Group", observed=False).size().reset_index(name="Count")
        chart8 = alt.Chart(df_group).mark_bar().encode(
            x=alt.X("Sleep Duration Group:N", title="Hours of Sleep", sort=sleep_labels),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color("Sleep Duration Group:N", legend=None)
        ).properties(width=400, height=400, title="Sleep Duration Distribution")
        st.altair_chart(chart8)

    with col3:

        st.write("Select number of bins for categories below")
        num_bins = st.slider("Number of bins", 2, 6, 4)
        bin_cols = ["Physical Activity Level", "Daily Steps", "Heart Rate"]

        for col in bin_cols:
            group_col = str(col) + " Group"

            min_value = df[col].min()
            max_value = df[col].max()

            bins = np.linspace(min_value, max_value, num_bins + 1).astype(int)
            bins = np.unique(bins)

            labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]    

            df[group_col] = pd.cut(df[col], bins=bins, labels=labels, right=False, duplicates='drop')
            df_group = df.groupby(group_col, observed=False).size().reset_index(name="Count")
            df_melted = df_group.melt(id_vars=[group_col], var_name="Metric", value_name="Value")

            chart = alt.Chart(df_melted).mark_bar().encode(
                x=alt.X(f"{group_col}:N", title=group_col, sort=labels),
                y="Value:Q",
                color=alt.Color(f"{group_col}:N", legend=None)
            ).properties(width=400, height=400, title=f"{col} Group Distribution")

            st.altair_chart(chart)

def filter_data(col):

    if col == 'Age':
        num_bins = st.slider("Select number of bins:", min_value=2, max_value=6, value=4)
        min_age = df['Age'].min()
        max_age = df['Age'].max()

        bins = np.linspace(min_age, max_age, num_bins + 1).astype(int)
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]    
        df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        col = 'Age Group'

    cols = st.columns(len(categorical_cols))
    for i, category in enumerate(categorical_cols):
        with cols[i]:
            st.write(f"#### Distribution of {category} by {col}")
            
            df_count = df.groupby([f'{col}', category], observed=False).size().reset_index(name="Count")
            sort_order = order_dict.get(category, None)

            chart = alt.Chart(df_count).mark_bar().encode(
                x=alt.X(f'{col}:N', title=f'{col}', axis=alt.Axis(labelAngle=-45), sort=sort_order),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color(f"{category}:N", title=category, sort=sort_order),
                xOffset=alt.XOffset(f"{category}:N", sort=sort_order)
            ).properties(width=700, height=500)  

            st.altair_chart(chart)

    numeric_option = st.multiselect('Which factors would you like to view? (Numeric Values)', numeric_cols, numeric_cols[0])
    
    df_group = df.groupby(f'{col}', observed=False)[numeric_option].mean()
    df_melted = df_group.reset_index().melt(id_vars=[f'{col}'], var_name="Metric", value_name="Value")

    st.write(f"#### Comparison by {col}")
    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X(f'{col}:N', title=f'{col}', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Value:Q", title="Average Value"),
        color="Metric:N", 
        xOffset="Metric:N"
    ).properties(width=800, height=500)

    st.altair_chart(chart)

def view_filtered_data():
    st.write("## View Filtered Data")

    filter_by_option = st.selectbox('How would you like to group the data?', filter_cols)

    filter_data(filter_by_option)
    
def check_correlation_by_variable():
    st.write("## Check Correlation by Variable")

    # st.write("### Correlation Matrix Heatmap")

    # fig, ax = plt.subplots(figsize=(10, 6))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 7}, ax=ax)
    # st.pyplot(fig)

    st.write("### Select correlation threshold")
    threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.6, 0.1)
    filtered_corr = corr_matrix[(corr_matrix >= threshold) | (corr_matrix <= -threshold)]

    st.write(f"### Filtered Correlation Matrix (|corr| ≥ {threshold})")

    col1, col2 = st.columns([2, 1]) 

    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#e8e8e8')
        sns.heatmap(filtered_corr, annot=True, cmap='viridis', fmt=".2f", annot_kws={"size": 6}, ax=ax, 
                linewidths=0.5, linecolor='gray', mask=filtered_corr.isna())
        ax.tick_params(axis='both', labelsize=6)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6)
        st.pyplot(fig)

    
    with col2:
        stacked = filtered_corr.stack()
        stacked = stacked[stacked.index.get_level_values(0) != stacked.index.get_level_values(1)]
        stacked.index = stacked.index.map(lambda x: tuple(sorted(x)))
        stacked = stacked[~stacked.index.duplicated(keep='first')]

        corr_pairs_df = stacked.reset_index()
        corr_pairs_df.columns = ['Variable 1', 'Variable 2', 'Correlation']

        corr_pairs_df['Abs Corr'] = corr_pairs_df['Correlation'].abs()
        corr_pairs_df = corr_pairs_df.sort_values(by='Abs Corr', ascending=False).drop(columns='Abs Corr')

        # st.table(corr_pairs_df.reset_index(drop=True))

        if st.button("Generate Detailed Explanation"):
            # Convert data to Markdown for prompt
            corr_md = corr_pairs_df.to_markdown(index=False)

            prompt = f"""The table below shows the correlation between Variable 1 and Variable 2:\n\n{corr_md}\n\n
            Please provide a brief and clear interpretation of these results. Highlight which variables are highly correlated, 
            explain what the correlations might imply, and share any potential insights (at least five insights) and any suggestions to stay healthy only based on the table data.
            And also, explain it in a way even for high school students to understand it easily, but don't mention hight school students."""

            # Query Groq for interpretation
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            st.write(response.choices[0].message.content)

            st.write(corr_md)


def linear_regression_analysis():
    st.write("## Linear Regression Analysis")
    
    correlation_matrix = df[numeric_cols].corr().abs()
    correlation_with_target = correlation_matrix.mean().sort_values(ascending=False)
    sorted_numeric_cols = correlation_with_target.index.tolist()

    col1, col2 = st.columns(2)

    with col1:
        x_option = st.selectbox("Select predictor variable (X):", sorted_numeric_cols)
    with col2:
        y_option = st.selectbox("Select response variable (Y):", [col for col in sorted_numeric_cols if col != x_option])
    
    if x_option and y_option:
        if x_option == y_option:
            st.error(" Predictor and response variables must be different. Please select different columns.")

    correlation = df[[x_option, y_option]].corr().iloc[0, 1]
    if abs(correlation) < 0.1: 
        st.warning(f"⚠ The correlation between {x_option} and {y_option} is very low ({correlation:.2f})."
            " This means that linear regression may not be meaningful.")

    if x_option and y_option:
        X = df[[x_option]].dropna()
        y = df[y_option].dropna()

        df_reg = pd.concat([X, y], axis=1).dropna()
        X = df_reg[[x_option]]
        y = df_reg[y_option]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        coef = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 1 - 1)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        evs = explained_variance_score(y_test, y_pred)
        residuals = y_test - y_pred
        dw_stat = durbin_watson(residuals)

        st.write("### Regression Equation")
        st.latex(f"{y_option} = {coef:.4f} * {x_option} + {intercept:.4f}")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("### Regression Plot")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=X_test[x_option], y=y_test, ax=ax, label="Actual Data")
            sns.lineplot(x=X_test[x_option], y=y_pred, color="red", ax=ax, label="Regression Line")
            ax.set_xlabel(x_option)
            ax.set_ylabel(y_option)
            ax.set_title("Linear Regression Plot")
            st.pyplot(fig)

            st.write("### Residuals Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(residuals, kde=True, bins=20, ax=ax)
            ax.set_title("Residuals Distribution")
            st.pyplot(fig)

        with col2:
            st.write("### Metrics of Evaluation")
            metrics_df = pd.DataFrame({
                "Metric": ["R² Score", "Adjusted R²", "Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error", "Correlation","Explained Variance Score", "Durbin-Watson Statistic"],
                "Value": [r2, adj_r2, mae, mse, rmse, correlation, evs, dw_stat]
            })
            st.table(metrics_df)

# tabs for navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Introduction",
    "Dataset analysis",
    "View filtered data",
    "Check correlation by variable",
    "Linear regression analysis"
])

with tab1:
    introduction()

with tab2:
    dataset_analysis()

with tab3:
    view_filtered_data()

with tab4:
    check_correlation_by_variable()

with tab5:
    linear_regression_analysis()