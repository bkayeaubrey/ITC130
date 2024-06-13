import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Function to load data from a CSV file and handle missing values
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    if df.isnull().sum().sum() > 0:
        st.warning('The dataset contains missing values.')
        st.dataframe(df[df.isnull().any(axis=1)])
    return df

# Function to detect outliers using z-score
@st.cache
def detect_outliers(df, columns):
    outlier_indices = []
    for col in columns:
        # Calculate z-score for each column
        z_scores = np.abs(stats.zscore(df[col]))
        # Define threshold for outliers (e.g., z-score > 3)
        threshold = 3
        # Find indices of outliers
        outlier_indices.extend(np.where(z_scores > threshold)[0])
    return list(set(outlier_indices))

# Load data
file_path = 'obesity.csv'
data = load_data(file_path)

# Title of the web app
st.title('Obesity Levels Prediction Insights\n by: Kaye Aubrey Bolotano & Frichie Ann Ortiz')

# Introduction text
st.write("""
This app showcases insights from the study "Prediction of Obesity Levels Based On Eating Habits and Physical Activities" 
which compares the performance of Decision Tree, Random Forest, and SVM algorithms.
""")

# Check if there are missing values
missing_values = data.isnull().sum().sum() > 0

# Display missing values information
if missing_values:
    st.warning('The dataset contains missing values.')
    st.write('Below are the rows with missing values:')
    st.dataframe(data[data.isnull().any(axis=1)])
else:
    st.success('No missing values found in the dataset.')

# Outlier Detection Section
numerical_columns = data.select_dtypes(include=np.number).columns
outliers = detect_outliers(data, numerical_columns)

# Display outliers
st.subheader('Outlier Detection')
if len(outliers) > 0:
    st.write('Below are the outliers identified in the dataset:')
    st.write(data.iloc[outliers])
else:
    st.write('No outliers detected in the dataset.')

# Display a summary of the data
st.subheader('Data Summary')
st.write('Here is a summary of the data:')
st.write(data.describe())

# Display algorithm performance data
# For demonstration, creating a dummy dataset as an example
# In a real scenario, this should be replaced with actual performance data from the study
performance_data = {
    'Algorithm': ['Random Forest', 'Decision Tree', 'SVM'],
    'Accuracy': [97, 90.31, 95.37],
    'Precision': [97.54, 90.41, 95.55],
    'F1 Score': [97.37, 90.35, 95.35],
    'MSE': [0.1123, 1.0132, 0.3546],
    'MAE': [0.0419, 0.2775, 0.1079]
}
performance_df = pd.DataFrame(performance_data)

# Display the performance data as a table
st.subheader('Algorithm Performance Data Table')
st.write('Below is the performance data of the algorithms:')
st.dataframe(performance_df)

# Bar chart for algorithm performance
st.subheader('Algorithm Performance Comparison')
metric = st.selectbox('Select a metric for comparison:', performance_df.columns[1:])
fig1, ax1 = plt.subplots()
sns.barplot(x='Algorithm', y=metric, data=performance_df, ax=ax1)
ax1.set_title(f'Comparison of {metric} among Algorithms')
ax1.set_ylabel(metric)
ax1.set_xlabel('Algorithm')
st.pyplot(fig1)
st.write(f'The bar chart displays the {metric} of different algorithms.')

# Scatter plot for numerical data
st.subheader('Scatter Plot')
x_col = st.selectbox('Select the x-axis variable:', data.select_dtypes(include=np.number).columns)
y_col = st.selectbox('Select the y-axis variable:', data.select_dtypes(include=np.number).columns)
fig2, ax2 = plt.subplots()
sns.scatterplot(x=x_col, y=y_col, data=data, ax=ax2)
ax2.set_title(f'Scatter Plot of {x_col} vs {y_col}')
st.pyplot(fig2)
st.write(f'The scatter plot shows the relationship between {x_col} and {y_col}.')

# Add a narrative for the data context
st.subheader('Study Context')
st.write("""
### Abstract:
The research evaluates the effectiveness of the Random Forest, Decision Tree, and Support Vector Machine (SVM) algorithms 
in predicting obesity outcomes using the "Obesity Levels" dataset. The study highlights key attributes, especially weight, 
as significant predictors of obesity and utilizes correlation analysis and feature importance scores to support these findings.

### Keywords:
- Machine Learning
- Obesity
- Random Forest
- Decision Tree
- Support Vector Machine (SVM)
- Feature Selection

### Introduction:
Obesity is a global health concern with rising prevalence and significant impacts on healthcare systems. It results from a 
combination of genetic, environmental, and lifestyle factors, particularly the imbalance between energy intake and expenditure. 
Machine learning algorithms can help identify patterns and relationships in data, making them useful for obesity research.

### Objectives:
- To identify key predictors of obesity.
- To assess the effectiveness of Random Forest, Decision Tree, and SVM algorithms in predicting obesity.
- To provide a comprehensive understanding of factors contributing to obesity.
- To offer a data-driven approach to mitigating obesity-related health issues.

### Conclusion:
The Random Forest algorithm outperformed both the Decision Tree and SVM models in predicting obesity levels. 
It achieved the highest accuracy, precision, and F1 score, along with the lowest MSE and MAE. 
The study suggests that Random Forest may be the most suitable choice for this predictive task.
""")

