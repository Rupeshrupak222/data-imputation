import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

st.set_page_config(page_title="Data Imputation Techniques", layout="wide")
st.title("🧠 Data Imputation Techniques Demonstration")

# Sample dataset
data = {
    'Age': [25, np.nan, 30, 22, np.nan, 28],
    'Salary': [50000, 60000, np.nan, 52000, 58000, np.nan],
    'Department': ['HR', 'IT', 'HR', np.nan, 'Finance', 'IT']
}
df_original = pd.DataFrame(data)

st.subheader("🔍 Original Data with Missing Values")
st.dataframe(df_original)

# Sidebar selection
option = st.sidebar.selectbox("Choose an imputation method", [
    "Simple Imputer (Mean/Median/Mode)",
    "KNN Imputer",
    "Iterative Imputer",
    "Forward Fill / Backward Fill",
    "Group-Based Imputation"
])

if option == "Simple Imputer (Mean/Median/Mode)":
    df = df_original.copy()
    mean_imputer = SimpleImputer(strategy='mean')
    df['Age'] = mean_imputer.fit_transform(df[['Age']])

    median_imputer = SimpleImputer(strategy='median')
    df['Salary'] = median_imputer.fit_transform(df[['Salary']])

    mode_imputer = SimpleImputer(strategy='most_frequent')
    df['Department'] = mode_imputer.fit_transform(df[['Department']])

    st.subheader("✅ After Simple Imputation")
    st.dataframe(df)

elif option == "KNN Imputer":
    df_knn = pd.DataFrame({
        'Age': [25, np.nan, 30, 22, np.nan, 28],
        'Salary': [50000, 60000, np.nan, 52000, 58000, np.nan]
    })
    knn_imputer = KNNImputer(n_neighbors=2)
    df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df_knn), columns=df_knn.columns)

    st.subheader("✅ After KNN Imputation")
    st.dataframe(df_knn_imputed)

elif option == "Iterative Imputer":
    df_iter = pd.DataFrame({
        'Age': [25, np.nan, 30, 22, np.nan, 28],
        'Salary': [50000, 60000, np.nan, 52000, 58000, np.nan]
    })
    iter_imputer = IterativeImputer()
    df_iter_imputed = pd.DataFrame(iter_imputer.fit_transform(df_iter), columns=df_iter.columns)

    st.subheader("✅ After Iterative Imputation")
    st.dataframe(df_iter_imputed)

elif option == "Forward Fill / Backward Fill":
    df_ffill = pd.DataFrame({
        'Age': [25, np.nan, 30, 22, np.nan, 28],
        'Salary': [50000, 60000, np.nan, 52000, 58000, np.nan]
    })

    st.subheader("➡️ Forward Fill")
    st.dataframe(df_ffill.fillna(method='ffill'))

    st.subheader("⬅️ Backward Fill")
    st.dataframe(df_ffill.fillna(method='bfill'))

elif option == "Group-Based Imputation":
    df_group = pd.DataFrame({
        'Gender': ['M', 'F', 'M', 'F', 'F', 'M'],
        'Income': [50000, np.nan, 52000, np.nan, 58000, 51000]
    })
    df_group['Income'] = df_group.groupby('Gender')['Income'].transform(lambda x: x.fillna(x.mean()))

    st.subheader("👥 Group-Based Imputation by Gender")
    st.dataframe(df_group)
