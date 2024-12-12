import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# Load the dataset
data = pd.read_csv("heart.csv")

# Data Cleaning (Remove incorrect values)
data = data[data['ca'] < 4]
data = data[data['thal'] > 0]

# Rename columns for better readability
data = data.rename(columns={
    'cp': 'chest_pain_type',
    'trestbps': 'resting_blood_pressure',
    'chol': 'cholesterol',
    'fbs': 'fasting_blood_sugar',
    'restecg': 'resting_electrocardiogram',
    'thalach': 'max_heart_rate_achieved',
    'exang': 'exercise_induced_angina',
    'oldpeak': 'st_depression',
    'slope': 'st_slope',
    'ca': 'num_major_vessels',
    'thal': 'thalassemia'
})

# Sidebar for navigation
st.sidebar.title("Heart Disease Dashboard")
option = st.sidebar.selectbox("Choose a visualization:", [
    "Target Distribution",
    "Correlation Heatmap",
    "Numerical Feature Distributions",
    "Categorical Feature Distributions",
    "Pairplot",
    "Chest Pain Type vs Target",
    "Age Distribution by Target",
    "Age vs Cholesterol",
    "Model Evaluation"
])

if option == "Target Distribution":
    st.title("Heart Disease Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='target', data=data, palette="viridis")
    ax.set_title("Heart Disease Distribution")
    ax.set_xlabel("Heart Disease (1 = Yes, 0 = No)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

elif option == "Correlation Heatmap":
    st.title("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif option == "Numerical Feature Distributions":
    st.title("Numerical Feature Distributions")
    num_features = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression']
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, feature in enumerate(num_features):
        sns.histplot(data[feature], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f"Distribution of {feature}")
    plt.tight_layout()
    st.pyplot(fig)

elif option == "Categorical Feature Distributions":
    st.title("Categorical Feature Distributions")
    cat_features = ['chest_pain_type', 'resting_electrocardiogram', 'exercise_induced_angina', 
                    'st_slope', 'thalassemia', 'num_major_vessels', 'fasting_blood_sugar', 'sex']
    fig, axes = plt.subplots(4, 2, figsize=(15, 15))
    axes = axes.flatten()
    for i, feature in enumerate(cat_features):
        sns.countplot(x=feature, hue='target', data=data, palette='viridis', ax=axes[i])
        axes[i].set_title(f"{feature} Distribution by Target")
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

elif option == "Pairplot":
    st.title("Pairplot of Numerical Features")
    selected_features = ['age', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 'st_depression', 'target']
    fig = sns.pairplot(data[selected_features], hue="target", palette="viridis", diag_kind="kde")
    st.pyplot(fig)

elif option == "Chest Pain Type vs Target":
    st.title("Chest Pain Type vs Target")
    fig, ax = plt.subplots()
    sns.countplot(x='chest_pain_type', hue='target', data=data, palette='coolwarm')
    ax.set_title("Chest Pain Type vs Target")
    st.pyplot(fig)

elif option == "Age Distribution by Target":
    st.title("Age Distribution by Heart Disease")
    fig, ax = plt.subplots()
    sns.boxplot(x='target', y='age', data=data, palette="Set2")
    ax.set_title("Age Distribution by Heart Disease")
    st.pyplot(fig)

elif option == "Age vs Cholesterol":
    st.title("Age vs Cholesterol by Target")
    fig, ax = plt.subplots()
    sns.scatterplot(x='age', y='cholesterol', hue='target', data=data, palette="viridis")
    ax.set_title("Age vs Cholesterol by Target")
    st.pyplot(fig)

elif option == "Model Evaluation":
    st.title("Model Evaluation")
    X = data.drop('target', axis=1)
    y = data['target']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train an XGBoost model
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_val)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_val, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="viridis", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
