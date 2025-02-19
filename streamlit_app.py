import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# App title and description
st.title("Iris Flower Classification App")
st.write("This app predicts the species of an iris flower based on its features.")

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map target numbers to species names
species_mapping = {i: species for i, species in enumerate(iris.target_names)}
df['species_name'] = df['species'].map(species_mapping)
st.write("Sample of the Iris Dataset:", df.head())

# Split the data for model training
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sidebar inputs for feature values
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

# Prediction button
if st.sidebar.button("Predict"):
    user_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)
    st.success(f"Predicted Species: {species_mapping[prediction[0]]}")
    st.write("Prediction Probabilities:", prediction_proba)

# Data visualization: Display the distribution of sepal length
st.subheader("Sepal Length Distribution")
fig, ax = plt.subplots()
sns.histplot(df['sepal length (cm)'], bins=20, kde=True, ax=ax)
st.pyplot(fig)
