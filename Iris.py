import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
X = iris.data
Y = iris.target
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X, Y)

st.title("Iris flower type prediction App")
st.header("Enter the measurements of an iris flower to predict its type")

sepal_length_min, sepal_length_max, sepal_length_mean = min(iris.data[:, 0]), max(iris.data[:, 0]), iris.data[:, 0].mean()
sepal_width_min, sepal_width_max, sepal_width_mean = min(iris.data[:, 1]), max(iris.data[:, 1]), iris.data[:, 1].mean()
petal_length_min, petal_length_max, petal_length_mean = min(iris.data[:, 2]), max(iris.data[:, 2]), iris.data[:, 2].mean()
petal_width_min, petal_width_max, petal_width_mean = min(iris.data[:, 3]), max(iris.data[:, 3]), iris.data[:, 3].mean()

sepal_length = st.slider("Sepal length", sepal_length_min, sepal_length_max, sepal_length_mean)
sepal_width = st.slider("Sepal width", sepal_width_min, sepal_width_max, sepal_width_mean)
petal_length = st.slider("Petal length", petal_length_min, petal_length_max, petal_length_mean)
petal_width = st.slider("Petal width", petal_width_min, petal_width_max, petal_width_mean)

# Define a prediction button
if st.button('Predict'):
    # Get the input values
    input_data = [[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]]

    # Make the prediction using the trained classifier
    prediction = rfc.predict(input_data)

    # Show the prediction result
    st.write(f"The predicted iris flower type is: {iris.target_names[prediction[0]]}")