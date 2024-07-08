import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

df = pd.read_csv('diabetes.csv')
df.drop(columns=['Pregnancies', 'SkinThickness'], inplace=True)
features = ['Glucose', 'BloodPressure', 'Insulin', 'Age', 'DiabetesPedigreeFunction']
X = df[features]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Predicción de Diabetes")

st.sidebar.header("Parámetros de Entrada")

def user_input_features():
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('BloodPressure', 0, 122, 70)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    age = st.sidebar.slider('Age', 21, 81, 33)
    diabetes_pedigree_function = st.sidebar.slider('DiabetesPedigreeFunction', 0.0, 2.4, 0.5)
    data = {'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'Insulin': insulin,
            'Age': age,
            'DiabetesPedigreeFunction': diabetes_pedigree_function}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('Parámetros de Entrada')
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.decision_function(input_df)

st.subheader('Predicción')
st.write('Diabetes' if prediction[0] == 1 else 'No Diabetes')

st.subheader('Probabilidad')
st.write(prediction_proba[0])
