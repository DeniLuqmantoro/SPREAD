import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import streamlit as st

# Load dataset
df = pd.read_csv('diabetes.csv')

# Split dataset menjadi fitur dan target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = model.predict(X_test)

# Fungsi prediksi untuk input pengguna
def predict(diabetes):
    pred = model.predict(diabetes)
    return pred[0]

# Tampilan web menggunakan Streamlit
st.set_page_config(page_title="SPREAD", page_icon=":hospital:")
st.title("Sistem Prediksi Penyakit Diabetes")

st.image('diabetes_image.jpg', use_column_width=True)  # Gantilah dengan nama file gambar yang sesuai

# Custom CSS
st.markdown("""
    <style>
            
        .styled-table {
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: sans-serif;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }
        .styled-table th {
            background-color: #009879;
            color: #ffffff;
            text-align: left;
        }
        .styled-table th, .styled-table td {
            padding: 12px 15px;
        }
        .styled-table tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .styled-table tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        .styled-table tbody tr:last-of-type {
            border-bottom: 2px solid #009879;
        }
    </style>
""", unsafe_allow_html=True)

# Input Form
with st.sidebar:
    st.subheader('Masukkan Data Pasien')
    Pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=17, value=0)
    Glucose = st.number_input("Kadar Glukosa", min_value=0, max_value=200, value=120)
    BloodPressure = st.number_input("Tekanan Darah Diastolik", min_value=0, max_value=122, value=69)
    SkinThickness = st.number_input("Ketebalan Kulit", min_value=0, max_value=99, value=20)
    Insulin = st.number_input("Kadar Insulin", min_value=0, max_value=846, value=79)
    BMI = st.number_input("BMI", min_value=0.0, max_value=67.1, value=0.0, step=0.1)
    DiabetesPedigreeFunction = st.number_input("Fungsi Silsilah Diabetes", min_value=0.078, max_value=2.42, value=0.078, step=0.001)
    Age = st.number_input("Umur", min_value=21, max_value=81, value=21)

    # Memasukkan data pasien ke dalam array
    diabetes = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

# Menampilkan hasil prediksi
if st.button("Prediksi"):
    result = predict(diabetes.reshape(1, -1))
    if result == 1:
        st.error("Pasien kemungkinan besar memiliki diabetes")
    else:
        st.success("Pasien kemungkinan tidak memiliki diabetes")
