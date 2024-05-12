# Mengimpor library yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

st.title('Prediksi Risiko Kanker Paru-paru')
st.write('Silakan isi data berikut untuk melakukan prediksi:')

# Membaca dataset
dataset = pd.read_csv('./survey_lung_cancer.csv')

# Menghapus fitur yang tidak terpakai
# dataset.drop(['PEER_PRESSURE'], axis=1, inplace=True)

# Menggunakan one-hot encoding untuk kolom 'GENDER'
dataset = pd.get_dummies(dataset, columns=['GENDER'])

# print(dataset.head())

# Memisahkan fitur (features) dan target (label)
X = dataset.drop('LUNG_CANCER', axis=1)
y = dataset['LUNG_CANCER']

# Membagi dataset menjadi data latih (train) dan data uji (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Decision Tree
model = DecisionTreeClassifier()

# Melatih model menggunakan data latih
model.fit(X_train, y_train)

# Menampilkan tingkat pentingnya setiap fitur
importance = model.feature_importances_

# Membuat DataFrame untuk menampilkan hasil
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})

# Mengurutkan berdasarkan tingkat pentingnya
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# # Menampilkan atribut paling berpengaruh
# print("Atribut yang paling berpengaruh:")
# print(feature_importance_df)    

# Membuat prediksi menggunakan data uji
y_pred = model.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
# print(f'Akurasi: {accuracy}')

# Menampilkan laporan klasifikasi
# print(classification_report(y_test, y_pred))

# CONTOH PREDIKSI DATA BARU
# print("Prediksi data baru\n")

gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
age = st.slider('Usia', min_value=0, max_value=150, step=1)
smoking = st.selectbox('Merokok?', ['Tidak', 'Ya'])
yellow_fingers = st.selectbox('Jari Kuning?', ['Tidak', 'Ya'])
anxiety = st.selectbox('Kecemasan?', ['Tidak', 'Ya'])
peer_pressure = st.selectbox('Tekanan dari Teman?', ['Tidak', 'Ya'])
chronic_disease = st.selectbox('Penyakit Kronis?', ['Tidak', 'Ya'])
fatigue = st.selectbox('Kelelahan?', ['Tidak', 'Ya'])
allergy = st.selectbox('Alergi?', ['Tidak', 'Ya'])
wheezing = st.selectbox('Napas Bersuara?', ['Tidak', 'Ya'])
alcohol_consuming = st.selectbox('Konsumsi Alkohol?', ['Tidak', 'Ya'])
coughing = st.selectbox('Batuk?', ['Tidak', 'Ya'])
shortness_of_breath = st.selectbox('Sesak Napas?', ['Tidak', 'Ya'])
swallowing_difficulty = st.selectbox('Kesulitan Menelan?', ['Tidak', 'Ya'])
chest_pain = st.selectbox('Nyeri Dada?', ['Tidak', 'Ya'])

# Mengkodekan fitur-fitur yang bersifat biner
gender_encoded = 2 if gender == 'Perempuan' else 1
smoking_encoded = 2 if smoking == 'Ya' else 1
yellow_fingers_encoded = 2 if yellow_fingers == 'Ya' else 1
anxiety_encoded = 2 if anxiety == 'Ya' else 1
peer_pressure_encoded = 2 if peer_pressure == 'Ya' else 1
chronic_disease_encoded = 2 if chronic_disease == 'Ya' else 1
fatigue_encoded = 2 if fatigue == 'Ya' else 1
allergy_encoded = 2 if allergy == 'Ya' else 1
wheezing_encoded = 2 if wheezing == 'Ya' else 1
alcohol_consuming_encoded = 2 if alcohol_consuming == 'Ya' else 1
coughing_encoded = 2 if coughing == 'Ya' else 1
shortness_of_breath_encoded = 2 if shortness_of_breath == 'Ya' else 1
swallowing_difficulty_encoded = 2 if swallowing_difficulty == 'Ya' else 1
chest_pain_encoded = 2 if chest_pain == 'Ya' else 1

if st.button('Prediksi'):
    if age > 0:
        new_data = {
            # YES=2 , NO=1.
            'AGE': [age],
            'SMOKING': [smoking_encoded],
            'YELLOW_FINGERS': [yellow_fingers_encoded],
            'ANXIETY': [anxiety_encoded],
            'PEER_PRESSURE': [peer_pressure_encoded],
            'CHRONIC_DISEASE': [chronic_disease_encoded],
            'FATIGUE': [fatigue_encoded],
            'ALLERGY': [allergy_encoded],
            'WHEEZING': [wheezing_encoded],
            'ALCOHOL_CONSUMING': [alcohol_consuming_encoded],
            'COUGHING': [coughing_encoded],
            'SHORTNESS_OF_BREATH': [shortness_of_breath_encoded],
            'SWALLOWING_DIFFICULTY': [swallowing_difficulty_encoded],
            'CHEST_PAIN': [chest_pain_encoded],
            'GENDER': [gender_encoded],
        }

        new_data_df = pd.DataFrame(new_data, columns=X_train.columns)

        prediction = model.predict(new_data_df)

        if prediction[0] == "YES":
            st.write(f'Hasil prediksi menyatakan anda MEMILIKI penyakit paru-paru.')
        else:
            st.write(f'Hasil prediksi menyatakan anda TIDAK MEMILIKI penyakit paru-paru.')
    else:
        st.write(f'Masukkan data usia dengan benar!')
