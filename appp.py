import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import time
import sklearn
# import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier




data = 'https://raw.githubusercontent.com/Lilik-murdiyah/Dataset/main/klinikparu800.csv'
df = pd.read_csv(data)

def fill_null_with_mode(df):
    for kolom in df.columns:
        modus = df[kolom].mode()[0]
        df[kolom].fillna(modus, inplace=True)
    return df

def normalize_data(df):
  columns_to_normalize = ['Umur', 'Saturasi', 'Paru Kesan', 'Tensi', 'Nafas', 'Nadi']
  scaler = MinMaxScaler()
  df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
  return df

def transform_categorical_columns(df, columns_to_transform):
    for column in columns_to_transform:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
    return df

def split_data(df_filled, test_size, random_state=42):
    x = df_filled.drop('Diagnosa', axis=1)
    y = df_filled['Diagnosa']
    # Bagi data menjadi set pelatihan dan pengujian
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test

def resample_data(x_train, y_train, max_imbalance_ratio=1.5, min_positive_ratio=0.4):
    total_no_train_data = sum(y_train == 0)
    total_asma_train_data = sum(y_train == 1)
    n_samples_to_add = 0
    while (total_no_train_data / (total_asma_train_data + n_samples_to_add) >= max_imbalance_ratio) and \
          (total_asma_train_data + n_samples_to_add) / (total_no_train_data + total_asma_train_data + n_samples_to_add) <= min_positive_ratio:
        n_samples_to_add += 1
    sm = SMOTE(random_state=42, sampling_strategy={1: total_asma_train_data + n_samples_to_add})
    x_resampled, y_resampled = sm.fit_resample(x_train, y_train)
    return x_resampled, y_resampled

def show_class_distribution(y_before, y_after, title_before, title_after):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Diagram distribusi sebelum SMOTE
    class_counts_before = Counter(y_before)
    class_labels = ["no/0", "asma/1"]  
    ax1.bar(class_labels, class_counts_before.values())
    for i, v in enumerate(class_counts_before.values()):
        ax1.text(i, v, str(v), ha='center', va='bottom')
    ax1.set_xlabel("Kelas")
    ax1.set_ylabel("Jumlah Sampel")
    ax1.set_title(title_before)
    # Diagram distribusi setelah SMOTE
    class_counts_after = Counter(y_after)
    ax2.bar(class_labels, class_counts_after.values())
    for i, v in enumerate(class_counts_after.values()):
        ax2.text(i, v, str(v), ha='center', va='bottom')
    ax2.set_xlabel("Kelas")
    ax2.set_ylabel("Jumlah Sampel")
    ax2.set_title(title_after)
    st.pyplot(fig)

def validate_input(col, value, min_value, max_value, message):
    try:
        while value < min_value or value > max_value:
            st.write(message)
    except TypeError: 
        st.write(message)
    return value




def home():
    st.title("Penyakit Asma")
    st.markdown("""<hr style="border: 4px solid black;">""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Latar Belakang", "Arsitektur Sistem", "Dataset"])

    with tab1 :
        st.subheader("Latar Belakang")
        st.write('Salah satu penyakit penyebab kematian di Indonesia adalah asma. Penyakit ini ditandai dengan terjadinya penyempitan hingga peradangan saluran udara. Hal ini menyebabkan  batuk, kesulitan bernapas, mengi, dan dada terasa berat. Diagnosis asma membutuhkan pemeriksaan fisik hingga tes fungsi paru-paru dengan berbagai kriteria. Hal ini terkadang sulit dikarenakan perlu pertimbangan khusus oleh dokter spesialis. Oleh karena itu perlu adanya klasifikasi penderita penyakit asma. Machine learning dapat melakukan proses klasifikasi salah satunya menggunakan metode Gradient Boosting. Penelitian ini bertujuan untuk membangun model klasifikasi asma menggunakan Gradient Boosting, serta menganalisis pengaruh penggunaan SMOTE untuk menangani ketidakseimbangan kelas pada data rekam medis asma di RSUD Sumberrejo Bojonegoro. Tujuannya untuk mengetahui risiko penyakit asma berdasarkan gejala yang dialami. Diharapkan penelitian ini dapat menghasilkan model klasifikasi penyakit asma yang efektif dan akurat, membantu tenaga medis untuk diagnosis asma lebih cepat dan tepat sehingga dapat memberikan penanganan yang sesuai terhadap pasien.')
    with tab2 :
        st.subheader("Arsitektur Sistem Usulan")
        image_url = 'https://raw.githubusercontent.com/Lilik-murdiyah/Dataset/main/arsitektur%20sistem.png'
        st.image(image_url, caption='Arsitektur Sistem', width=700)
        st.write('Bagian ini berisi mengenai gambaran dan penjelasan arsitektur sistem penerapan metode Gradient Boosting untuk klasifikasi penyakit asma. Membangun desain arsitektur sistem bertujuan untuk mempermudah dalam analisis proses pada suatu sistem, dengan begitu sistem yang akan dibangun sesuai dengan apa yang diharapkan. Arsitektur sistem pada penelitian ini, alurnya akan dijelaskan secara rinci dengan menggunakan diagram I-P-O yang memberikan gambaran tahap input, proses, hingga output.Pada bagian input akan dimasukkan rekap data penyakit asma yang diperoleh dari Rumah Sakit Umum Daerah Sumberrejo, Kabupaten Bojonegoro, Jawa Timur. Data diinputkan dalam ekstensi csv, data ini yang kemudian akan diolah untuk pelatihan dalam membangun sistem klasifikasi penyakit asma. ')
        st.write('Bagian selanjutnya adalah proses yang terdiri dari preprocessing, SMOTE, hyperparameter tuning dengan Grid Search, pelatihan model Gradient Boosting, dan evaluasi. Tahap preprocessing yang dilakukan antara lain mengatasi missing value, transformasi data, normalisasi data, dan split data. Saat melakukan split data, data dibagi menjadi data training dan data testing. Lalu setelah dilakukan split data, data training akan melalui proses SMOTE, yaitu penyeimbangan jumlah banyaknya data, dikarenakan antara data dengan label asma dan tidak asma jumlahnya sangat tidak seimbang, maka data perlu diseimbangkan dengan penggunaan teknik SMOTE. Selanjutnya data yang telah di SMOTE diproses dengan Grid Search untuk mencari parameter yang optimal untuk pelatihan, proses ini dikenal dengan istilah hyperparameter tuning. Setelah mendapat parameter yang optimal, setiap data training akan dilatih dengan model Gradient Boosting hasil dari proses Grid Search. Model tersebut kemudian dievaluasi menggunakan Confusion Matriks untuk mengetahui nilai accuracy, precision, recall, dan f1-score. Kemudian bagian output merupakan keluaran yang menghasilkan model klasifikasi penyakit asma yang optimal dari berbagai skenario model yang telah diusulkan. ')

    with tab3 :
        st.subheader("Dataset")
        st.write('Penelitian ini akan menggunakan data primer dan sekunder yang didapatkan secara langsung dari Rumah Sakit Umum Daerah Sumberrejo, Kabupaten Bojonegoro, Jawa Timur. Data yang diperoleh merupakan data rekam medis dari salah satu klinik yang ada di RSUD Sumberrejo yaitu klinik paru-paru, sehingga data tersebut merupakan data dari hasil pemeriksaan pasien check up atau rawat jalan. Diagnosis asma dilakukan dengan menjalani pemeriksaan spirometri untuk mengukur kinerja paru-paru berdasarkan volume udara dan jumlah total udara yang dihembuskan atau bisa dilakukan dengan tes difusi gas, tes ini berguna dalam mengetahui kemampuan paru-paru menyerap oksigen dan gas lain dalam proses pernapasan. Pemeriksaan gejala batuk, sesak napas dan pola pernapasan pasien juga dilakukan. Dataset yang didapatkan berjumlah 800 record yang diambil pada rentang waktu 3 tahun, yaitu mulai tahun 2022 hingga tahun 2024. ')
        st.write('Dataset yang diperoleh memiliki 10 fitur. Fitur-fitur ini dipilih berdasarkan penelitian langsung yang dibimbing oleh dokter klinik paru-paru RSUD Sumberrejo. Dalam pemilihan fitur tersebut, tidak hanya melibatkan pengetahuan dan pengalaman klinis dari dokter, tetapi juga didukung oleh pertimbangan dari literatur ilmiah dan jurnal penelitian yang relevan. Proses ini memastikan bahwa setiap fitur yang dipilih memiliki hubungan yang relevan dengan diagnosis penyakit asma. Berikut merupakan dataset yang digunakan dalam penelitian ini:')
        st.dataframe(df, use_container_width=True, hide_index=True)

def program():
    st.title("Implementasi Model")
    st.markdown("""<hr style="border: 4px solid black;">""", unsafe_allow_html=True)
    tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Exploratory Data Analysis", "Data Cleaning", "Transformasi","Normalisasi","Penyeimbangan Data", "Klasifikasi"])

    with tab4 :
        # Menghitung jumlah data asma dan yang tidak asma
        jumlah_diagnosa = df['Diagnosa'].value_counts()
        plt.figure(figsize=(1.5,1.5))
        plt.pie(jumlah_diagnosa, labels=jumlah_diagnosa.index, autopct='%1.1f%%')
        plt.title('Diagram Distribusi Data Asma')
        plt.axis('equal')
        st.pyplot(plt)

    with tab5 :
        st.subheader("Missing value")
        st.write('Jumlah Data Kosong')
        st.write(df.isnull().sum())  # Menampilkan jumlah data kosong
        df_filled= fill_null_with_mode(df.copy())
        st.write('Dataframe Setelah Pengisian Nilai Null dengan Nilai Modus:')  
        st.write(df_filled)  # Menampilkan dataframe setelah pengisian nilai null
        st.write('Jumlah Data Kosong setelah Pengisian Tiap Kolom:')
        st.write(df_filled.isnull().sum())  # Menampilkan jumlah data kosong

        st.subheader('Validasi Type Data')
        st.write('Tipe Data Awal:')
        st.write(df.dtypes)  # Menampilkan tipe data awal
        df_filled['Saturasi'] = df_filled['Saturasi'].astype('int64') # Konversi kolom tertentu ke tipe data int64
        df_filled['Nafas'] = df_filled['Nafas'].astype('int64')
        st.write('Tipe Data  Fitur Saturasi, Nafas seharusnya bertipe data integer, maka harus dikonversi menjadi :')
        st.write(df_filled.dtypes)  # Menampilkan tipe data setelah konversi

        st.subheader('Duplikat Data')
        duplicate_rows = df_filled[df_filled.duplicated()]
        num_duplicate_rows = len(duplicate_rows)
        st.write(f"Jumlah baris duplikat sebelum penghapusan: {num_duplicate_rows}")
        df_filled = df_filled.drop_duplicates()
        # cek duplikat data lagi
        duplicate_rows = df_filled[df_filled.duplicated()]
        num_duplicate_rows = len(duplicate_rows)
        st.write(f"Jumlah baris duplikat setelah penghapusan: {num_duplicate_rows}")
        st.dataframe(df_filled)

    with tab6 :
        st.subheader("Transformasi")
        st.write(" Data Sebelum Transformasi")
        st.write(df_filled)
        st.write(" Data Setelah Transformasi")
        columns_to_transform = ['JK', 'Batuk', 'Sesak', 'Paru Kesan', 'Tensi', 'Diagnosa']
        df_filled = transform_categorical_columns(df_filled, columns_to_transform)
        st.write(df_filled) 

    with tab7 :
        st.subheader("Normalisasi")
        st.write(" Data Sebelum Normalisasi")
        st.write(df_filled)
        st.write(" Data Setelah Normalisasi")
        df_filled = normalize_data(df_filled)
        st.write(df_filled)
     

    with tab8 :
        st.subheader("Split dan Penyeimbangan Data")
        st.write('Proses dari split data') 
        # Pilihan ukuran test set
        test_size_options = {
            "70:30": 0.3,
            "80:20": 0.2,
            "90:10": 0.1
        }
        selected_option = st.selectbox("Pilih ukuran test set:", options=list(test_size_options.keys()))
        # Lakukan pembagian data secara otomatis
        x_train, x_test, y_train, y_test = split_data(df_filled, test_size=test_size_options[selected_option])
        st.write("Ukuran data pelatihan:", x_train.shape)
        st.write("Ukuran data pengujian:", x_test.shape)

        st.write('Proses dari penyeimbangan data')
        x_resampled, y_resampled = resample_data(x_train, y_train, max_imbalance_ratio=1.5, min_positive_ratio=0.4)
        show_class_distribution(y_train, y_resampled, "Distribusi Kelas Sebelum SMOTE", "Distribusi Kelas Setelah SMOTE")



    with tab9 :
        st.subheader("Klasifikasi")
        x_train, x_test, y_train, y_test = split_data(df_filled, test_size=0.3)
        x_resampled, y_resampled = resample_data(x_train, y_train, max_imbalance_ratio=1.5, min_positive_ratio=0.4)

        # Train model
        model = GradientBoostingClassifier(
            learning_rate=0.2,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=4,
            n_estimators=12
        )
        model.fit(x_resampled, y_resampled)

        # Prediksi
        y_pred = model.predict(x_test)

        joblib.dump(model, 'my_model.joblib')

        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.write("Akurasi Training:", model.score(x_train, y_train))
        st.write("Akurasi Testing:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1-Score:", f1)

        # Waktu komputasi
        start_time = time.time()
        model.fit(x_train, y_train)
        end_time = time.time()
        computation_time = end_time - start_time
        st.write("Waktu komputasi:", computation_time, "detik")

        image_url = 'https://raw.githubusercontent.com/Lilik-murdiyah/Dataset/main/smote70-cf.png'
        st.image(image_url, caption='Gambar Confussion Matriks', width=700)

        image_url = 'https://raw.githubusercontent.com/Lilik-murdiyah/Dataset/main/smote70-graf.png'
        st.image(image_url, caption='Gambar Perbandingan Akurasi Train dan Test', width=700)

        image_url = 'https://raw.githubusercontent.com/Lilik-murdiyah/Dataset/main/treee.png'
        st.image(image_url, caption='Hasil Pohon Percabangan', width=1500)

def uji_coba():
    st.title('Prediksi Data Baru')
    st.write('Masukkan data untuk di prediksi:')

    # Load model
    model = joblib.load('my_model.joblib')
    input_data = {}
    for col in df.columns[0:9]:
        if col == 'Umur':
            input_data[col] = validate_input({col},st.number_input('Input Umur', min_value=1), 1, None, "Umur harus lebih dari 0.")
        elif col == 'JK':
            gender = st.selectbox(f'Input Jenis Kelamin', ('Laki-laki', 'Perempuan'))
            input_data[col] = 1 if gender == 'Perempuan' else 0
        elif col == 'Batuk' or col == 'Sesak':
            yes_or_no = st.selectbox(f'Input Status {col}', ('Ya', 'Tidak'))
            input_data[col] = 0 if yes_or_no == 'Ya' else 1
        elif col == 'Saturasi':
            input_data[col] = validate_input({col},st.number_input('Input Saturasi (%)', min_value=1, max_value=100), 1, 100, "Saturasi harus antara 1-100.")
        elif col == 'Paru Kesan':
            paru_kesan = st.selectbox(f'Input {col}', ('dbn', 'whezing', 'ronkhi'))
            input_data[col] = 0 if paru_kesan == 'dbn' else (1 if paru_kesan == 'ronkhi' else 2)
        elif col == 'Tensi':
            tensi = st.selectbox(f'Input {col}', ('rendah', 'normal', 'tinggi'))
            input_data[col] = 0 if tensi == 'rendah' else (1 if tensi == 'normal' else 2)
        elif col == 'Nafas':
            input_data[col] = validate_input({col},st.number_input('Input Nafas(/menit)', min_value=10, max_value=40), 10, 40, "Napas harus antara 10-40.")
        elif col == 'Nadi':
            input_data[col] = validate_input({col},st.number_input('Input Nadi (/menit)', min_value=50, max_value=150), 50, 150, "Nadi harus antara 50-150.")
        else:
            st.write('tidak ada')

    prediction = model.predict(np.array(list(input_data.values())).reshape(1, -1))

    if prediction == 1:
        st.write("Prediksi: Pasien kemungkinan terkena asma")
    else:
        st.write("Prediksi: Pasien kemungkinan tidak terkena asma")






# Konfigurasi sidebar dan menu
with st.sidebar:
    selected = option_menu(
        "Klasifikasi Asma",
        ["Home", "Implementasi", "Uji Coba"],
        icons=["house", "gear", "bar-chart-line", "play-circle"],
        default_index=0
    )
# Menampilkan halaman sesuai dengan pilihan pengguna
if selected == "Home":
    home()
elif selected == "Implementasi":
    program()
elif selected == "Uji Coba":
    uji_coba()
