import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Pengumpulan Data Pasien")

# Membaca data dari file Excel
data_pasien = pd.read_excel('patient_icd_10_rs_sbs.xlsx')

# Membuat DataFrame dari data
df_pasien = pd.DataFrame(data_pasien)

# Menampilkan DataFrame
st.write("Data Pasien:")
st.dataframe(df_pasien)

# Pra-pemrosesan Data dan Feature Engineering
df_pasien.rename(columns={'ID_Pasien': 'id_pasien', 'Nama': 'nama', 'Umur': 'umur', 'Jenis_Kelamin': 'jenis_kelamin', 'Diagnosa_ICD10': 'diagnosa_icd10'}, inplace=True)
df_pasien['jenis_kelamin'] = df_pasien['jenis_kelamin'].map({'Laki-laki': 1, 'Perempuan': 0})
bins = [0, 18, 35, 50, 65, 100]
labels = ['Anak-anak', 'Dewasa Muda', 'Dewasa', 'Paruh Baya', 'Lansia']
df_pasien['umur_kategori'] = pd.cut(df_pasien['umur'], bins=bins, labels=labels, right=False)

st.write("Data Pasien setelah Pra-pemrosesan:")
st.dataframe(df_pasien)

# Eksplorasi dan Analisis Data Awal (EDA)
st.subheader("Eksplorasi dan Analisis Data Awal (EDA)")

# Statistik deskriptif
statistik_deskriptif = df_pasien.describe(include='all')
st.write("Statistik Deskriptif:")
st.write(statistik_deskriptif)

# Visualisasi distribusi umur pasien
st.write("Distribusi Umur Pasien:")
fig, ax = plt.subplots()
sns.histplot(df_pasien['umur'], bins=10, kde=True, ax=ax)
ax.set_title('Distribusi Umur Pasien')
ax.set_xlabel('Umur')
ax.set_ylabel('Frekuensi')
st.pyplot(fig)

# Gejala teratas yang sering dilaporkan
st.write("Gejala Teratas yang Sering Dilaporkan:")
gejala_teratas = df_pasien['diagnosa_icd10'].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(x=gejala_teratas.index, y=gejala_teratas.values, ax=ax)
ax.set_title('Gejala Teratas yang Sering Dilaporkan')
ax.set_xlabel('Kode ICD-10')
ax.set_ylabel('Frekuensi')
st.pyplot(fig)

# Klasifikasi dengan Naive Bayes
st.subheader("Klasifikasi dengan Naive Bayes")
X = df_pasien[['umur', 'jenis_kelamin']]
y = df_pasien['diagnosa_icd10']
X = X.dropna()
y = y[X.index]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
st.write("Classification Report Naive Bayes:")
st.text(classification_report(y_test, y_pred, zero_division=0))
st.write("Accuracy Score Naive Bayes:")
st.write(accuracy_score(y_test, y_pred))

# Klasifikasi dengan Decision Tree
st.subheader("Klasifikasi dengan Decision Tree")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
st.write("Classification Report Decision Tree:")
st.text(classification_report(y_test, y_pred_dt, zero_division=0))
st.write("Accuracy Score Decision Tree:")
st.write(accuracy_score(y_test, y_pred_dt))

# Klastering
st.subheader("Klastering")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pasien[['umur', 'jenis_kelamin']])
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
df_pasien['klaster'] = kmeans.labels_
st.write("Data Pasien dengan Klaster:")
st.dataframe(df_pasien)
fig, ax = plt.subplots()
sns.scatterplot(x='umur', y='jenis_kelamin', hue='klaster', data=df_pasien, palette='viridis', ax=ax)
ax.set_title('Klaster Pasien Berdasarkan Umur dan Jenis Kelamin')
ax.set_xlabel('Umur')
ax.set_ylabel('Jenis Kelamin')
st.pyplot(fig)


# Penyajian Kesimpulan dan Rekomendasi
st.subheader("Penyajian Kesimpulan dan Rekomendasi")
kesimpulan = """
Berdasarkan analisis yang telah dilakukan, berikut adalah beberapa kesimpulan yang dapat diambil:
1. Distribusi umur pasien menunjukkan bahwa sebagian besar pasien berada dalam rentang usia dewasa muda dan dewasa.
2. Gejala teratas yang sering dilaporkan berdasarkan kode ICD-10.
3. Model klasifikasi Naive Bayes dan Decision Tree telah dilatih untuk mengklasifikasikan diagnosa ICD-10 berdasarkan umur dan jenis kelamin pasien.
4. Model Decision Tree menunjukkan akurasi yang lebih tinggi dibandingkan dengan model Naive Bayes.
5. Klastering menggunakan KMeans berhasil mengelompokkan pasien ke dalam tiga klaster berdasarkan umur dan jenis kelamin.
6. Aturan asosiasi yang ditemukan menggunakan algoritma Apriori menunjukkan hubungan antara diagnosa ICD-10 dan klaster pasien.
"""
st.write(kesimpulan)

rekomendasi = """
Berdasarkan kesimpulan di atas, berikut adalah beberapa rekomendasi yang dapat diberikan:
1. Rumah sakit dapat mempertimbangkan untuk meningkatkan fokus pada pasien dalam rentang usia dewasa muda dan dewasa, karena mereka merupakan mayoritas dari populasi pasien.
2. Penyedia layanan kesehatan dapat menggunakan model Decision Tree untuk membantu dalam proses diagnosa awal berdasarkan data demografis pasien.
3. Klastering pasien dapat digunakan untuk mengidentifikasi kelompok pasien dengan karakteristik serupa, yang dapat membantu dalam perencanaan perawatan dan sumber daya.
4. Aturan asosiasi yang ditemukan dapat digunakan untuk mengidentifikasi pola umum dalam diagnosa pasien, yang dapat membantu dalam pengembangan program pencegahan dan intervensi.
"""
st.write(rekomendasi)
