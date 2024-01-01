# Import library yang diperlukan
from flask import Flask, request, jsonify, render_template  # Untuk membuat aplikasi web Flask
import pandas as pd  # Untuk manipulasi data, terutama CSV
from sklearn.model_selection import train_test_split  # Untuk membagi dataset menjadi data latih dan uji
from sklearn.feature_extraction.text import TfidfVectorizer  # Untuk mengubah teks menjadi fitur numerik
from sklearn.ensemble import RandomForestClassifier  # Model klasifikasi yang digunakan
import joblib  # Untuk menyimpan dan memuat model machine learning
import os  # Untuk operasi file dan direktori

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# ---------------------- Persiapan Data ----------------------

# Path ke file CSV yang berisi data untuk pelatihan model
CSV_PATH = 'Data analysis.csv'

# Mencoba untuk membaca file CSV, jika gagal akan menampilkan pesan error
try:
    data = pd.read_csv(CSV_PATH, sep=';', on_bad_lines='skip')  # Membaca data CSV
except Exception as e:
    print(f"Error saat membaca file CSV: {e}")  # Menangkap dan menampilkan error jika terjadi
    exit()  # Menghentikan program jika file tidak dapat dibaca

# Memastikan bahwa data memiliki kolom 'text' dan 'label'
if 'text' not in data.columns or 'label' not in data.columns:
    raise ValueError("File CSV harus memiliki kolom 'text' dan 'label'!")  # Menangani kasus jika kolom tidak ada

# Memisahkan kolom 'text' sebagai fitur dan 'label' sebagai target
X = data['text']
y = data['label']

# Mengubah teks menjadi representasi numerik menggunakan TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)  # Melakukan transformasi pada data teks

# Membagi data menjadi data latih dan data uji dengan rasio 80% latih dan 20% uji
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Path untuk menyimpan model yang telah dilatih
MODEL_PATH = "stacking_ensemble_model.pkl"

# Melatih model RandomForest jika model belum tersedia
model = RandomForestClassifier(n_estimators=50, random_state=42)  # Inisialisasi model Random Forest
model.fit(X_train, y_train)  # Melatih model dengan data latih

# Menyimpan model dan vectorizer menggunakan joblib untuk digunakan di kemudian hari
joblib.dump({'model': model, 'vectorizer': tfidf}, MODEL_PATH)

# ---------------------- Flask Routes ----------------------

@app.route('/')
def home():
    """
    Menangani route untuk halaman utama aplikasi.
    """
    return render_template('index.html')  # Mengembalikan halaman utama (HTML)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Menangani route untuk menerima input dari pengguna dan melakukan prediksi.
    """
    try:
        # Mengambil input teks dari form yang dikirimkan melalui POST
        input_text = request.form.get('text')  
        if not input_text:
            return jsonify({'error': 'Input teks tidak boleh kosong!'})  # Menangani jika input kosong

        # Mengubah input teks menjadi bentuk numerik menggunakan vectorizer
        input_vectorized = tfidf.transform([input_text])

        # Melakukan prediksi dengan model yang telah dilatih
        prediction = model.predict(input_vectorized)[0]
        return jsonify({'prediction': prediction})  # Mengembalikan hasil prediksi dalam format JSON
    except Exception as e:
        return jsonify({'error': str(e)})  # Menangani error dan mengembalikan pesan error

# ---------------------- Menjalankan Server ----------------------

if __name__ == '__main__':
    """
    Menjalankan server Flask dengan mode debug diaktifkan untuk pengembangan.
    """
    app.run(debug=True)  # Menjalankan aplikasi dengan debug mode aktif
