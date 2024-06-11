from flask import Flask, render_template, request
from joblib import load
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Muat model SVM dan CountVectorizer menggunakan joblib
model = load('svm_model.pkl')
vectorizer = load('count_vectorizer.pkl')

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

app = Flask(__name__)

def preprocess_text(text):
    # Normalisasi (membersihkan tanda baca)
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Normalisasi kata-kata yang sering disingkat atau salah
    text_normalized = text_clean
    text_normalized = re.sub(r'\bgk\b|\bg\b|\bngk\b|\bgak\b|\bngak\b|\bnggak\b|\bgx\b|\btdk\b|\bga\b', 'tidak', text_normalized)
    text_normalized = re.sub(r'\bgabisa\b', 'tidak bisa', text_normalized)
    text_normalized = re.sub(r'\btp\b|\btapi\b', 'namun', text_normalized)
    text_normalized = re.sub(r'\bok\b|\boke\b|\bokee\b|\bokk\b', 'baik', text_normalized)
    text_normalized = re.sub(r'\bsdh\b|\budh\b|\budah\b', 'sudah', text_normalized)
    text_normalized = re.sub(r'\bapk\b', 'aplikasi', text_normalized)
    text_normalized = re.sub(r'\beror\b', 'error', text_normalized)
    text_normalized = re.sub(r'\bbs\b|\bbsa\b', 'bisa', text_normalized)
    text_normalized = re.sub(r'\bdpt\b|\bdpat\b|\bdapet\b', 'dapat', text_normalized)
    text_normalized = re.sub(r'\byg\b', 'yang', text_normalized)
    text_normalized = re.sub(r'\bjgn\b|\bjngn\b|\bjangn\b', 'jangan', text_normalized)
    text_normalized = re.sub(r'\bblm\b|\bblum\b', 'belum', text_normalized)
    text_normalized = re.sub(r'\bkrn\b|\bkrna\b', 'karena', text_normalized)
    text_normalized = re.sub(r'\bjos\b', 'sangat baik', text_normalized)
    text_normalized = re.sub(r'\butk\b|\buntk\b', 'untuk', text_normalized)
    
    # Case folding
    text_lower = text_normalized.lower()
    
    # Tokenisasi
    tokens = text_lower.split()
    
    # Stemming dengan Sastrawi
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Gabungkan kembali menjadi satu string
    preprocessed_text = ' '.join(stemmed_tokens)
    
    return text_clean, text_normalized, text_lower, tokens, stemmed_tokens, preprocessed_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Preprocess the text input
    text_clean, text_normalized, text_lower, tokens, stemmed_tokens, preprocessed_text = preprocess_text(text)
    text_transformed = vectorizer.transform([preprocessed_text])
    
    # Debugging: Cetak semua tahap preprocessing
    print("Original text: ", text)
    print("Cleaned text: ", text_clean)
    print("Normalized text: ", text_normalized)
    print("Lowercase text: ", text_lower)
    print("Tokens: ", tokens)
    print("Stemmed tokens: ", stemmed_tokens)
    print("Transformed text (vectorized): ", text_transformed.toarray())
    
    # Prediksi dengan model
    prediction = model.predict(text_transformed)[0]
    
    # Debugging: Cetak hasil prediksi
    print("Prediction: ", prediction)

    if prediction == 'positif':
        sentiment = 'Sentimen Positif'
    else:
        sentiment = 'Sentimen Negatif'
    
    return render_template('index.html', prediction=sentiment, original_text=text, cleaned_text=text_clean, 
                           normalized_text=text_normalized, lowercase_text=text_lower, tokens=tokens, 
                           stemmed_tokens=stemmed_tokens, preprocessed_text=preprocessed_text)

if __name__ == '__main__':
    app.run(debug=True)
