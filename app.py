from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# 1. Load model dan scaler yang sudah diekspor dari Jupyter Notebook
# Karena 'model.pkl' berisi list [model_dt, model_svc], kita load ke dalam variabel 'loaded_models'
loaded_models = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Pilih salah satu model untuk digunakan (indeks 0 = Decision Tree, indeks 1 = SVC)
# Di sini kita gunakan Decision Tree (indeks 0) sesuai default jobsheet
model = loaded_models[0] 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. Ambil data inputan dari form HTML
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bloodpressure = float(request.form['bloodpressure'])
        skinthickness = float(request.form['skinthickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        # 3. Gabungkan data menjadi array 2D untuk prediksi
        features = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])

        # 4. Lakukan transformasi (scaling) menggunakan scaler yang sudah di-load
        features_scaled = scaler.transform(features)

        # 5. Lakukan prediksi
        prediction = model.predict(features_scaled)

        # 6. Interpretasikan hasil prediksi (0 = Negatif, 1 = Positif)
        if prediction[0] == 1:
            result = "Positif Diabetes"
            alert_class = "alert-danger"
        else:
            result = "Negatif Diabetes"
            alert_class = "alert-success"

        return render_template('index.html', prediction_text=f'Hasil Prediksi: {result}', alert_class=alert_class)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}', alert_class='alert-warning')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)