from flask import Flask, render_template, request
import joblib
import pandas as pd

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load pipeline model yang sudah disimpan
model = joblib.load('Engagement-Prediction/best_model.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ambil data dari form
        data = {
            'Age': int(request.form['Age']),
            'Gender': request.form['Gender'],
            'Marital Status': request.form['Marital_Status'],
            'Occupation': request.form['Occupation'],
            'Monthly Income': request.form['Monthly_Income'],
            'Educational Qualifications': request.form['Educational_Qualifications'],
            'Family size': int(request.form['Family_size']),
            'Feedback': request.form['Feedback']
        }

        # Ubah ke dataframe
        input_df = pd.DataFrame([data])

        # Prediksi
        prediction = model.predict(input_df)[0]

        # Hasil ke HTML
        return render_template('result.html', prediction=prediction)

# Jalankan app
if __name__ == '__main__':
    app.run(debug=True)
