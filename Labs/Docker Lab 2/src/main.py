from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, static_folder='statics')

# Load the trained model and scaler
model, sc = joblib.load('iris_rf_model.joblib')
class_labels = ['Setosa', 'Versicolor', 'Virginica']


@app.route('/')
def home():
    return "Welcome to the Iris Random Forest Classifier API!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form
            sepal_length = float(data['sepal_length'])
            sepal_width = float(data['sepal_width'])
            petal_length = float(data['petal_length'])
            petal_width = float(data['petal_width'])

            # Prepare and scale input
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = sc.transform(input_data)

            # Make prediction
            predicted_class = class_labels[model.predict(input_scaled)[0]]

            return jsonify({"predicted_class": predicted_class})
        except Exception as e:
            return jsonify({"error": str(e)})
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return "Unsupported HTTP method"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)
