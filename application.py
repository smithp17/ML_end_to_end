from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Create Flask app
application = Flask(__name__)

app= application

# Load the saved model and preprocessing pipeline (if any)
model = pickle.load(open('artifacts/model.pkl', 'rb'))
preprocessor = pickle.load(open('artifacts/preprocessor.pkl', 'rb'))  # Assuming you have a saved preprocessor

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting form data
        data = {
            'gender': [request.form['gender']],
            'race_ethnicity': [request.form['race_ethnicity']],
            'parental_level_of_education': [request.form['parental_level_of_education']],
            'lunch': [request.form['lunch']],
            'test_preparation_course': [request.form['test_preparation_course']],
            'reading_score': [float(request.form['reading_score'])],
            'writing_score': [float(request.form['writing_score'])]
        }
        
        # Convert data to DataFrame
        data_df = pd.DataFrame(data)
        
        # Apply the same preprocessing steps as during training
        final_data = preprocessor.transform(data_df)  # Use the preprocessor you saved

        # Make a prediction
        prediction = model.predict(final_data)

        # Render the result
        return render_template('result.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")
