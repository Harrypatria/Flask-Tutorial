Flask Tutorial: Building Modern Web Applications with Python

Welcome to the Flask Tutorial, a comprehensive guide to building scalable and dynamic web applications using Flask, a lightweight and flexible Python web framework. Whether you're a beginner or an intermediate developer, this tutorial will walk you through Flask's core concepts, setup, and a practical example of deploying a machine learning model with a user-friendly interface.

Table of Contents

What is Flask?
Why Choose Flask?
Prerequisites
Project Setup
Sample Project: ML Model Deployment
Project Structure
Step-by-Step Implementation


Running the Application
Testing the Application
Best Practices
Additional Resources
License


What is Flask?
Flask is a micro web framework for Python, designed to be simple, flexible, and extensible. It provides the essentials for building web applications without imposing rigid structures, allowing developers to choose their tools and libraries. Flask is ideal for small to medium-sized projects and serves as a foundation for larger applications with its modular design.

Why Choose Flask?

Lightweight and Modular: Flask is minimal, giving you control over components and dependencies.
Easy to Learn: Its simple API and clear documentation make it beginner-friendly.
Extensible: Integrate with databases, ORMs, and other tools like SQLAlchemy, Flask-RESTful, and more.
Active Community: A vibrant ecosystem with plugins and tutorials for rapid development.
Production-Ready: Used by companies like Netflix, Airbnb, and Reddit for scalable applications.


Prerequisites
Before starting, ensure you have the following installed:

Python 3.8+: Download from python.org.
pip: Python's package manager (included with Python).
Virtualenv (optional but recommended): For isolated project environments.
Basic knowledge of Python, HTML, and REST APIs.

Required Python packages:

Flask
scikit-learn
pandas
numpy
requests

Install them using:
pip install flask scikit-learn pandas numpy requests


Project Setup

Create a Project Directory:
mkdir flask-ml-deployment
cd flask-ml-deployment


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Create a requirements.txt file:
flask==3.0.3
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
requests==2.32.3

Install with:
pip install -r requirements.txt




Sample Project: ML Model Deployment
This tutorial demonstrates a Flask application that deploys a machine learning model to predict startup profits based on input features (e.g., R&D spend, marketing spend). Users can input data via a web interface or API, and the app returns predictions.
Project Structure
flask-ml-deployment/
├── static/
│   └── css/
│       └── style.css        # Custom styles for the web interface
├── templates/
│   └── index.html           # HTML template for user input and output
├── data/
│   └── 50_Startup.csv       # Dataset for training the ML model
├── model.pkl                # Serialized ML model
├── app.py                   # Flask application with API endpoints
├── model.py                 # Script to train and serialize the ML model
├── request.py               # Script to test API endpoints
├── requirements.txt         # Project dependencies
└── README.md                # This file

Step-by-Step Implementation
1. Prepare the Dataset
The 50_Startup.csv dataset contains columns: R&D Spend, Administration, Marketing Spend, State, and Profit. Download it from a reliable source or create a sample dataset.
2. Train the ML Model (model.py)
This script trains a linear regression model and saves it as model.pkl.
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv('data/50_Startup.csv')

# Features and target
X = data[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y = data['Profit']

# Preprocessing: One-hot encode 'State' column
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['State'])
    ], remainder='passthrough')

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl")

3. Build the Flask App (app.py)
This script creates a Flask app with a web interface and API endpoint.
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data
        rd_spend = float(request.form['rd_spend'])
        administration = float(request.form['administration'])
        marketing_spend = float(request.form['marketing_spend'])
        state = request.form['state']

        # Create DataFrame for prediction
        input_data = pd.DataFrame([[rd_spend, administration, marketing_spend, state]],
                                  columns=['R&D Spend', 'Administration', 'Marketing Spend', 'State'])

        # Predict
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction=f'Predicted Profit: ${prediction:,.2f}')
    
    return render_template('index.html', prediction=None)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    rd_spend = float(data['rd_spend'])
    administration = float(data['administration'])
    marketing_spend = float(data['marketing_spend'])
    state = data['state']

    input_data = pd.DataFrame([[rd_spend, administration, marketing_spend, state]],
                              columns=['R&D Spend', 'Administration', 'Marketing Spend', 'State'])

    prediction = model.predict(input_data)[0]
    return jsonify({'profit': prediction})

if __name__ == '__main__':
    app.run(debug=True)

4. Create the Web Interface (templates/index.html)
This HTML template provides a form for user input and displays predictions.
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Startup Profit Predictor</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Startup Profit Predictor</h1>
        <form method="POST">
            <label>R&D Spend ($):</label>
            <input type="number" name="rd_spend" step="0.01" required><br>
            <label>Administration ($):</label>
            <input type="number" name="administration" step="0.01" required><br>
            <label>Marketing Spend ($):</label>
            <input type="number" name="marketing_spend" step="0.01" required><br>
            <label>State:</label>
            <select name="state" required>
                <option value="California">California</option>
                <option value="Florida">Florida</option>
                <option value="New York">New York</option>
            </select><br>
            <button type="submit">Predict</button>
        </form>
        {% if prediction %}
            <h2>{{ prediction }}</h2>
        {% endif %}
    </div>
</body>
</html>

5. Add Styling (static/css/style.css)
This CSS file enhances the web interface.
body {
    font-family: Arial, sans-serif;
    background-color: #f4f4f9;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 600px;
    margin: 50px auto;
    padding: 20px;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    color: #333;
}

form {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

label {
    font-weight: bold;
    color: #555;
}

input, select {
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 16px;
}

button {
    padding: 10px;
    background-color: #28a745;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
}

button:hover {
    background-color: #218838;
}

h2 {
    text-align: center;
    color: #28a745;
    margin-top: 20px;
}

6. Test the API (request.py)
This script tests the /api/predict endpoint.
import requests

url = 'http://127.0.0.1:5000/api/predict'
data = {
    'rd_spend': 100000,
    'administration': 200000,
    'marketing_spend': 300000,
    'state': 'California'
}

response = requests.post(url, json=data)
print('Prediction:', response.json())


Running the Application

Train the Model:
python model.py

This generates model.pkl.

Start the Flask Server:
python app.py

The app runs on http://127.0.0.1:5000.

Access the Web Interface:Open a browser and navigate to http://127.0.0.1:5000. Enter values in the form to get predictions.

Test the API:Run:
python request.py

Check the console for the API response.



Testing the Application

Web Interface: Input various values for R&D Spend, Administration, Marketing Spend, and State. Verify the predicted profit is displayed correctly.
API Endpoint: Use tools like Postman or request.py to send POST requests to http://127.0.0.1:5000/api/predict with JSON payloads.
Edge Cases: Test with extreme values (e.g., zero or negative inputs) to ensure the app handles them gracefully.


Best Practices

Use Environment Variables: Store sensitive data (e.g., API keys) in a .env file using python-dotenv.
Error Handling: Add try-except blocks in app.py to manage invalid inputs.
Logging: Implement logging to monitor app performance and errors.
Testing: Write unit tests using unittest or pytest for model and API endpoints.
Deployment: Use WSGI servers like Gunicorn and deploy on platforms like Heroku, AWS, or Render.


Additional Resources

Flask Official Documentation
Scikit-Learn Documentation
Real Python: Flask Tutorials
Deploying Flask Apps


License
This project is licensed under the MIT License. See the LICENSE file for details.
