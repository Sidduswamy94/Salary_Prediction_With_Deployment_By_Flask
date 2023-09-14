# Import libraries
import numpy as np
from flask import Flask, request, render_template
import pickle

# create the instance of the Flask()
# load the model into the model.
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

# create a user define function to run the html file
# run the html file
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api',methods=['POST'])
def predict():
    prediction_text = ''
    if request.method == 'POST':
        # Get user input
        experience = int(request.form['experience'])
        test_score = int(request.form['test_score'])
        interview_score = int(request.form['interview_score'])

        # Prepare input for prediction
        input_features = np.array([experience, test_score, interview_score]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)
        prediction_text = 'Employee Salary should be $ {:.2f}'.format(prediction[0])

    return render_template('index.html', prediction_text=prediction_text)

# run the flask file in local server
if __name__ == "__main__":
    app.run(debug=True)