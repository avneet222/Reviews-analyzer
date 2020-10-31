from flask import Flask, render_template, request
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd

classify = pickle.load(open("classifier.pkl", "rb"))
cv = pickle.load(open("BOW.pkl","rb"))

app = Flask(__name__)

@app.route('/')
@cross_origin()
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = classify.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
