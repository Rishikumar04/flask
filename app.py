import flask
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

model = joblib.load('pipe.pkl')

app = Flask(__name__)

@app.route('/')
def app_run():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    test_pd = pd.DataFrame([request.form])
    value = model.predict(test_pd)[0]
    value = str(round(value,2))
    return render_template("predict.html",prices = value +" Lakhs")


if __name__ == "__main__":
    app.run(port=8000, debug = True)

