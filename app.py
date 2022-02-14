import numpy as np
import nltk
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import *


app = Flask(__name__, template_folder='template')

model = pickle.load(open(r'C:\Users\Aishwarya Pai\Documents\fasal\TC_forecast.pkl','rb'))
model_1 = pickle.load(open(r'C:\Users\Aishwarya Pai\Documents\fasal\HUM_forecast.pkl','rb'))

@app.route('/')
def home():
    return render_template(r'index.html')

@app.route('/predict',methods=['POST'])
def predict():
    X_test = int(request.form["Obs"])
    X_test=np.asarray([X_test])
    X_test=X_test.reshape((-1,1))
    poly = PolynomialFeatures(degree=4)
    Xt = poly.fit_transform(X_test)
    output = model.predict(Xt)    
    poly1 = PolynomialFeatures(degree=3)
    Xt1 = poly1.fit_transform(X_test)
    output_1 = model_1.predict(Xt1)
    return(render_template(r'index.html',prediction_text="Temperature is $ {}".format(output), prediction_text1="Humidity is $ {}".format(output_1)))


if __name__ == '__main__':
    app.run(port=5000, debug=True)




