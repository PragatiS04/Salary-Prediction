from flask import Flask, render_template,request
import numpy as np
import pandas
import pickle
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('file.html')

@app.route('/predict',methods = ['POST'])
def get_result():
    poly = pickle.load(open('transform.pkl' , 'rb'))
    model = pickle.load(open('model.pkl','rb'))
    query = [[float(request.form["Experience"])]]
    X_query = poly.transform(query)
    sal = model.predict(X_query)
    return 'Dear ' + request.form["Name"] + ' Your predicted Salary after ' + request.form["Experience"] + ' Experience is : ' + str(sal)

if __name__ == '__main__':
    app.run(debug = True)