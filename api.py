from flask import Flask, request
from data.Configuration import config as cf
import numpy as np


app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        prediction = cf.model.predict(data)  # runs globally loaded model on the data
        print("Welcome to API implementation and testing of ML layer: ")
    return str(prediction[0])


if __name__ == '__main__':
    cf.load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)


