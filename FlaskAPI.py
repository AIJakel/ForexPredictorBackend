from flask import Flask, jsonify, request
from flask_cors import CORS
import operationsAPI, getPredictionData
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
import datetime

#This file contains all the api information

#initalizes the tensorflow graph (needed for tf to work with flask)
app = Flask(__name__)
CORS(app)
def init():
    global model_aud_usd,model_eur_usd,model_gbp_usd,model_nzd_usd,model_usd_cad,model_usd_chf,model_usd_jpy,graph, lastCalled, timeNow
    global p_aud_usd,p_eur_usd,p_gbp_usd,p_nzd_usd,p_usd_cad,p_usd_chf,p_usd_jpy
    # load the pre-trained Keras model
    model_aud_usd = load_model('model_predictFutureCandle_aud_usd.model')
    model_eur_usd = load_model('model_predictFutureCandle_eur_usd.model')
    model_gbp_usd = load_model('model_predictFutureCandle_gbp_usd.model')
    model_nzd_usd = load_model('model_predictFutureCandle_nzd_usd.model')
    model_usd_cad = load_model('model_predictFutureCandle_usd_cad.model')
    model_usd_chf = load_model('model_predictFutureCandle_usd_chf.model')
    model_usd_jpy = load_model('model_predictFutureCandle_usd_jpy.model')

    timeNow = datetime.datetime.now()
    lastCalled = datetime.datetime.now()

    graph = tf.get_default_graph()

# end point for getting all the historic data for a specified pair
@app.route('/historical/<string:curr_Pair>', methods=['GET'])
def get_AllHistorical(curr_Pair):
    return operationsAPI.getAllHistorical(curr_Pair)

@app.route('/getDate/<string:curr_Pair>', methods=['GET'])
def get_Date(curr_Pair):
    return operationsAPI.getNowDateForCurr(curr_Pair)

#end point for getting the prediction for the next hour for a specified pair
@app.route('/prediction/<string:curr_Pair>', methods=['GET'])
def get_Prediction(curr_Pair):
    inputFeature = operationsAPI.getCurrData(curr_Pair)
    if curr_Pair == 'aud_usd':
        with graph.as_default():
            raw_prediction = model_aud_usd.predict(inputFeature)
        prediction = pd.DataFrame(raw_prediction, columns=["p_open","p_close","p_high","p_low"])
        prediction = prediction.astype(float).round(4)
        return prediction.to_json(orient='records')
    
    elif curr_Pair == 'eur_usd':
        with graph.as_default():
            raw_prediction = model_eur_usd.predict(inputFeature)
        prediction = pd.DataFrame(raw_prediction, columns=["p_open","p_close","p_high","p_low"])
        prediction = prediction.astype(float).round(4)
        return prediction.to_json(orient='records')
    
    elif curr_Pair == 'gbp_usd':
        with graph.as_default():
            raw_prediction = model_gbp_usd.predict(inputFeature)
        prediction = pd.DataFrame(raw_prediction, columns=["p_open","p_close","p_high","p_low"])
        prediction = prediction.astype(float).round(4)
        return prediction.to_json(orient='records')
    
    elif curr_Pair == 'nzd_usd':
        with graph.as_default():
            raw_prediction = model_nzd_usd.predict(inputFeature)
        prediction = pd.DataFrame(raw_prediction, columns=["p_open","p_close","p_high","p_low"])
        prediction = prediction.astype(float).round(4)
        return prediction.to_json(orient='records')
    
    elif curr_Pair == 'usd_cad':
        with graph.as_default():
            raw_prediction = model_usd_cad.predict(inputFeature)
        prediction = pd.DataFrame(raw_prediction, columns=["p_open","p_close","p_high","p_low"])
        prediction = prediction.astype(float).round(4)
        return prediction.to_json(orient='records')
    
    elif curr_Pair == 'usd_chf':
        with graph.as_default():
            raw_prediction = model_usd_chf.predict(inputFeature)
        prediction = pd.DataFrame(raw_prediction, columns=["p_open","p_close","p_high","p_low"])
        prediction = prediction.astype(float).round(4)
        return prediction.to_json(orient='records')
    
    elif curr_Pair == 'usd_jpy':
        with graph.as_default():
            raw_prediction = model_usd_jpy.predict(inputFeature)
        prediction = pd.DataFrame(raw_prediction, columns=["p_open","p_close","p_high","p_low"])
        prediction = prediction.astype(float).round(4)
        return prediction.to_json(orient='records')

#test api end point
@app.route("/")
def helloWorld():
    return jsonify({"Status":"Test"})

if __name__ == "__main__":
    init()
    app.run(threaded=True, debug=True)