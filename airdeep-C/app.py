from flask import Flask, jsonify, request
# from sklearn.externals
import joblib
import pandas as pd
from utils import preprocess
import json
import logging
from collections import Counter
import numpy as np


# config
with open('config/config.json', 'r') as r:
    config = json.load(r)

def load_models(config):
    models = dict()
    model_path_config = config['model_path']
    print(model_path_config)
    for model_name, model_path in model_path_config.items():
        models[model_name] = joblib.load(model_path)
    
    return models

models = load_models(config)
app = Flask(__name__)

# gunicorn logging
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

@app.route('/health', methods=['GET'])
def test():
    return "YaHooooooooo"

# predict
@app.route('/predict/<model>', methods=['POST'])
def predict(model):
    # get queries
    data = [request.json]

    query_df = pd.DataFrame(data)
    query = preprocess.prepro_smoke_data(query_df)
    
    # append prediction result
    query_df['pred'] = models[model].predict(query)

    # df to json
    result =  query_df.to_json(orient='records')
    result = json.loads(result)

    log = {'result' : result, 'model_name' : model}
    app.logger.info(str(log))
    
    return json.dumps(result)  


@app.route('/predict/compare/<model>', methods=['POST'])
def predict_compare(model):
    # get queries
    data = request.json['data']
  
    query_df = pd.DataFrame(data)
    query = preprocess.prepro_smoke_data(query_df)
    
    # append prediction result
    pred = models[model].predict(query)
    query_df['pred'] = pred

    # voting pred results 
    pred_vote = Counter(pred).most_common(1)[0][0]

    # df to json
    result =  query_df.to_json(orient='records')
    result = json.loads(result)

    # logging 
    result = {'data' : result, 'model_name' : model, 'vote' : int(pred_vote)}
    app.logger.info(str(result))

    return json.dumps(result)


@app.route('/predict/seq/<model>', methods=['POST'])
def predict_seq(model):
    # get queries
    data = request.json['data']
    query_df = pd.DataFrame(data)
    query_df=query_df[['tvoc','pm10','pm2.5']]
    query = np.array([query_df])
    
    # append prediction result
    pred = models[model].predict(query)
    pred += 1

    # logging 
    result = {'data' : data, 'model_name' : model, 'vote' : int(pred)}
    app.logger.info(str(result))

    return json.dumps(result)   

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)
