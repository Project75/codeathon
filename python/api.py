from flask import Flask, request
from flask import jsonify
from flask import make_response
from flask_cors import CORS
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

with open('./finalized_model.sav', 'rb') as handle:
    loaded_model = pickle.load(handle)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': '404: Not found'}), 404)

@app.errorhandler(500)
def not_found2(error):
    return make_response(jsonify({'error': '500: Internal Server Error'}), 500)	
@app.route('/')
def index():
    return "Flask server"

@app.route('/iotprediction', methods = ['POST'])
def prediction():
    data = request.get_json()
    print(data)
	#TODO: json to dict
    #hardcode for testing
    dict_data = {"atemp":[1],"PID":[6],"outpressure":[35],"inpressure":[7],"temp":[22]}
    #call loaded model
    df_data =pd.DataFrame.from_dict(dict_data)
    result_arr=loaded_model.predict(df_data).tolist()
		
    resp = make_response(jsonify({"Prediction": result_arr[0]}), 201)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers["Content-Type"] = "application/json"
    resp.headers["X-Content-Type-Options"] = "nosniff"

    return resp

	

if __name__ == "__main__":
    app.run(port=5000)
	
