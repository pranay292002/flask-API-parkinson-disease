from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and scaler
try:
    with open('parkinsons_model.pkl', 'rb') as model_file:
        clf = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    print("Error: Model files not found. Make sure you have the necessary model files in the correct location.")
    clf = None
    scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    if clf is None or scaler is None:
        return jsonify("Error: Model not loaded. Please check server logs for details.")

    data = request.get_json(force=True)
    mdvp_fo_hz = data['mdvp_fo_hz']
    mdvp_fhi_hz = data['mdvp_fhi_hz']
    mdvp_flo_hz = data['mdvp_flo_hz']
    mdvp_jitter_percent = data['mdvp_jitter_percent']
    mdvp_jitter_abs = data['mdvp_jitter_abs']
    mdvp_rap = data['mdvp_rap']
    mdvp_ppq = data['mdvp_ppq']
    jitter_ddp = data['jitter_ddp']
    mdvp_shimmer = data['mdvp_shimmer']
    shimmer_db = data['shimmer_db']
    shimmer_apq3 = data['shimmer_apq3']
    shimmer_apq5 = data['shimmer_apq5']
    mdvp_apq = data['mdvp_apq']
    shimmer_dda = data['shimmer_dda']
    nhr = data['nhr']
    hnr = data['hnr']
    rpde = data['rpde']
    dfa = data['dfa']
    spread1 = data['spread1']
    spread2 = data['spread2']
    d2 = data['d2']
    ppe = data['ppe']

    user_input = np.array([[mdvp_fo_hz, mdvp_fhi_hz, mdvp_flo_hz, mdvp_jitter_percent, mdvp_jitter_abs, mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, shimmer_db, shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
    user_input_scaled = scaler.transform(user_input)
    
    try:
        prediction = clf.predict(user_input_scaled)

        if prediction == 0:
            result = "Good News! You don't have Parkinson's disease."
        else:
            result = "Sorry to say, You may have Parkinson's disease."

        return jsonify(result)
    except Exception as e:
        return jsonify("Error: {}".format(str(e)))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
