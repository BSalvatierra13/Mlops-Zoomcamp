from flask import Flask, request, jsonify
import mlflow

mlflow.set_tracking_uri("sqlite:///../../mlflow.db")

model_id = "m-ecabd03133c64726878a8b04eaf675c9"
model = mlflow.sklearn.load_model(f"models:/{model_id}")

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PUlocationID'], ride['DOlocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    y_pred = model.predict([features])  # el pipeline hace el DictVectorizer internamente
    return y_pred[0]

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)