from flask import Flask, render_template
from flask_restful import Resource, Api, reqparse
from keras.models import model_from_json
import pandas as pd
import pickle

app = Flask(__name__)
api = Api(app)

with open('model_1.json', 'r') as f:
    loaded_model_json = f.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("model_1.h5")
    model._make_predict_function()

with open('model_1_scale.pickle', 'rb') as f:
    scaler = pickle.load(f)

brands_supported = ['audi', 'bmw', 'ford', 'mercedes-benz', 'nissan', 'opel', 'skoda', 'toyota', 'volkswagen', 'volvo']
gears_supported = ['automatic', 'manual', 'unknown-g']
fuels_supported = ['gasoline', 'diesel', 'unknown-f']
model_col_order = brands_supported + gears_supported + fuels_supported + ['year', 'odo']


class PredictorApi(Resource):

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('odo', type=int, help='Car odometer reading')
        parser.add_argument('year', type=int, help='Car manufacturing year')
        parser.add_argument('brand', type=str, choices=brands_supported, help='Car brand')
        parser.add_argument('gear', type=str, choices=gears_supported, help='Car gear type')
        parser.add_argument('fuel', type=str, choices=fuels_supported, help='Car fuel type')
        args = parser.parse_args(strict=True)
        onehotted = {'odo': args['odo'], 'year': args['year']}
        onehotted[args['brand']] = 1
        onehotted[args['gear']] = 1
        onehotted[args['fuel']] = 1
        df = pd.DataFrame([onehotted], columns=model_col_order).fillna(0)
        x_data = scaler.transform(df.values)
        args['price_prediction'] = int(model.predict(x_data)[0][0])
        return args


@app.route('/')
def index():
    return render_template('index.html')

api.add_resource(PredictorApi, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0')