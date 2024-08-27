from flask import Flask, render_template, request
import jsonify
import pickle
import numpy as np

from mappings import mappings

with open(r'P:\Personal Projects\Vehicle Price Prediction\Vehicle Price Prediction\Models\random_forest_regressor_model.pkl', 'rb') as f:
    predictor = pickle.load(f)

with open(r'P:\Personal Projects\Vehicle Price Prediction\Vehicle Price Prediction\Models\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

def encode_input(input_dict, mapping):
    encoded_values = []
    for key, value in input_dict.items():
        # Ensure the value is in lowercase if it's a string
        if isinstance(value, str):
            value = value.lower()
        if key in mapping:
            # Get the encoded value or use a default value (e.g., -1 for unknown values)
            encoded_values.append(mapping[key].get(value, -1))
    
    return np.array(encoded_values).reshape(1, -1)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':

        brand = request.form['Brand']
        model = request.form['Model']
        body_type = request.form['Body Type']
        drivetrain = request.form['Drivetrain']
        interior_colour = request.form['Interior Colour']
        exterior_colour = request.form['Exterior Colour']
        fuel_type = request.form['Fuel Type']
        transmission_type = request.form['Transmission Type']
        doors = int(request.form['Doors'])
        cylinder = int(request.form['Cylinder Count'])
        distance = int(request.form['Distance (km)'])
        mileage = int(request.form['Mileage (kms/Lt)'])
        years_used = int(request.form['Years Used'])

        input = {
            'Brand': brand,
            'Model': model,
            'Body Type': body_type,
            'Drivetrain': drivetrain,
            'Interior Colour': interior_colour,
            'Exterior Colour': exterior_colour,
            'Fuel Type': fuel_type,
            'Transmission Type': transmission_type,
        }

        encoded_input = encode_input(input, mappings)
        full_input = np.hstack((encoded_input, [[doors, cylinder, distance, mileage, years_used]]))

        scaled_input = scaler.transform(full_input)
        prediction = predictor.predict(scaled_input)

        output = round(prediction[0],2)

        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))
        
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)