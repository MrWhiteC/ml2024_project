from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and the pre-trained encoder (for one-hot encoding of month)
with open('CO2Predicting', 'rb') as f:
    model = pickle.load(f)

with open('merged_df_encoded.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Load the scaler to standardize the features
with open('scaler_standard.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/co2_emission')
# def co2_emission():
#     return render_template('co2_emission.html')

# Route for the CO2 emission calculator page and prediction
@app.route('/co2_emission', methods=['GET', 'POST'])
def co2_emission():
    if request.method == 'POST':
        # Get data from the form input
        data = request.get_json()
        
        # Safely extract the values or use default values if they are missing
        electricity_capacity = data['electricity_capacity']
        
        coal_percentage = data.get('coal_percentage')
        natural_gas_percentage = data.get('natural_gas_percentage')
        fuel_oil_percentage = data.get('fuel_oil_percentage')
        renewable_percentage = data.get('renewable_percentage')
        hydro_percentage = data.get('hydro_percentage')
        
        # Apply default values if necessary
        coal_percentage = float(coal_percentage) if coal_percentage is not None else 16.0
        natural_gas_percentage = float(natural_gas_percentage) if natural_gas_percentage is not None else 69.0
        fuel_oil_percentage = float(fuel_oil_percentage) if fuel_oil_percentage is not None else 0.0
        renewable_percentage = float(renewable_percentage) if renewable_percentage is not None else 11.0
        hydro_percentage = float(hydro_percentage) if hydro_percentage is not None else 4.0
        
        # Set month as one-hot encoding (default: Jan)
        month_input = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # One-hot encoded for January
        
        # Preprocess the power plant input data
        electricity_capacity_per_month = electricity_capacity * 0.0776922
        coal_input = (coal_percentage / 100) * electricity_capacity_per_month
        natural_gas_input = (natural_gas_percentage / 100) * electricity_capacity_per_month
        fuel_oil_input = (fuel_oil_percentage / 100) * electricity_capacity_per_month
        renewable_input = (renewable_percentage / 100) * electricity_capacity_per_month
        hydro_input = (hydro_percentage / 100) * electricity_capacity_per_month
        
        # Combine all inputs into a feature vector (5 power plant distribution features + 12 one-hot encoded month features)
        features = np.concatenate([np.array([coal_input, fuel_oil_input, hydro_input, natural_gas_input, renewable_input]), month_input])

        # Ensure the features array is 2D (which is required by StandardScaler)
        features = features.reshape(1, -1)
        
        # Standardize the features using the scaler
        features_scaled = scaler.transform(features)
        
        # Make the CO₂ emission prediction using the model
        prediction = model.predict(features_scaled)

        # Multiple month to year
        prediction = prediction * 12.8099757426394
        
        # Return the CO₂ emission result as JSON
        return jsonify({'co2_emission': prediction[0]})

    # If GET request, render the form for CO₂ emission calculator page
    return render_template('co2_emission.html')



@app.route('/planting')
def planting():
    return render_template('planting.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
