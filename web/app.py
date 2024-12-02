from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model and the pre-trained encoder (for one-hot encoding of month)
with open('CO2Predicting', 'rb') as f:
    model = pickle.load(f)

# Load the scaler to standardize the features (MinMaxScaler)
with open('scaler_minmax.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

# Route for the CO2 emission calculator page and prediction
@app.route('/co2_emission', methods=['GET', 'POST'])
def co2_emission():
    if request.method == 'POST':
        try:
            # Get data from the form input
            data = request.get_json()

            # Safely extract the values or use default values if they are missing
            electricity_capacity = data.get('electricity_capacity')
            coal_percentage = data.get('coal_percentage')
            natural_gas_percentage = data.get('natural_gas_percentage')
            fuel_oil_percentage = data.get('fuel_oil_percentage')
            renewable_percentage = data.get('renewable_percentage')
            hydro_percentage = data.get('hydro_percentage')

            # Check if any required fields are missing or invalid
            if not electricity_capacity or not isinstance(electricity_capacity, (int, float)):
                return jsonify({"error": "Invalid or missing electricity_capacity"}), 400

            # Validate the percentages
            for field, value in {
                "coal_percentage": coal_percentage,
                "natural_gas_percentage": natural_gas_percentage,
                "fuel_oil_percentage": fuel_oil_percentage,
                "renewable_percentage": renewable_percentage,
                "hydro_percentage": hydro_percentage
            }.items():
                if value is not None and not isinstance(value, (int, float)):
                    return jsonify({"error": f"Invalid {field}, must be a number"}), 400

            # Apply default values if necessary
            coal_percentage = float(coal_percentage) if coal_percentage is not None else 16.0
            natural_gas_percentage = float(natural_gas_percentage) if natural_gas_percentage is not None else 69.0
            fuel_oil_percentage = float(fuel_oil_percentage) if fuel_oil_percentage is not None else 0.0
            renewable_percentage = float(renewable_percentage) if renewable_percentage is not None else 11.0
            hydro_percentage = float(hydro_percentage) if hydro_percentage is not None else 4.0

            # Check if all percentages are zero
            if (coal_percentage == 0 and natural_gas_percentage == 0 and 
                fuel_oil_percentage == 0 and renewable_percentage == 0 and 
                hydro_percentage == 0):
                return jsonify({'co2_emission': 0})

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

            # Ensure the features array is 2D (required by MinmaxScaler)
            features = features.reshape(1, -1)

            # Apply MinMax scaling to the features, excluding the month-related part
            # Extract only the power generation input features to scale
            power_generation_inputs = features[:, :5]  # First 5 are power generation related

            # Scale the power generation features
            scaled_power_generation_inputs = scaler.transform(power_generation_inputs)

            # Replace the original unscaled features with the scaled features
            features[:, :5] = scaled_power_generation_inputs

            # Make the CO₂ emission prediction using the model
            prediction = model.predict(features)

            # Multiply by 12.8099757426394 to convert to annual CO₂ emissions
            prediction = prediction * 12.8099757426394

            # Return the CO₂ emission result as JSON
            return jsonify({'co2_emission': prediction[0]})

        except Exception as e:
            # Log the exception (optional)
            print(f"Error occurred: {e}")
            return jsonify({"error": "An error occurred while processing your request"}), 500

    # If GET request, render the form for CO₂ emission calculator page
    return render_template('co2_emission.html')

@app.route('/planting')
def planting():
    return render_template('planting.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

#Test web