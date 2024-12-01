import unittest
import pickle
import json
import numpy as np
from app import app  # assuming your Flask app is in a file named app.py

# Load the model and scaler for testing
model_path = 'web/CO2Predicting'  # Ensure the path is correct relative to the test file location
model = pickle.load(open(model_path, 'rb'))

scaler_path = 'web/scaler_minmax.pkl'  # Ensure the path is correct
scaler = pickle.load(open(scaler_path, 'rb'))

# Sample input data
electricity_capacity = 220000
coal_percentage = 16
natural_gas_percentage = 69
fuel_oil_percentage = 0
renewable_percentage = 11
hydro_percentage = 4
month_input = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# Preprocess the sample data
electricity_capacity_per_month = electricity_capacity * 0.0776922
coal_input = (coal_percentage / 100) * electricity_capacity_per_month
natural_gas_input = (natural_gas_percentage / 100) * electricity_capacity_per_month
fuel_oil_input = (fuel_oil_percentage / 100) * electricity_capacity_per_month
renewable_input = (renewable_percentage / 100) * electricity_capacity_per_month
hydro_input = (hydro_percentage / 100) * electricity_capacity_per_month

# Create feature vector for model input
sample = np.array([coal_input, fuel_oil_input, hydro_input, natural_gas_input, renewable_input] + month_input)
sample = sample.reshape(1, -1)  # Reshape to match the model input format

# Apply MinMax scaling to the features (same columns as the notebook)
sample_scaled = scaler.transform(sample[:, :5])  # Only scale the first 5 features
sample = np.concatenate([sample_scaled, sample[:, 5:]], axis=1)  # Add the month inputs back

# Flask app test case
class FlaskAppTestCase(unittest.TestCase):
    
    def test_model_input(self):
        """Test that the model produces the expected output."""
        output = model.predict(sample)
        
        # Print the actual output to help debug
        print(f"Model output: {output}")
        
        # Check if the output is within a reasonable range (e.g., 8000 to 9000)
        self.assertTrue(7000 <= output[0] <= 9000, f"Expected output to be in range [8000, 9000], but got {output[0]}")

    def test_model_output_shape(self):
        """Test the shape of the model output."""
        output = model.predict(sample)
        self.assertEqual(output.shape, (1,), f"Expected shape (1,) but got {output.shape}")

if __name__ == '__main__':
    unittest.main()
#test web
