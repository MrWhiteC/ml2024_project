import unittest
import pickle
import json
import numpy as np
from app import app  # assuming your Flask app is in a file named app.py

# Load the model and scaler for testing
model_path = 'web/CO2Predicting'  # Ensure the path is correct relative to the test file location
model = pickle.load(open(model_path, 'rb'))

scaler_path = 'web/scaler_standard.pkl'  # Ensure the path is correct
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
sample = np.concatenate([np.array([coal_input, fuel_oil_input, hydro_input, natural_gas_input, renewable_input]), month_input])
sample = sample.reshape(1, -1)

# Flask app test case
class FlaskAppTestCase(unittest.TestCase):
    
    def test_model_input(self):
        """Test that the model produces the expected output."""
        output = model.predict(sample)
        # Adjust the expected output based on your model's expected prediction range
        self.assertAlmostEqual(output[0], 0, delta=1e-5)  # Use a delta for floating-point comparison

    def test_model_output_shape(self):
        """Test the shape of the model output."""
        output = model.predict(sample)
        self.assertEqual(output.shape, (1,), f"Expected shape (1,) but got {output.shape}")

if __name__ == '__main__':
    unittest.main()
