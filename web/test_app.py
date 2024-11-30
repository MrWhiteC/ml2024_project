import unittest
import pickle
from app import app  # assuming your Flask app is in a file named app.py
import json
import numpy as np

model_path = 'CO2Predicting'
model = pickle.load(open(model_path, 'rb'))

scaler_path = 'scaler_standard.pkl'
scaler = pickle.load(open(scaler_path, 'rb'))

electricity_capacity = 220000
        
coal_percentage = 16
natural_gas_percentage = 69
fuel_oil_percentage = 0
renewable_percentage = 11
hydro_percentage = 4
month_input = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

electricity_capacity_per_month = electricity_capacity * 0.0776922
coal_input = (coal_percentage / 100) * electricity_capacity_per_month
natural_gas_input = (natural_gas_percentage / 100) * electricity_capacity_per_month
fuel_oil_input = (fuel_oil_percentage / 100) * electricity_capacity_per_month
renewable_input = (renewable_percentage / 100) * electricity_capacity_per_month
hydro_input = (hydro_percentage / 100) * electricity_capacity_per_month

sample = np.concatenate([np.array([coal_input, fuel_oil_input, hydro_input, natural_gas_input, renewable_input]), month_input])
sample = sample.reshape(1, -1)

class FlaskAppTestCase(unittest.TestCase):
    
   def test_model_input():
    output = model.predict(sample)
    assert output == 0

    def test_model_output_shape():
    output = model.predict(sample)
    assert output.shape == (1,), f"Expecting the shape to be (1,1) but got {output.shape=}"

if __name__ == '__main__':
    unittest.main()

