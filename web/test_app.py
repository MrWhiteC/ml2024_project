import unittest
from app import app
import CO2Predicting  # assuming your Flask app is in a file named app.py
import json

class FlaskAppTestCase(unittest.TestCase):
    
    # Set up the test client
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True  # This will enable exception handling

    def test_index(self):
        """Test the index page (GET request to '/')"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<!DOCTYPE html>", response.data)  # Check if the page contains HTML

    def test_planting(self):
        """Test the planting page (GET request to '/planting')"""
        response = self.app.get('/planting')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<!DOCTYPE html>", response.data)  # Check if the page contains HTML

    def test_co2_emission_post(self):
        """Test the POST request to the /co2_emission route"""
        
        # Create mock data for the CO2 emission calculation
        mock_data = {
            'electricity_capacity': 1000,  # Example value
            'coal_percentage': 20,
            'natural_gas_percentage': 30,
            'fuel_oil_percentage': 10,
            'renewable_percentage': 25,
            'hydro_percentage': 15
        }

        # Send POST request to /co2_emission with the mock data
        response = self.app.post('/co2_emission', 
                                 data=json.dumps(mock_data),
                                 content_type='application/json')

        self.assertEqual(response.status_code, 200)

        # Ensure the response contains a 'co2_emission' key
        response_data = json.loads(response.data)
        self.assertIn('co2_emission', response_data)

        # Optionally: check if the value is a float or a valid number
        self.assertIsInstance(response_data['co2_emission'], float)

    def test_co2_emission_post_missing_data(self):
        """Test POST request to /co2_emission with missing fields"""
        
        # Simulate missing data (e.g., no 'natural_gas_percentage')
        mock_data = {
            'electricity_capacity': 1000,
            'coal_percentage': 20,
            'fuel_oil_percentage': 10,
            'renewable_percentage': 25,
            'hydro_percentage': 15
        }

        response = self.app.post('/co2_emission', 
                                 data=json.dumps(mock_data),
                                 content_type='application/json')

        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertIn('co2_emission', response_data)
        self.assertIsInstance(response_data['co2_emission'], float)

    def test_co2_emission_post_invalid_data(self):
        """Test POST request to /co2_emission with invalid data"""
        
        # Simulate invalid data (e.g., a string where a number is expected)
        mock_data = {
            'electricity_capacity': "invalid",  # Invalid input
            'coal_percentage': 20,
            'natural_gas_percentage': 30,
            'fuel_oil_percentage': 10,
            'renewable_percentage': 25,
            'hydro_percentage': 15
        }

        response = self.app.post('/co2_emission', 
                                 data=json.dumps(mock_data),
                                 content_type='application/json')

        # Expecting a 400 or similar error due to invalid input
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()

