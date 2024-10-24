# Stats-Honours-Project

## Environment

This code is compatible with Python 3.8 (more recent versions may not be supported)
All the necessary packages may be set up by running `pip install -r requirements.txt`

## Data 

The data must be downloaded and placed in a folder called "data" within the working directory.

## Run

To run the code there are 8 options:

### For Air Quality dataset

For the univariate case: `python main.py AirQuality False`
For the multivariate case: `python main.py AirQuality True`   

### For Appliance Energy Prediction dataset

For the univariate case: python main.py ApplianceEnergy False  
For the multivariate case with d=8: `python main.py ApplianceEnergy True 8`  
For the multivariate case with d=9: `python main.py ApplianceEnergy True 9`  
For the multivariate case with d=11: `python main.py ApplianceEnergy True 11`   

### For Geometric Interpretation Plot

python main.py GeomInterp
