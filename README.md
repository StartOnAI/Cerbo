# Cerbo

Cerbo is "brain" in Esperanto. 

It is a high-level API wrapping Scikit-Learn, Tensorflow and Keras. Allowing, you to efficiently perform ML modelling and preprocessing.

## Install

Installing Cerbo:
```
pip install cerbo
```

or

```
python -m pip install cerbo
```

## Writing your first program!

Currently, Cerbo performs efficient ML/DL modelling in a couple lines with limited preprocessing capabilites, we are adding new ones daily. Currently, to train a model from a CSV file all you have to do is call 

```python
from cerbo.preprocessing import *

data, col_names = load_custom_data("path_to_csv", "column_you_want_to_predict", num_features=4, id=False)
```

*data* is a dictionary containing X and y values, for training.


*col_names* is a list of features 


Note: set id to true when there is an Id column in the CSV File, and set Num_Features to any value(as long it is within the # of colunns in the file"


After running this you will get 2 .png files labelled correlation, and features respectively.
* Correlation.png
  * Will show a correlation matrix of all of the features in the CSV file
* feature.png
  * Will show a Pandas Scatter Matrix of with a N x N grid with N being *num_features*.
 

To train a model on this data just do


```python
gb, preds = Boosting("r", data, algo="gb", seed=42) 
```


Which quickly trains a Gradient Boosting Regressor on this data. 


You can also do 

```python
dt, preds = DecisionTree("c", data, seed=42)
```

To train a quick DT Classifier. 


## Authors
* Karthik Bhargav 
* Siddharth Sharma
* Sauman Das
* Andy Phung
* Felix Liu
* Anaiy Somalwar
* Nathan Z.
* Aurko Routh
* Keshav Shah
* Navein Suresh
* Ayush Karupakula
* Ishan Jain
* Shrey Gupta
