<div align="center">

![Logo](images/cerbo_logo_2.png)

[![Downloads](https://static.pepy.tech/personalized-badge/cerbo?period=total&units=none&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/cerbo)
[![Python Versions](https://img.shields.io/pypi/pyversions/cerbo.svg?style=plastic)](https://https://pypi.org/project/cerbo/0.1.2/)

# Cerbo

Cerbo means "brain" in Esperanto. 

It is a high-level API wrapping Scikit-Learn, Tensorflow, and TensorFlow-Keras that allows for efficient machine learning and deep learning modelling and data preprocessing while enjoying large layers of abstraction. Cerbo was originally developed to help teach students the fundamental elements of machine learning and deep learning without requiring prerequisite knowledge in Python. It also allows students to train machine learning and deep learning models easily as there is in-built error proofing.

</div>

## Install

There are two simple ways of installing Cerbo.

First, you can try:
```
pip install cerbo
```

or

```
python -m pip install cerbo
```

It is important to note that there are several packages that must already be installed to install Cerbo. The full list and versions can be found in requirements.txt, and nearly all can simply be installed through pip. If you are having trouble installing any of the prerequisite packages, a quick Google search and online coding forums such as StackOverFlow should explain how to install them correctly.

## Writing your first program!

Currently, Cerbo performs efficient ML/DL modelling in a couple lines with limited preprocessing capabilites, we are adding new ones daily. Currently, to train a model from a CSV file all you have to do is call 

```python
import cerbo.preprocessing as cp
import cerbo.ML as cml

data, col_names = cp.load_custom_data("path_to_csv", "column_you_want_to_predict", num_features=4, id=False)
```

*data* is a dictionary containing X and y values, for training.


*col_names* is a list of features 


Note: set id to true when there is an Id column in the CSV File, and set Num_Features to any value(as long it is within the # of colunns in the file"


After running this you will get 2 .png files labelled correlation, and features respectively.
* correlation.png
  * Will show a correlation matrix of all of the features in the CSV file
* feature.png
  * Will show a Pandas Scatter Matrix of with a N x N grid with N being *num_features*.
 

To train a model on this data just do


```python
gb = cml.Boosting(task="r", data=data, algo="gb", seed=42)
cml.save_model(gb) 
```


Which quickly trains and saves a Gradient Boosting Regressor on this data. 


You can also do 
```python
dt = cml.DecisionTree(task="c", data=data, seed=42)
cml.save_model(dt)
```

To train and save a quick Decision Tree Classifier. 
