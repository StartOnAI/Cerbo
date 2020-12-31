import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
import plotly.express as px
import plotly.graph_objects as go
from pandas.plotting import scatter_matrix
import seaborn as sns

from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def plot_subplots(df, cols, num=5):
    scatter_matrix(df[cols[: num]], figsize=(12, 8))
    plt.savefig("features.png")


def correlations(df, cols):
    plt.figure(figsize=(16, 12))
    imgs = sns.heatmap(df[cols].corr(), cmap="YlGnBu", annot=True, fmt='.2f', vmin=0)
    plt.savefig("correlation.png")
    return imgs


def load_dataset(name, num_features=5, random_state=42, flatten=False, show_corr_matrix=True, show_subplots=True):
    '''
  Args:
    name (str): name of dataset ('mnist', 'fmnist', 'iris', 'breast_cancer', 'diabetes', 'housing')
    num_features (int): number of features to view in subplots, if None then num_features includes all features
    random_state (int): specify random state
    flatten (bool): returns image with shape (-1, 28, 28) if True, else returns image with shape (-1, 784)
    show_corr_matrix (bool): whether to show correlation matrix
    show_subplots (bool): whether to show subplots comparing the num_features

  Returns:
    (X_train, y_train): training data (numpy arrays)
    (X_test, y_test): testing data (20% of total data)
    col_names (list): feature names
  '''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        if flatten:
            X_train = X_train.reshape(-1, 784)
            X_test = X_test.reshape(-1, 784)
            X_train = X_train / 255.0
            X_test = X_test / 255.0
        return (X_train, y_train), (X_test, y_test)

    elif name == 'fashion_mnist' or name == 'fmnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        if flatten:
            X_train = X_train.reshape(-1, 784)
            X_test = X_test.reshape(-1, 784)
        return (X_train, y_train), (X_test, y_test)

    elif name == 'iris':
        data = datasets.load_iris()
        X = data.data
        y = data.target
        col_names = data.feature_names
        df = pd.DataFrame(X, columns=col_names)
        num = num_features
        if num_features == None:
            num = len(col_names)

        if show_subplots:
            plot_subplots(df, col_names, num)
        if show_corr_matrix:
            corr_img = correlations(df, col_names)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    elif name == 'cancer' or name == 'breast_cancer':
        data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
        col_names = data.feature_names
        df = pd.DataFrame(X, columns=col_names)
        num = num_features
        if num_features == None:
            num = len(col_names)
        if show_subplots:
            plot_subplots(df, col_names, num)
        if show_corr_matrix:
            corr_img = correlations(df, col_names)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    elif name == 'diabetes':
        data = datasets.load_diabetes()
        X = data.data
        y = data.target
        col_names = data.feature_names
        df = pd.DataFrame(X, columns=col_names)
        num = num_features
        if num_features == None:
            num = len(col_names)

        if show_subplots:
            plot_subplots(df, col_names, num)
        if show_corr_matrix:
            corr_img = correlations(df, col_names)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    elif name == 'house' or name == 'california_housing' or name == 'housing':
        data = datasets.fetch_california_housing()
        X = data.data
        y = data.target
        col_names = data.feature_names
        df = pd.DataFrame(X, columns=col_names)
        num = num_features
        if num_features == None:
            num = len(col_names)
        if show_subplots:
            plot_subplots(df, col_names, num)
        if show_corr_matrix:
            corr_img = correlations(df, col_names)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if show_corr_matrix:
            plt.title("Correlation Matrix")
    return (X_train, y_train), (X_test, y_test), col_names


def load_custom_data(path, predicted_variable, num_features=5, random_state=42, show_corr_matrix=True,
                     show_subplots=True, id=False, cat=True, cats=None):
    '''
  Args:
  path(str): Path to the dataset 
  predicted_variable(str): The variable you are trying to get the output for
  num_features(int): The number of features that you want to see on the Scatter_Matrix
  random_state(int): Seed for the validation set(to be added)
  show_corr_matrix(bool): If you want to see the correlation matrx or not
  show_subplots(bool): If you want to see subplots or not

  Returns:
  data(dict): Containing X, and Y data
  col_names(list): Contains a list of the column names


  '''

    df = pd.read_csv(path)
    X = df.drop(predicted_variable, axis=1)
    if id:
        X = X.drop("id", axis=1)
    y = df[predicted_variable]
    col_names = X.columns
    num = num_features
    if num_features == None:
        num = len(col_names)

    if show_subplots:
        plot_subplots(df, col_names, num)

    if show_corr_matrix:
        corr_img = correlations(df, col_names)

    data = {
        "X": X,
        "y": y,
    }



    return data, col_names


def kaggle_submission_csv(model, test_path, X, y, start=True):
    '''
  Args:
  model(ML Model): The name of the machine learning model you trained on the data set
  test_path(str): Path to the testing data
  X(str): Name of the column containing all of the numbers(to identify testing data)
  y(str): Name of the column containing the output of the testing data(corresponding to X)

  Returns:
  Nothing

  It just converts it to a csv and done!    
  '''

    test = pd.read_csv(test_path)
    print(test.shape)
    columns = test.columns
    predictions = (model.predict(test[columns]))
    if start:
        submit = pd.DataFrame({X: range(0, predictions.size), y: predictions})
        submit.to_csv("submission.csv", index=False)
    else:
        submit = pd.DataFrame({X: range(1, predictions.size + 1), y: predictions})
        submit.to_csv("submission.csv", index=False)
    print("Conversion Successful")


def visualize_data(X, y, column_index=0, task="scatter", color=None):
    '''
  Args:
  X: The x_axis you want(or first comparator for barplot)
  y: The y_axis you want(or second compartor for barplot)
  column_index(int) = The specific column from the X training data you are analyzing
  task(string): The specific task you are trying to accomplish(line, scatter, bar, and histogram)
  color(col_name): Distinguish different values(countries, flowers, etc.)
  test_path(str): Path to the testing data

  Returns:
  Nothing

  But it shows a graph! 
  '''

    flag = False
    if task == "line":
        fig = px.line(x=X[:, column_index], y=y, color=color)
        flag = True
    elif task == "scatter":
        fig = px.scatter(x=X[:, column_index], y=y)
        flag = True
    elif task == "bar":
        fig = px.bar(x=X[:, column_index], y=y)
        flag = True
    elif task == "hist":
        fig = px.histogram(x=X[:, column_index])
        flag = True
    elif task == "compBP":
        sns.barplot(x=X, y=y)
        plt.savefig("BarPlot.png")

    if flag:
        fig.show()


def plot_3d(x, y, z):
    '''
  Args:
    x: numpy array with shape (n,)
    y: numpy array with shape (n,)
    z: numpy array with shape (n,)
    
  Returns:
    Shows 3 dimensional PlotLy scatter plot
  '''
    fig = px.scatter_3d(x=x, y=y, z=z)
    fig.show()


def regression_data(X_min=0, X_max=2, n_samples=100, n_features=1, noise='high'):
    '''
  Args:
    X_min (int or list): if list, length of X_min should be n_features, specifies bounds for each feature
                         if value specified is int, then X_min is same for all features

    X_max (int or list): if list, length of X_max should be n_features, specifies bounds for each feature
                         if value specified is int, then X_max is same for all features

    n_samples (int): number of points to generate
    n_features (int): number of features
    noise (str): 3 possible levels ('high', 'medium', 'low')
    
  Returns:
    (X_train, y_train): numpy array with train data
    (X_test, y_test): numpy array with test data
    if n_features is 1: returns scatter plot with data (PlotLy)
    if n_features is 2: returns 3d scatter plot with data (PlotLy)
    if n_features > 2: return 2d plot comparing first feature with target value
  '''
    if type(X_max) == 'list':
        data_range = np.array(X_max) - np.array(X_min)
    else:
        data_range = np.ones(n_features) * (X_max - X_min)
        X_min = np.ones(n_features) * (X_min)

    df = {}
    df['X1'] = (data_range[0]) * np.random.rand(n_samples) + X_min[0]

    coef = 20 * np.random.rand() - 10
    bias = (100) * np.random.rand() - 100
    if noise == 'high':
        y = bias + coef * df['X1'] + 1 * coef * data_range[0] * np.random.rand(n_samples) - X_min[0]

    elif noise == 'medium':
        y = bias + coef * df['X1'] + 0.7 * coef * data_range[0] * np.random.rand(n_samples) - X_min[0]

    elif noise == 'low':
        y = bias + coef * df['X1'] + 0.5 * coef * data_range[0] * np.random.rand(n_samples) - X_min[0]

    if n_features > 1:
        for i in range(n_features - 1):
            coef = 20 * np.random.rand() - 10
            index = str(i + 2)
            val = f"X{index}"
            df[val] = ((data_range[i + 1]) * np.random.rand(n_samples) + X_min[i + 1])
            y += coef * df[f'X{i + 2}']
    df['y'] = y
    df = pd.DataFrame(df)

    if n_features == 2:
        plot_3d(df['X1'], df['X2'], df['y'])

    if n_features != 2:
        fig = px.scatter(x=df['X1'], y=y)
        fig.show()
    X_train, X_test, y_train, y_test = train_test_split(np.array(df.drop('y', axis=1, inplace=False)),
                                                        np.array(df['y']), test_size=0.2)
    return (X_train, y_train), (X_test, y_test)


def classification_data(n_clusters=2, n_samples=100):
    '''
  Args:
    n_clusters (int): number of clusters to generate
    n_samples (int): number of points to generate in each cluster

  Returns:
    (X_train, y_train): numpy array with train data
    (X_test, y_test): numpy array with test data
    function call also displays 2d scatter plot with clusters
  '''

    distance = 50 * (np.log(n_clusters))
    data = {}
    data['x'] = []
    data['y'] = []
    data['class'] = []
    for i in range(n_clusters):
        center1 = (100 * ((n_clusters / 2) + 2 * n_clusters) * np.random.rand(),
                   (100 * ((n_clusters / 2) + 2 * n_clusters) * np.random.rand()))
        x1 = np.random.uniform(center1[0], center1[0] + distance, size=(n_samples,))

        for x in x1:
            data['x'].append(x)
        y1 = np.random.normal(center1[1], distance, size=(n_samples,))

        for y in y1:
            data['y'].append(y)

        for label in np.ones(n_samples) * i:
            data['class'].append(label)

    data = pd.DataFrame(data)

    data_2 = data.sample(frac=1).reset_index(drop=True)
    arr = np.array(data_2)
    data['class'] = [str(int(label)) for label in data['class']]
    fig = px.scatter(data, x='x', y='y', color='class')
    fig.show()
    X_train, X_test, y_train, y_test = train_test_split(np.array(data.drop(['class'], axis=1, inplace=False)),
                                                        np.array(data['class']), test_size=0.2)
    return (X_train, y_train), (X_test, y_test)


# train, test = classification_data()

def read_images_from_dataframe(df, IMAGE_DIR, file_col='files', class_col='class', class_mode='raw',
                               target_size=(224, 224), batch_size=32, shuffle=True, validation_split=0.2):
    img_datagen = ImageDataGenerator(rescale=1. / 255,
                                     validation_split=validation_split)

    train_generator = img_datagen.flow_from_dataframe(df,
                                                      directory=IMAGE_DIR,
                                                      x_col=file_col,
                                                      y_col=class_col,
                                                      target_size=target_size,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      class_mode=class_mode,
                                                      subset='training')

    test_generator = img_datagen.flow_from_dataframe(df,
                                                     directory=IMAGE_DIR,
                                                     x_col=file_col,
                                                     y_col=class_col,
                                                     target_size=target_size,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     class_mode=class_mode,
                                                     subset='validation')

    return train_generator, test_generator


def read_images_from_directory(IMAGE_DIR, target_size=(100, 100), class_mode='categorical',
                               batch_size=16, shuffle=True, validation_split=0.2):
    img_datagen = ImageDataGenerator(rescale=1. / 255,
                                     validation_split=validation_split)

    train_generator = img_datagen.flow_from_directory(
        directory=IMAGE_DIR,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode=class_mode,
        subset='training')

    test_generator = img_datagen.flow_from_directory(
        directory=IMAGE_DIR,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode=class_mode,
        subset='validation')

    return train_generator, test_generator


def data_scaling(data, task="minmax"):
    X = data["X"]
    y = data["y"]

    if task == "minmax" or task == "mm":
        scaler = MinMaxScaler()
    elif task == "standard" or task == "s":
        scaler = StandardScaler()

    scaler.fit_transform(X)

    scaled_data = {
        "X": X,
        "y": y
    }
    return scaled_data
