import pickle

from xgboost import XGBRegressor, XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, \
    AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, SGDClassifier, SGDRegressor



# ----------------------------------------------------------------- DT
def DecisionTree(task, data, split=0.3, max_depths=None, seed=42):
    """
    Simplified Decision Tree
    
    Parameters
    ----------
    task : str 
        String describing if the task is Classification or Regression
    data : dict
        Dictionary containing features and values for given features
    split : float
        Train/Test Split for the data inside dict
    max_depths : int
        Maximum Depth the Decision Tree can go to
    seed : int
        Value that controls shuffling of data
    

    Returns
    -------
    model
        The actual Decision Tree Model fitted to the training data
    
    """ 
    
    task = task.lower()
    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    
    if task == "c" or task == "classify" or task == "classification":
        model = DecisionTreeClassifier(max_depth=max_depths, random_state=seed)
        model.fit(X_train, y_train)
        
        train_preds = model.score(X_train, y_train)
        print("Decision Tree Training Accuracy: " + str(train_preds*100) + "%")
        print("Decision Tree Testing Accuracy:  " + str(model.score(X_test, y_test) * 100) + "%") 

        return model

    elif task == "r" or task == "reg" or task == "regression":
        model = DecisionTreeRegressor(max_depth=max_depths, random_state=seed)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        train_rmse = mean_squared_error(y_train, train_preds, squared=False)
        print("Decision Tree Training RMSE: " + str(train_rmse) + "")

        test_preds = model.predict(X_test)
        test_rmse = mean_squared_error(y_test, test_preds, squared=False)
        print("Decision Tree Testing RMSE: " + str(test_rmse) + "")
        
        return model     
    else:
        raise NameError('Task should be classification or regression')


# ----------------------------------------------------------------- KNN

def KNN(task, data, split=0.3, neighbors=5, weights="uniform", seed=42): 
    """
    Simple Implementation of a KNearestNeighbors

    Parameters
    ----------
    task : str 
        String describing if the task is Classification or Regression
    data : dict
        Dictionary containing features and values for given features
    split : float
        Train/Test Split for the data inside dict
    neighbors : int
        Number of Neighbors for the KNN Algorithm
    weights : str
        Weight Function used in calculating final KNN Prediction
    seed : int
        Value that controls shuffling of data

    Returns
    -------
    model
        The actual KNN Model fitted to the training data
    
    """ 
    
    task = task.lower()
    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    if task == "c" or task == "classify" or task == "classification":
        model = KNeighborsClassifier(n_neighbors=neighbors, weights=weights)
        model.fit(X_train, y_train)
        
        train_preds = model.score(X_train, y_train)
        print("KNN Training Accuracy: " + str(train_preds*100) + "%")
        print("KNN Testing Accuracy: " + str(model.score(X_test, y_test) * 100) + "%")
    
        return model

    elif task == "r" or task == "reg" or task == "regression":
        model = KNeighborsRegressor(n_neighbors=neighbors, weights=weights)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        train_rmse = mean_squared_error(y_train, train_preds, squared=False)
        print("KNN Training RMSE: " + str(train_rmse))

        test_preds = model.predict(X_test)
        test_rmse = mean_squared_error(y_test, test_preds, squared=False)
        print("KNN Testing RMSE: " + str(test_rmse))
        
        return model

    else:
        raise NameError('Task should be Classification or Regression')


# ----------------------------------------------------------------- Random Forests
def RandomForest(task, data, split=0.3, n_estimators=100, max_depths=None, seed=42):
    """
    Easy to Use Random Forest Algorithm

    Parameters
    ----------
    task : str 
        String describing if the task is Classification or Regression
    data : dict
        Dictionary containing features and values for given features
    split : float
        Train/Test Split for the data inside dict
    n_estimators : int
        Number of Trees within the Forest
    max_depths : int
        Maximum Depth the Decision Tree can go to
    seed : int
        Value that controls shuffling of data


    Returns
    -------
    model
        The Actual Random Forest Model fitted to the training data
    
    """

    task = task.lower()
    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    
    if task == "c" or task == "classify" or task == "classification":
        model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depths, random_state=seed)
        model.fit(X_train, y_train)

        train_preds = model.score(X_train, y_train)
        print("RandomForest Training Accuracy: " + str(train_preds*100) + "%")
        print("RandomForest Testing Accuracy:  " + str(model.score(X_test, y_test) * 100) + "%")

        return model

    elif task == "r" or task == "reg" or task == "regression":
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depths, random_state=seed)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        train_rmse = mean_squared_error(y_train, train_preds, squared=False)
        print("RandomForest Training RMSE: " + str(train_rmse) + "")

        test_preds = model.predict(X_test)
        test_rmse = mean_squared_error(y_test, test_preds, squared=False)
        print("RandomForest Testing RMSE: " + str(test_rmse) + "")
        
        return model  

    else:
        raise NameError('Task should be Regression or Classification')

# ----------------------------------------------------------------- Boosting
def Boosting(task, data, split=0.3, algo="xgb", N_estimators=75, LR=0.5, Max_Depth=3, seed=42):
    algo = algo.lower()
    task = task.lower()

    X = data["X"]
    y = data["y"]


    if algo == "xgb" or task == "xgboost":
        if task == "reg" or task == "r" or task == "regression":
            boost = XGBRegressor(n_estimators=N_estimators, learning_rate=LR, max_depth=Max_Depth, random_state=42)
        elif task == "classify" or task == "c" or task == "classification":
            boost = XGBClassifier(n_estimators=N_estimators, learning_rate=LR, max_depth=Max_Depth, random_state=42)

    elif algo == "gb" or algo == "gradientboosting" or algo == "gradient":
        if task == "reg" or task == "r" or task == "regression":
            boost = GradientBoostingRegressor(n_estimators=N_estimators, learning_rate=LR, max_depth=Max_Depth, random_state=42)
        elif task == "classify" or task == "c" or task == "classification":
            boost = GradientBoostingClassifier(n_estimators=N_estimators, learning_rate=LR, max_depth=Max_Depth, random_state=42)

    elif algo == "ada" or algo == "adaboost" or algo == "ab":
        if task == "reg" or task == "r" or task == "regression":
            boost = AdaBoostRegressor(n_estimators=N_estimators, learning_rate=LR, random_state=42)
        elif task == "classify" or task == "c" or task == "classification":
            boost = AdaBoostClassifier(n_estimators=N_estimators, learning_rate=LR, random_state=42)
    else:
        raise NameError('Algorithm should be AdaBoost, Gradient Boosting or XGBoost')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    boost.fit(X_train, y_train)
    train_preds = boost.score(X_train, y_train)
    print("Boosting Training Accuracy: " + str(train_preds * 100) + "%")
    preds = boost.score(X_test, y_test)
    print("Boosting Testing Accuracy: " + str(preds * 100) + "%")
    return boost


# -------------------------------------------------------------------------- SGD
def SGD(task, data, split=0.3, lr="optimal", alpha=0.0001, seed=42):
    task = task.lower()

    X = data["X"]
    y = data["y"]

    if task == "r" or task == "reg" or task == "regression":
        sgd = SGDRegressor(learning_rate=lr, alpha=alpha, random_state=seed)
    elif task == "c" or task == "classify" or task == "classification":
        sgd = SGDClassifier(learning_rate=lr, alpha=alpha, random_state=seed)
    else:
        raise NameError('Task should be either regression or classification')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    sgd.fit(X_train, y_train)
    train_preds = sgd.score(X_train, y_train)
    print("Boosting Training Accuracy: " + str(train_preds * 100) + "%")
    preds = sgd.score(X_test, y_test)
    print("Boosting Testing Accuracy: " + str(preds * 100) + "%")
    return sgd


# -------------------------------------------------------------------------- SVMs

def SVM(task, data, split=0.3, C=1, kernel='rbf', gamma='scale', class_weight=None, verbose=True, seed=42):
    task = task.lower()

    X = data["X"]
    y = data["y"]


    if task == "r" or task == "reg" or task == "regression":
        SVM = svm.SVC(C=C, kernel=kernel, gamma=gamma, class_weight=class_weight, verbose=False)
    elif task == "c" or task == "classify" or task == "classification":
        SVM = svm.SVR(C=C, kernel=kernel, gamma=gamma, verbose=False)
    else:
        raise NameError('task should be either regression or classification')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    SVM.fit(X_train, y_train)
    train_preds = SVM.score(X_train, y_train)
    print("Boosting Training Accuracy: " + str(train_preds * 100) + "%")
    preds = SVM.score(X_test, y_test)
    print("Boosting Testing Accuracy: " + str(preds * 100) + "%")
    return SVM


# ------------------------------------------------------------------------ Logistic Regression
def LogisticReg(data, split=0.3, solver="lbfgs", seed=42):
    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    lr = LogisticRegression(solver=solver, random_state=42)

    lr.fit(X_train, y_train)
    train_preds = lr.score(X_train, y_train)
    print("Logistic Regression Training Accuracy: " + str(train_preds * 100) + "%")
    preds = lr.score(X_test, y_test)
    print("Logistic Regression Testing Accuracy: " + str(preds * 100) + "%")

    return lr


# ------------------------------------------------------------------------ Regression Models
def Regression(data, alpha=1.0, split=0.3, task="linear", seed=42):
    task = task.lower()

    X = data["X"]
    y = data["y"]

    model_name = ""

    if task == "linear" or task == "li":
        model = LinearRegression()
        model_name = "Linear"
    elif task == "ridge" or task == "r":
        model = Ridge(alpha=alpha, random_state=seed)
        model_name = "Ridge"
    elif task == "lasso" or task == "la":
        model = Lasso(alpha=alpha, random_state=seed)
        model_name = "Lasso"
    else:
        raise NameError('Specify a correct regression algorithm')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    model.fit(X_train, y_train)

    train_preds = model.score(X_train, y_train)
    print(f"{model_name} Regression Training Accuracy: " + str(train_preds * 100) + "%")
    preds = model.score(X_test, y_test)
    print(f"{model_name} Regression Testing Accuracy: " + str(preds * 100) + "%")

    return model

# ------------------------------------------------------------------------- Save Models

def save_model(model):
    with open('model.pkl', 'wb') as fid:
        pickle.dump(model, fid)
        print("Saved to disk!")


def load_model(path):
    with open(path, 'rb') as fid:
        model = pickle.load(fid)
    print("Loaded!")
    return model
