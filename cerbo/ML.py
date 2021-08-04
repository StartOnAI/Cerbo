import pickle

from xgboost import XGBRegressor, XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
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

    else:
        raise('Task should be classification or regression')

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
        Maximum Depth each Tree can go to
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
def Boosting(task, data, split=0.3, algo="xgb", n_estimators=75, lr=0.5, seed=42):
    """
    Simplified Boosting Algorithm

    Parameters
    ----------
    task : str 
        String describing if the task is Classification or Regression
    data : dict
        Dictionary containing features and values for given features
    split : float
        Train/Test Split for the data inside dict
    algo : str
        Specific Boosting Algorithm to be used[XGBoost, Gradient Boosting]
    n_estimators : int
        Number of Trees within the Forest
    lr : float
        The amount the contribution of each tree(n_estimators) is decreased by
    seed : int
        Value that controls shuffling of data

    Returns
    -------
    model
        The Actual Boosting Model fitted to the training data

    """ 
    
    algo = algo.lower()
    task = task.lower()
    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    if algo == "xgb" or task == "xgboost":
        if task == "c" or task == "classify" or task == "classification":
            model = XGBClassifier(n_estimators=n_estimators, learning_rate=lr, random_state=seed)
            model.fit(X_train, y_train)

            train_preds = model.score(X_train, y_train)
            print("XGBoost Training Accuracy: " + str(train_preds*100) + "%")
            print("XGBoost Testing Accuracy: " + str(model.score(X_test, y_test)*100) + "%")

            return model 

        elif task == "r" or task == "reg" or task == "regression":
            model = XGBRegressor(n_estimators=n_estimators, learning_rate=lr, random_state=seed)
            model.fit(X_train, y_train)
            
            train_preds = model.predict(X_train)
            train_rmse = mean_squared_error(y_train, train_preds, squared=False)
            print("XGBoost Training RMSE: " + str(train_rmse))

            test_preds = model.predict(X_test)
            test_rmse = mean_squared_error(y_test, test_preds, squared=False)
            print("XGBoost Testing RMSE: " + str(test_rmse))

            return model

        else: 
            raise NameError('Task should be Regression or Classification')
   
    elif algo == "gb" or algo == "gradient" or algo == "gradientboosting":
        if task == "c" or task == "classify" or task == "classification":
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, random_state=seed)
            model.fit(X_train, y_train)

            train_preds = model.score(X_train, y_train)
            print("Gradient Boosting Training Accuracy: " + str(train_preds*100) + "%")
            print("Gradient Boosting Testing Accuracy: " + str(model.score(X_test, y_test)*100) + "%")

            return model 

        elif task == "r" or task == "reg" or task == "regression":
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr, random_state=seed)
            model.fit(X_train, y_train)
            
            train_preds = model.predict(X_train)
            train_rmse = mean_squared_error(y_train, train_preds, squared=False)
            print("Gradient Boosting Training RMSE: " + str(train_rmse))

            test_preds = model.predict(X_test)
            test_rmse = mean_squared_error(y_test, test_preds, squared=False)
            print("Gradient Boosting Testing RMSE: " + str(test_rmse))

            return model

        else:
            raise NameError('Task should be Regression or Classification')

    elif algo == "ada" or algo == "adaboost":
        if task == "c" or task == "classify" or task == "classification":
            model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=lr, random_state=42)
            model.fit(X_train, y_train)

            train_preds = model.score(X_train, y_train)
            print("AdaBoost Training Accuracy: " + str(train_preds*100) + "%")
            print("AdaBoost Testing Accuracy: " + str(model.score(X_test, y_test)*100) + "%")

            return model 

        elif task == "r" or task == "reg" or task == "regression":
            model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=lr, random_state=42)
            model.fit(X_train, y_train)
            
            train_preds = model.predict(X_train)
            train_rmse = mean_squared_error(y_train, train_preds, squared=False)
            print("AdaBoost Training RMSE: " + str(train_rmse))

            test_preds = model.predict(X_test)
            test_rmse = mean_squared_error(y_test, test_preds, squared=False)
            print("AdaBoost Testing RMSE: " + str(test_rmse))

            return model

        else:
            raise NameError('Task should be Regression or Classification')

    else:
        raise NameError('Algorithm should be AdaBoost, Gradient Boosting or XGBoost')



# -------------------------------------------------------------------------- SGD
def SGD(task, data, split=0.3, alpha=0.0001, seed=42):
    """
    Stochastic Gradient Descent 

    Parameters
    ----------
    task : str 
        String describing if the task is Classification or Regression
    data : dict
        Dictionary containing features and values for given features
    split : float
        Train/Test Split for the data inside dict
    alpha : float
        Constant that multiplies the regularization term
    seed : int
        Value that controls shuffling of data

    Returns
    -------
    model
        SGD Model Fitted to Training Data

    """ 
    
    task = task.lower()
    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    
    if task == "c" or task == "classify" or task == "classification":
        model = SGDClassifier(alpha=alpha, random_state=seed)
        model.fit(X_train, y_train)

        train_preds = model.score(X_train, y_train)
        print("SGD Training Accuracy: " + str(train_preds*100) + "%")
        print("SGD Testing Accuracy: " + str(model.score(X_test, y_test) * 100) + "%")

        return model

    elif task == "r" or task == "reg" or task == "regression":
        model = SGDRegressor(alpha=alpha, random_state=seed)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        train_rmse = mean_squared_error(y_train, train_preds, squared=False)
        print("SGD Training RMSE: " + str(train_rmse))

        test_preds = model.predict(X_test)
        test_rmse = mean_squared_error(y_test, test_preds, squared=False)
        print("SGD Testing RMSE: " + str(test_rmse))

        return model
  
    else:
        raise NameError('Task should be either regression or classification')

# -------------------------------------------------------------------------- SVMs
def SVM(task, data, split=0.3, C=1, seed=42):
    """
    Easy to Use Support Vector Machine

    Parameters
    ----------
    task : str 
        String describing if the task is Classification or Regression
    data : dict
        Dictionary containing features and values for given features
    split : float
        Train/Test Split for the data inside dict
    C : float
        Regularization Parameter
    seed : int
        Value that controls shuffling of data


    Returns
    -------
    model 
        The Actual SVM Model fitted to the Training Data

    """


    task = task.lower()
    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    if task == "c" or task == "classify" or task == "classification":
        model = svm.SVC(C=C)
        model.fit(X_train, y_train)
        
        train_preds = model.score(X_train, y_train)
        print("Support Vector Machines Training Accuracy: " + str(train_preds*100) + "%")
        print("Support Vector Machines Testing Accuracy: " + str(model.score(X_test, y_test)*100) + "%")

        return model
    elif task == "r" or task == "reg" or task == "regression":
        model = svm.SVR(C=C)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        train_rmse = mean_squared_error(y_train, train_preds, squared=False)
        print("Support Vector Machines Training RMSE: " + str(train_rmse))

        test_preds = model.predict(X_test)
        test_rmse = mean_squared_error(y_test, test_preds, squared=False)
        print("Support Vector Machines Testing RMSE: " + str(test_rmse))

        return model

    else:
        raise NameError('task should be either regression or classification')


# ------------------------------------------------------------------------ Logistic Regression
def LogisticReg(data, split=0.3, solver="lbfgs", seed=42):
    """
    Logistic Regression Model

    Parameters
    ----------
    data : dict
        Dictionary containing features and values for given features
    split : float
        Train/Test Split for the data inside dict
    solver : str
        Algorithm used for Optimization problem for Log. Reg
        Other Solvers:
            lbfgs
            newton-cg
            lib-linear
            sag
            saga
    seed : int
        Value that controls shuffling of data
    
    Returns
    -------
    model
        Logistic Regression Model fitted to the training data

    """

    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    model = LogisticRegression(solver=solver, random_state=42)
    model.fit(X_train, y_train)
    train_preds = model.score(X_train, y_train)
    print("Logistic Regression Training Accuracy: " + str(train_preds * 100) + "%")
    test_preds = model.score(X_test, y_test)
    print("Logistic Regression Testing Accuracy: " + str(test_preds * 100) + "%")
    
    return model 


# ------------------------------------------------------------------------ Regression Models
def Regression(data, split=0.3, task="linear", seed=42):
    """
    Regression Model

    Parameters
    ----------
    data : dict
        Dictionary containing features and values for given features
    split : float
        Train/Test Split for the data inside dict
    task : str
        Variable so you can choose between different types of Regression
        Options:
            Linear
            Lasso
            Ridge
    seed : int
        Value that controls shuffling of data

    Returns
    -------
    model
        Regression Model fitted to training data

    """ 
    
    task = task.lower()
    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    if task == "linear" or task == "li":
        model = LinearRegression()
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        train_rmse = mean_squared_error(y_train, train_preds, squared=False)
        print("Linear Regression Training RMSE: " + str(train_rmse))

        test_preds = model.predict(X_test)
        test_rmse = mean_squared_error(y_test,test_preds,squared=False)
        print("Linear Regression Testing RMSE: " + str(test_rmse))

        return model
    elif task == "ridge" or task == "r":
        model = Ridge()
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        train_rmse = mean_squared_error(y_train, train_preds, squared=False)
        print("Ridge Regression Training RMSE: " + str(train_rmse))

        test_preds = model.predict(X_test)
        test_rmse = mean_squared_error(y_test,test_preds,squared=False)
        print("Ridge Regression Testing RMSE: " + str(test_rmse))

        return model

    elif task == "lasso" or task == "la":
        model = Lasso()
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        train_rmse = mean_squared_error(y_train, train_preds, squared=False)
        print("Lasso Regression Training RMSE: " + str(train_rmse))

        test_preds = model.predict(X_test)
        test_rmse = mean_squared_error(y_test,test_preds,squared=False)
        print("Lasso Regression Testing RMSE: " + str(test_rmse))

        return model
    else:
        raise NameError('Specify a correct regression algorithm')

# ------------------------------------------------------------------------- Mode Hel
def save_model(model):
    with open('model.pkl', 'wb') as fid:
        pickle.dump(model, fid)
        print("Saved to disk!")


def load_model(path):
    with open(path, 'rb') as fid:
        model = pickle.load(fid)
    print("Loaded!")
    return model
