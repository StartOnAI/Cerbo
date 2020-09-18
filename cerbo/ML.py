from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
# from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, \
    AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn import svm
import pickle


# ----------------------------------------------------------------- DT
def DecisionTree(task, data, split=0.3, criterion="gini", max_depths=None, class_weights=None, min_samples_splits=2, max_features = None, seed=42):
    # task - string "c " or "r"; data - list of inputs; labels - list of outputs; split - train-test split;
    # class_weights - dictionary; visualization - boolean
    task = task.lower()

    X = data["X"]
    y = data["y"]

    if task == "classify" or task == "c" or task == "classification":
        model = DecisionTreeClassifier(max_depth=max_depths, criterion=criterion, class_weight=class_weights, min_samples_split=min_samples_splits, max_features=max_features, random_state=seed)

    elif task == "reg" or task == "r" or task == "regression":
        model = DecisionTreeRegressor(max_depth=max_depths, criterion=criterion, class_weight=class_weights, min_samples_split=min_samples_splits, max_features=max_features, random_state=seed)

    else:
        raise NameError('Task should be classification or regression')


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    model.fit(X_train, y_train)
    train_preds = model.score(X_train, y_train)
    print("Decision Tree Training Accuracy: " + str(train_preds * 100) + "%")
    preds = model.predict(X_test)
    print("Decision Tree Testing Accuracy:  " + str(model.score(X_test, y_test) * 100) + "%")
    return model


# ----------------------------------------------------------------- KNN

def KNN(task, data, neighbors=5, weights="uniform", split=0.3, seed=42):  # data should be a dict containing X and y; split is the size of the test set
    # data preprocessing
    task = task.lower()

    X = data["X"]
    y = data["y"]

    if task == "classify" or task == "c" or task == "classification":
        knn = KNeighborsClassifier(n_neighbors=neighbors, weights=weights)
    elif task == "reg" or task == "r" or task == "regression":
        knn = KNeighborsRegressor(n_neighbors=neighbors, weights=weights)
    else:
        raise NameError('Task should be Classification or Regression')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    knn.fit(X_train, y_train)

    train_preds = knn.score(X_train, y_train)
    print("KNN Training Accuracy: " + str(train_preds * 100) + "%")
    print("KNN Testing Accuracy: " + str((knn.score(X_test, y_test)) * 100) + "%")
    return knn


# ----------------------------------------------------------------- Random Forests
def RandomForest(task, data, split=0.3, N_Estimators=100, criterion="gini", Max_Depth=None, Max_Features="auto", Min_Samples_Split=2, seed=42):
    task = task.lower()

    X = data["X"]
    y = data["y"]

    if task == "reg" or task == "r" or task == "regression":
        rf = RandomForestRegressor(n_estimators=N_Estimators, criterion=criterion, max_depth=Max_Depth, max_features=Max_Features, min_samples_split=Min_Samples_Split,
                                   random_state=seed)
    elif task == "classify" or task == "c" or task == "classification":
        rf = RandomForestClassifier(n_estimators=N_Estimators, criterion=criterion,  max_depth=Max_Depth, max_features=Max_Features, min_samples_split=Min_Samples_Split,
                                    random_state=seed)
    else:
        raise NameError('Task should be Regression or Classification')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    rf.fit(X_train, y_train)
    train_preds = rf.score(X_train, y_train)
    print("Random Forest Training Accuracy: " + str(train_preds * 100) + "%")
    preds = rf.score(X_test, y_test)
    print("Random Forest Testing Accuracy: " + str(preds * 100) + "%")
    return rf


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

# ------------------------------------------------------------------------- Apriori
def apriori(data, min_support, min_confidence, min_lift, min_length):

    rules = apriori(data, min_support, min_confidence, min_lift, min_length)
    #data is a list of lists where the inner lists are various transactions
    results = list(rules)
    #results is a list of all the rules generated
    num = 0
    for association in results:
      num += 1
      transaction = [item for item in association[0]]
      print("Association Rule #" + num + " : " + transaction[0] + " --> " + transaction[1])
    return results

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
