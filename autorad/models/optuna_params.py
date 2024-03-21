from typing import Callable
from sklearn.base import is_classifier

from optuna.trial import Trial


def get_param_fn(model_name) -> Callable:
    param_map = {"Random Forest": params_RandomForest,
                 "XGBoost": params_XGBoost,
                 "Logistic Regression": params_LogReg,
                 "SVM": params_SVM,
                 "KNN": params_KNN,
                 "MLP": params_MLP,
                 "DecisionTreeClassifier":params_DecisionTreeClassifier,
                 "Catboost": params_Catboost
                 }

    param_fn = param_map.get(model_name, None)
    if param_fn is None:
        raise ValueError(f"Optuna parameters for {model_name} not implemented!" )

    return param_fn


def params_DecisionTreeClassifier(trial: Trial) -> dict:
    params = {
        'criterion': trial.suggest_categorical('decisiontree_criterion', ['gini', 'entropy', 'log_loss']),
        'splitter': trial.suggest_categorical('decisiontree_splitter', ['best', 'random']),
    }
    return params


def params_MLP(trial: Trial) -> dict:
    num_hidden_layers = trial.suggest_int('mlp_num_hidden_layers',1,5)
    hidden_layer_sizes=[]
    for layer in range(num_hidden_layers):
        hidden_layer_sizes.append(trial.suggest_int(f'mlp_hidden_layer_{layer}', 1,200))
    params ={
        'hidden_layer_sizes': hidden_layer_sizes
    }
    return params

def params_KNN(trial: Trial) -> dict:
    params={
        'n_neighbors': trial.suggest_int('knn_n_neighbors', 1, 30),
        'weights': trial.suggest_categorical('knn_weights', ['uniform', 'distance']),
        'leaf_size': trial.suggest_int('knn_leaf_size', 1,100)
    }
    return params

def params_RandomForest(trial: Trial) -> dict:
    params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("rf_max_depth", 2, 50),
        "max_features": trial.suggest_categorical(
            "rf_max_features", ["sqrt", "log2"]
        ),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
        "bootstrap": trial.suggest_categorical("rf_bootstrap", [True, False]),
    }
    return params


def params_XGBoost(trial: Trial) -> dict:
    params = {
        "lambda": trial.suggest_float("xgb_lambda", 1e-8, 10.0),
        "alpha": trial.suggest_float("xgb_alpha", 1e-8, 10.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "subsample": 1.0,  # trial.suggest_float("xgb_subsample", 0.2, 1.0),
        "booster": trial.suggest_categorical(
            "xgb_booster", ["gbtree", "gblinear", "dart"]
        ),
    }
    if params["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        params["max_depth"] = trial.suggest_int("xgb_max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        params["min_child_weight"] = trial.suggest_int(
            "xgb_min_child_weight", 2, 10
        )
        params["eta"] = trial.suggest_float("xgb_eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        params["gamma"] = trial.suggest_float("xgb_gamma", 1e-8, 1.0, log=True)
        params["grow_policy"] = trial.suggest_categorical(
            "xgb_grow_policy", ["depthwise", "lossguide"]
        )

    if params["booster"] == "dart":
        params["sample_type"] = trial.suggest_categorical(
            "xgb_sample_type", ["uniform", "weighted"]
        )
        params["normalize_type"] = trial.suggest_categorical(
            "xgb_normalize_type", ["tree", "forest"]
        )
        params["rate_drop"] = trial.suggest_float(
            "xgb_rate_drop", 1e-8, 1.0, log=True
        )
        params["skip_drop"] = trial.suggest_float(
            "xgb_skip_drop", 1e-8, 1.0, log=True
        )
    return params


def params_SVM(trial: Trial) -> dict:
    params = {
        "kernel": trial.suggest_categorical(
            "svm_kernel", ["linear", "poly", "rbf", "sigmoid"]
        ),
        "C": trial.suggest_float("svm_C", 1e-3, 10.0),
        "gamma": trial.suggest_float("svm_gamma", 1e-3, 10.0),
        "degree": trial.suggest_int("svm_degree", 1, 5, 1),
    }
    return params


def params_LogReg(trial: Trial) -> dict:
    penalty = trial.suggest_categorical("lr_penalty", ["l2", "l1"])
    if penalty == "l1":
        solver = "saga"
    else:
        solver = "lbfgs"
    params = {
        "penalty": penalty,
        "C": trial.suggest_float("lr_C", 1e-3, 10.0),
        "solver": solver,
    }
    return params


def params_preprocessing(trial):
    params = {
        "oversampling_method": trial.suggest_categorical(
            "oversampling_method",
            ["placeholder"],
        ),
    }
    return params

def params_Catboost(trial: Trial) -> dict:
    params = {
        "eta": trial.suggest_categorical("catboost_eta", [0.0000001,0.000001,0.0001,0.001,0.01,0.0005,0.005,0.05,0.1,0.15,0.2,0.3,0.4,0.5,]),
        "depth": trial.suggest_int("catboost_depth",1,12, step=1),
        "n_estimators": trial.suggest_int("catboost_n_estimators",10,300,step=10),
        "random_strength": trial.suggest_float("catboost_random_strength",0,0.8, step=0.1),
        "l2_leaf_reg": trial.suggest_categorical("catboost_l2_leaf_reg", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100, 200])


    }
