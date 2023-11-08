
import pandas as pd

# Functions from other py files
from Training import train_model, run_TabNetClassifier, run_RandomForestClassifier, run_GradientBoostingClassifier
from Evaluation import eval_models
from Explainability import exp_models

if __name__ == '__main__':

    # load split data (results from test set were done separately)

    X_valid = pd.read_csv('X_valid.csv')
    X_train = pd.read_csv('X_train.csv')
    y_valid = pd.read_csv('y_valid.csv')
    y_train = pd.read_csv('y_train.csv')

    estimator_list = ["GradientBoostingClassifier", "RandomForestClassifier", "TabNetClassifier"]
    model_list = []
    # Train models
    for est in estimator_list:
        model_list.append(train_model(classifier=est, X_train=X_train, y_train=y_train))

    # Evaluate models
    eval_models(model_list, X_valid, y_valid)

    # Explainability with FeatureImportances
    exp_models(model_list, X_valid, y_valid)
