
import sys

sys.path.append("C:\\Python39\\lib\\site-packages")
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from itertools import product
from scipy import stats
import optuna
from optuna import trial
from transformers import IdentityTransformer
from classifiers.baseline_classifier import BaselineClassifier
from data_utils import DataUtils
from model_utils import ModelUtils
from model_split_utils import ModelSplitUtils
from consts import Metrics, ModelConsts, DataConsts
from classifiers.base_optimizer import BaseOptimizer
from classifiers.geriatric_optimizer import GeriatricOptimizer
from model_param_grid import ModelParams

target_var = DataConsts.target_var
patient_id_col = DataConsts.patient_id_col
RANDOM_STATE = ModelConsts.RANDOM_STATE
TRAIN_SIZE = ModelConsts.TRAIN_SIZE

if __name__ == '__main__':
    #data prep
    data = DataUtils.read_clean_data()
    labelled_data = DataUtils.get_labelled_data(data)
    X, X_train, X_test, y, y_train, y_test = ModelSplitUtils.get_X_y_split(data)
    kf = ModelSplitUtils.get_folder()
    labelled_X, labelled_X_train, labelled_X_test, labelled_y, labelled_y_train, labelled_y_test = ModelSplitUtils.get_labelled_split(
        labelled_data, X, X_train, X_test, y, y_train, y_test)
    # baseline
    false_baseline_classifier = BaselineClassifier(class_guess=0)
    true_baseline_classifier = BaselineClassifier(class_guess=1)
    base_logistic_model = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced', penalty='l2',
                                             random_state=RANDOM_STATE)
    base_random_forest_model = RandomForestClassifier(class_weight='balanced', max_depth=3, n_estimators=100,
                                                      random_state=RANDOM_STATE)

    baseline_models = [false_baseline_classifier, true_baseline_classifier, base_logistic_model,
                       base_random_forest_model]
    baseline_x = [X, labelled_X]
    baseline_y = [y, labelled_y]
    baseline_model_names = ["always guess false", "always guess true", "base random forest", "base logistic"]
    data_names = ["all data", "all labelled data"]
    baseline_stats = []
    for model_arch, data_arch in list(
            product(list(zip(baseline_models, baseline_model_names)), list(zip(baseline_x, baseline_y, data_names)))):
        model, model_name = model_arch
        bx, by, d_name = data_arch
        try:
            baseline_stats.append(
                ModelUtils.model_score_report(bx, by, model, folder=kf, model_name=model_name, data_name=d_name))
        except:
            pass
    print("The following are baseline models")
    pd.DataFrame(baseline_stats)[
        ['model_name', 'data_name', 'test_precision', 'test_accuracy', 'test_balanced_accuracy']].dropna()

    # straight away vectorization, without using textual labels data

    g_optimizer = GeriatricOptimizer(verbose=1)
    g_optimizer.create_study(study_name="g_study", metric="f01")
    print(f"optimizer optimization metric is {g_optimizer.metric}")
    X_train_folds, y_train_folds, X_test_folds, y_test_folds = ModelSplitUtils.get_cross_val_folds(X, y, folder=kf)
    for this_X_train, this_y_train, this_X_test, this_y_test in list(
            zip(X_train_folds, y_train_folds, X_test_folds, y_test_folds)):
        print(this_X_train.shape, this_y_train.shape, this_X_test.shape, this_y_test.shape)
    verbose = 0
    n_trials = 20
    optimizers = []
    for i, (this_X_train, this_X_test, this_y_train, this_y_test) in enumerate(
            list(zip(X_train_folds, X_test_folds, y_train_folds, y_test_folds))):
        this_g_optimizer = GeriatricOptimizer(verbose=0)
        this_g_optimizer.create_study(study_name=f"study_fold_{i + 1}", metric="balanced_accuracy")
        this_g_optimizer.optimize_study(X_train=this_X_train, y_train=this_y_train, X_test=this_X_test,
                                        y_test=this_y_test,
                                        folder=kf, n_trials=n_trials)
        optimizers.append(this_g_optimizer)
        this_results_list = this_g_optimizer.get_best_k_optimal_results(3)
        for k, results in enumerate(this_results_list):
            print(f"scores for {i} results of balanced_accuracy optimization: {results.get('scores')}")
            this_df = pd.DataFrame(results.get('test_prob_preds'), index=this_X_test.index)
            this_df = pd.concat([this_df, this_X_test], axis=1).to_csv(
                f"balanced_accuracy_opt_{k}_test_prob_preds_fold_{i}.csv")
            this_df = pd.DataFrame(results.get('original_train_preds'), index=this_X_train.index)
            this_df = pd.concat([this_df, this_X_train], axis=1).to_csv(
                f"balanced_accuracy_opt_{k}_train_prob_preds_fold{i}.csv")

        this_g_optimizer_prec = GeriatricOptimizer(verbose=0)
        this_g_optimizer_prec.create_study(study_name=f"study_fold_{i + 1}", metric="f001")
        this_g_optimizer_prec.optimize_study(X_train=this_X_train, y_train=this_y_train, X_test=this_X_test,
                                             y_test=this_y_test, folder=kf, n_trials=n_trials)
        optimizers.append(this_g_optimizer_prec)
        this_results_list = this_g_optimizer_prec.get_best_k_optimal_results(3)
        for k, results in enumerate(this_results_list):
            print(f"scores for {i} results of f001 optimization: {results.get('scores')}")
            this_df = pd.DataFrame(results.get('test_prob_preds'), index=this_X_test.index)
            this_df = pd.concat([this_df, this_X_test], axis=1).to_csv(f"f001_opt_{k}_test_prob_preds_fold_{i}.csv")
            this_df = pd.DataFrame(results.get('original_train_preds'), index=this_X_train.index)
            this_df = pd.concat([this_df, this_X_train], axis=1).to_csv(f"f001_opt_{k}_train_prob_preds_{i}.csv")
        try:
            this_g_optimizer_prec.study.trials_dataframe().to_csv(f"f001_fold_{i}_trials.csv")
            this_g_optimizer.study.trials_dataframe().to_csv(f"balanced_accuracy_fold_{i}_trials.csv")
        except:
            pass
    from sklearn.feature_selection import SelectPercentile, f_classif
    from sklearn.feature_extraction.text import TfidfVectorizer

    result = this_results_list[2]
    params = result['params']
    print(params)
    model = result['model']
    print(model)
    vectorization_params = {k.removeprefix("embed_"): v for k, v in params.items() if
                            k.startswith("embed") and k != "embed_vectorizer_type"}
    vectorization_params = {k.removeprefix("tfidf_"): v for k, v in params.items() if k.startswith("tfidf")}
    vectorizer = TfidfVectorizer(**vectorization_params)
    selector = SelectPercentile(f_classif, percentile=params.get("select_best_percentile"))
    vectorizer.fit(X_train_folds[0][DataConsts.text_col_english])
    train_vecs = vectorizer.transform(X_test_folds[0][DataConsts.text_col_english])
    print(train_vecs.shape)
    selector.fit(train_vecs, y_test_folds[0])
    train_vecs = selector.transform(train_vecs, y_test_folds[0])
    print(train_vecs)

