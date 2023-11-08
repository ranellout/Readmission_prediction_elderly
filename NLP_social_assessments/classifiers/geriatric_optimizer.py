import optuna
from optuna import trial
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, roc_auc_score, recall_score, \
    f1_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import MinMaxScaler
from classifiers.base_optimizer import BaseOptimizer
from imblearn.over_sampling import RandomOverSampler
from consts import ModelConsts
from model_param_grid import ModelParams
from numpy.linalg import LinAlgError
import optuna
from optuna import trial
from classifiers.base_optimizer import BaseOptimizer
from model_param_grid import ModelParams
from model_utils import ModelUtils
from transformers import IdentityTransformer


class GeriatricOptimizer(BaseOptimizer):
    DEFAULT_PARAMS = ModelParams

    def __init__(self, params=None, **kwargs):
        params = params if params is not None else self.DEFAULT_PARAMS
        super(GeriatricOptimizer, self).__init__(params, **kwargs)

    def get_trial_params(self, trial):
        params = {}
        random_state = trial.suggest_categorical("random_state", self.params.random_states)
        params.update({"random_state": random_state})
        # general pathways params
        reduce_dimensions = trial.suggest_categorical("_reduce_dimensions", self.params.reduce_dimensions_params)
        select_features = trial.suggest_categorical("_select_features", self.params.select_features_params)
        data_imbalance = trial.suggest_categorical("_data_imbalance", self.params.data_imbalance_params)
        params.update({"reduce_dimensions": reduce_dimensions,
                       "select_features": select_features,
                       "data_imbalance": data_imbalance})
        # input feature params
        input_text = trial.suggest_categorical("input_text", self.params.input_feature_params)
        params.update({"input_text":
                           input_text})
        # text vectorization params
        vectorizer_type = trial.suggest_categorical("embed_vectorizer_type", self.params.vectorizer_type_params)
        analyzer = trial.suggest_categorical("embed_analyzer", self.params.analyzer_params)
        ngram_range = trial.suggest_categorical("embed_ngram_range", self.params.ngram_range_params)
        binary = trial.suggest_categorical("embed_binary", self.params.binary_params)
        max_features = trial.suggest_int("embed_max_features", self.params.max_features_min,
                                         self.params.max_features_max)
        min_df = trial.suggest_float("embed_min_df", self.params.min_df_min, self.params.min_df_max)
        max_df = trial.suggest_float("embed_max_df", self.params.max_df_min, self.params.max_df_max)
        if input_text == 'social_assessment_hebrew':
            stop_words = trial.suggest_categorical("heb_embed_stop_words", [None])
        else:
            stop_words = trial.suggest_categorical("eng_embed_stop_words", self.params.stop_words_params)
        tfidf_norm = trial.suggest_categorical("embed_tfidf_norm", self.params.tfidf_norm_params)
        params.update({"vectorizer_type": vectorizer_type,
                       "analyzer": analyzer,
                       "ngram_range": ngram_range,
                       "binary": binary,
                       "max_features": max_features,
                       "min_df": min_df,
                       "max_df": max_df,
                       "stop_words": stop_words,
                       "tfidf_norm": tfidf_norm})
        # data imbalance params
        if data_imbalance:
            data_imbalance_strategy = trial.suggest_categorical("data_imbalance_strategy",
                                                                self.params.data_imbalance_strategy_params)
            params.update({"data_imbalance_strategy": data_imbalance_strategy})

        # dimension reduction params
        if reduce_dimensions:
            number_or_explained_variance = trial.suggest_categorical("number_or_explained_variance",
                                                                     self.params.number_or_explained_variance_params)
            n_components = trial.suggest_int("n_components", self.params.pca_n_components_min,
                                             self.params.pca_n_components_max)
            if number_or_explained_variance == 'explained variance':
                n_components = n_components / 100.
            params.update({"number_or_explained_variance": number_or_explained_variance,
                           "n_components": n_components})
        # feature selection params
        select_best_percentile = trial.suggest_float("select_best_percentile",
                                                     self.params.select_best_percentile_min,
                                                     self.params.select_best_percentile_max)
        select_best_percentile_func = trial.suggest_categorical("feature_selection_function_func",
                                                                self.params.select_best_percentile_func_params)
        params.update({"select_best_percentile": select_best_percentile,
                       "select_best_percentile_func": select_best_percentile_func})

        # model hyper-params
        model_type = trial.suggest_categorical("model_type", self.params.model_type_params)
        params.update({"model_type": model_type})
        if model_type == 'random_forest':
            rf_max_depth = trial.suggest_int("rf_max_depth_min", self.params.rf_max_depth_min,
                                             self.params.rf_max_depth_max)
            rf_n_estimators = trial.suggest_int("rf_max_depth_min", self.params.rf_n_estimators_min,
                                                self.params.rf_n_estimators_max)
            params.update({"rf_max_depth": rf_max_depth, "rf_n_estimators": rf_n_estimators})
        elif model_type == 'gradient_boost':
            gb_max_depth = trial.suggest_int("gb_max_depth_min", self.params.gb_max_depth_min,
                                             self.params.gb_max_depth_max)
            gb_n_estimators = trial.suggest_int("gb_max_depth_min", self.params.gb_n_estimators_min,
                                                self.params.gb_n_estimators_max)
            params.update({"gb_max_depth": gb_max_depth, "gb_n_estimators": gb_n_estimators})
        elif model_type == 'logistic':
            lg_max_iter = trial.suggest_int("lg_max_iter", self.params.lg_max_iter_min, self.params.lg_max_iter_max)
            lg_C = trial.suggest_float("lg_C", self.params.lg_C_min, self.params.lg_C_max)
            lg_solver = trial.suggest_categorical("lg_solver", self.params.lg_solver_params)
            lg_penalty = trial.suggest_categorical("lg_penalty", self.params.lg_penalty_params)
            params.update({"lg_max_iter": lg_max_iter,
                           "lg_C": lg_C,
                           "lg_solver": lg_solver,
                           "lg_penalty": lg_penalty})
        return params

    def get_model_from_params(self, trial, params):
        pass

    def get_model_transformed_data_from_params(self, trial, params, X_train, X_test, y_train, y_test):
        trial_X_train = X_train.copy()
        trial_X_test = X_test.copy()
        trial_y_train = y_train.copy()
        trial_y_test = y_test.copy()

        if self.verbose >= 1:
            print(
                f"X_train: {trial_X_train.shape}, y_train: {trial_y_train.shape}, X_test: {trial_X_test.shape}, y_test: {trial_y_test.shape}")
        # text vectorization
        vectorizer_type = params.get("vectorizer_type")
        if vectorizer_type == 'tfidf':
            vectorization_params = {'max_features': params.get("max_features"),
                                    'ngram_range': params.get("ngram_range"),
                                    'stop_words': params.get("stop_words"),
                                    'norm': params.get("tfidf_norm"),
                                    'min_df': params.get("min_df"),
                                    'max_df': params.get("max_df"),
                                    'binary': params.get("binary"),
                                    'analyzer': params.get("analyzer")}
            vectorizer = TfidfVectorizer(**vectorization_params)
        elif vectorizer_type == 'count':
            vectorization_params = {'max_features': params.get("max_features"),
                                    'ngram_range': params.get("ngram_range"),
                                    'stop_words': params.get("stop_words"),
                                    'min_df': params.get("min_df"),
                                    'max_df': params.get("max_df"),
                                    'binary': params.get("binary"),
                                    'analyzer': params.get("analyzer")}
            vectorizer = CountVectorizer(**vectorization_params)
        elif vectorizer_type == 'dist':
            vectorization_params = {'max_features': params.get("max_features"),
                                    'ngram_range': params.get("ngram_range"),
                                    'stop_words': params.get("stop_words"),
                                    'min_df': params.get("min_df"),
                                    'max_df': params.get("max_df"),
                                    'binary': params.get("binary"),
                                    'analyzer': params.get("analyzer")}
            vectorizer = CountVectorizer(**vectorization_params)

        trial_X_train = vectorizer.fit_transform(trial_X_train[params.get("input_text")]).toarray()
        trial_X_test = vectorizer.transform(trial_X_test[params.get("input_text")]).toarray()
        if vectorizer_type == 'dist':
            trial_X_train = pd.DataFrame(trial_X_train).apply(lambda x: x / x.sum(), axis=1)
            trial_X_test = pd.DataFrame(trial_X_test).apply(lambda x: x / x.sum(), axis=1)
        trial_X_train = pd.DataFrame(trial_X_train).fillna(0.)
        trial_X_test = pd.DataFrame(trial_X_test).fillna(0.)

        if self.verbose >= 1:
            print("after vectorization:")
            print(
                f"X_train: {trial_X_train.shape}, y_train: {trial_y_train.shape}, X_test: {trial_X_test.shape}, y_test: {trial_y_test.shape}")

        # data imbalance
        if params.get("data_imbalance"):
            if params.get("data_imbalance_strategy") == "over_sampling":
                sampler_params = {"random_state": params.get("random_state")}
                sampler = RandomOverSampler(**sampler_params)
        else:
            sampler = IdentityTransformer()
        trial_X_train, trial_y_train = sampler.fit_resample(trial_X_train, trial_y_train)
        if self.verbose >= 1:
            print("after data imbalance:")
            print(
                f"X_train: {trial_X_train.shape}, y_train: {trial_y_train.shape}, X_test: {trial_X_test.shape}, y_test: {trial_y_test.shape}")

        # dimension reduction
        n_components = params.get("n_components")
        if params.get("reduce_dimensions"):
            if (n_components >= (trial_X_train.shape[1])) or (n_components >= (trial_X_train.shape[0])):
                n_components = trial_X_train.shape[1]
        pca = PCA(random_state=params.get("random_state"), n_components=n_components)
        try:
            trial_X_train = pca.fit_transform(trial_X_train)
            trial_X_test = pca.transform(trial_X_test)
        except LinAlgError:
            pass

        if self.verbose >= 1:
            print("after dimension reduction:")
            print(
                f"X_train: {trial_X_train.shape}, y_train: {trial_y_train.shape}, X_test: {trial_X_test.shape}, y_test: {trial_y_test.shape}")

        ######################### feature selection ######################################

        if params.get("select_features"):
            if params.get("select_best_percentile_func") == 'chi2':
                scaler = MinMaxScaler()
                selector = SelectPercentile(chi2, percentile=params.get("select_best_percentile"))
            elif params.get("select_best_percentile_func") == 'f_classif':
                scaler = IdentityTransformer()
                selector = SelectPercentile(f_classif, percentile=params.get("select_best_percentile"))
            elif params.get("select_best_percentile_func") == 'mutual_info_classif':
                scaler = IdentityTransformer()
                selector = SelectPercentile(mutual_info_classif, percentile=params.get("select_best_percentile"))
        else:
            scaler = IdentityTransformer()
            selector = IdentityTransformer()

        if self.verbose >= 1:
            print(
                f'before selection: X_train: {trial_X_train.shape}, y_train: {trial_y_train.shape}, X_test: {trial_X_test.shape}, y_test: {trial_y_test.shape}')
        trial_X_train = scaler.fit_transform(trial_X_train)
        trial_X_test = scaler.transform(trial_X_test)
        trial_X_train = selector.fit_transform(trial_X_train, trial_y_train)
        trial_X_test = selector.transform(trial_X_test)
        if self.verbose >= 1:
            print(
                f'after selection:  X_train: {trial_X_train.shape}, y_train: {trial_y_train.shape}, X_test: {trial_X_test.shape}, y_test: {trial_y_test.shape}')

        ###################################################################################

        ####################### classifier ################################################
        model_type = params.get("model_type")
        if model_type == 'logistic':
            model_params = {'max_iter': params.get("lg_max_iter"),
                            'solver': params.get("lg_solver"),
                            'class_weight': 'balanced',
                            'penalty': params.get("lg_penalty"),
                            'random_state': params.get("random_state"),
                            'C': params.get("lg_C")}
            model = LogisticRegression(**model_params)
        elif model_type == 'random_forest':
            model_params = {'n_estimators': params.get("rf_n_estimators"),
                            'class_weight': 'balanced',
                            'max_depth': params.get("rf_max_depth"),
                            'random_state': params.get("random_state")}
            model = RandomForestClassifier(**model_params)
        elif model_type == 'gradient_boost':
            model_params = {'n_estimators': params.get("gb_n_estimators"),
                            'max_depth': params.get("gb_max_depth"),
                            'random_state': params.get("random_state")}
            model = GradientBoostingClassifier(**model_params)
        trial.set_user_attr('final_features_dimension', trial_X_train.shape[1])
        trial_X_train = pd.DataFrame(trial_X_train)
        trial_X_test = pd.DataFrame(trial_X_test)

        return model, trial_X_train, trial_X_test, trial_y_train, trial_y_test

    def get_transformed_data_from_params(self, params, X_train, X_test, y_train, y_test):
        pass

    def get_scores_from_model_params_data(self, trial, params, model,
                                          trial_X_train, trial_X_test,
                                          trial_y_train, trial_y_test):
        scores = {}
        try:
            if self.verbose >= 1:
                print(f"trying to fit model to the data")
                print(
                    f"X_train: {trial_X_train.shape}, y_train: {trial_y_train.shape}, X_test: {trial_X_test.shape}, y_test: {trial_y_test.shape}")
                print(
                    f"data balance:\n train: {trial_y_train.value_counts().to_dict()} \n test: {trial_y_test.value_counts().to_dict()}")

            if self.verbose >= 2:
                print(f"splitting to train-dev")
            this_X_train, this_X_dev, this_y_train, this_y_dev = train_test_split(trial_X_train, trial_y_train,
                                                                                  test_size=0.3)
            if self.verbose >= 2:
                print(f"splitting to train-dev successful \n trying to fit model")
            model.fit(this_X_train, this_y_train)
            if self.verbose >= 2:
                print("fitting initial model successful")
            train_prob_preds = model.predict_proba(this_X_train)
            dev_prob_preds = model.predict_proba(this_X_dev)
            optimal_theta = ModelUtils.get_optimal_theta(train_prob_preds, dev_prob_preds, this_y_train, this_y_dev,
                                                         thetas=None, optimal_metric=self.metric)
            trial.set_user_attr("optimal_theta", optimal_theta)
            all_scores = ModelUtils.get_fold_scores_on_threshold(model, trial_X_train, trial_y_train,
                                                                 folder=self.folder,
                                                                 theta=optimal_theta, verbose=self.verbose)
            scores = all_scores.mean(axis=0).round(3).to_dict()
            # check on test scores in advance

            test_prob_preds = model.predict_proba(trial_X_test)
            test_preds = ModelUtils.update_preds_by_threshold(test_prob_preds, theta=optimal_theta)
            test_scores = ModelUtils.get_scores_from_preds(true_labels=trial_y_test, preds=test_preds, suffix="test")
            scores.update(**test_scores)
            unsampled_X_train = trial_X_train.round(8).drop_duplicates()
            unsampled_y_train = trial_y_train.iloc[unsampled_X_train.index]
            unsampled_train_prob_preds = model.predict_proba(unsampled_X_train)
            unsampled_train_preds = ModelUtils.update_preds_by_threshold(unsampled_train_prob_preds, theta=optimal_theta)
            unsampled_train_scores = ModelUtils.get_scores_from_preds(true_labels=unsampled_y_train,
                                                                      preds=unsampled_train_preds, suffix="unsampled_train")
            scores.update(**unsampled_train_scores)

            for score_name, score in scores.items():
                trial.set_user_attr(score_name, score)
            if self.verbose >= 1:
                print("model fitting finished")
                print(f"fold scores are: {scores}")
        except ValueError:
            print(f"fitting not successful because of value error, could be problem with {model}")
        except:
            print("fitting not successful for some unknown reason")

        if self.get_all_results:

            train_preds = model.predict(trial_X_train)
            train_prob_preds = model.predict_proba(trial_X_train)
            # test_preds = model.predict(trial_X_test)
            test_prob_preds = model.predict_proba(trial_X_test)
            test_preds = ModelUtils.update_preds_by_threshold(test_prob_preds, theta=optimal_theta)
            test_scores = ModelUtils.get_scores_from_preds(true_labels=trial_y_test, preds=test_preds, suffix="test")
            scores.update(**test_scores)
            results = {'scores': scores,
                       'train_preds': train_preds, 'train_prob_preds': train_prob_preds,
                       'test_preds': test_preds, 'test_prob_preds': test_prob_preds,
                       'original_train_preds': unsampled_train_preds, 'original_train_prob_preds': unsampled_train_prob_preds,
                       'train_vecs': trial_X_train, 'test_vecs': trial_X_test, 'original_train_vecs': unsampled_X_train,
                       'y_train': trial_y_train, 'y_test': trial_y_test, 'original_y_train': unsampled_y_train,
                       'model': model, 'optimal_theta': optimal_theta}
            if self.verbose >= 1:
                print(f"returning results: {results}")
            return results
        else:
            metric = self.metric
            if self.verbose >= 1:
                print(f"metric is {metric}")
                print(scores)
                print(scores.get(f'dev_{metric}'))
            trial_score = scores.get(f'{metric}_dev', 0.)
            # mean_score = trial_score.mean()
            if self.verbose >= 1:
                print(f"returning results: {trial_score}")
            return trial_score

    def get_scores_from_model_params_data_old(self, trial, params, model,
                                              trial_X_train, trial_X_test,
                                              trial_y_train, trial_y_test):
        scores = {}
        try:
            if self.verbose >= 1:
                print(f"trying to fit model to the data")
                print(
                    f"X_train: {trial_X_train.shape}, y_train: {trial_y_train.shape}, X_test: {trial_X_test.shape}, y_test: {trial_y_test.shape}")
                print(f"data balance:\n train: {trial_y_train.value_counts()} \n test: {trial_y_test.value_counts()}")

            scores, optimal_theta = ModelUtils.get_optimal_theta_from_folds(trial_X_train, trial_y_train)
            fold_scores = cross_validate(model, X=trial_X_train,
                                         y=trial_y_train,
                                         scoring=ModelConsts.default_scoring,
                                         return_train_score=True)
            if self.verbose >= 1:
                print("model fitting finished")
                print(f"fold scores are: {fold_scores}")
            for score_name, score_list in fold_scores.items():
                if score_name.startswith("test"):
                    score_name = score_name.replace("test", "dev")
                score_list_agg = score_list.mean()
                trial.set_user_attr(score_name, score_list_agg)
                scores.update(**{score_name: score_list_agg})
        except:
            pass

        trial.set_user_attr('final_features_dimension', trial_X_train.shape[1])

        if self.get_all_results:
            model.fit(trial_X_train, trial_y_train)
            train_preds = model.predict(trial_X_train)
            train_prob_preds = model.predict_proba(trial_X_train)
            test_preds = model.predict(trial_X_test)
            test_prob_preds = model.predict_proba(trial_X_test)
            test_accuracy = accuracy_score(trial_y_test, test_preds)
            test_balanced_accuracy = balanced_accuracy_score(trial_y_test, test_preds)
            test_precision = precision_score(trial_y_test, test_preds)
            test_roc_auc = roc_auc_score(trial_y_test, test_prob_preds[:, 1])
            test_recall = recall_score(trial_y_test, test_preds)
            test_f1 = f1_score(trial_y_test, test_preds)
            scores.update(**{'test_accuracy': test_accuracy,
                             'test_recall': test_recall,
                             'test_precision': test_precision,
                             'test_roc_auc': test_roc_auc,
                             'test_balanced_accuracy': test_balanced_accuracy,
                             'test_f1': test_f1})
            # scores
            results = {'scores': scores, 'train_preds': train_preds, 'train_prob_preds': train_prob_preds,
                       'test_preds': test_preds, 'test_prob_preds': test_prob_preds, 'model': model,
                       'train_vecs': trial_X_train, 'test_vecs': trial_X_test, 'y_train': trial_y_train,
                       'y_test': trial_y_test}
            if self.verbose >= 1:
                print(f"returning results: {results}")
            return results
        else:
            metric = self.metric
            print(f"metric is {metric}")
            print(scores)
            print(scores.get(f'dev_{metric}'))
            trial_score = scores.get(f'dev_{metric}', np.array([0.]))
            # mean_score = trial_score.mean()
            if self.verbose >= 1:
                print(f"returning results: {trial_score}")
            return trial_score

    def get_optimization_score_from_scores(self, trial, scores):
        pass

    def objective(self, trial):
        trial_X_train = self.X_train.copy()
        trial_X_test = self.X_test.copy()
        trial_y_train = self.y_train.copy()
        trial_y_test = self.y_test.copy()

        # step 1 - get params
        # step 2 - get model (pipeline)
        # step 3 - get scores from model and data
        trial_params = self.get_trial_params(trial)
        # divide train into train-validation, train model on train_set, find threshold optimal for score (precision) on validation set
        # product -> optimal theta

        # for score optimization - cross-validation on all train-validation set with previously found trheshold
        # final scoring (after optimization) - on test set

        if self.verbose >= 1:
            print(f"trial params are: {trial_params}")
        trial_model, trial_X_train, trial_X_test, trial_y_train, trial_y_test = self.get_model_transformed_data_from_params(
            trial, trial_params, trial_X_train, trial_X_test, trial_y_train, trial_y_test)
        trial_scores = self.get_scores_from_model_params_data(trial, trial_params, trial_model, trial_X_train,
                                                              trial_X_test, trial_y_train, trial_y_test)
        if self.get_all_results:
            return trial_scores
        else:
            # optimization_score = self.get_optimization_score_from_scores(trial, trial_scores)
            if self.verbose >= 1:
                print(f"returning optimization score. optimization score is {trial_scores}")
            return trial_scores
