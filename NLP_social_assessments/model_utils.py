from sklearn.model_selection import cross_validate
from consts import ModelConsts
from typing import Dict
from sklearn.metrics import fbeta_score, make_scorer, precision_score, recall_score, balanced_accuracy_score, f1_score, \
    roc_auc_score
from typing import List
import pandas as pd
import numpy as np

from model_split_utils import ModelSplitUtils


class ModelUtils:
    @classmethod
    def model_score_report(cls, X, y, model, scoring: Dict = ModelConsts.default_scoring, folder=None,
                           model_name: str = None,
                           data_name: str = None, print_stuff: bool = False):
        """
        create some default report
        :param X:
        :param y:
        :param model:
        :param scoring:
        :param folder:
        :param model_name:
        :param data_name:
        :param print_stuff:
        :return:
        """
        score = cross_validate(model, X, y, cv=folder.split(X, y), scoring=scoring, return_train_score=True)
        mean_precision = score['test_precision'].mean().round(3)
        mean_accuracy = score['test_accuracy'].mean().round(3)
        mean_balanced_accuracy = score['test_balanced_accuracy'].mean().round(3)
        if print_stuff:
            print(
                f'for model {model_name} on data {data_name}: accuracy: {mean_accuracy}, precision: {mean_precision}, balanced accuracy: {mean_balanced_accuracy}')
        return {**{k: v.mean().round(3) for (k, v) in score.items()},
                **{'model_name': model_name, 'data_name': data_name}}

    @classmethod
    def get_scores_from_preds(cls, true_labels, preds, prob_preds=None, suffix: str = None):
        """
        function for getting a bunch of different score from true labels vs preds
        :param true_labels: pd.Series of labels
        :param preds: pd.Series of binary predictions
        :param suffix: some suffix to add to the score names
        :return:
        """
        if suffix is None:
            suffix = ""

        this_precision = precision_score(true_labels, preds)
        this_recall = recall_score(y_true=true_labels, y_pred=preds)
        this_f2 = fbeta_score(y_true=true_labels, y_pred=preds, beta=2)
        this_f05 = fbeta_score(y_true=true_labels, y_pred=preds, beta=0.5)
        this_f01 = fbeta_score(y_true=true_labels, y_pred=preds, beta=0.1)
        this_f001 = fbeta_score(y_true=true_labels, y_pred=preds, beta=0.01)
        this_f1 = f1_score(y_true=true_labels, y_pred=preds)
        this_balanced_accuracy = balanced_accuracy_score(true_labels, preds)
        scores = {"precision": this_precision,
                  "recall": this_recall,
                  "f2": this_f2,
                  "f1": this_f1,
                  "f05": this_f05,
                  "f01": this_f01,
                  "f001": this_f001,
                  "balanced_accuracy": this_balanced_accuracy}

        if prob_preds is not None:
            this_roc_auc = roc_auc_score(true_labels, prob_preds[:, 1])
            scores.update(**{"roc_auc": this_roc_auc})

        scores = {f"{k}_{suffix}": round(v, 3) for k, v in scores.items()}
        return scores

    @classmethod
    def update_preds_by_threshold(cls, prob_preds, theta: float = 0.5):
        """
        helper function, get vector of probability scores predictions and a threshold and then binarize to 0-1 preds
        :param prob_preds: vecotr of scores
        :param theta: float threshole in range [0,1]
        :return: updated_prob_preds
        """
        updated_prob_preds = pd.DataFrame(prob_preds).apply(lambda x: 1 if x[1] > theta else 0, axis=1)
        return updated_prob_preds

    @classmethod
    def get_all_threshold_scores(cls, train_prob_preds, dev_prob_preds, train_y, dev_y, thetas: List[float] = None,
                                 **const_kwargs):
        """
        function to get a bunch of scores (defined by get scores from preds) for a lot of different thresholds
        :param train_prob_preds:
        :param dev_prob_preds:
        :param train_y:
        :param dev_y:
        :param thetas:
        :param const_kwargs:
        :return:
        """
        scores = []
        if thetas is None:
            thetas = np.linspace(0, 1, 101)[1: -1]

        for theta in thetas:
            this_theta_preds_train = cls.update_preds_by_threshold(prob_preds=train_prob_preds, theta=theta)
            this_theta_preds_dev = cls.update_preds_by_threshold(prob_preds=dev_prob_preds, theta=theta)
            this_scores = {"theta": theta}
            this_scores.update(**cls.get_scores_from_preds(train_y, this_theta_preds_train, suffix="train"))
            this_scores.update(**cls.get_scores_from_preds(dev_y, this_theta_preds_dev, suffix="dev"))
            this_scores.update(**const_kwargs)
            scores.append(this_scores)
        scores = pd.DataFrame(scores)
        return scores

    @classmethod
    def get_optimal_theta(cls, train_prob_preds, dev_prob_preds, train_y, dev_y, thetas: List[int] = None,
                          optimize_by: str = "dev",
                          optimal_metric: str = "precision", return_scores: bool = False, **const_kwargs):
        """
        get optimal theta for single train, dev probability predictions
        :param train_prob_preds:
        :param dev_prob_preds:
        :param train_y:
        :param dev_y:
        :param thetas:
        :param optimal_metric:
        :param return_scores: if want to return the data frame containing all the scores also
        :param const_kwargs:
        :return: the optimal theta
        """
        scores = cls.get_all_threshold_scores(train_prob_preds, dev_prob_preds, train_y, dev_y, thetas, **const_kwargs)
        best_theta = float(
            pd.DataFrame(scores).sort_values(f"{optimal_metric}_{optimize_by}", ascending=False).head(1)[
                'theta'].values)
        if return_scores:
            return best_theta, scores
        else:
            return best_theta

    @classmethod
    def get_optimal_threshold_from_list(cls, best_thetas_list: List[float]):
        """
        get the optimal theta from an optimal_theta list
        the rule is - ignore edges (min and max) and then average over the rest
        :param best_thetas_list: list of best thetas from folds
        :return: the optimal aggregated theta
        """
        min_theta = np.min(best_thetas_list)
        max_theta = np.max(best_thetas_list)
        agg_theta = np.mean([theta for theta in best_thetas_list if theta not in [min_theta, max_theta]])
        return agg_theta

    @classmethod
    def get_fold_scores_on_threshold(cls, model, X_train, y_train, folder=None, theta=0.5, **kwargs):
        """
        get the scores of some data and some model, use cross validation and define the threshold as given
        :return:
        """
        all_scores = []
        verbose = kwargs.get("verbose", 0)
        if folder is None:
            folder = ModelSplitUtils.get_folder()
        X_train_folds, y_train_folds, X_dev_folds, y_dev_folds = ModelSplitUtils.get_cross_val_folds(X_train, y_train,
                                                                                                     folder=folder)
        if verbose >= 2:
            print(
                f"starting cross validation process on threshold {theta} and model {model}. Data initial size is {X_train.shape}")
        for i, (this_train_X, this_dev_X, this_train_y, this_dev_y) in enumerate(
                list(zip(X_train_folds, X_dev_folds, y_train_folds, y_dev_folds))):
            if verbose >= 2:
                print(f"fold number {i + 1}")
                print(this_train_X.shape, this_train_y.shape, this_dev_X.shape, this_dev_y.shape)

            model.fit(this_train_X, this_train_y)

            train_prob_preds = model.predict_proba(this_train_X)
            train_preds = cls.update_preds_by_threshold(train_prob_preds, theta=theta)
            train_scores = cls.get_scores_from_preds(this_train_y, train_preds, train_prob_preds, suffix="train")
            dev_prob_preds = model.predict_proba(this_dev_X)
            dev_preds = cls.update_preds_by_threshold(dev_prob_preds, theta=theta)
            dev_scores = cls.get_scores_from_preds(this_dev_y, dev_preds, dev_prob_preds, suffix="dev")

            if verbose >= 1:
                print(train_prob_preds.shape, train_preds.shape, dev_prob_preds.shape, dev_preds.shape)
            if verbose >= 1:
                print(f"for fold #{i + 1} train scores are: {train_scores} and dev scores are {dev_scores}")

            all_scores.append({**train_scores, **dev_scores, **{'fold': i + 1}})

        if verbose >= 2:
            print(f"finished cross validation")
        all_scores = pd.DataFrame(all_scores)
        return all_scores

    @classmethod
    def get_optimal_theta_from_folds(cls, model, X_train, y_train, folder=None, return_scores: bool = False,
                                     thetas_list=None, **kwargs):
        best_thetas = []
        all_scores = []
        verbose = kwargs.get("verbose", 0)
        if folder is None:
            folder = ModelSplitUtils.get_folder()
        X_train_folds, y_train_folds, X_dev_folds, y_dev_folds = ModelSplitUtils.get_cross_val_folds(X_train, y_train,
                                                                                                     folder=folder)

        for i, (this_train_X, this_dev_X, this_train_y, this_dev_y) in enumerate(
                list(zip(X_train_folds, X_dev_folds, y_train_folds, y_dev_folds))):
            # vectorizer = CountVectorizer(max_features=2000)
            # X_train_vecs = vectorizer.fit_transform(this_train_X[DataConsts.text_col_english])
            # X_dev_vecs = vectorizer.transform(this_dev_X[DataConsts.text_col_english])
            if verbose >= 2:
                print(f"fold number {i + 1}")
                print(this_train_X.shape, this_train_y.shape, this_dev_X.shape, this_dev_y.shape)
            model.fit(this_train_X, this_train_y)
            train_prob_preds = model.predict_proba(this_train_X)
            train_preds = model.predict(this_train_X)
            dev_prob_preds = model.predict_proba(this_dev_X)
            dev_preds = model.predict(this_dev_X)
            if verbose >= 1:
                print(train_prob_preds.shape, train_preds.shape, dev_prob_preds.shape, dev_preds.shape)

            this_best_theta, this_scores = cls.get_optimal_theta(train_prob_preds, dev_prob_preds, this_train_y,
                                                                 this_dev_y,
                                                                 return_scores=True, fold=i + 1, thetas=thetas_list)
            best_thetas.append(this_best_theta)
            all_scores.append(this_scores)
        all_scores = pd.concat(all_scores)
        optimal_agg_theta = cls.get_optimal_threshold_from_list(best_thetas)
        if return_scores:
            return all_scores, optimal_agg_theta, best_thetas
        else:
            return optimal_agg_theta
