import optuna
from abc import ABC, abstractmethod, abstractclassmethod
from consts import DataConsts, ModelConsts, Metrics
from model_param_grid import ModelParams
from model_split_utils import ModelSplitUtils


class BaseOptimizer(ABC):
    DEFAULT_VERBOSE = 0
    DEFAULT_DIRECTION = 'maximize'
    DEFAULT_METRIC = 'precision'
    DEFAULT_STUDY_NAME = 'my_study'
    DEFAULT_N_TRIALS = 30
    DEFAULT_PARAMS = ModelParams

    def __init__(self, params=None, **kwargs):
        self.verbose = kwargs.get("verbose", self.DEFAULT_VERBOSE)
        self.get_all_results = False
        self.params = params if params is not None else self.DEFAULT_PARAMS
        # all parameters/attributes that will be defined later initialized as None
        self.metric = None
        self.study = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.folder = None

    def create_study(self, **kwargs):
        direction = kwargs.get('direction', self.DEFAULT_DIRECTION)
        self.metric = kwargs.get('metric', self.DEFAULT_METRIC)
        study_name = kwargs.get('study_name', self.DEFAULT_STUDY_NAME)
        self.study = optuna.create_study(direction=direction, study_name=study_name)
        self.get_all_results = False
        if self.verbose >= 1:
            print(f"created study {self.study} successfully")

    def get_data_for_optimization(self, X_train=None, y_train=None, X_test=None, y_test=None, folder=None, data=None,
                                  data_consts=DataConsts,
                                  **kwargs):
        if (X_train is None) and (y_train is None) and (X_test is None) and (y_test is None) and (data is not None):
            X, X_train, X_test, y, y_train, y_test = ModelSplitUtils.get_X_y_split(data=data)
            self.X = X.copy()
            self.y = y.copy()
        if folder is None:
            folder = ModelSplitUtils.get_folder()
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.folder = folder

    def optimize_study(self, X_train=None, y_train=None, X_test=None, y_test=None, folder=None, data=None, **kwargs):
        self.get_data_for_optimization(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                       folder=folder, data=data, **kwargs)
        n_trials = kwargs.get('n_trials', self.DEFAULT_N_TRIALS)
        self.study.optimize(self.objective, n_trials=n_trials)

    @abstractmethod
    def get_trial_params(self, trial):
        params = None
        return params

    @abstractmethod
    def get_model_from_params(self, trial, params):
        pass

    @abstractmethod
    def get_transformed_data_from_params(self, trial, params, X_train, X_test, y_train, y_test):
        """
        in case we want to transform the data separately from the model
        :param trial:
        :param params:
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :return:
        """
        pass

    @abstractmethod
    def get_model_transformed_data_from_params(self, params, X_train, X_test, y_train, y_test):
        pass

    @abstractmethod
    def get_scores_from_model_params_data(self, trial, params, model, trial_X_train, trial_X_test,
                                                    trial_y_train, trial_y_test):
        pass

    @abstractmethod
    def get_optimization_score_from_scores(self, trial, scores):
        pass

    @abstractmethod
    def objective(self, trial):
        trial_X_train = self.X_train.copy()
        trial_X_test = self.X_test.copy()
        trial_y_train = self.y_train.copy()
        trial_y_test = self.y_test.copy()

        # step 1 - get params
        # step 2 - get model (pipeline)
        # step 3 - get scores from model and data
        trial_params = self.get_trial_params(trial)
        trial_model = self.get_model_from_params(trial, params=trial_params)
        trial_scores = self.get_scores_from_model_params_data(trial, trial_params, trial_model, trial_X_train,
                                                                        trial_X_test, trial_y_train, trial_y_test)
        if self.get_all_results:
            return trial_scores
        else:
            optimization_score = self.get_optimization_score_from_scores(trial, trial_scores)
            return optimization_score

    def get_full_results(self, trial):
        self.get_all_results = True
        results = self.objective(trial)
        self.get_all_results = False
        return results

    def get_optimal_results(self):
        return self.get_full_results(self.study.best_trial)

    def get_best_k_optimal_results(self, k: int = 3):
        results_list = []
        best_trial_idxs = self.study.trials_dataframe()['value'].nlargest(k).index.tolist()
        for trial_idx in best_trial_idxs:
            this_trial = self.study.trials[trial_idx]
            this_trial_params = this_trial.params
            this_trial_results = self.get_full_results(this_trial)
            this_trial_results['params'] = this_trial_params
            this_trial_results['trial_number'] = this_trial.number
            results_list.append(this_trial_results)
        return results_list