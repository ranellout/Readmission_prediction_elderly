import pandas as pd
from consts import ModelConsts, DataConsts
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class ModelSplitUtils:

    @classmethod
    def get_X_y_split(cls, data: pd.DataFrame,
                      target_var: str = DataConsts.target_var,
                      random_state: int = ModelConsts.RANDOM_STATE,
                      train_size: float = ModelConsts.TRAIN_SIZE,
                      shuffle: bool = ModelConsts.cross_val_shuffle):
        """
        get X y split, default parameters defined by ModelConsts
        :param data: the data frame
        :param target_var: the target variable name
        :param random_state:
        :param train_size:
        :param shuffle:
        :return:
        """
        X = data[data.columns[~data.columns.isin([target_var])]]
        y = data[target_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, shuffle=shuffle,
                                                            train_size=train_size, stratify=y)
        return X, X_train, X_test, y, y_train, y_test

    @classmethod
    def get_folder(cls, n_splits: int = ModelConsts.cross_val_n_splits, shuffle: bool = ModelConsts.cross_val_shuffle,
                   random_state: int = ModelConsts.RANDOM_STATE):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return kf

    @classmethod
    def get_labelled_split(cls, labelled_data: pd.DataFrame = None,
                           X: pd.DataFrame = None,
                           X_train: pd.DataFrame = None,
                           X_test: pd.DataFrame = None,
                           y: pd.Series = None,
                           y_train: pd.Series = None,
                           y_test: pd.Series = None,
                           data_consts=DataConsts,):
        """
        get split of labelled data from the split (if no split provided create split)
        :param labelled_data:
        :param X:
        :param X_train:
        :param X_test:
        :param y:
        :param y_train:
        :param y_test:
        :param data_consts:
        :return:
        """
        # labelled_data
        labeller = LabelEncoder()
        scaler = MinMaxScaler()
        if (X is None) or (y is None):
            X, X_train, X_test, y, y_train, y_test = cls.get_X_y_split(data = data_consts.data_filename)

        labelled_X = X[X[data_consts.patient_id_col].isin(labelled_data[data_consts.patient_id_col])].apply(labeller.fit_transform)
        labelled_y = y[labelled_X.index]
        labelled_X = scaler.fit_transform(labelled_X)
        labelled_X_train = X_train[X_train[data_consts.patient_id_col].isin(labelled_data[data_consts.patient_id_col])].apply(
            labeller.fit_transform)
        labelled_y_train = y_train[labelled_X_train.index]
        labelled_X_train = scaler.fit_transform(labelled_X_train)
        labelled_X_test = X_test[X_test[data_consts.patient_id_col].isin(labelled_data[data_consts.patient_id_col])].apply(
            labeller.fit_transform)
        labelled_y_test = y_test[labelled_X_test.index]
        labelled_X_test = scaler.fit_transform(labelled_X_test)
        return labelled_X, labelled_X_train, labelled_X_test, labelled_y, labelled_y_train, labelled_y_test

    @classmethod
    def get_cross_val_folds(cls, X: pd.DataFrame, y: pd.Series, folder = None):
        """
        get splits, usually split on train for train-dev splits
        :param X: data frame
        :param y: target var series
        :param folder: form of folding
        :return:
        """
        if folder is None:
            folder = cls.get_folder()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        X_train_folds = []
        X_dev_folds = []
        y_train_folds = []
        y_dev_folds = []
        for this_X_train_idxs, this_X_dev_idxs in folder.split(X, y):
            this_X_train = X.iloc[this_X_train_idxs]
            this_y_train = y.iloc[this_X_train_idxs]
            this_X_dev = X.iloc[this_X_dev_idxs]
            this_y_dev = y.iloc[this_X_dev_idxs]
            X_train_folds.append(this_X_train)
            X_dev_folds.append(this_X_dev)
            y_train_folds.append(this_y_train)
            y_dev_folds.append(this_y_dev)
        return X_train_folds, y_train_folds, X_dev_folds, y_dev_folds

