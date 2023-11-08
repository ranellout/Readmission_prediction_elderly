import numpy as np
import pandas as pd
from consts import DataConsts

class DataUtils:
    @classmethod
    def read_clean_data(cls, data_consts = DataConsts):
        """
        helper function for doing all prepatory stuff
        :param data_consts: a class containing all consts for data process
            all other parameters to be read from DataConsts
        :return:
        """
        data = pd.read_csv(data_consts.data_filename)
        data = data.drop(columns=data_consts.data_columns_to_drop)
        # data cleaning
        data["help_at_home_hours"] = data["help_at_home_hours"].apply(lambda x: int(x) if str(x).isnumeric() else np.nan)
        data["marital_status"] = data["marital_status"].apply(lambda x: x if x != "narried" else "married")
        data["deceased"] = data["deceased"].apply(lambda x: x if x != "low" else False)
        # some transformations
        data["#admissions"] = data.groupby(["patient_identifier"])["event_identifier"].transform('count')
        # rename typos columns
        data = data.rename(columns = data_consts.data_columns_rename_dict)

        # remove unlabelled target - none existing but just to be sure :)
        data = data[data[data_consts.target_var].notna()]
        return data

    @classmethod
    def get_labelled_data(cls, data, data_consts = DataConsts):
        # create labelled data
        labelled_data = data[data[data_consts.text_feature_cols].notna().sum(axis=1) > 0]
        binary_text_col_mapping = {"yes": True, "no": False}
        for col in data_consts.assumed_false_if_not_mentioned_cols:
            data[col].iloc[labelled_data.index] = data[col].iloc[labelled_data.index].fillna(False)
        for col in data_consts.boolean_text_feature_cols:
            data[col].iloc[labelled_data.index] = data[col].iloc[labelled_data.index].replace(binary_text_col_mapping)
        if data_consts.CREATE_UNKNOWN_CLASS:
            for col in data_consts.discrete_text_feature_cols:
                data[col].iloc[labelled_data.index] = data[col].iloc[labelled_data.index].fillna("unknown")
        labelled_data = data[data[data_consts.text_feature_cols].notna().sum(axis=1) > 0]
        return labelled_data
