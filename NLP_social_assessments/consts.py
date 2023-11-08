from sklearn.metrics import make_scorer, fbeta_score

class Metrics:
    f_two_score = make_scorer(fbeta_score, beta=2)
    f_half_score = make_scorer(fbeta_score, beta=0.5)
    f_1_10_score = make_scorer(fbeta_score, beta=0.1)
    f_1_100_score = make_scorer(fbeta_score, beta=0.01)

class ModelConsts:
    RANDOM_STATE = 42
    TRAIN_SIZE = 0.8
    cross_val_n_splits = 5
    cross_val_shuffle = True
    default_scoring = {'precision': 'precision',
                       'accuracy': 'accuracy',
                       'balanced_accuracy': 'balanced_accuracy',
                       'f1': 'f1',
                       'accuracy': 'accuracy',
                       'recall': 'recall',
                       'f2_score': Metrics.f_two_score,
                       'f0.5_score': Metrics.f_half_score,
                       'f0.1_score': Metrics.f_1_10_score,
                       'f0.01_score': Metrics.f_1_100_score,
                       'roc_auc': 'roc_auc'}

class DataConsts:
    data_filename = "social_assesments.csv"
    data_columns_to_drop = ["Unnamed: 0"]
    data_columns_rename_dict = {'social_assesment_hebrew': 'social_assessment_hebrew'}
    # columns definitions
    text_col_hebrew = "social_assessment_hebrew"
    text_col_english = "social_assesment"
    target_var = "readmission_criteria"
    patient_id_col = "patient_identifier"
    text_feature_cols = ["sex", "age", "immigrant", "year_of_immigration", "marital_status", "children",
                         "closest_relative", "closest_supporting_relative", "help_at_home_hours",
                         "seeking_help_at_home", "is_holocaust_survivor", "is_exhausted", "needs_extreme_nursing",
                         "has_extreme_nursing", "is_confused", "is_dementic", "residence", "recommended_residence",
                         "deceased"]
    assumed_false_if_not_mentioned_cols = ['immigrant', 'is_holocaust_survivor']
    boolean_text_feature_cols = ['immigrant', 'seeking_help_at_home', 'is_holocaust_survivor', 'is_exhausted',
                                 'needs_extreme_nursing', 'has_extreme_nursing', "is_confused", "is_dementic",
                                 'deceased']
    discrete_text_feature_cols = ['sex', 'immigrant', 'marital_status', 'closest_relative',
                                  'closeset_supporting_relative', 'seeking_help_at_home', 'is_holocaust_survivor',
                                  'is_exhausted', "needs_extreme_nursing", "has_extreme_nursing", "is_confused",
                                  "is_dementic", "residence", "recommended_residence"]
    continuous_text_feature_cols = ['age', 'year_of_immigration', 'children', 'help_at_home_hours']
    # data prep decisions
    """
    parameters along the way
    - text_features prep - create or not create a 'not available' class (not only true / false)
    - treat readmissions as separate entities - did not look into how many patients were readmitted
    - input features - text straight away / try to use manual textual feature labels / both
    - text prep - vectorization straight away / use NER to reduce dimension+make possibly more indicative for input features classification
    """
    CREATE_UNKNOWN_CLASS = False