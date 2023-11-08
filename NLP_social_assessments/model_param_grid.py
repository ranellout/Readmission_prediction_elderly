from consts import DataConsts, ModelConsts

class ModelParams:
    # general
    random_states = [1, 2, 42]
    # general decisions
    reduce_dimensions_params = [True, False]
    select_features_params = [True, False]
    data_imbalance_params = [True, False]
    input_feature_params = [DataConsts.text_col_english, DataConsts.text_col_hebrew]
    # text vectorization methods and parameters
    vectorizer_type_params = ['count', 'tfidf', 'dist']
    binary_params = [True, False]
    ngram_range_params = [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (1, 4), (2, 4)]
    max_features_min = 50
    max_features_max = 5000
    stop_words_params = [None, "english"]
    tfidf_norm_params = ['l1', 'l2']
    max_df_min = 0.5
    max_df_max = 1.0
    min_df_min = 0.
    min_df_max = 0.02
    analyzer_params = ['word', 'char', 'char_wb']
    # data imbalance params
    data_imbalance_strategy_params = ["over_sampling"]
    # feature selection params
    select_best_percentile_min = 0.01
    select_best_percentile_max = 100
    select_best_percentile_func_params = ['chi2', 'f_classif', 'mutual_info_classif']
    # dimension reduction params
    number_or_explained_variance_params = ["number", "explained_variance"]
    pca_n_components_min = 5
    pca_n_components_max = 99
    # model params
    model_type_params = ['logistic', 'random_forest', 'gradient_boost']
    rf_max_depth_min = 2
    rf_max_depth_max = 50
    rf_n_estimators_min = 30
    rf_n_estimators_max = 400

    gb_max_depth_min = 2
    gb_max_depth_max = 30
    gb_n_estimators_min = 30
    gb_n_estimators_max = 200

    lg_C_min = 0.01
    lg_C_max = 100
    lg_max_iter_min = 50
    lg_max_iter_max = 1500
    lg_penalty_params = ['l2', None]
    lg_solver_params = ['lbfgs', 'newton-cg', 'newton-cholesky']
