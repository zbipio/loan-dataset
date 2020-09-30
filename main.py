import numpy as np

from model_functions import (
    run_cv,
    train_and_test_model,
)

from parameter_functions import (
    get_model_params,
    AVAILABLE_MODELS,
)

from preprocessing_functions import (
    read_csv_file,
    drop_rows_with_nans,
    create_target_variable,
    extract_number_from_text,
    create_date_features,
    create_log_features,
    create_factorizing_dict,
    factorize_categorical_features,
    optimize_dtypes,
    prepare_train_test_sets,
)


###### READ DATA ######
usecols = [
    'annual_inc', 'application_type', 'disbursement_method', 'earliest_cr_line', 'grade',
    'home_ownership', 'installment', 'int_rate', 'issue_d', 'loan_amnt', 'loan_status',
    'pub_rec_bankruptcies', 'sub_grade', 'term', 'verification_status',
]
loans = read_csv_file('accepted_2007_to_2018Q4.csv', usecols)
# loans = loans.sample(frac=0.1)  # can be used for a quicker debug process

###### CLEANING AND FEATURE ENGINEERING ######
loans = drop_rows_with_nans(loans)
loans = create_target_variable(loans)
loans = extract_number_from_text(loans)
loans = create_date_features(loans)
loans = create_log_features(loans)

categorical_cols = [
    'application_type', 'disbursement_method', 'grade', 'home_ownership', 'sub_grade', 'verification_status',
]
factorized_dict = create_factorizing_dict(loans, categorical_cols)
loans = factorize_categorical_features(loans, factorized_dict)

###### OPTIMIZE COLUMNS ######
cols_to_optimize = {
    'application_type_cat': np.int8, 'disbursement_method_cat': np.int8, 'earliest_cr_line_month': np.int8,
    'grade_cat': np.int8, 'home_ownership_cat': np.int8, 'issue_d_month': np.int8, 'sub_grade_cat': np.int8,
    'sub_grade_digit': np.int8, 'term_month': np.int8, 'verification_status_cat': np.int8,
    'earliest_cr_line_year': np.int16, 'issue_d_year': np.int16,
    'days_between_earliest_cr_and_issue': np.int32,
}

loans = optimize_dtypes(loans, cols_to_optimize)

###### FEATURE SELECTION ######
sel_cols = [
    # as are from table
    'int_rate', 'installment',

    # factorized categorical features or extracted from text
    'application_type_cat', 'home_ownership_cat',
    'verification_status_cat', 'disbursement_method_cat',
    'grade_cat', 'sub_grade_cat', 'sub_grade_digit',

    # date time columns
    'issue_d_month', 'issue_d_year', 'days_between_earliest_cr_and_issue',

    # Log columns
    'annual_inc_log'
]

###### MODEL TRAINING & TESTING ######
X_train, X_test, y_train, y_test = prepare_train_test_sets(loans[sel_cols], loans['bad_loan'])

chosen_model = 'LGBM'  # 'XGB' or 'LGBM' or 'DecisionTree'
hyperopt = False

params, hp_trials = get_model_params(chosen_model, X_train, y_train, X_test, y_test, hyperopt, max_evals=10)
model = AVAILABLE_MODELS[chosen_model](**params)

cv_mean_score, cv_score_std = run_cv(model, X_train, y_train)
test_score, y_pred = train_and_test_model(model, X_train, y_train, X_test, y_test)
print('Model {} results:'.format(chosen_model))
print('CV score: {}, CV std: {}'.format(cv_mean_score, cv_score_std))
print('Test score: {}'.format(test_score))
