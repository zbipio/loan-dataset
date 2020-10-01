import gc
from pathlib import Path

from typing import (
    Dict,
    List,
    Tuple,
)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from logger import (
    log_and_stop,
    LOAN_LOGGER,
)


def read_csv_file(file: str, usecols: List[str]) -> pd.DataFrame:
    """
    Read the specified file consisting loan data with specified usecols.
    Columns issue_d and earliest_cr_line are parsed as datetime.
    Numerical columns have explicitly specified dtypes for memory optimization.
    Function returns a read file.
    """
    try:
        file_path = Path(__file__).parents[0].absolute() / 'data' / file

        read_file = pd.read_csv(file_path,
                                usecols=usecols,
                                parse_dates=['issue_d', 'earliest_cr_line'],
                                dtype={'int_rate': np.float16, 'installment': np.float32, 'annual_inc': np.float32})

        LOAN_LOGGER.info('File read')
        return read_file
    except:
        message = 'File NOT read'
        log_and_stop(LOAN_LOGGER, message)


def drop_rows_with_nans(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows which contain NaN in one of the 3 columns: `annual_inc`, `earliest_cr_line`, 'pub_rec_bankruptcies'.
    Function returns a new dataset with removed NaN rows.
    """
    try:
        df = dataset.copy()
        df = df.dropna(subset=['annual_inc', 'earliest_cr_line'])

        LOAN_LOGGER.info('Rows containing NaNs removed')
        return df
    except:
        message = 'Rows containing NaNs NOT removed'
        log_and_stop(LOAN_LOGGER, message)


def create_target_variable(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Create the column bad_loan which is True if loan_status has any of the following values:
    `Charged Off`, `Late (31-120 days)`, 'Late (16-30 days)',
    `Does not meet the credit policy. Status:Charged Off`, `Default`.
    Function returns a new dataset with the target variable.
    """
    try:
        df = dataset.copy()
        bad_status = [
            'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)',
            'Does not meet the credit policy. Status:Charged Off', 'Default'
        ]
        df['bad_loan'] = df['loan_status'].isin(bad_status)

        LOAN_LOGGER.info('Target Variable created')
        return df
    except:
        message = 'Target Variable NOT created'
        log_and_stop(LOAN_LOGGER, message)


def extract_number_from_text(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the number of months from the term column
    and the digit from the sub_grade column.
    Function returns a new dataset with new features.
    """
    try:
        df = dataset.copy()
        df['term_month'] = df['term'].map(lambda x: int(x.strip()[:2]))
        df['sub_grade_digit'] = df['sub_grade'].map(lambda x: int(x[1]))

        LOAN_LOGGER.info('Number values extracted from text columns')
        return df
    except:
        message = 'Number values NOT extracted from text columns'
        log_and_stop(LOAN_LOGGER, message)


def create_date_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on datetime columns: issue_d, earliest_cr_line.
    Function returns a new dataset with new features.
    """
    try:
        df = dataset.copy()
        df['issue_d_month'] = df['issue_d'].dt.month
        df['issue_d_year'] = df['issue_d'].dt.year
        df['earliest_cr_line_month'] = df['earliest_cr_line'].dt.month
        df['earliest_cr_line_year'] = df['earliest_cr_line'].dt.year
        df['days_between_earliest_cr_and_issue'] = (df['issue_d'] - df['earliest_cr_line']).dt.days

        LOAN_LOGGER.info('Date features created')
        return df
    except:
        message = 'Date features NOT created'
        log_and_stop(LOAN_LOGGER, message)


def create_log_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the logarithms for skewed the column annual_inc.
    Function returns a dataset with new features.
    """
    try:
        df = dataset.copy()
        df['annual_inc_log'] = np.log1p(df['annual_inc'].values)

        LOAN_LOGGER.info('Logarithm features created')
        return df
    except:
        message = 'Logarithm features NOT created'
        log_and_stop(LOAN_LOGGER, message)


def create_factorizing_dict(dataset: pd.DataFrame, cat_feats: List[str]) -> Dict:
    """
    Create a dictionary which contains all label-number relations for the given categorical variables.
    Function returns the created dictionary.
    """
    try:
        df = dataset.copy()
        factorized_dict = {}
        for cat_feat in cat_feats:
            factorized_dict[cat_feat] = pd.factorize(df[cat_feat])[1]

        LOAN_LOGGER.info('Factorized dictionary created')
        return factorized_dict
    except:
        message = 'Factorized dictionary NOT created'
        log_and_stop(LOAN_LOGGER, message)


def factorize_categorical_features(dataset: pd.DataFrame, factorized_dict: Dict) -> pd.DataFrame:
    """
    Changing all categorical variables into numerical variables based on the given factorized dictionary.
    Function returns a new dataset where factorized columns end with `_cat`.
    """
    try:
        df = dataset.copy()
        for cat_feat in factorized_dict.keys():
            df['{}_cat'.format(cat_feat)] = df[cat_feat].map(
                lambda x: factorized_dict[cat_feat].get_loc(x))

        LOAN_LOGGER.info('Categorical features factorized')
        return df
    except:
        message = 'Categorical features NOT factorized'
        log_and_stop(LOAN_LOGGER, message)


def optimize_dtypes(dataset: pd.DataFrame, dtype_cols) -> pd.DataFrame:
    """
    Optimize the dtype for the given set of columns in order to use less memory.
    Additionally garbage collection is run.
    Function returns a new dataset with new dtypes.
    """
    try:
        df = dataset.copy()
        for key in dtype_cols.keys():
            df.loc[:, key] = df[key].astype(dtype_cols[key])

        gc.collect()

        LOAN_LOGGER.info('dtypes optimized for given columns')
        return df
    except:
        message = 'dtypes NOT optimized for given columns'
        log_and_stop(LOAN_LOGGER, message)


def prepare_train_test_sets(features, target) -> Tuple:
    """
    Change the DataFrame and Series into numpy arrays and divide the data into a training and test set.
    Test set will consist 20% of observations.
    Function returns X and y divided into training and test sets.
    """
    try:
        X = np.array(features, dtype=np.float)
        y = np.array(target, dtype=np.float)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020, stratify=y)

        LOAN_LOGGER.info('Train and test sets prepared')
        return X_train, X_test, y_train, y_test
    except:
        message = 'Train and test sets NOT prepared'
        log_and_stop(LOAN_LOGGER, message)


def preprocess_data(file: str) -> pd.DataFrame:
    """
    Read the loan file and preprocess all data.
    Useful function to minimize visible code e.g. for visualization purposes.
    Returns the preprocessed loan dataset.
    """
    usecols = [
        'annual_inc', 'application_type', 'disbursement_method', 'earliest_cr_line', 'grade',
        'home_ownership', 'installment', 'int_rate', 'issue_d', 'loan_amnt', 'loan_status',
        'pub_rec_bankruptcies', 'sub_grade', 'term', 'verification_status',
    ]
    df = read_csv_file(file, usecols)

    df = drop_rows_with_nans(df)
    df = create_target_variable(df)
    df = extract_number_from_text(df)
    df = create_date_features(df)
    df = create_log_features(df)

    categorical_cols = [
        'application_type', 'disbursement_method', 'grade', 'home_ownership', 'sub_grade', 'verification_status',
    ]
    factorized_dict = create_factorizing_dict(df, categorical_cols)
    df = factorize_categorical_features(df, factorized_dict)

    cols_to_optimize = {
        'application_type_cat': np.int8, 'disbursement_method_cat': np.int8, 'earliest_cr_line_month': np.int8,
        'grade_cat': np.int8, 'home_ownership_cat': np.int8, 'issue_d_month': np.int8, 'sub_grade_cat': np.int8,
        'sub_grade_digit': np.int8, 'term_month': np.int8, 'verification_status_cat': np.int8,
        'earliest_cr_line_year': np.int16, 'issue_d_year': np.int16,
        'days_between_earliest_cr_and_issue': np.int32,
    }

    df = optimize_dtypes(df, cols_to_optimize)
    return df
