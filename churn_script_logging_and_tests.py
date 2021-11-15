"""
Project: Logging and tests for churn_library.py file

Author: Evidence Nwangwa
Date: 2021
"""

#import os
import logging
from churn_library import import_data, perform_eda,encoder_helper, cols_to_keep, perform_feature_engineering

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Dataframe has columns and rows")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(eda):
    '''
    test perform eda function
    '''
    df = import_data("./data/bank_data.csv")
    try:
        eda(df)
        logging.info("Images saved successfully")
    except BaseException:
        logging.error("Testing perform_eda: The images were not saved")
    # Don't know how to do assert statement to check the results


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df = import_data("./data/bank_data.csv")
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    response = "Churn"
    try:
        encoder_helper(df, cat_columns, response)
        logging.info("Encoding was successful")
    except BaseException:
        logging.error(
            "Testing encoding_helper: categorical columns were not successfully encoded")
    # Don't know how to do assert statement to check the results


def test_cols_to_keep(cols_to_keep):
    '''
    test cols_to_keep
    '''
    df = import_data("./data/bank_data.csv")
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    response = "Churn"
    df2 = encoder_helper(df, cat_columns, response)
    try:
        cols_to_keep(df2)
        logging.info("Feature engineering was successful")
    except BaseException:
        logging.error(
            "Testing feature_engineering: columns were not transformed")
    # Don't know how to do assert statement to check the results


def test_train_models(training_models):
    '''
    test perform_feature_engineering
    '''
    df = import_data("./data/bank_data.csv")
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    response = "Churn"
    df2 = encoder_helper(df, cat_columns, response)
    X, y = cols_to_keep(df2)
    try:
        training_models(X, y)
        logging.info("Model split into train and test was successful")
    except BaseException:
        logging.error(
            "test_train_models: Training and test data was not split sucessfully")
    # Don't know how to do assert statement to check the results


if __name__ == "__main__":
    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_cols_to_keep(cols_to_keep)
    test_train_models(perform_feature_engineering)
