"""
Project: Transforming customer churn project from notebook into production ready python code

Author: Evidence Nwangwa
Date: 2021
"""

# Project libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    # churn histogram
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig("images/eda/churn_hist_img.png")

    # customer_age_histogram
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig("images/eda/customer_age_hist_img.png")

    # marital status histogram
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig("images/eda/marital_status_hist_img.png")

    # distribution plot
    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig("images/eda/dist_plot_img.png")

    # heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig("images/eda/heat_map_img.png")


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns
    '''

    for column in category_lst:
        column_lst = []
        group = df.groupby(column).mean()['Churn']
        for val in df[column]:
            column_lst.append(group.loc[val])
            column_name = f"{column}_{response}"
        df[column_name] = column_lst
    return df


def cols_to_keep(df):
    """
    Input:
            df: pandas dataframe

    Output:
            X: New pandas dataframe that has only the columns that wants to be kept.
            y: The target values from the dataframe.
    """
    y = df['Churn']
    X = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]

    return X, y


def perform_feature_engineering(X, y):
    '''
    input:
              X: The pandas dataframe
              y: the target pandas series
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # train test split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def logistic_model(X_train, X_test, y_train):
    '''
    train a logistic regression model and save the output

    input:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
    output:
            None
    '''
    lrc = LogisticRegression(solver='liblinear')

    # fit the model
    lrc.fit(X_train, y_train)

    # predict with the model
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save model.
    joblib.dump(lrc, './models/logistic_model.pkl')

    return y_train_preds_lr, y_test_preds_lr


def random_forest_model(X_train, X_test, y_train):
    '''
    train a random forest model and save the output

    input:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
    output:
            None
    '''
    rfc = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # fit the model
    cv_rfc.fit(X_train, y_train)

    # predict with the model
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # save model.

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')

    return y_train_preds_rf, y_test_preds_rf


def lrc_plot(X_test, y_test):
    '''
            Produces roc_curve for linear regression
    input:
            y_test:  test response values
            X_test: The pandas dataframe

    output:
            None
    '''
    # load logistic regression model.
    lr_model = joblib.load('./models/logistic_model.pkl')

    # logistic regression roc_curve
    plt.plot_roc_curve(lr_model, X_test, y_test)
    plt.savefig("images/results/lrc_roc_curve.png")


def rfc_lrc_roc_curve(X_test, y_test):
    '''
            Produces roc_curve for linear regression and random forest.
    input:
            y_test:  test response values
            X_test: The pandas dataframe

    output:
            None
    '''

    # load random forest model.
    rfc_model = joblib.load('./models/rfc_model.pkl')
    # load logistic regression model.
    lr_model = joblib.load('./models/logistic_model.pkl')

    # logistic regression roc_curve
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    # random forest + logistic regression roc_curve
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.show()
    plt.savefig("images/results/rfc_lrc_roc_curve.png")


def shap_tree_image(X_test):
    """
    Input:
            X_test: The test pandas dataframe
    Output:
            None
    """
    rfc_model = joblib.load('./models/rfc_model.pkl')
    explainer = shap.TreeExplainer(rfc_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    plt.savefig("images/results/rfc_shap_tree_explainer.png")


def feature_importance_plot(X_test):
    """
    Input:
            X_test: The test pandas dataframe
    Output:
            None
    """
    # load random forest model.
    rfc_model = joblib.load('./models/rfc_model.pkl')

    # Calculate feature importances
    importances = rfc_model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances

    names = [X_test.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_test.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_test.shape[1]), names, rotation=90)

    plt.savefig("images/results/rfc_feature_importance.png")


# RANDOM FOREST classification report

def rf_classification_report(
        y_train,
        y_train_preds_rf,
        y_test,
        y_test_preds_rf):
    """
    Input:
            Y_train - training data
            y_train_preds_rf - Predicted values from logistic regression using train data set
            y_test - test data
            y_test_preds_rf - Predicted values from logistic regression using test data set
    Output:
            None
    """

    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    plt.savefig("images/results/random_forest_classification_report.png")

# LOGISTIC REGRESSION classification report


def lrc_classification_report(
        y_train,
        y_train_preds_lr,
        y_test,
        y_test_preds_lr):
    """
    Input:
            Y_train - training data
            y_train_preds_lr - Predicted values from logistic regression using train data set
            y_test - test data
            y_test_preds_lr - Predicted values from logistic regression using test data set
    Output:
            None
    """

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    plt.savefig("images/results/logistic_regression_classification_report.png")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


if __name__ == "__main__":
    # Build Inputs and test code.
    pth = "./data/bank_data.csv"
    df = import_data(pth)

    perform_eda(df)

    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    response = "Churn"

    df2 = encoder_helper(df, category_lst, response)

    X, y = cols_to_keep(df2)

    X_train, X_test, y_train, y_test = perform_feature_engineering(X, y)

    print(X_train.head())
    print(X_test.head())
    print(y_train[:5])
    print(y_test[:5])

    y_train_preds_lr, y_test_preds_lr = logistic_model(
        X_train, X_test, y_train)

    y_train_preds_rf, y_test_preds_rf = random_forest_model(
        X_train, X_test, y_train)

    print(y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf)

