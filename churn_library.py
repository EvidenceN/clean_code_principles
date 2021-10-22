"""
Project: Transforming customer churn project from notebook into production ready python code

Author: Evidence Nwangwa
Date: 2021
"""

# Project libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    df = pd.read_csv(pth)
    return df

def perform_eda(df):
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        '''
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

        # churn histogram
        plt.figure(figsize=(20,10)) 
        df['Churn'].hist()
        plt.savefig("images/eda/churn_hist_img.png")

        # customer_age_histogram
        plt.figure(figsize=(20,10)) 
        df['Customer_Age'].hist()
        plt.savefig("images/eda/customer_age_hist_img.png")

        # marital status histogram
        plt.figure(figsize=(20,10)) 
        df.Marital_Status.value_counts('normalize').plot(kind='bar');
        plt.savefig("images/eda/marital_status_hist_img.png")

        # distribution plot
        plt.figure(figsize=(20,10)) 
        sns.distplot(df['Total_Trans_Ct'])
        plt.savefig("images/eda/dist_plot_img.png") 

        # heatmap
        plt.figure(figsize=(20,10)) 
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
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
                df: pandas dataframe with new columns for
        '''

        # gender encoded column
        category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

        for column in category_lst:
                column_lst = []
                group = df.groupby(column).mean()['Churn']
                for val in df[column]:
                        column_lst.append(group.loc[val])
                        column_name = f"{column}_{response}"
                df[column_name] = column_lst
        
        


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


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

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass

if __name__ == "__main__":
        pth = "./data/bank_data.csv"
        df = import_data(pth)
        perform_eda(df)