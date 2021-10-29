# Build Inputs and test code. 
import churn_library

from churn_library import *

pth = "./data/bank_data.csv"
df = import_data(pth)

perform_eda(df)

category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

response = "Churn"

df2 = encoder_helper(df, category_lst, response)

X, y = cols_to_keep(df2)

X_train, X_test, y_train, y_test = perform_feature_engineering(X, y)

y_train_preds_lr, y_test_preds_lr = logistic_model(X_train, X_test, y_train)

y_train_preds_rf, y_test_preds_rf = random_forest_model(X_train, X_test, y_train)

lrc_plot (X_test, y_test)

rfc_lrc_roc_curve(X_test, y_test)


print(X_train.head())
print(X_test.head())
print(y_train[:5])
print(y_test[:5])
print(y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf )
