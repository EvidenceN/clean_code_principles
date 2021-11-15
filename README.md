# Use "Predict Customer Churn" Project to apply clean code principles 


## Project *Practicing Clean Code Principles*

## Project Description
This project predicts customer churn. The primary goal of this project is to take the notebook code located at churn_notebook.ipynb and transform it into production ready code in the notebook churn_library.py. 

Furthermore, this project practices test_driven_development. This project also focuses on creating a test for the project in the file churn_script_logging_and_tests.py


## Running Files
How do you run your files? What should happen when you run your files?

Instructions on how to run the project files on local computer using pipenv or anaconda

* Create a virtual environment using pipenv or anaconda
* Install the libraries in requirements.txt
* Follow the instructions below to execute the code for test file and python file

**How to run *churn_library.py* and what the output should be**

To execute the code, run `ipython churn_library.py`

To get a pep8 score, run `pylint churn_library.py`

The output on the console should be these values: `X_train.head(), X_test.head(), y_train[:5], y_test[:5], y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf`

**How to run test file *churn_script_logging_and_tests.py***

To test the code, run `ipython churn_script_logging_and_tests.py`

To test the pylint score, run `pylint churn_script_logging_and_tests.py`

Output should be nothing on the console. But in churn_library.log, there should be messages indicating success or failure of tests. 


