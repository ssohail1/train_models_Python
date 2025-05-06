# train_models_Python

* Python libraries required:
* NumPy
* pandas
* seaborn
* matplotlib
* scikit-learn
  
* multiclass_classification.py:

   Python script which applies machine learning to train and test model by:
    - scaling the numerical data and combine the scaled numerical data to the original dataset
    - copy the categorical columns to a variable and remove target_col from that variable
    - use one-hot encoding on the categorical variables (minus the target_col)
    - combine the encoded categorical variable to the concat scaled data
    - Encode the target_col and save to new variable
    - set X and y variables where X has all data minus target_col and y has only target_col
    - apply train_test_split() on the X and y variables
    - fit a logistic regression model on the training data followed by model prediction on X_test
