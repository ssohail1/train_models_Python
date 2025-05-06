# train_models_Python
* multiclass_classification.py:

   Python script which applies machine learning to train and test model by:
    - 1st: scaling the numerical data and combine the scaled numerical data to the original dataset
    - 2nd: copy the categorical columns to a variable and remove target_col from that variable
    - 3rd: use one-hot encoding on the categorical variables (minus the target_col)
    - 4th: combine the encoded categorical variable to the concat scaled data
    - 5th: 
