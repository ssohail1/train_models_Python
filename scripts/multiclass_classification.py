def model_pipeline(data_path, test_size, target_col):
    
    data = pd.read_csv(data_path)

  
    # STANDARDIZE NUMERICAL FEATURES
    continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuous_columns])
  
    # Convert to a DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
  
    # Combine with the original dataset
    scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

    
    # STANDARDIZE CATEGORICAL COLUMNS
    # Identifying categorical columns
    categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove(target_col)  # Exclude target column
  
    # Applying one-hot encoding to convert categorical variables to numerical format
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
  
    # Converting to a DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
  
    # Combining with the original dataset
    prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
    
    # Encoding the target variable
    prepped_data[target_col] = prepped_data[target_col].astype('category').cat.codes
    prepped_data.head()

    # Preparing the final data
    X = prepped_data.drop(target_col, axis=1)
    y = prepped_data[target_col]

    
    # MODEL TRAINING AND EVALUATION LOGISTIC REGRESSION
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=42,stratify=y)

    # Training logistic regression model
    model_multinom = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model_multinom.fit(X_train,y_train)
    y_pred = model_multinom.predict(X_test)
    
    # Evaluation metrics
    return f"Multinomial Strategy \n Accuracy: {np.round(100*accuracy_score(y_test, y_pred),2)}%"



accuracy = model_pipeline(file_path, test_size=0.2, target_col='target_column')

# Example data visualization code for visualizing model feature importance
importance = np.mean(np.abs(model_multinom.coef_),axis=0)
sns.barplot(x=importance, y=X.columns)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()
