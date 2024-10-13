# rf_active_smote.RfActiveSMOTE

A random forest model (uses `sklearn.ensemble.RandomForestClassifier`) integrated with Active SMOTE (uses `pneumonia_predictor.backend.active_smote.ActiveSMOTE`).

The source code for this model can be accessed in `pneumonia_predictor.backend.rf_active_smote`. Some of the parameters' default values can be found in `pneumonia_predictor.config`.

<h2>
<code>rf_active_smote.RfActiveSMOTE(X_train, y_train, X_test, y_test, target_name, num_est, num_clusters, sampling_ratio)</code>
</h2>


### Parameters

- `X_train` : `pandas.DataFrame` - the training feature set
- `y_train` : `pandas.DataFrame` - the target labels for training set
- `X_test` : `pandas.DataFrame` - the test feature set
- `y_test` : `pandas.DataFrame` - the target labels for test set
- `target_name` : `str` - the name of the target feature
- `num_est` : `int`, default `config.N_ESTIMATORS` - the name of the target feature
- `num_clusters` : `int`, default `config.N_CLUSTERS` - number of clusters for *k*-means clustering part of Active SMOTE
- `sampling_ratio` : `float`, default `config.SAMPLING_RATIO`

### Methods

#### `train(n_iterations)`

Method that trains the model.

##### Parameters

- `n_iterations` : `int` (default `config.N_ITERATIONS`) - number of retrains the model will do

#### `display_results(opt)`

Display line graph of model training's performance report.

##### Parameters

- `opt` : `str` - chosen metric to display; options are the following: `acc` (accuracy), `min` (minority class' performance), `maj` (majority class performance), and `avg` (weighted average of both class' performance)

#### `save(model_name)`

Method to save the model as a pickle (`.pkl`) file. The destination is located at `project_root/saved_models` by default. You can change it in `config.SAVED_MODELS_PATH`.

##### Parameters

- `model_name` : `str` - name of the pickle file
