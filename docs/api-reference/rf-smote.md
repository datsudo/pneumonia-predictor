# rf_smote.RfSMOTE

`RfSMOTE` acts as the baseline for the `RfActiveSMOTE` model. This is a combination of random forest and traditional, plain SMOTE method.

The source code for this model can be accessed in `pneumonia_predictor.backend.rf_smote`. Some of the parameters' default values can be found in `pneumonia_predictor.config`.

<h2>
<code>rf_smote.RfSMOTE(X_train, y_train, X_test, y_test, target_name, num_est)</code>
</h2>

### Parameters

- `X_train` : `pandas.DataFrame` - the training feature set
- `y_train` : `pandas.DataFrame` - the target labels for training set
- `X_test` : `pandas.DataFrame` - the test feature set
- `y_test` : `pandas.DataFrame` - the target labels for test set
- `target_name` : `str` - the name of the target feature
- `num_est` : `int`, default `config.N_ESTIMATORS` - amount of decision trees the random forest should build

### Methods

#### `train()`

Method that trains the model.

#### `save(model_name)`

Method to save the model as a pickle (`.pkl`) file. The destination is located at `project_root/saved_models` by default. You can change it in `config.SAVED_MODELS_PATH`.
