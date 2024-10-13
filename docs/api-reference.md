# Model Usage

Assuming that your dataset already undergoes a proper preprocessing pipeline and data splitting (into training and testing sets), you are now
ready to use both `RfActiveSMOTE` and `RfSMOTE`.


- `pneumonia_predictor.backend.rf_smote`

Some of the parameters' default values can be found in `pneumonia_predictor.config`.

---

## rf_smote.RfSMOTE

### `rf_smote.RfSMOTE(X_train, y_train, X_test, y_test, target_name, num_est)`

#### Parameters

- `X_train` : `pandas.DataFrame`
- `y_train` : `pandas.DataFrame`
- `X_test` : `pandas.DataFrame`
- `y_test` : `pandas.DataFrame`
