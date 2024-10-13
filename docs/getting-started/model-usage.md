# Model Usage

Assuming that your dataset already undergoes a proper preprocessing pipeline and data splitting (into training and testing sets), you are now
ready to use both `RfActiveSMOTE` and `RfSMOTE`. See the [API reference](../api-reference/index.md) for more details.

```python
from pneumonia_predictor.backend.rf_active_smote import RfActiveSMOTE
from pneumonia_predictor.backend.rf_smote import RfSMOTE

rf_active_smote = RfActiveSMOTE(
    X_train, y_train,
    X_test, y_test,
    "pneumonia_admission"
)

rf_smote = RfSMOTE(
    X_train, y_train,
    X_test, y_test,
    "pneumonia_admission"
)
```

Both models have `train()` methods, except that `rf_active_smote.train()` can optionally accepts `n_iterations` parameter to state that the model should do *N* number of retrains (default value is `config.N_ITERATIONS = 5`).

```python
rf_active_smote.train(4)
rf_smote.train()
```

> ℹ️ Logging is enabled by default&mdash;every step of model training is being logged in `logs.txt`. To turn this off, change the value of `config.LOGFILE_ENABLED` to `False`.

Both models have `save()` method for saving trained models as a pickle file (`.pkl`) with the help of [joblib](https://joblib.readthedocs.io/). The pickles are saved in `project-root/saved_models` by default (specified in `config.SAVED_MODELS_PATH`).

```python
# will be saved as saved_models/rf_active_smote_model.pkl
rf_active_smote.save("rf_active_smote_model")

# will be saved as saved_models/rf_smote_model.pkl
rf_smote.save("rf_smote_model")
```
