from typing import Callable, Dict, Tuple, Union, List
from pandas import Series
import numpy as np
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from mlops.utils.models.sklearn import load_class, tune_hyperparameters

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def hyperparameter_tuning(
    training_set: Union[Dict[str, Union[Series, csr_matrix, list]], list],
    model_class_name: Union[str, list] = None,
    *args,
    **kwargs,
) -> Tuple[
    Dict[str, Union[bool, float, int, str]],
    Union[csr_matrix, None],
    Union[Series, None],
    Dict[str, Union[Callable[..., BaseEstimator], str]],
]:
    if isinstance(training_set, dict):
        build_data = training_set.get('build')
    else:
        build_data = training_set
    
    build_data = tuple(build_data) if isinstance(build_data, list) else build_data

    print(f"build_data: {build_data}")  # Optional debug print to check initial value

    if len(build_data) == 7:
        X, X_train, X_val, y, y_train, y_val, _ = build_data
    elif len(build_data) == 2:
        X, y = build_data
        X_train = X_val = y_train = y_val = None
    else:
        # Add debugging output to capture the unexpected build_data
        print(f"Debug: Unexpected build_data content: {build_data}")
        print(f"Debug: Length of build_data: {len(build_data)}")
        raise ValueError(f"Unexpected number of elements in build_data: expected 7 or 2, but got {len(build_data)}")

    if isinstance(model_class_name, list):
        model_class_name = model_class_name[0] if model_class_name else None

    if not isinstance(model_class_name, str) or not model_class_name:
        print(f"Received model_class_name: {model_class_name}")
        raise ValueError("model_class_name must be a non-empty string or list with at least one element.")

    model_class = load_class(model_class_name)

    def reshape_data(data):
        if data is not None and not isinstance(data, str):
            try:
                return check_array(data, ensure_2d=True)
            except ValueError:
                return data.reshape(-1, 1) if data.ndim == 1 else data
        return data

    # Only reshape if the data is not a string or None
    X_train, y_train = reshape_data(X_train), reshape_data(y_train)
    X_val, y_val = reshape_data(X_val), reshape_data(y_val)

    if X_train is not None and y_train is not None:
        hyperparameters = tune_hyperparameters(
            model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            max_evaluations=kwargs.get('max_evaluations'),
            random_state=kwargs.get('random_state'),
        )
    else:
        hyperparameters = {}

    return hyperparameters, X, y, dict(cls=model_class, name=model_class_name)
