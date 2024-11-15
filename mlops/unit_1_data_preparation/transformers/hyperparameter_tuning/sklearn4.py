from typing import Callable, Dict, Tuple, Union

from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils.models.sklearn import load_class, tune_hyperparameters

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def hyperparameter_tuning(
    training_set: Dict[str, Union[Series, csr_matrix]],
    model_class_name: str,
    *args,
    **kwargs,
) -> Tuple[
    Dict[str, Union[bool, float, int, str]],
    csr_matrix,
    Series,
    Callable[..., BaseEstimator],
]:
    # Check type and content of training_set
    print(f"Type of training_set: {type(training_set)}")
    print(f"Contents of training_set: {training_set}")
    
    # If training_set is a string or unexpected type, raise an error to help debug
    if isinstance(training_set, str):
        raise ValueError("Unexpected type for `training_set`. Expected a dictionary or similar structure, but got a string.")

    # Assuming we confirmed it is a list or dictionary structure after the above
    build_output = training_set.get('build', None) if isinstance(training_set, dict) else training_set[0]
    
    if build_output is None:
        raise ValueError("`build` key or expected output not found in training_set.")
    
    # Unpack and continue as before if `build_output` is structured correctly
    print(f"Type of build_output: {type(build_output)}")
    print(f"Contents of build_output: {build_output}")
    
    # Proceed with unpacking if it contains 7 elements
    X, X_train, X_val, y, y_train, y_val, dv_info = build_output

    model_class = load_class(model_class_name)

    hyperparameters = tune_hyperparameters(
        model_class,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_evaluations=kwargs.get('max_evaluations'),
        random_state=kwargs.get('random_state'),
    )

    return hyperparameters, X, y, dict(cls=model_class, name=model_class_name)
