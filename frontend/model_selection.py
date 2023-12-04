import streamlit as st

# Define available models
CLASSIFICATION_MODELS = {
    'logistic': 'Logistic Regression',
    'decision_tree_C': 'Decision Tree Classifier',
    'random_forest_C': 'Random Forest Classifier',
    'svm_C': 'SVC'
}

REGRESSION_MODELS = {
    'linear': 'Linear Regression',
    'decision_tree_R': 'Decision Tree Regressor',
    'random_forest_R': 'Random Forest Regressor',
    'svm_R': 'SVR'
}

def add_model_section(key_suffix, current_config=None):
    if current_config is None:
        current_config = {
            'model_type': 'Classification',
            'chosen_models': [],
            'hyperparameters': []  # This will be a list of booleans
        }

    default_model_type = current_config['model_type']
    model_type = st.selectbox(
        "Choose model type",
        ["Classification", "Regression"],
        key=f"model_type_{key_suffix}"
    )

    # Define the options based on the model type
    options = CLASSIFICATION_MODELS if model_type == "Classification" else REGRESSION_MODELS

    # Model selection based on the model type
    chosen_models = st.multiselect(
        f"Select {model_type} Models",
        options.values(),
        key=f"models_{key_suffix}"
    )

    # Initialize the hyperparameters list with False for each chosen model
    hyperparameters = [False] * len(chosen_models)

    # Loop through each chosen model
    for index, model in enumerate(chosen_models):
        model_key = next(key for key, value in options.items() if value == model)
        # Create a checkbox for hyperparameters tuning
        hyperparameters[index] = st.checkbox(
            f"Enable Hyperparameter Tuning for {model}",
            value=current_config['hyperparameters'][index] if index < len(current_config['hyperparameters']) else False,
            key=f"hyperparam_{model_key}_{key_suffix}"
        )

    # Map selected model names back to their keys
    selected_model_keys = [key for key, value in options.items() if value in chosen_models]

    return {
        "model_type": model_type.lower(),
        "chosen_models": selected_model_keys,
        "hyperparameters": hyperparameters  # This is now a list of booleans
    }

def model_selection_ui(default_configs=None):
    if default_configs is None:
        default_configs = {
            'model_type': 'Classification',
            'chosen_models': [],
            'hyperparameters': []  # This will be a list of booleans
        }

    if 'model_configs' not in st.session_state:
        st.session_state['model_configs'] = [default_configs]

    with st.expander("Model Configuration", expanded=True):
        st.session_state['model_configs'][0] = add_model_section('main', st.session_state['model_configs'][0])

    return st.session_state['model_configs'][0]