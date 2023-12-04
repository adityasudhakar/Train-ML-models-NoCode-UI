
import streamlit as st
import pandas as pd
from io import StringIO
from frontend.model_selection import model_selection_ui
from frontend.slicing import slicing_ui, get_column_details
import os
# from frontend.fineval_details import train
from services.project_service import ProjectService
from multiprocessing import freeze_support
import time
from frontend.analytics import find_and_display_analytics_json



def save_dataframe_to_csv(dataframe, file_name, folder_name="data"):
    """
    Saves a DataFrame to a CSV file and returns the path of the saved file.
    If the file already exists, it will be overwritten.
    
    :param dataframe: The DataFrame to save.
    :param folder_name: The name of the folder to save the CSV in.
    :param file_name: The name of the CSV file to save the DataFrame as.
    :return: The path to the saved CSV file.
    """
    # Create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Construct the full file path
    csv_file_path = os.path.join(folder_name, f'{file_name}.csv')
    
    # Save the DataFrame to CSV, overwrite if the file already exists
    dataframe.to_csv(csv_file_path, index=False)
    
    # Return the full path of the saved CSV file
    return csv_file_path

def load_train_project_details(project_name):
    """
    Load project details from the Streamlit session state.

    Parameters:
    - project_name: str, the name of the project

    Returns:
    - dict: a dictionary containing the details of the project
    """
    default_details = {
        "csv": None,
        "out_label": None,
        "drop_cols": [],
        "final_dataframe": None,
        'model_details':{},
        'project_type': 'train',
        'train': -2,
        'args': None,
        "path": None,
        "problem_type": None,
        "models": [],
        "hyper_parameters": [],
        "ordinal_col": None,
        "fill_strategy": None,
        "test_size": None,
        "slice_conditions" : {},
        "dataframe": None,
        "hyperparameters":[],
        "completed_status": {}

    }
    return st.session_state['projects'].get(project_name, default_details)

def save_train_project_details(project_name, details):
    """
    Save project details to the Streamlit session state.

    Parameters:
    - project_name: str, the name of the project
    - details: dict, a dictionary containing project details such as DataFrame, target column, etc.
    """
    st.session_state['projects'][project_name] = details

def training_det(project_name):    
    """
    Render the UI elements for a specific project to upload data, select target/drop columns,
    and configure model selection.
    """
    st.header(f"Project: {project_name}")
    details = load_train_project_details(project_name)

    # File upload handling
    uploaded_file = st.file_uploader("Upload a CSV", type=['csv'], key=f"{project_name}_upload")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        details.update({
            "csv": df.to_json(),
            "out_label": None,
            "drop_cols": [],
            "final_dataframe": None,
            "dataframe": None,
            "model_details": {},
            "project_type:": 'train',
            'train': -2,
            'args': None,
            "path": None,
            "problem_type": None,
            "models": [],
            "hyper_parameters": None,
            "ordinal_col": None,
            "fill_strategy": None,
            "test_size": None,
            "slice_conditions" : {},
            "hyperparameters":[],
            "completed_status": {}
        })
        save_train_project_details(project_name, details)  # Save after upload

    if details['csv']:
        df = pd.read_json(StringIO(details['csv']))
        details['dataframe'] = df


        # Target column selection
        out_label = st.selectbox(
            'Select Target Column',
            df.columns,
            index=df.columns.get_loc(details['out_label']) if details['out_label'] else 0,
            key=f"{project_name}_out_label"
        )
        details['out_label'] = out_label

        # Drop column selection
        columns_to_drop = st.multiselect(
            'Select Columns to Drop',
            [col for col in df.columns if col != out_label],
            default=details['drop_cols'],
            key=f"{project_name}_drop_cols"
        )



        non_numerical_columns = [col for col in df.select_dtypes(include=['object']).columns]
        ordinal_col = st.multiselect(
            'Ordinal Column',
            [col for col in df.columns if col != out_label and col not in columns_to_drop and col in non_numerical_columns],
            default=details['ordinal_col'],
            key=f"{project_name}_ordinal_col"
        )
        details["ordinal_col"] = ordinal_col
        details['train'] = 1
        fill_strategy = st.selectbox(
            'Choose Process Type',
            ['mean', 'median', 'mod'],
            index=0 if details['fill_strategy'] not in ['mean', 'median','mod'] else ['mean', 'median','mod'].index(details['fill_strategy']),
            key=f"{project_name}_fill_strategy"
        )
        details["fill_strategy"] = fill_strategy

        test_size = st.slider("Test Dataset Size", min_value=0.2, max_value=0.5, value=0.2, key=f'slider_test_size')
        details["test_size"] = test_size

        column_details = get_column_details(df.drop(columns=columns_to_drop))
        filtered_df = slicing_ui(project_name, df.drop(columns=columns_to_drop), column_details)
        # df = filtered_df.copy() if filtered_df is not None else df
        slice_conditions_original = st.session_state['projects'][project_name]['conditions'] if 'conditions' in st.session_state['projects'][project_name] else []
        
        slice_conditions = {}
        for condition in slice_conditions_original:
            slice_conditions[condition[0]] = condition[1] + str(condition[2])
        details['slice_conditions'] = slice_conditions
        details['drop_cols'] = columns_to_drop

        # Model selection using the provided model_selection_ui function
        
        if not details['model_details']:
            details['model_details'] = {
                'model_type': 'classification',  # preset to 'classification' or use your own logic
                'chosen_models': [],  # preselect models or leave empty
                'hyperparameters': []  # set default hyperparameter tuning choice
            }
        model_choices = model_selection_ui(details['model_details'])
        

        if model_choices != {}:
            details['model_details'] = model_choices
            details['problem_type'] =  model_choices['model_type']
            details['models'] =  model_choices['chosen_models']
            details['hyper_parameters'] =  model_choices['hyperparameters']
        print("model_choices:   ",model_choices)
            # details['hyper_parameters'] = [1 if model_choices[0]['tune_hyperparameters'] else 0] * len(model_choices[0]['chosen_models'])
        final_df = df.copy()
        final_df = filtered_df.copy() if filtered_df is not None else final_df
        # final_df = final_df.drop(columns=details['drop_cols'])
        
        details['final_dataframe'] = final_df.to_json()
        csv_path = save_dataframe_to_csv(dataframe=df,file_name=project_name)
        details['path'] = csv_path


        # Confirm project settings and finalize the DataFrame
        if st.button('Confirm Project', key=f"{project_name}_confirm"):
            print("Session_Storage",st.session_state['projects'][project_name])
            # print("Path",csv_path)
            save_train_project_details(project_name, details)
            st.success(f"Project '{project_name}' confirmed!")
            # train(project_name=project_name,details=details)
        train_button = st.button("Train")
        if train_button and project_name:
            args = {
            'train': details['train'],
            'proj_name': project_name,
            'path': details['path'],
            'out_label': details['out_label'],
            'drop_cols': details['drop_cols'],
            'slice_conditions': details['slice_conditions'],
            'problem_type': details['problem_type'],
            'models': details['models'],
            'hyper_parameters': details['hyper_parameters'],
            'ordinal_col': details['ordinal_col'],
            'fill_strategy': details['fill_strategy'],
            'test_size': details['test_size']
            }
            print(args)
            if 'ProjectService' not in st.session_state['projects'][project_name]:
                st.session_state['projects'][project_name]['ProjectService'] = ProjectService(args)
            freeze_support()
            st.session_state['projects'][project_name]['ProjectService'].start_service()


        # Display the final DataFrame if available
        if details['final_dataframe']:
            st.write('Final DataFrame:')
            final_df = pd.read_json(StringIO(details['final_dataframe']))
            st.dataframe(final_df)
        else:
            # Always display the current CSV data
            st.write('Current CSV data:')
            st.dataframe(df)
        if 'ProjectService' in st.session_state['projects'][project_name]:
            status = st.session_state['projects'][project_name]['ProjectService'].get_status()
            if status:
                for model, state in status.items():
                    st.session_state['projects'][project_name][model] = state
                    st.write(f"{model}: {state}")
                    if state == 'Training':
                        st.session_state['projects'][project_name]['completed_status'][model] = False
                    if state == 'Completed':
                        st.session_state['projects'][project_name]['completed_status'][model] = True
            if all(st.session_state['projects'][project_name]['completed_status'].values()):
                # st.experimental_rerun()
                for xm in details['models']:
                    if st.session_state['projects'][project_name]['train'] == 1:
                        find_and_display_analytics_json('Projects/'+project_name+"/"+xm+"/train")
                    
            else:
                st.write("Waiting for training to complete...")
                time.sleep(5)
                st.experimental_rerun()
                        
                    
        
            # Re-run the script every 5 seconds to update the status
            