
import streamlit as st
import pandas as pd
from io import StringIO
import os
import yaml
from services.project_service import ProjectService
from multiprocessing import freeze_support
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



def find_projects_with_args(base_path='Projects'):
    """
    Find immediate subfolders within the base path that contain an 'args.yaml' file somewhere in their subdirectories.
    Returns a dictionary mapping subfolder names to the first found 'args.yaml' file path.
    """
    projects_with_args = {}
    for project_name in os.listdir(base_path):
        project_path = os.path.join(base_path, project_name)
        if os.path.isdir(project_path):
            for root, dirs, files in os.walk(project_path):
                if 'args.yaml' in files:
                    projects_with_args[project_name] = os.path.join(root, 'args.yaml')
                    break  # Only the first occurrence is needed
    return projects_with_args

def find_models(project_path):
    """
    Find all 'args.yaml' files within the given project path and extract model names.
    Returns a list of unique model names found within the project.
    """
    model_names = set()
    for root, dirs, files in os.walk(project_path):
        if 'args.yaml' in files:
            args_path = os.path.join(root, 'args.yaml')
            try:
                with open(args_path, 'r') as file:
                    args = yaml.safe_load(file)
                    if 'models' in args:
                        model_names.update(args['models'])
            except Exception as e:
                st.error(f"Error reading {args_path}: {e}")
    return list(model_names)

def load_fv_project_details(project_name):
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
        'model_details':[],
        'project_type': 'fine_val',
        'train': -2,
        'args': None,
        'path': None,
        "completed_status": {}
    }
    return st.session_state['projects'].get(project_name, default_details)

def save_fv_project_details(project_name, details):
    """
    Save project details to the Streamlit session state.

    Parameters:
    - project_name: str, the name of the project
    - details: dict, a dictionary containing project details such as DataFrame, target column, etc.
    """
    st.session_state['projects'][project_name] = details

def fv_det(project_name):    
    """
    Render the UI elements for a specific project to upload data, select target/drop columns,
    and configure model selection.
    """
    st.header(f"Project: {project_name}")
    details = load_fv_project_details(project_name)

    # File upload handling
    uploaded_file = st.file_uploader("Upload a CSV", type=['csv'], key=f"{project_name}_upload")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        details.update({
            "csv": df.to_json(),
            "out_label": None,
            "drop_cols": [],
            "final_dataframe": None,
            "model_details": [],
            "project_type:": 'fine_val',
            'train': -2,
            'args': None,
            'path': None,
            "completed_status": {}
        })
        save_fv_project_details(project_name, details)  # Save after upload

    if details['csv']:
        df = pd.read_json(StringIO(details['csv']))

        # Target column selection
        process_type = st.selectbox(
            'Choose Process Type',
            ['finetune', 'predict'],
            index=0 if details['out_label'] not in ['finetune', 'predict'] else ['finetune', 'predict'].index(details['out_label']),
            key=f"{project_name}_process_type"
        )
        if process_type == "finetune":
            details['train'] = -1
        else:
            details['train'] = 0
        run_directories = find_projects_with_args()
        if run_directories:  # Check if the list is not empty
            selected_run = st.selectbox("Select a run folder", list(run_directories.keys()), key=f"{project_name}_run_folder")
            selected_project_run = os.path.join('Projects', selected_run)
            # models, args_final = get_models_from_run(selected_run) if selected_run else []
            # selected_models = st.multiselect("Select models", models, key=f"{project_name}_model_selection")
            args_file_path = run_directories[selected_run]
            try:
                with open(args_file_path, 'r') as file:
                    project_args = yaml.safe_load(file)
                    st.success(f"'args.yaml' successfully loaded for project: {selected_run}")
            except Exception as e:
                st.error(f"Failed to load 'args.yaml' file for project {selected_run}: {e}")

            # Find and list models from the selected project
            models_list = find_models(selected_project_run)
            if models_list:
                selected_models = st.multiselect("Select models", models_list)
                # Assuming you want to do something with the selected models...
            else:
                st.write("No models found in the selected project.")
            details['args'] = project_args
            details['args']["models"] = selected_models
            details['args']['train'] = details['train']

            
        


        csv_path = save_dataframe_to_csv(dataframe=df,file_name=project_name)
        details['path'] = csv_path
        # print(csv_path)
        details['args']["path"] = csv_path
        # column_details = get_column_details(df)
        # filtered_df = slicing_ui(project_name, df, column_details)
        # df = filtered_df.copy() if filtered_df is not None else df
        details['conitions'] = st.session_state['projects'][project_name]['conditions'] if 'conditions' in st.session_state['projects'][project_name] else []

        # Model selection using the provided model_selection_ui function
        # model_choices = model_selection_ui(details['model_details'])
        # details['model_details'] = model_choices

        # Confirm project settings and finalize the DataFrame
        if st.button('Confirm Project', key=f"{project_name}_confirm"):
            final_df = df
            details['final_dataframe'] = final_df.to_json()
            
            print(details['args'])
            save_fv_project_details(project_name, details)
            st.success(f"Project '{project_name}' confirmed!")

        # train(project_name=project_name,details=details)
        train_button = st.button("Train")
        if train_button and project_name:
            if 'ProjectService' not in st.session_state['projects'][project_name]:
                st.session_state['projects'][project_name]['ProjectService'] = ProjectService(details['args'])
            freeze_support()
            st.session_state['projects'][project_name]['ProjectService'].start_service()
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
                for xm in details["args"]['models']:
                    if st.session_state['projects'][project_name]['train'] == 0:
                        find_and_display_analytics_json('Projects/'+details["args"]["proj_name"]+"/"+xm+"/_prediction")
                    if st.session_state['projects'][project_name]['train'] == -1:
                        find_and_display_analytics_json('Projects/'+details["args"]["proj_name"]+"/"+xm+"/fine-tuning")
                        print('Projects/'+details["args"]["proj_name"]+"/"+xm+"/fine-tuning")
                
            else:
                st.write("Waiting for training to complete...")
                time.sleep(5)
                st.experimental_rerun()
        # Display the final DataFrame if available
        if details['final_dataframe']:
            st.write('Final DataFrame:')
            final_df = pd.read_json(StringIO(details['final_dataframe']))
            st.dataframe(final_df)
        else:
            # Always display the current CSV data
            st.write('Current CSV data:')
            st.dataframe(df)