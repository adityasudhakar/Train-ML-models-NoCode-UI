import streamlit as st
import pandas as pd
from io import StringIO
# from frontend.model_selection import model_selection_ui
from frontend.training_details import training_det
from frontend.fineval_details import fv_det
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

# Ensure that the 'projects' dictionary exists within the Streamlit session state for storing project data
if 'projects' not in st.session_state:
    st.session_state['projects'] = {}

def clear_projects_callback():
    """
    Clear all project-related information from the session state.
    """
    st.session_state['projects'] = {}
    # Delete all session state keys except 'projects'
    for key in list(st.session_state.keys()):
        if key != 'projects':
            del st.session_state[key]
    st.experimental_rerun()

def create_project_callback():
    """
    Create a new project with the given name and initialize its details in session state.
    """
    new_project_name = st.session_state['new_project_name'].strip()
    new_project_type = st.session_state['new_project_type']
    if new_project_name:
        if new_project_name not in st.session_state['projects']:
            st.session_state['projects'][new_project_name] = {
                "csv": None, "out_label": None, "drop_cols": [], "final_dataframe": None, 'model_details':{},'project_type':new_project_type, 'train':-2,'args': None, "path": None,"problem_type": None,"models": [],"hyper_parameters": [], "ordinal_col": None, "fill_strategy": None, "test_size": None, "slice_conditions" : {}, "dataframe": None,"hyperparameters":[], "completed_status": {}
            }
            st.session_state['current_project'] = new_project_name
            st.sidebar.success(f"Project '{new_project_name}' created!")
            st.session_state['new_project_name'] = ''
        else:
            st.sidebar.error("Project already exists.")
    else:
        st.sidebar.error("Please enter a valid project name.")

def project_details(project_name):
    details = st.session_state['projects'][project_name]

    if details['project_type'] == 'train':
        training_det(project_name)
    if details['project_type'] == 'Fine-tunning/Prediction':
        fv_det(project_name)

def main():
    """
    The main function that initializes the Streamlit UI components and handles navigation.
    """
    st.sidebar.title("Projects")

    if 'current_project' not in st.session_state:
        st.session_state['current_project'] = None
    
    project_type = st.sidebar.selectbox(
        "Select Project Type",
        ["train", "Fine-tunning/Prediction"],
        key="new_project_type"
    )

    project_names = list(st.session_state['projects'].keys())
    new_project_name = st.sidebar.text_input("Enter the name of the project", key="new_project_name")
    st.sidebar.button("Create New Project", on_click=create_project_callback)

    for project_name in project_names:
        if st.sidebar.button(project_name):
            st.session_state['current_project'] = project_name

    if st.session_state.get('current_project'):
        project_details(st.session_state['current_project'])

    if st.sidebar.button("Clear Projects"):
        clear_projects_callback()

if __name__ == "__main__":
    main()