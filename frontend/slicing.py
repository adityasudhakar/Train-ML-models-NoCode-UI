import streamlit as st
import pandas as pd

# Get column details
def get_column_details(df):
    details = {}
    for column in df.columns:
        unique_values = df[column].unique()
        if df[column].dtype == 'object' or len(unique_values) < 4:
            details[column] = ('categorical', unique_values)
        else:
            details[column] = ('numerical', (df[column].min(), df[column].max()))
    return details

# Function to remove a condition from the session state
def remove_condition(project_name,index):
    """
    Remove a condition from the current project's conditions list and re-run the Streamlit app.

    Parameters:
    index (int): The index of the condition to remove.
    """
    try:
        current_project_conditions = st.session_state.projects[project_name]['conditions']
        del current_project_conditions[index]
        st.session_state.projects[project_name]['conditions'] = current_project_conditions
        st.rerun()
    except Exception as e:
        st.error(f"Failed to remove condition: {e}")

# Function to display the conditions table in Streamlit
def display_conditions_table(project_name):
    """
    Display a table of current conditions with the option to delete each condition.
    """
    with st.expander("Current Conditions", expanded=True):
        try:
            current_project_conditions = st.session_state.projects[project_name]['conditions']
            conditions_df = pd.DataFrame(current_project_conditions, columns=['Column', 'Condition', 'Value'])
            
            for index, condition in conditions_df.iterrows():
                cols = st.columns((2, 1, 1, 1))
                cols[0].write(condition['Column'])
                cols[1].write(condition['Condition'])
                cols[2].write(condition['Value'])
                if cols[3].button("Delete", key=f"delete_{index}"):
                    remove_condition(project_name,index)
        except Exception as e:
            st.error(f"Failed to display conditions table: {e}")

# Function to display the form for adding slicing conditions
def display_slicing_form(project_name,df, column_details):
    """
    Display a form in Streamlit to add slicing conditions.

    Parameters:
    df (DataFrame): The DataFrame to which the conditions will be applied.
    column_details (dict): A dictionary containing details about DataFrame columns.
    """
    with st.expander("Add Slicing Conditions:", expanded=True):
        try:
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_column = st.selectbox("Columns", options=df.columns, index=0, key='selected_column')
            column_type, column_data = column_details[selected_column]

            with col2:
                condition_options = ['==', '!='] if column_type == 'categorical' else ['<', '<=', '==', '!=', '>=', '>']
                selected_condition = st.selectbox("Condition", options=condition_options, index=0, key=f'condition_{selected_column}')

            with col3:
                if column_type == 'categorical':
                    selected_value = st.selectbox("Value", options=column_data, index=0, key=f'value_{selected_column}')
                else:
                    selected_value = st.slider("Value", min_value=min(column_data), max_value=max(column_data), value=min(column_data), key=f'slider_{selected_column}')

            submit_button = st.button(label='Add Condition')
            if submit_button:
                if 'conditions' not in st.session_state.projects[project_name]:
                    st.session_state.projects[project_name]['conditions'] = []                
                current_project_conditions = st.session_state.projects[project_name]['conditions']
                current_project_conditions.append((selected_column, selected_condition, selected_value))
                st.session_state.projects[project_name]['conditions'] = current_project_conditions
        except Exception as e:
            st.error(f"Failed to display slicing form: {e}")

# Apply conditions to the DataFrame
def apply_conditions(df, conditions):
    """
    Apply the specified conditions to filter the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to filter.
    conditions (list of tuples): Conditions to apply to the DataFrame.

    Returns:
    DataFrame: The filtered DataFrame after applying conditions.
    """
    try:
        for col, cond, val in conditions:
            if cond == '==':
                df = df[df[col] == val]
            elif cond == '!=':
                df = df[df[col] != val]
            elif cond == '<':
                df = df[df[col] < val]
            elif cond == '<=':
                df = df[df[col] <= val]
            elif cond == '>':
                df = df[df[col] > val]
            elif cond == '>=':
                df = df[df[col] >= val]
        return df
    except Exception as e:
        st.error(f"An error occurred while applying conditions: {e}")
        return df

# Main UI function for slicing the DataFrame based on user-defined conditions
def slicing_ui(project_name, df, column_details):
    """
    Main UI function for adding data slicing conditions and applying them to the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to be sliced.
    column_details (dict): A dictionary containing details about DataFrame columns.
    """
    st.write("### Data Slicing Conditions")
    filtered_df = None
    
    display_slicing_form(project_name,df, column_details)
    if 'conditions' in st.session_state.projects[project_name]:
        display_conditions_table(project_name)
        
        

                
            # try
        current_project_conditions = st.session_state.projects[project_name]['conditions']
        filtered_df = apply_conditions(df, current_project_conditions)
        filtered_df = filtered_df.reset_index(drop=True)
        # except Exception as e:
        #     st.error(f"An error occurred while filtering the data: {e}")

    return filtered_df
