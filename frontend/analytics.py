import streamlit as st
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    st.subheader("Confusion Matrix:")
    st.pyplot(fig)

def plot_classification_report(cr):
    report_data = []
    lines = cr.split('\n')
    for line in lines[2:-5]:
        row = {}
        parsed_line = [v for v in line.split("  ") if v]  # Filter out empty strings
        row['class'] = parsed_line[0].strip()
        row['precision'] = float(parsed_line[1])
        row['recall'] = float(parsed_line[2])
        row['f1_score'] = float(parsed_line[3])
        row['support'] = float(parsed_line[4])
        report_data.append(row)
    report_df = pd.DataFrame.from_dict(report_data)
    report_df.set_index('class', inplace=True)
    st.subheader("Classification Report:")
    st.dataframe(report_df)
def display_additional_metrics(metrics):
    for key, value in metrics.items():
        if key not in ['confusion_matrix', 'classification_report']:
            st.subheader(f"{key.capitalize()}:")
            if isinstance(value, (list, dict)):
                st.json(value)
            else:
                st.write(value)
def find_and_display_analytics_json(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'analytics.json':  # Look specifically for 'analytics.json'
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Extract the model name from the directory above the 'train' directory
                model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
                
                # Read the .json file
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    # Display the model name
                    st.header(model_name)
                    
                    # Plot and display metrics
                    if 'confusion_matrix' in data:
                        cm = np.array(data['confusion_matrix'])
                        # Assuming the classes are numbered sequentially from 0
                        classes = list(range(len(cm)))
                        plot_confusion_matrix(cm, classes)
                    
                    if 'classification_report' in data:
                        plot_classification_report(data['classification_report'])
                    
                    # Display the rest of the metrics
                    display_additional_metrics(data)

