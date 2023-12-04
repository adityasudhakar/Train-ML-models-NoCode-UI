
# Train-ML-models-NoCode-UI


## Introduction
Welcome ! This Streamlit-based interface allows users to train, fine-tune, and predict machine learning models without writing any code. It's designed to be intuitive and user-friendly, making machine learning more accessible to everyone.

## Getting Started
To get started with `Train-ML-models-NoCode-UI`, clone the repository and set up your environment:

```bash
git clone https://github.com/Ahmadshahzad2/Train-ML-models-NoCode-UI.git
cd Train-ML-models-NoCode-UI
pip install -r requirements.txt
```

## Usage
To run the application, use the following command:

```bash
code
streamlit run app.py
```

Once the app is running, you can perform the following actions through the UI:

**Create a new project**: Assign a name to your new project and it will be created for you.

**Upload a datase**t: Upload your CSV file to start training your model on your dataset.

**Configure your model**: Choose your target column, drop unnecessary columns, and specify ordinal columns.

**Data Preprocessing**: Fill null values and split your dataset into training and testing subsets.

**Set Conditions**: Apply slicing conditions to your data for more precise training.

**Model Selection**: Choose between classification or regression and select the model(s) you wish to train.

After configuring your settings, you can train your model and view the results, including confusion matrices and accuracy reports.

## Features:
Intuitive Streamlit UI for easy interaction
Support for training, fine-tuning, and predictions
Multiple model support with the ability to compare results
Preprocessing options including handling null values and data slicing
