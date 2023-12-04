# utils/data_processing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler
import pickle
import os


class DataProcessing:

    def __init__(self,args):
        self._args=args
        self._df=pd.read_csv(args['path'])
        self._out_label=args['out_label']
        self._X=None
        self._y= None
        self._label_map={}

    def _select_col(self):
        """
        Restricts DataFrame 'X' to only the specified columns.

        Args:
        - self._args['_select_col']: List of column names to keep.

        Returns:
        None
        """
        self._df.drop(columns=self._args['drop_cols'],inplace=True)
        

    def upload_csv(file):
        """
        Read an uploaded CSV file into a pandas DataFrame.

        Args:
        - file: The uploaded file object from Streamlit.

        Returns:
        - DataFrame: The read data from the CSV file.
        """
        df = pd.read_csv(file)
        return df

    def infer_data_types(df):
        """
        Infer the data types of columns in the DataFrame.

        Args:
        - df: The DataFrame whose data types need to be inferred.

        Returns:
        - dict: A dictionary mapping each column to its inferred data type.
        """
        data_types = df.dtypes.to_dict()
        return data_types

    def _slice_data(self):
        """
        Slice the DataFrame based on given criteria.

        Args:
        - df: The DataFrame to be sliced.
        - criteria: A dict of criteria. Example: {"age": ">18", "income": "<=50000", "gender": "==male"}

        Returns:
        - DataFrame: The sliced data.
        """
        criteria=self._args['slice_conditions']
        df=self._df
        for column, condition in criteria.items():
            if pd.api.types.is_numeric_dtype(df[column]):
                if ">=" in condition:
                    value = float(condition.split(">=")[1])
                    df = df[df[column] >= value]
                elif "<=" in condition:
                    value = float(condition.split("<=")[1])
                    df = df[df[column] <= value]
                elif ">" in condition:
                    value = float(condition.split(">")[1])
                    df = df[df[column] > value]
                elif "<" in condition:
                    value = float(condition.split("<")[1])
                    df = df[df[column] < value]
            else:
                if "==" in condition:
                    value = condition.split("==")[1]
                    df = df[df[column] == value]
                elif "!=" in condition:
                    value = condition.split("!=")[1]
                    df = df[df[column] != value]
                # Add more conditions as required for categorical data.
        return df


    
    def _encode_categorical(df, columns):
        """
        Convert categorical columns to numerical by label encoding.

        Args:
        - df: The DataFrame whose columns need to be encoded.
        - columns: List of column names to be encoded.

        Returns:
        - DataFrame: The DataFrame with encoded columns.
        """
        label_encoders = {}
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        return df, label_encoders

    def _one_hot_encode(df, columns):
        """
        One-hot encode specific columns of the DataFrame.

        Args:
        - df: The DataFrame whose columns need to be one-hot encoded.
        - columns: List of column names to be one-hot encoded.

        Returns:
        - DataFrame: The DataFrame with one-hot encoded columns.
        """
        df = pd.get_dummies(df, columns=columns)
        return df

    def _split_data(self):

        """
        Split the DataFrame into training and test sets.

        Args:
        - df: The DataFrame to be split.
        - target_column: Name of the target column.
        - test_size: Proportion of the dataset to include in the test split.

        Returns:
        - X_train, X_test, y_train, y_test: Training and test sets.
        """
        
        X_train, X_test, y_train, y_test = train_test_split(self._X, self._y, test_size=self._args["test_size"])
        return X_train, X_test, y_train, y_test
    
    def __autonomous_preprocessing(self):
        """
        Autonomously preprocess the given DataFrame.

        Args:
        - df: input dataframe
        - ordinal_columns: list of columns that are ordinal. 
                           All other object type columns will be assumed to be nominal.

        Returns:
        - DataFrame: The preprocessed DataFrame.
        """


        ### pre-process the feature

        df=self._X
        ordinal_columns=[]
        # If no ordinal columns are specified, initialize as empty list

        if self._args["ordinal_col"]:
            ordinal_columns=self._args["ordinal_col"]

       

        # Identify nominal columns
        nominal_columns = [col for col in df.select_dtypes(include=['object']).columns 
                           if col not in ordinal_columns]
        
        # Identify numerical columns (excluding ordinal columns)
        numerical_columns = [col for col in df.select_dtypes(exclude=['object']).columns 
                             if col not in ordinal_columns]

        # Apply one-hot encoding to nominal columns
        for col in nominal_columns:
            ohe = OneHotEncoder(drop='first', sparse=False)
            new_ohe_features = ohe.fit_transform(df[col].values.reshape(-1, 1))
            new_features_df = pd.DataFrame(new_ohe_features, columns=[f"{col}_{category}" 
                                               for category in ohe.categories_[0][1:]],
                                               index=df.index)
            df = pd.concat([df, new_features_df], axis=1)
            df.drop(col, axis=1, inplace=True)

        # Apply label encoding to ordinal columns
        label_encoders = {}  # Dictionary to hold the LabelEncoders for each column

        for col in ordinal_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Now save this dictionary for future ineferences using label encorder
        project_path=os.path.join(os.getcwd(), self._args['proj_directory'])
        folder_pth=os.path.join(project_path,'label_encorders')
        os.makedirs(folder_pth,exist_ok=True)
        with open(os.path.join(folder_pth,'label_encoders_inputs.pkl'), 'wb') as file:
            pickle.dump(label_encoders, file)
            

            
        # Normalize numerical columns
        scaler = MinMaxScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        self._X=df


        ### pre_process the output label


        if self._args['problem_type']=='classification':
            label_encoders = {} 

            le = LabelEncoder()
            self._y=le.fit_transform(self._y)
            label_encoders[self._args['out_label']] = le
            with open(os.path.join(folder_pth,'label_encoders_output.pkl'), 'wb') as file:
                pickle.dump(label_encoders, file)

    
    def _autonomous_data_cleaning(self):
        """
        df: input dataframe
        strategy: strategy to handle missing numerical values. Can be 'mean', 'median', or 'mode'.
                For categorical values, 'mode' will be used regardless of this parameter.
        """

        df=self._df
        strategy=self._args["fill_strategy"]

        # Drop columns with more than 35% missing values
        threshold_col = 0.35 * len(df)
        df.dropna(axis=1, thresh=threshold_col, inplace=True)
        
        # Drop rows with missing values in more than 15% of its columns
        threshold_row = 0.15 * len(df.columns)
        df.dropna(axis=0, thresh=df.shape[1] - threshold_row, inplace=True)

        # Handle remaining missing values
        # For numerical columns
        for col in df.select_dtypes(include=['float64', 'int64']):
            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

        # For categorical columns
        for col in df.select_dtypes(include=['object']):
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Remove duplicate rows
        df.drop_duplicates(inplace=True)

        # Remove constant columns
        for col in df.columns:
            if len(df[col].unique()) == 1:
                df.drop(col, axis=1, inplace=True)

        # # Convert string numbers to actual numbers
        # for col in df.select_dtypes(include=['object']):
        #     try:
        #         df[col] = pd.to_numeric(df[col])
        #     except ValueError:
        #         pass  # column cannot be converted to a number

        self._df=df

    def pre_process(self):
        self._select_col()
        self._slice_data()
        self._autonomous_data_cleaning()
        # print(self._df.head())

        self._y= self._df[self._args['out_label']]
        self._X=self._df.drop(columns=self._args['out_label'],axis=1)
        self.__autonomous_preprocessing()
        if self._args['train']:
            data_arr= self._split_data()
    
        else:
            data_arr=self._X, self._y

        return data_arr


# Example usage
if __name__ == "__main__":

    general_args = {
        'train': 1,
        'proj_name': 'P5',
        'proj_directory': "Projects/P1",
        'path': 'data/sample_data.csv',
        'out_label': 'salary',
        'drop_cols': ['Unnamed: 0'],
        'slice_conditions': {"age": ">18"},
        'problem_type': 'regression',
        'models': ['linear', 'decision_tree_R', 'random_forest_R', 'svm_R'],
        'hyper_parameters': [0, 1, 0, 1],
        'ordinal_col': ['education_level'],
        'fill_strategy': 'mean',
        'test_size': 0.3
    }

    my_data=DataProcessing(general_args)

    

    X_train, X_test, y_train, y_test=my_data.pre_process()

    print('stop')

    # data = {
    #     'age': [25, 30, 35, 40, 45],
    #     'gender': ['Male', 'Female', 'Male', 'Female', 'Other'],
    #     'education_level': ['Bachelor', 'High School', 'PhD', 'Masters', 'Bachelor'],
    #     'salary': [50000, 55000, 70000, 68000, 62000]
    # }

    # df = pd.DataFrame(data)
    # df.to_csv('data/sample_data.csv')

    # print("Original DataFrame:")
    # print(df)

    # processed_df = DataProcessing.__autonomous_preprocessing(df, ordinal_columns=['education_level'])

    # print("\nProcessed DataFrame:")
    # print(processed_df)

    