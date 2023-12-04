# utils/model_training.py
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import os
import json
import yaml
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


from utils.data_processing import DataProcessing
# from data_processing import DataProcessing




def convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list


class MachineLearningModel:
    
    def __init__(self, args,hyper_param,model_name,data):
        """
        Initializes the MachineLearningModel with the specified arguments.
        :param args: Dictionary with keys 'task_type', 'models_to_train', '_hyperparameter_tuning'
                     'task_type' is a string that can be 'classification' or 'regression'
                     'models_to_train' is a list of models to be trained
                     '_hyperparameter_tuning' is a boolean indicating whether to perform tuning
        """
        self._args=args
        self._status=0
        self._task_type = args.get('problem_type')
        self._hyper_param = hyper_param
        self._model_zoo = {
                'logistic': LogisticRegression(),
                'decision_tree_C': DecisionTreeClassifier(),
                'random_forest_C': RandomForestClassifier(),
                'svm_C': SVC(probability=True),
                'linear': LinearRegression(),
                'decision_tree_R': DecisionTreeRegressor(),
                'random_forest_R': RandomForestRegressor(),
                'svm_R': SVR()
            }
        
        self._model = None
        self._model_name=model_name
        self._data=data
        self._predictions= []

        self._grid_search_params_zoo = {
                'logistic': {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            },
            'decision_tree_C': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'random_forest_C': {
                'n_estimators': [10, 50, 100, 200],
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm_C': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto']
            },
          
            'linear': {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            'decision_tree_R': {
                'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'random_forest_R': {
                'n_estimators': [10, 50, 100, 200],
                'criterion': ['mse', 'mae'],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm_R': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'epsilon': [0.1, 0.2, 0.3, 0.4]
            }
        }

        self._model_hyperm=self._grid_search_params_zoo[self._model_name]
              
    def get_status(self):
        return self._status
    
    def process_train(self):
        train_dir= os.path.join(self._args["proj_directory"],self._model_name,'train')
        os.makedirs(train_dir,exist_ok=True)

        self._model=self._model_zoo[self._model_name]


        print("training", self._model_name)
        if not self._hyper_param:
            self._train_model()
        else:

            self._hyperparameter_tuning()

        print("trained", self._model_name)

        self._test_model()
        self._performace_analytics()
        self._save_weights()
        self.__save_args()


    def process_finetune(self):

        finetune_dir= os.path.join(self._args["proj_directory"],self._model_name,'fine-tuning')
        os.makedirs(finetune_dir,exist_ok=True)

        model_load_pth=os.path.join(self._args["proj_directory"],self._model_name,'train','model_weights_analytics',self._model_name+'.pkl')
        self._model= self._load_modelload_model(model_load_pth)
        print("training", self._model_name)

        if not self._hyper_param:
            self._train_model()
        else:
            self._hyperparameter_tuning()

        print("trained", self._model_name)

        self._test_model()
        print("test", self._model_name)

        self._performace_analytics()
        print("analytics", self._model_name)

        self._save_weights()
        print('saved_weights')
        self.__save_args()
        print('saved_args')

        self._status=1



    def process__predict(self):

        _prediction_dir= os.path.join(self._args["proj_directory"],self._model_name,'_prediction')
        os.makedirs(_prediction_dir,exist_ok=True)

        model_load_pth=os.path.join(self._args["proj_directory"],self._model_name,'train','model_weights_analytics',self._model_name+'.pkl')
        
        self._model= self._load_modelload_model(model_load_pth)
        self._predict()
        np.savetxt(os.path.join(_prediction_dir,'pred.csv'), self._predictions.astype(int), delimiter=",")
        self._performace_analytics()
        self.__save_args()
        self._status=1



    def _predict(self):
        self._predictions = self._model.predict(self._data[0])


    def _train_model(self):
        X_train, _, y_train, _=self._data
    
        self._model.fit(X_train, y_train)
    
    def _test_model(self):
        _, x_test, _, _=self._data
        self._predictions = self._model.predict(x_test)

    def _performace_analytics(self):
        _predictions=self._predictions

        if self._args['train']:
            _, _, _, y_test=self._data
        else:
            y_test=self._data[1]

        if self._args['train']==1:
            project_path=os.path.join(os.getcwd(), self._args['proj_directory'])
            folder_pth=os.path.join(project_path,self._model_name,'train','model_weights_analytics')
        elif self._args['train']==-1:
            folder_pth= os.path.join(self._args["proj_directory"],self._model_name,'fine-tuning','model_weights_analytics')
        else:
            folder_pth= os.path.join(self._args["proj_directory"],self._model_name,'_prediction',"model_weights_analytics")



        os.makedirs(folder_pth,exist_ok=True)


        task_type=self._args['problem_type']

        results = {}

        if task_type == "classification":
            # Accuracy
            results['accuracy'] = accuracy_score(y_test, _predictions)
            
            # Classification report (precision, recall, f1-score, and support)
            results['classification_report'] = classification_report(y_test, _predictions)
            
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_test, _predictions)
            
            # AUC-ROC curve for binary classification
            if len(np.unique(y_test))  == 2 and max(y_test)<2:
                fpr, tpr, thresholds = roc_curve(y_test, _predictions)
                results['roc_auc'] = auc(fpr, tpr)

                # Plotting AUC-ROC curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % results['roc_auc'])
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
            # Save figure
                img_path = os.path.join(folder_pth, 'roc_curve.png')
                plt.savefig(img_path)
                results['roc_auc_image'] = img_path

                plt.close()  # Close the plot
        elif task_type == "regression":
            # Mean Squared Error
            results['MSE'] = mean_squared_error(y_test, _predictions)
            
            # R2 Score
            results['R2'] = r2_score(y_test, _predictions)
            
            # Mean Absolute Error
            results['MAE'] = mean_absolute_error(y_test, _predictions)

        with open(os.path.join(folder_pth,'analytics.json'), 'w') as json_file:
            json.dump(results, json_file,default=convert,indent=2)

    def _save_weights(self):
        if self._args['train']==1:
            project_path=os.path.join(os.getcwd(), self._args['proj_directory'])
            folder_pth=os.path.join(project_path,self._model_name,'train','model_weights_analytics')
        elif self._args['train']==-1:
            folder_pth= os.path.join(self._args["proj_directory"],self._model_name,'fine-tuning','model_weights_analytics')
        self._save_model(folder_pth)
        
    def __save_args(self):
        if self._args['train']==1:
            project_path=os.path.join(os.getcwd(), self._args['proj_directory'],self._model_name,'train')
        elif self._args['train']==-1:
            project_path= os.path.join(self._args["proj_directory"],self._model_name,'fine-tuning')
        else:
            project_path= os.path.join(self._args["proj_directory"],self._model_name,'_prediction')

        # import pdb;pdb.set_trace()
        print(project_path)
        args_yaml = yaml.dump(self._args, default_flow_style=False)
    
        with open(os.path.join(project_path,'args.yaml'), 'w') as file:
            file.write(args_yaml)

    def _hyperparameter_tuning(self):
        X_train, _, y_train, _=self._data


        grid_search = GridSearchCV(self._model, self._model_hyperm, cv=5)
        grid_search.fit(X_train, y_train)
        self._model = grid_search.best_estimator_



        # grid_search_params = self.get_grid_search_params()
        # for model_name in self._models_to_train:
        #     model = self._models.get(model_name)
        #     if model:
        #         grid_search = GridSearchCV(model, grid_search_params.get(model_name, {}), cv=5)
        #         grid_search.fit(X_train, y_train)
        #         self.best_models[model_name] = grid_search.best_estimator_
        # return self.best_models


    def _save_model(self, model_path):
        path = os.path.join(model_path, f'{self._model_name}.pkl')
        joblib.dump(self._model, path)
        return path

    def _load_modelload_model(self, model_path):
        return joblib.load(model_path)



if __name__ == "__main__":

    general_args={
        'train': 0,
        'proj_directory': "Projects/P3",
        'path': 'data/sample_data.csv' ,
        'out_label': 'gender',
        'drop_cols': ['Unnamed: 0'],
        'slice_conditions': {"age": ">18", "salary": "<=50000", "gender": "==Male"},
        'problem_type': 'classification',
        'models': ['logistic','decision_tree_C','random_forest_C','svm_C'],
        'hyper_parameters': [0,1,0,1],
        'ordinal_col': ['education_level'],
        'fill_strategy':'mean',
        'test_size': 0.3
    }

    my_data=DataProcessing(general_args)

    

    data_arr=my_data.pre_process()

  

    for model,hyper in zip(general_args['models'],general_args['hyper_parameters']):

        my_model=MachineLearningModel(general_args,hyper,model,data_arr)

        if general_args['train']:
            my_model.process_train()
        else:
            my_model.process__predict()

    




