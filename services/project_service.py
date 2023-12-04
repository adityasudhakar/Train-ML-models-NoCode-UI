import multiprocessing
from utils.data_processing import DataProcessing
import os
from utils.model_training import MachineLearningModel
from multiprocessing import Process, Manager

 



class ProjectService:
    def __init__(self,args):
        self.args=args
        self.data_lodaer=None
        self.models=self.args.get('models')
        self.proj_name=self.args.get('proj_name')
        self.models_train= []
        self.status_dict = dict()

    def get_status(self):
        return self.status_dict

    def start_service(self):
        project_path=os.path.join(os.getcwd(), 'Projects',self.proj_name)
        os.makedirs(project_path,exist_ok=True)
        self.args["proj_directory"]=project_path

        self.data_lodaer=DataProcessing(self.args)

        processed_data=self.data_lodaer.pre_process()

        processes = []



        
        for model,hyper in zip(self.models,self.args['hyper_parameters']):
            self.models_train.append(MachineLearningModel(self.args,hyper,model,processed_data))
            if self.args['train']==1:
                p = multiprocessing.Process(target=self.models_train[-1].process_train)
                self.status_dict[model]='Training'

            elif self.args['train']==-1:
                self.status_dict[model]='Training'


                p = multiprocessing.Process(target=self.models_train[-1].process_finetune)
            else:
                self.status_dict[model]='Training'


                p = multiprocessing.Process(target=self.models_train[-1].process__predict)

            processes.append(p)
            p.start()

        for p in processes:
            p.join()
            self.status_dict[model]='Completed'

        
        
        






                

if __name__ == "__main__":

    general_args={
        'proj_name': 'P2',
        'path': 'data/sample_data.csv' ,
        'out_label': 'gender',
        'drop_cols': ['Unnamed: 0'],
        'slice_conditions': {"age": ">18", "salary": "<=50000", "gender": "==Male"},
        'problem_type': 'classification',
        'models': ['logistic','decision_tree','random_forest_C','svm_C'],
        'hyper_parameters': [0,1,0,1],
        'ordinal_col': ['education_level'],
        'fill_strategy':'mean',
        'test_size': 0.3
    }

    my_service=ProjectService(general_args)

    my_service.start_service()



