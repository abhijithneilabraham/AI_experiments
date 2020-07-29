

from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import logging

x_hotel = ['book a hotel', 'need a nice place to stay','need to spend the night','find a hotel']
x_weather = ['what is the weather like', 'is it hot outside','will it rain today', 'is it hot to go out']
class Classifier:
    def __init__(self,model_type,model_name,use_cuda=True):
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)
    
        train_args={
            'overwrite_output_dir': True,
            'num_train_epochs': 10,
        }
    
        # Create a ClassificationModel
        self.model = ClassificationModel(model_type, model_name, num_labels=2, use_cuda=use_cuda, cuda_device=0, args=train_args)

    def fit(self,x1,x2,split=0.3):
        train_data=[]
        eval_data=[]
        x1t, x1e, x2t, x2e = train_test_split( x1,x2, test_size=split, random_state=42)
        for x1,x2 in zip(x1t,x2t):
            train_data.append([x1,0])
            train_data.append([x2,1])
        for x1,x2 in zip(x1e,x2e):
            eval_data.append([x1,0])
            eval_data.append([x2,1])
        train_df = pd.DataFrame(train_data)
        eval_df = pd.DataFrame(eval_data)

        # Train the model
        self.model.train_model(train_df, eval_df=eval_df)

    # Evaluate the model
        result, model_outputs, wrong_predictions = self.model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)


    def predict(self,x):
        predictions, raw_outputs = self.model.predict(x)
        return predictions
    
clf=Classifier('roberta', 'roberta-base',use_cuda=False,)
clf.fit(x_hotel,x_weather)
print(clf.predict(["book me a hotel","is weather to be hot"]))

    

