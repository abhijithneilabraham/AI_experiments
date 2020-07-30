

from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn
import logging
from sklearn import preprocessing


x_hotel = ['book a hotel', 'need a nice place to stay','need to spend the night','find a hotel']
x_weather = ['what is the weather like', 'is it hot outside','will it rain today', 'is it hot to go out']
class Classifier:
    def __init__(self,model_type,model_name,use_cuda=True):
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)



        # Create a ClassificationModel
        self.model_type=model_type
        self.model_name=model_name
        self.use_cuda=use_cuda
        self.dat={}
        self.rerun=False

    def add(self,X,Y):
        self.dat[Y]=X
        
    def train(self,split=0.7,num_epochs=10):
        self.le=preprocessing.LabelEncoder()
        print(list(self.dat.keys()))
        self.le.fit(list(self.dat.keys()))
        
        train_data=[]
        eval_data=[]
        for k,v in self.dat.items():
            len_train=int(round(len(v)*split))
            train_data.extend([[i,self.le.transform([k])[0]] for i in v[:len_train]])

            eval_data.extend([[i,self.le.transform([k])[0]] for i in v[len_train:]])


        print(train_data,eval_data)
        train_df = pd.DataFrame(train_data)
        eval_df = pd.DataFrame(eval_data)
        train_args={
            'overwrite_output_dir': True,
            'num_train_epochs': num_epochs,
        }
        self.model = ClassificationModel(self.model_type, self.model_name, num_labels=len(list(self.dat.keys())), use_cuda=self.use_cuda, cuda_device=0, args=train_args)
        # Train the model
        self.model.train_model(train_df, eval_df=eval_df)

    # Evaluate the model
        result, model_outputs, wrong_predictions = self.model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)


    def predict(self,x):
        predictions, raw_outputs = self.model.predict(x)
        return self.le.inverse_transform(predictions)

clf=Classifier('roberta', 'roberta-base',use_cuda=False,)
clf.add(x_hotel,"hotel")
clf.add(x_weather,"weather")
clf.add(['good','bad','ugly','better'],"looks")
clf.train()
print(clf.predict(["book me a hotel","is the weather hot today"]))

