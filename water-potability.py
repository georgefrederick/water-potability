import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score
import yaml

class waterPotability:
    
    def __init__(self, c, kernel):
        '''Pass the dataset to train on'''
        
        self.c = c
        self.kernel = kernel
        self.df = pd.read_csv('water_potability.csv')
        
    def print_results(self, results):
        '''Prints out the training and testing results for the model'''
        
        print('Best params : {}'.format(results.best_params_))
        means = results.cv_results_['mean_test_score']
        stds = results.cv_results_['std_test_score']

        for mean, std, params in zip(means, stds, results.cv_results_['params']):
            print('{} (+/-{}) for {}'.format(round(mean,3), round(std*2,3), params))
  
    def train(self):
        '''Impute values on missing values'''
        
        self.df['ph'].fillna(self.df['ph'].mean(), inplace = True)
        self.df['Sulfate'].fillna(self.df['Sulfate'].median(), inplace = True)
        self.df['Trihalomethanes'].fillna(self.df['Trihalomethanes'].mean(), inplace = True)

        X = self.df.drop('Potability',axis=1)
        y = self.df['Potability'].values
        
        X_train,X_test,y_train,y_test = train_test_split(X,y, random_state =42, test_size = 0.2)
        
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        svc = SVC(cache_size = 100)
        
        params = {'C':self.c,
                  'kernel':self.kernel}
        
        gsvm = GridSearchCV(svc,params, cv =5)
        
        self.print_results(gsvm.fit(X_train,y_train))
        
        joblib.dump(gsvm.best_estimator_,'./waterpotability_SVC.pkl')
        
        
        # Load model and predcit test set
        model = joblib.load('./waterpotability_SVC.pkl')
        
        predictions = model.predict(X_test)

        accu = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions, average='micro')
        precision = precision_score(y_test,predictions, average='micro')
        print('\n+==================================Test results==============================================+\n')
        print('Acuracy : {}, Recall: {}, Precision_score: {}'.format(accu,recall,precision)) 

def main():
    conf = yaml.load(open('./config.yaml'))

    wp = waterPotability(conf['C'],conf['kernel'])
    wp.train()
    
if __name__ == '__main__':
    main()