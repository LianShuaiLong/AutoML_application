#-*-codin:utf8-*-
import xgboost as xgb
import pandas as pd
import nni
import argparse
import logging
import glob
import os
import yaml

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,roc_auc_score
from nni.trial import get_experiment_id

logging.basicConfig(level=logging.INFO,format='%asctime)s-%(name)s-%(levelname)s-%(message)s')
LOG = logging.getLogger('auto-xgb')


parser = argparse.ArgumentParser()
parser.add_argument('--train_datapath','-tr',type=str,default='..')
parser.add_argument('--test_datapath','-ev',type=str,default='..')
parser.add_argument('--label_name','-ln',type=str,default='label')
parser.add_argument('--save_dir','-sd',type=str,default='today',help='folder save best model')
args = parser.parse_args()
return args

def load_data(train_datapath,test_datapath,label_name):
    df_train = pd.read_csv(train_datapath)
    df_test = pd.read_csv(test_datapath)

    # train data
    x_train = df_train
    y_train = x_train.pop(label_name)

    x_test = df_test
    y_test = x_test.pop(label_name)

    return x_train,y_train,x_test,y_test

def get_default_parameters():
     params = {
          'learning_rate': 0.02,
          'n_estimators': 2000, 
          'max_depth': 4,
          'min_child_weight':2,
          'gamma':0.9,
          'subsample':0.8,
          'colsample_bytree':0.8,
          'objective':'binary:logistic',
          'nthread':-1,
          'scale_pos_weight':1,
          'eval_metric':'auc'
     }
     return params

def get_model(PARAMS):
     model = xgb.XGBClassifier()
     model.learning_rate = PARAMS.get("learning_rate")
     model.max_depth = PARAMS.get("max_depth")
     model.subsample = PARAMS.get("subsample")
     model.colsample_bytree = PARAMS.get("colsample_bytree")
     model.verbose = True
     model.eval_metric = 'auc'
     model.objective = 'binary:logistic'
     return model

def get_config(config_path):
    f = open(config_path)
    cfg = yaml.load(f.read())
    maxTrialNum = cfg['maxTrialNum']
    maxExecDuration = cfg['maxExecDuration']

    return (maxTrialNum,maxExecDuration)

def run(x_train, y_train,xtest,y_test,model):
 
     eval_set = [(x_test,y_test)]
     model.fit(x_train,y_train,early_stopping_rounds=10,eval_metric = 'auc',eval_set = eval_set,verbose = True)
     y_pred = model.predict_proba(x_test,ntree_limit = model.best_ntree_limit)[:,1]
     auc = roc_auc_score(y_test, y_pred)
     score = roc_auc_score(y_train,model.predict_proba(x_train,ntree_limit = model.best_ntree_limit)[:,1])
     trail_id = get_experiment_id
     LOG.info('trail id :{}\ttrain auc:{}\ttest auc{}' %(trail_id,score,auc))
     save_dir = args.save_dir
     model_best = glob.glob('{}/*.model'.format(save_folder))
     if len(model_best)!=0:
         model_best = model_best[0]
         model_best_auc = float(model_best.strip('.model').split('_')[-1])
         if auc > model_best_auc:
            model.save_model('{}/xgb_{}.model'.format(save_folder,auc))
            os.system('rm {}'.format(model_best))
     else:
         model.save_model('{}/xgb_{}.model'.format(save_folder,auc))
     nni.report_final_result(auc)


if __name__ == '__main__':
     
     train_datapath,test_datapath,label_name = args.train_datapath,args.test_datapath,args.label_name
     x_train,y_train,x_test,y_test = load_data(train_datapath,test_datapath,label_name)
     print('load data finish')
    
     try:
         RECEIVED_PARAMS = nni.get_next_parameter()
         LOG.debug(RECEIVED_PARAMS)
         PARAMS = get_default_parameters()
         PARAMS.update(RECEIVED_PARAMS)
         LOG.debug(PARAMS)
         model = get_model(PARAMS)
         #train
         run(x_train, y_train,x_test,y_test,model)
     except Exception as exception:
         LOG.exception(exception)
         raise

