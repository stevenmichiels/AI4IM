# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import libs
import numpy as np
from numpy import arange
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import pickle
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from dtreeviz.trees import dtreeviz
from scipy.stats import reciprocal, uniform
import os
import scipy.io
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from scipy.interpolate import interp1d
from sklearn.preprocessing import PowerTransformer

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVR, SVR,LinearSVC
import xgboost
import lightgbm as lgbm
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_decomposition import PLSRegression

homedir=os.getcwd()
resultsdir=os.path.join(homedir,'Results')

# user input
## Correlation
min_var_threshold=0.005
PCA_n_components=0.99
max_number_of_features=300

use_feature_correlation_selection=1
use_new_features=1
use_tsfresh_filtered=0
number_iter = 1000
number_cv = 5
search_grid=0
increased_weight=0

tune_hyperparams_lgb_class=0
tune_hyperparams_rf_class=0

class_optim_func="neg_log_loss"
## 

features_corr='No' if use_feature_correlation_selection==0 else 'Yes'
new_features='Yes' if use_new_features==1 else 'No'
searchname='_RandomSearch_' if search_grid==0 else '_GridSearch_'
class_optim='LL_WeightIncreased' if (class_optim_func=="neg_log_loss" and increased_weight==1) else 'LL_WeightIncreased'
file_suffix='NewFeatures_'+new_features+'_FeatureCorr_'+features_corr+searchname+class_optim

grid_fit_class_params={"cv":number_cv,"scoring" : class_optim_func,"verbose" : 2, "n_jobs": -1}
random_fit_class_params={"cv":number_cv,"scoring" : class_optim_func,"verbose" : 2, "n_jobs": -1, "n_iter":number_iter}

####################################################################################
# Utility function to report best scores
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
####################################################################################

## classification performance metrics
def calculate_spec_sens_acc_from_confusionmatrix(cm):
    # foute conventie tn,fp, fn, tp = cm.ravel()
    tp,fp, fn, tn = cm.ravel()
    acc=(tn+tp)/cm.sum()
    spec= tn / (tn+fp)
    sens= tp / (tp+fn)
    return acc,spec,sens
####################################################################################

# extraheren ground truth en predictie
def extract_gt_and_prediction(y_test, y_pred, labelname, df_Y):
    y_gt_and_pred_df=pd.DataFrame(np.stack((y_test[:,df_Y.columns.get_loc(labelname)], y_pred), axis=1), columns=['gt','pred'])
    y_gt_and_pred=y_gt_and_pred_df.sort_values(by='gt').values
    return y_gt_and_pred

# plot prediction errors
def plot_prediction_errors(y_gt_and_pred, modelname, xlab,ylab, legende, filesuffix, resultsdirectory):      
    plt.scatter(y_gt_and_pred[:,0], y_gt_and_pred[:,0]-y_gt_and_pred[:,1], alpha=0.5)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.axvline(x=y_gt_and_pred[:,0].mean(), linestyle='dashed')
    plt.legend([legende, 'prediction error'])
    plt.title(modelname+' model: Prediction errors on test set'+ '\n ' + filesuffix)
    plt.savefig(os.path.join(resultsdirectory,modelname+' ' + filesuffix + '.png'))
    plt.show()
    
### inladen data ###
if use_new_features==0:
    ####################################################################################
    #### load data
    cleaned_datafiles_dir=os.path.join(os.getcwd(),'cleaned_data')
    kul_datafiles_dir=os.path.join(os.getcwd(),'KUL_data')

    cleaned_datafiles=os.listdir(cleaned_datafiles_dir)[:-1]
    kul_datafiles=os.listdir(kul_datafiles_dir)

    ####################################################################################
    ##### create X from cleaned data
    features_mat=scipy.io.loadmat('feature_names.mat')
    feature_categories=list(features_mat.keys())[-4:]
    feature_names=[]
    feature_names_applied_to_curves=['vultijd','vultijdsensor']

    for index in range(len(features_mat[feature_categories[0]][0])):
        feature_names.append(feature_categories[0]+'_'+features_mat[feature_categories[0]][0][index][0])

    for num in feature_categories[1:]:
        for index in range(len(features_mat[num][0][0][0][0])):
            feature_names_applied_to_curves.append(num+'_'+features_mat[num][0][0][0][0][index][0])

    mat_X = [scipy.io.loadmat(os.path.join(cleaned_datafiles_dir,num)) for num in cleaned_datafiles]
    X_comb=[mat_X[num]['X'] for num in range(len(cleaned_datafiles))]
    X=np.vstack(X_comb)
    np.save('X.npy', X)

    ####################################################################################
    ##### create Y from KUL-data
    mat_run_Y = [scipy.io.loadmat(os.path.join(kul_datafiles_dir,num)) for num in kul_datafiles]
    mat_run_Y_array = [mat_run_Y[num]['run'] for num in range(len(kul_datafiles))]
    mat_run_Y_names=[mat_run_Y_array[num].dtype.names for num in range(len(kul_datafiles))]

    def mat_to_nparray(matfile, matfileindex, labelstring):
        extracted=matfile[matfileindex][labelstring].ravel()
        ### correctie o.w.v. afwezig gewicht in mat-file zijnde het laatste datapunt in twee files
        extracted=np.array([extracted[num].item() for num in range(extracted.shape[0])]) if matfileindex==0 else np.array([extracted[num].item() for num in range(extracted.shape[0]-1)])
        return extracted

    mat_Y_0=np.stack([mat_to_nparray(mat_run_Y_array, 0, 'gewicht'), mat_to_nparray(mat_run_Y_array, 0, 'opening')], axis=1)
    mat_Y_1=np.stack([mat_to_nparray(mat_run_Y_array, 1, 'gewicht'), mat_to_nparray(mat_run_Y_array, 1, 'opening')], axis=1)
    mat_Y_2=np.stack([mat_to_nparray(mat_run_Y_array, 2, 'gewicht'), mat_to_nparray(mat_run_Y_array, 2, 'opening')], axis=1)

    Y=np.concatenate((mat_Y_0,mat_Y_1, mat_Y_2), axis=0)
    np.save('Y.npy', Y)

    # with open('X.npy', 'rb') as f:
    #     X = np.load(f)

    # with open('Y.npy', 'rb') as f:
    #     Y = np.load(f)

    ####################################################################################
    # dataframes maken
    df_X=pd.DataFrame(X, columns=feature_names_applied_to_curves)
    df_Y=pd.DataFrame(Y, columns=['gewicht', 'opening'])

    def calculate_valid(df, labelname,sigmas):
        df['valid_'+labelname+'_'+str(sigmas)+'sigma'] = np.where(np.abs((df[labelname]-df.describe()[labelname]['mean'])/df.describe()[labelname]['std'])>sigmas,0,1)
        return df
       
    df_Y=calculate_valid(df_Y, 'gewicht',1)
    df_Y=calculate_valid(df_Y, 'gewicht',2)
    df_Y=calculate_valid(df_Y, 'opening',1)
    df_Y=calculate_valid(df_Y, 'opening',2)
          
    df_X_Y=pd.concat([df_X, df_Y], axis=1)

else:
    X_filtered=np.load("tsfresh_filtered_features_array.npy")
    columnnames_filtered=np.load("tsfresh_filtered_features_names.npy")
    columnnames_filtered=[i for i in columnnames_filtered]
    X=np.load("tsfresh_extracted_features_array.npy")
    columnnames=np.load("tsfresh_extracted_features_names.npy")
    columnnames=[i for i in columnnames]
    
    y_gewicht=np.load('y_gewicht.npy')
    y_opening=np.load('y_opening.npy')
    
    Y=np.concatenate((y_gewicht,y_opening), axis=1)
    
    df_Y=pd.DataFrame(Y, columns=['gewicht', 'opening'])
    
    df_X_filtered=pd.DataFrame(X_filtered, columns=columnnames_filtered)
    df_X=pd.DataFrame(X, columns=columnnames)
    
    def calculate_valid(df, labelname,sigmas):
        df['valid_'+labelname+'_'+str(sigmas)+'sigma'] = np.where(np.abs((df[labelname]-df.describe()[labelname]['mean'])/df.describe()[labelname]['std'])>sigmas,0,1)
        return df
       
    df_Y=calculate_valid(df_Y, 'gewicht',1)
    df_Y=calculate_valid(df_Y, 'gewicht',2)
    df_Y=calculate_valid(df_Y, 'opening',1)
    df_Y=calculate_valid(df_Y, 'opening',2)
        
    df_X_Y=pd.concat([df_X, df_Y], axis=1)

####################################################################################
####################################################################################
# Test set hold out
sss1=StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for trainval_index, test_index in sss1.split(df_X.values if use_tsfresh_filtered ==0 else df_X_filtered.values , df_Y['valid_opening_1sigma'].values):
    print("TRAINVAL:", trainval_index, "TEST:", test_index)
    X_trainval, X_test = df_X.values[trainval_index], df_X.values[test_index]
    y_train, y_test = df_Y.values[trainval_index], df_Y.values[test_index]
df_Y_test=df_Y.iloc[test_index]
df_Y_trainval=df_Y.iloc[trainval_index]
df_X_test=df_X.iloc[test_index]
df_X_trainval=df_X.iloc[trainval_index]

# trainCV en validation set
sss2=StratifiedShuffleSplit(n_splits=1, test_size=0.11, random_state=42)
for trainCV_index, val_index in sss2.split(X_trainval, df_Y_trainval['valid_opening_1sigma'].values):
    print("TRAINCV:", trainCV_index, "VAL:", val_index)
    X_trainCV, X_val = df_X.values[trainCV_index], df_X.values[val_index]
    y_trainCV, y_val = df_Y.values[trainCV_index], df_Y.values[val_index]
df_Y_val=df_Y.iloc[val_index]
df_Y_trainCV=df_Y.iloc[trainCV_index]
df_X_val=df_X.iloc[val_index]
df_X_trainCV=df_X.iloc[trainCV_index]

y_openingtrainCV=y_trainCV[:,df_Y.columns.get_loc("opening")]
y_opening_mu_trainCV=df_Y_trainCV.opening.mean()
y_opening_stdev_trainCV=df_Y_trainCV.opening.std()

y_opening_valid_2sigmatrainCV=y_trainCV[:,df_Y.columns.get_loc("valid_opening_2sigma")]
y_gewicht_valid_2sigmatrainCV=y_trainCV[:,df_Y.columns.get_loc("valid_gewicht_2sigma")]

y_gewichttrainCV=y_trainCV[:,df_Y.columns.get_loc("gewicht")]
y_gewicht_mu_trainCV=df_Y_trainCV.gewicht.mean()
y_gewicht_stdev_trainCV=df_Y_trainCV.gewicht.std()

y_opening_valid_2sigma_class_weight=compute_class_weight('balanced',classes=np.unique(y_opening_valid_2sigmatrainCV),y=y_opening_valid_2sigmatrainCV)
y_gewicht_valid_2sigma_class_weight=compute_class_weight('balanced',classes=np.unique(y_gewicht_valid_2sigmatrainCV),y=y_gewicht_valid_2sigmatrainCV)

if increased_weight==1:
    y_opening_valid_2sigma_class_weight[0]*=4
    y_gewicht_valid_2sigma_class_weight[0]*=4

####################################################################################
# EXPLORATION

## scalen en PCA
# alleen filteren op variantie - EXPLOR
df_X_trainCV_varfiltered=df_X_trainCV.loc[:, df_X_trainCV.std()/df_X_trainCV.mean() > min_var_threshold]

df_X_trainCV_varfiltered_Y=df_X_trainCV_varfiltered.join(df_Y)
df_X_trainCV_varfiltered_Y_corr=df_X_trainCV_varfiltered_Y.corr()
df_X_trainCV_varfiltered_Y_corr.valid_opening_2sigma.abs().sort_values().tail(max_number_of_features)
df_X_corr_features_selection=df_X_trainCV_varfiltered_Y_corr.valid_opening_2sigma.abs().sort_values().tail(max_number_of_features)

X_correlated_feature_names=[df_X_corr_features_selection.index[i] for i in np.arange(df_X_corr_features_selection.shape[0]) if df_X_corr_features_selection.index[i] not in df_Y.columns]


if use_feature_correlation_selection==1:
     X_trainCV, X_val = df_X[X_correlated_feature_names].values[trainCV_index], df_X[X_correlated_feature_names].values[val_index]

# Random Forest

# lgbm classification for opening

pipeLGB_opening_class = Pipeline([("sc",StandardScaler()), ("lgb",lgbm.LGBMClassifier())])
pipeLGB_gewicht_class = Pipeline([("sc",StandardScaler()), ("lgb",lgbm.LGBMClassifier())])
sorted(pipeLGB_opening_class.get_params().keys())
sorted(pipeLGB_gewicht_class.get_params().keys())

lgbn_estimators=[100,250,500,1000,2000]
lgbreg_alpha=[0, 1e-1, 1, 2, 5, 7, 10, 50, 100]
lgbreg_lambda=[0, 1e-1, 1, 5, 10, 20, 50, 100]
lgbsubsample=[0.1,0.5,0.8,1]
lgbmin_child_weight = [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
lgbmin_child_samples = [100,250,500]
lgbnum_leaves = [5,10,25,50]

lgb_opening_class_weight = [{0:y_opening_valid_2sigma_class_weight[0],1:y_opening_valid_2sigma_class_weight[1]}]
lgb_gewicht_class_weight = [{0:y_gewicht_valid_2sigma_class_weight[0],1:y_gewicht_valid_2sigma_class_weight[1]}]
lgb_opening_class_weight=lgb_opening_class_weight+lgb_opening_class_weight
lgb_gewicht_class_weight=lgb_gewicht_class_weight+lgb_gewicht_class_weight

paramsLGB_opening_distributions = {"lgb__class_weight":lgb_opening_class_weight,"lgb__n_estimators":lgbn_estimators, "lgb__reg_alpha":lgbreg_alpha,"lgb__reg_lambda":lgbreg_lambda,"lgb__subsample":lgbsubsample,"lgb__min_child_weight": lgbmin_child_weight, "lgb__min_child_samples":lgbmin_child_samples, "lgb__num_leaves":lgbnum_leaves }
paramsLGB_gewicht_distributions = {"lgb__class_weight":lgb_gewicht_class_weight,"lgb__n_estimators":lgbn_estimators, "lgb__reg_alpha":lgbreg_alpha,"lgb__reg_lambda":lgbreg_lambda,"lgb__subsample":lgbsubsample,"lgb__min_child_weight": lgbmin_child_weight, "lgb__min_child_samples":lgbmin_child_samples, "lgb__num_leaves":lgbnum_leaves }

                     



if search_grid==0:
    lgb_opening_class_search_cv = RandomizedSearchCV(pipeLGB_opening_class, paramsLGB_opening_distributions, **random_fit_class_params)                       
    lgb_gewicht_class_search_cv = RandomizedSearchCV(pipeLGB_gewicht_class, paramsLGB_gewicht_distributions, **random_fit_class_params)  
    searchname='Random_'     
else:
    lgb_opening_class_search_cv = GridSearchCV(pipeLGB_opening_class, paramsLGB_opening_distributions, **grid_fit_class_params)                       
    lgb_gewicht_class_search_cv = GridSearchCV(pipeLGB_gewicht_class, paramsLGB_gewicht_distributions, **grid_fit_class_params)
    searchname='Grid_'   

if tune_hyperparams_lgb_class==1:
    lgb_opening_class_search_cv.fit(X_trainCV, y_opening_valid_2sigmatrainCV)
    lgb_gewicht_class_search_cv.fit(X_trainCV, y_gewicht_valid_2sigmatrainCV)
    
    pipeLGB_opening_classBEST=lgb_opening_class_search_cv.best_estimator_
    pipeLGB_gewicht_classBEST=lgb_gewicht_class_search_cv.best_estimator_
    pickle.dump(pipeLGB_opening_classBEST, open(file_suffix+'lgb_opening_class_best_estimator.pkl','wb'))
    pickle.dump(pipeLGB_opening_classBEST, open(file_suffix+'lgb_gewicht_class_best_estimator.pkl','wb'))
    pickle.dump(lgb_opening_class_search_cv, open(file_suffix+'lgb_opening_class.pkl','wb'))
    pickle.dump(lgb_gewicht_class_search_cv, open(file_suffix+'lgb_gewicht_class.pkl','wb'))

else:
    pipeLGB_opening_classBEST=pd.read_pickle(file_suffix+'lgb_opening_class_best_estimator.pkl')
    pipeLGB_gewicht_classBEST=pd.read_pickle(file_suffix+'lgb_gewicht_class_best_estimator.pkl')
    lgb_opening_class_search_cv=pd.read_pickle(file_suffix+'lgb_opening_class.pkl')
    lgb_gewicht_class_search_cv=pd.read_pickle(file_suffix+'lgb_gewicht_class.pkl')

report(lgb_opening_class_search_cv.cv_results_,5)
report(lgb_gewicht_class_search_cv.cv_results_,5)

        
# retrain and evaluate on validation set
pipeLGB_opening_classBEST.fit(X_trainCV, y_opening_valid_2sigmatrainCV)
pipeLGB_gewicht_classBEST.fit(X_trainCV, y_gewicht_valid_2sigmatrainCV)

   
y_opening_class_predLGB=pipeLGB_opening_classBEST.predict(X_val)
y_gewicht_class_predLGB=pipeLGB_gewicht_classBEST.predict(X_val)
y_opening_class_proba_predLGB=pipeLGB_opening_classBEST.predict_proba(X_val)
y_gewicht_clas_probas_predLGB=pipeLGB_gewicht_classBEST.predict_proba(X_val)


# ROC-curve
from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(y_val[:,-1],  y_opening_class_proba_predLGB[:,-1])

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# evaluate prediction for opening
modelnaam='LGBClassOpening'
modelnaam_naief='NaiveClassOpening'
y_opening_gt_and_pred_lgb=extract_gt_and_prediction(y_val,y_opening_class_predLGB, "valid_opening_2sigma", df_Y)
y_opening_gt_and_pred_naive=np.stack((y_opening_gt_and_pred_lgb[:,0],y_opening_gt_and_pred_lgb[:,0]*0+1),axis=1)
opening_modelcm = confusion_matrix(y_opening_gt_and_pred_lgb[:,0],y_opening_gt_and_pred_lgb[:,1])
opening_naivecm=confusion_matrix(y_opening_gt_and_pred_naive[:,1],y_opening_gt_and_pred_naive[:,0])
opening_modelacc,opening_modelspec,opening_modelsens=calculate_spec_sens_acc_from_confusionmatrix(opening_modelcm)
opening_naiveacc,opening_naivespec,opening_naivesens=calculate_spec_sens_acc_from_confusionmatrix(opening_naivecm)
rounding_number=3
line1=(file_suffix+" - "+ modelnaam+ "- "+"accuracy:"+str(np.round(opening_modelacc,rounding_number))+", specificity: "+str(0 if np.isnan(opening_modelspec) else np.round(opening_modelspec,rounding_number))+", sensitivity: "+str(np.round(opening_modelsens,rounding_number)))
line2=(file_suffix+" - Naive model for opening - ""accuracy:"+str(np.round(opening_naiveacc,rounding_number))+", specificity: "+str(0 if np.isnan(opening_naivespec) else np.round(opening_naivespec,rounding_number))+", sensitivity: "+str(np.round(opening_naivesens,rounding_number)))
lines = [line1,line2]
print(line1 + '\n ' + line2 + '\n ')
with open(os.path.join(resultsdir,modelnaam +'_' + file_suffix+'.txt'), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')

# evaluate prediction for weight
modelnaam='LGBClassGewicht'
modelnaam_naief='NaiveClassGewicht'
y_gewicht_gt_and_pred_lgb=extract_gt_and_prediction(y_val,y_gewicht_class_predLGB, "valid_gewicht_2sigma", df_Y)
y_gewicht_gt_and_pred_naive=np.stack((y_gewicht_gt_and_pred_lgb[:,0],y_gewicht_gt_and_pred_lgb[:,0]*0+1),axis=1)
gewicht_modelcm = confusion_matrix(y_gewicht_gt_and_pred_lgb[:,0],y_gewicht_gt_and_pred_lgb[:,1])
gewicht_naivecm=confusion_matrix(y_gewicht_gt_and_pred_naive[:,1],y_gewicht_gt_and_pred_naive[:,0])
gewicht_modelacc,gewicht_modelspec,gewicht_modelsens=calculate_spec_sens_acc_from_confusionmatrix(gewicht_modelcm)
gewicht_naiveacc,gewicht_naivespec,gewicht_naivesens=calculate_spec_sens_acc_from_confusionmatrix(gewicht_naivecm)
line1=(file_suffix+" - "+ modelnaam+ "- "+"accuracy:"+str(np.round(gewicht_modelacc,rounding_number))+", specificity: "+str(0 if np.isnan(gewicht_modelspec) else np.round(gewicht_modelsens,rounding_number))+", sensitivity: "+str(np.round(opening_modelsens,rounding_number)))
line2=(file_suffix+" - Naive model for gewicht - ""accuracy:"+str(np.round(gewicht_naiveacc,rounding_number))+", specificity: "+str(0 if np.isnan(gewicht_naivespec) else np.round(gewicht_modelsens,rounding_number))+", sensitivity: "+str(np.round(gewicht_naivesens,rounding_number)))
lines = [line1,line2]
print(line1 + '\n ' + line2 + '\n ')
with open(os.path.join(resultsdir,modelnaam +'_' + file_suffix+'.txt'), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')


## feature importance
fi_opening=lgb_opening_class_search_cv.best_estimator_[1]
fi_opening.fit(StandardScaler().fit_transform(X_trainCV), y_opening_valid_2sigmatrainCV)
fi_opening.feature_importances_
X_correlated_feature_names

