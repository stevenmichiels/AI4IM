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
modelsdir=os.path.join(homedir,'Models')
datadir=os.path.join(homedir,'Data')
# user input
## Correlation
min_var_threshold=0.005
PCA_n_components=0.99
max_number_of_features=300

use_new_features=1
use_feature_correlation_selection=1
use_tsfresh_filtered=0


number_iter = 1000
number_cv = 5
search_grid=0

tune_hyperparams_lgb_reg=0
tune_hyperparams_rfr_reg=0
reg_optim_func=["neg_mean_squared_error", "neg_mean_absolute_error"]
reg_optim_func=reg_optim_func[1]
class_optim_func="neg_log_loss"
## 

features_corr='No' if use_feature_correlation_selection==0 else 'Yes'
new_features='Yes' if use_new_features==1 else 'No'
searchname='RandomSearch_' if search_grid==0 else 'GridSearch_'
reg_optim='MAE_' if reg_optim_func=="neg_mean_absolute_error" else 'MSE_'
tsfreshfilter='Yes' if use_tsfresh_filtered==1 else 'No'
file_suffix='NewFeatures_'+new_features+'_FeatureCorr_'+features_corr+'_TSFreshFilter_'+tsfreshfilter+'_'+searchname+reg_optim

grid_fit_reg_params={"cv":number_cv,"scoring" : reg_optim_func,"verbose" : 2, "n_jobs": -1}
random_fit_reg_params={"cv":number_cv,"scoring" : reg_optim_func,"verbose" : 2, "n_jobs": -1, "n_iter":number_iter}
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
    features_mat=scipy.io.loadmat(os.path.join(cleaned_datafiles_dir,'feature_names.mat'))
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
    np.save(os.path.join(cleaned_datafiles_dir,'X.npy'), X)

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
    np.save(os.path.join(cleaned_datafiles_dir,'Y.npy'), Y)



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
    X_VALIDfiltered=np.load(os.path.join(datadir,"tsfresh_filtered_features_array.npy"))
    columnnames_VALID_filtered=np.load(os.path.join(datadir,"tsfresh_filtered_features_names.npy"))
    columnnames_VALID_filtered=[i for i in columnnames_VALID_filtered]
    X_OPENINGfiltered=np.load(os.path.join(datadir,"tsfresh_OPENINGfiltered_features_array.npy"))
    columnnames_OPENING_filtered=np.load(os.path.join(datadir,"tsfresh_OPENINGfiltered_features_names.npy"))
    columnnames_OPENING_filtered=[i for i in columnnames_OPENING_filtered]
    X_GEWICHTfiltered=np.load(os.path.join(datadir,"tsfresh_GEWICHTfiltered_features_array.npy"))
    columnnames_GEWICHT_filtered=np.load(os.path.join(datadir,"tsfresh_GEWICHTfiltered_features_names.npy"))
    columnnames_GEWICHT_filtered=[i for i in columnnames_GEWICHT_filtered]
    
    X=np.load(os.path.join(datadir,"tsfresh_extracted_features_array.npy"))
    columnnames=np.load(os.path.join(datadir,"tsfresh_extracted_features_names.npy"))
    columnnames=[i for i in columnnames]
    
    y_gewicht=np.load(os.path.join(datadir,'y_gewicht.npy'))
    y_opening=np.load(os.path.join(datadir,'y_opening.npy'))
    
    Y=np.concatenate((y_gewicht,y_opening), axis=1)
    
    df_Y=pd.DataFrame(Y, columns=['gewicht', 'opening'])
    
    df_X_OPENINGfiltered=pd.DataFrame(X_OPENINGfiltered, columns=columnnames_OPENING_filtered)
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
for trainval_index, test_index in sss1.split(df_X.values if use_tsfresh_filtered ==0 else df_X_OPENINGfiltered.values , df_Y['valid_opening_1sigma'].values):
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


####################################################################################
# EXPLORATION

## scalen en PCA
# alleen filteren op variantie - EXPLOR


df_X_trainCV_varfiltered=df_X_trainCV.loc[:, df_X_trainCV.std()/df_X_trainCV.mean() > min_var_threshold]
df_X_trainCV_varfiltered_Y=df_X_trainCV_varfiltered.join(df_Y)

# eigen
df_X_trainCV_varfiltered_Y_corr=df_X_trainCV_varfiltered_Y.corr()
df_X_trainCV_varfiltered_Y_corr.opening.abs().sort_values().tail(max_number_of_features)
df_X_corr_features_selection=df_X_trainCV_varfiltered_Y_corr.opening.abs().sort_values().tail(max_number_of_features)

if use_tsfresh_filtered==0:
    X_correlated_feature_names=[df_X_corr_features_selection.index[i] for i in np.arange(df_X_corr_features_selection.shape[0]) if df_X_corr_features_selection.index[i] not in df_Y.columns]
else: 
    X_correlated_feature_names=columnnames_OPENING_filtered


# nu met StandardScaling en PCA
# df_X_trainCV_varfiltered_scaled=StandardScaler().fit_transform(df_X_trainCV_varfiltered.values)
# pca=PCA(n_components=PCA_n_components)
# df_X_trainCV_varfiltered_scaled_pca=pca.fit_transform(df_X_trainCV_varfiltered_scaled)
# pca.explained_variance_ratio_.cumsum()[-1]

# df_X_PCA=pd.DataFrame(df_X_trainCV_varfiltered_scaled)
# df_X_PCA_Y=df_X_PCA.join(df_Y)
# df_X_PCA_Ycorr=df_X_PCA_Y.corr()
# df_X_PCA_Ycorr.opening.abs().sort_values().tail(40)

# Partial Least Squares Regression

###################################################

# Random Forest


if use_feature_correlation_selection==1:
     X_trainCV, X_val = df_X[X_correlated_feature_names].values[trainCV_index], df_X[X_correlated_feature_names].values[val_index]
       
         
# pca__n_components=[i for i in np.arange(90,100,10)]

# rf__n_estimators=[500,2000]
# rf__max_depth=[5,10,25,50]
# rf__max_leaf_nodes=[5,10,25,50]
# rf__min_samples_split=[5,10,25,50]
# rf__min_samples_leaf=[5,10,25,50]
# rf__max_features=[i for i in np.arange(20,70,10)]


# newpipeRFR=Pipeline(steps=[("sc",StandardScaler()), ('rf', RandomForestRegressor())])
# sorted(newpipeRFR.get_params().keys())
# paramsRFR_distributions = {"rf__n_estimators":rf__n_estimators, "rf__max_depth":rf__max_depth, 'rf__max_leaf_nodes':rf__max_leaf_nodes, 'rf__min_samples_split':rf__min_samples_split, 'rf__min_samples_leaf': rf__min_samples_leaf, 'rf__max_features': rf__max_features}

 
# if search_grid==0:
#     new_rfr_opening_reg_search_cv = RandomizedSearchCV(newpipeRFR, paramsRFR_distributions, **random_fit_reg_params)                       
#     new_rfr_gewicht_reg_search_cv = RandomizedSearchCV(newpipeRFR, paramsRFR_distributions, **random_fit_reg_params)
# else:
#     new_rfr_opening_reg_search_cv = GridSearchCV(newpipeRFR, paramsRFR_distributions, **grid_fit_reg_params)                       
#     new_rfr_gewicht_reg_search_cv = GridSearchCV(newpipeRFR, paramsRFR_distributions, **grid_fit_reg_params)

# if tune_hyperparams_rfr_reg==1:
#     new_rfr_opening_reg_search_cv.fit(X_trainCV, y_openingtrainCV)
#     new_rfr_gewicht_reg_search_cv.fit(X_trainCV, y_gewichttrainCV)
#     new_pipeRFR_opening_regBEST=new_rfr_opening_reg_search_cv.best_estimator_
#     new_pipeRFR_gewicht_regBEST=new_rfr_gewicht_reg_search_cv.best_estimator_

#     pickle.dump(new_pipeRFR_opening_regBEST, open(file_suffix+'rfr_opening_reg_best_estimator.pkl','wb'))
#     pickle.dump(new_pipeRFR_gewicht_regBEST, open(file_suffix+'rfr_gewicht_reg_best_estimator.pkl','wb'))
#     report(new_rfr_opening_reg_search_cv.cv_results_,5)
#     report(new_rfr_gewicht_reg_search_cv.cv_results_,5)
# else:
#     new_pipeRFR_opening_regBEST=pd.read_pickle(file_suffix+'rfr_opening_reg_best_estimator.pkl')
#     new_pipeRFR_gewicht_regBEST=pd.read_pickle(file_suffix+'rfr_gewicht_reg_best_estimator.pkl')
        
# # retrain and evaluate on validation set
# new_pipeRFR_opening_regBEST.fit(X_trainCV, y_openingtrainCV)
# new_pipeRFR_gewicht_regBEST.fit(X_trainCV, y_gewichttrainCV)

          
# y_opening_reg_predRFR=new_pipeRFR_opening_regBEST.predict(X_val)
# y_gewicht_reg_predRFR=new_pipeRFR_gewicht_regBEST.predict(X_val)

       
# # evaluate prediction for opening
# modelnaam='RFRegOpening'
# modelnaam_naief='NaiveRegOpening'
# y_opening_gt_and_pred_rfr=extract_gt_and_prediction(y_val,y_opening_reg_predRFR, "opening", df_Y)
# y_opening_gt_and_pred_naive=np.stack((y_opening_gt_and_pred_rfr[:,0],y_opening_gt_and_pred_rfr[:,0]*0+y_opening_mu_trainCV),axis=1)
# line1=(file_suffix+" - MAE "+ modelnaam +str(np.round(mean_absolute_error(y_opening_gt_and_pred_rfr[:,0],y_opening_gt_and_pred_rfr[:,1]),4)))
# line2=(file_suffix+" - MAE opening Naive: " +str(np.round(mean_absolute_error(y_opening_gt_and_pred_naive[:,0],y_opening_gt_and_pred_naive[:,1]),4)))
# plot_prediction_errors(y_opening_gt_and_pred_rfr, modelnaam, "Ground truth opening [mm]", "Prediction error [mm]", "Average opening",file_suffix, resultsdir)
# plot_prediction_errors(y_opening_gt_and_pred_naive, modelnaam_naief, "Ground truth opening [mm]", "Prediction error [mm]", "Average opening",file_suffix, resultsdir)
# opening_valtbinnendegrenzen_voorspelling=(y_opening_reg_predRFR<y_opening_mu_trainCV+2*y_opening_stdev_trainCV)*(y_opening_reg_predRFR>y_opening_mu_trainCV-2*y_opening_stdev_trainCV).astype(int)
# opening_valtbinnendegrenzen_groundtruth=df_Y_val['valid_opening_2sigma'].values
# opening_modelcm = confusion_matrix(opening_valtbinnendegrenzen_groundtruth,opening_valtbinnendegrenzen_voorspelling)
# opening_naivecm=confusion_matrix(np.ones(opening_valtbinnendegrenzen_groundtruth.shape), opening_valtbinnendegrenzen_groundtruth)
# opening_modelacc,opening_modelspec,opening_modelsens=calculate_spec_sens_acc_from_confusionmatrix(opening_modelcm)
# opening_naiveacc,opening_naivespec,opening_naivesens=calculate_spec_sens_acc_from_confusionmatrix(opening_naivecm)
# rounding_number=3
# line3=(file_suffix+" - "+ modelnaam+ "- "+"accuracy:"+str(np.round(opening_modelacc,rounding_number))+", specificity: "+str(0 if np.isnan(opening_modelspec) else np.round(opening_modelspec,rounding_number))+", sensitivity: "+str(np.round(opening_modelsens,rounding_number)))
# line4=(file_suffix+" - Naive model for opening - "+"accuracy:"+str(np.round(opening_naiveacc,rounding_number))+", specificity: "+str(0 if np.isnan(opening_naivespec) else np.round(opening_naivespec,rounding_number))+", sensitivity: "+str(np.round(opening_naivesens,rounding_number)))
# lines = [line1,line2,line3,line4]
# print(line1 + '\n ' + line2 + '\n ' + line3 + '\n ' + line4)
# with open(os.path.join(resultsdir,modelnaam +'_' + file_suffix+'.txt'), 'w') as f:
#     for line in lines:
#         f.write(line)
#         f.write('\n')

# # evaluate prediction for weight
# modelnaam='RFRegGewicht'
# modelnaam_naief='NaiveRegGewicht'
# y_gewicht_gt_and_pred_rfr=extract_gt_and_prediction(y_val,y_gewicht_reg_predRFR, "gewicht", df_Y)
# y_gewicht_gt_and_pred_naive=np.stack((y_gewicht_gt_and_pred_rfr[:,0],y_gewicht_gt_and_pred_rfr[:,0]*0+y_gewicht_mu_trainCV),axis=1)
# line1=(file_suffix+" - MAE "+ modelnaam +str(np.round(mean_absolute_error(y_gewicht_gt_and_pred_rfr[:,0],y_gewicht_gt_and_pred_rfr[:,1]),4)))
# line2=(file_suffix+" - MAE gewicht Naive: " +str(np.round(mean_absolute_error(y_gewicht_gt_and_pred_naive[:,0],y_gewicht_gt_and_pred_naive[:,1]),4)))
# plot_prediction_errors(y_gewicht_gt_and_pred_rfr, modelnaam, "Ground truth weight [g]", "Prediction error [g]", "Average weight",file_suffix, resultsdir)
# plot_prediction_errors(y_gewicht_gt_and_pred_naive, modelnaam_naief, "Ground truth weight [g]", "Prediction error [g]", "Average weight",file_suffix, resultsdir)
# gewicht_valtbinnendegrenzen_voorspelling=(y_gewicht_reg_predRFR<y_gewicht_mu_trainCV+2*y_gewicht_stdev_trainCV)*(y_gewicht_reg_predRFR>y_gewicht_mu_trainCV-2*y_gewicht_stdev_trainCV).astype(int)
# gewicht_valtbinnendegrenzen_groundtruth=df_Y_val['valid_gewicht_2sigma'].values
# gewicht_modelcm = confusion_matrix(gewicht_valtbinnendegrenzen_voorspelling, gewicht_valtbinnendegrenzen_groundtruth)
# gewicht_naivecm=confusion_matrix(np.ones(gewicht_valtbinnendegrenzen_groundtruth.shape), gewicht_valtbinnendegrenzen_groundtruth)
# gewicht_modelacc,gewicht_modelspec,gewicht_modelsens=calculate_spec_sens_acc_from_confusionmatrix(gewicht_modelcm)
# gewicht_naiveacc,gewicht_naivespec,gewicht_naivesens=calculate_spec_sens_acc_from_confusionmatrix(gewicht_naivecm)
# line3=(file_suffix+" - "+ modelnaam+ "- "+"accuracy:"+str(np.round(gewicht_modelacc,rounding_number))+", specificity: "+str(0 if np.isnan(gewicht_modelspec) else np.round(gewicht_modelspec,rounding_number))+", sensitivity: "+str(np.round(gewicht_modelsens,rounding_number)))
# line4=(file_suffix+" - Naive model for weight - "+"accuracy:"+str(np.round(gewicht_naiveacc,rounding_number))+", specificity: "+str(0 if np.isnan(gewicht_naivespec) else np.round(gewicht_naivespec,rounding_number))+", sensitivity: "+str(np.round(gewicht_naivesens,rounding_number)))
# lines = [line1,line2,line3,line4]
# print(line1 + '\n ' + line2 + '\n ' + line3 + '\n ' + line4)
# with open(os.path.join(resultsdir,modelnaam +'_' + file_suffix+'.txt'), 'w') as f:
#     for line in lines:
#         f.write(line)
#         f.write('\n')
# #################################################
# lgbm regression for opening en gewicht

new_pipeLGB_opening = Pipeline([("sc",StandardScaler()), ("lgb",lgbm.LGBMRegressor())])
new_pipeLGB_gewicht = Pipeline([("sc",StandardScaler()), ("lgb",lgbm.LGBMRegressor())])
sorted(new_pipeLGB_opening.get_params().keys())
sorted(new_pipeLGB_gewicht.get_params().keys())



lgbn_estimators=[100,250,500,1000,2000]
lgbreg_alpha=[0, 1e-1, 1, 2, 5, 7, 10, 50, 100]
lgbreg_lambda=[0, 1e-1, 1, 5, 10, 20, 50, 100]
lgbsubsample=[0.1,0.5,0.8,1]
lgbmin_child_weight = [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
lgbmin_child_samples = [100,250,500]
lgbnum_leaves = [5,10,25,50,100,150,200]
lgbnum_leaves = [100,150,200]

paramsLGB_distributions = {"lgb__n_estimators":lgbn_estimators, "lgb__reg_alpha":lgbreg_alpha,"lgb__reg_lambda":lgbreg_lambda,"lgb__subsample":lgbsubsample,"lgb__min_child_weight": lgbmin_child_weight, "lgb__min_child_samples":lgbmin_child_samples, "lgb__num_leaves":lgbnum_leaves }

if search_grid==0:
    new_lgb_opening_reg_search_cv = RandomizedSearchCV(new_pipeLGB_opening, paramsLGB_distributions, **random_fit_reg_params)                       
    new_lgb_gewicht_reg_search_cv = RandomizedSearchCV(new_pipeLGB_gewicht, paramsLGB_distributions, **random_fit_reg_params)
    searchname='Random_'     
else:
    new_lgb_opening_reg_search_cv = GridSearchCV(new_pipeLGB_opening, paramsLGB_distributions, **grid_fit_reg_params)                       
    new_lgb_gewicht_reg_search_cv = GridSearchCV(new_pipeLGB_opening, paramsLGB_distributions, **grid_fit_reg_params)
    searchname='Grid_'   

if tune_hyperparams_lgb_reg==1:
    new_lgb_opening_reg_search_cv.fit(X_trainCV, y_openingtrainCV)
    new_lgb_gewicht_reg_search_cv.fit(X_trainCV, y_gewichttrainCV)
    new_pipeLGB_opening_regBEST=new_lgb_opening_reg_search_cv.best_estimator_
    new_pipeLGB_gewicht_regBEST=new_lgb_gewicht_reg_search_cv.best_estimator_
    pickle.dump(new_pipeLGB_opening_regBEST, open(os.path.join(modelsdir,file_suffix+'lgb_opening_reg_best_estimator.pkl'),'wb'))
    pickle.dump(new_pipeLGB_gewicht_regBEST, open(os.path.join(modelsdir,file_suffix+'lgb_gewicht_reg_best_estimator.pkl'),'wb'))
    report(new_lgb_opening_reg_search_cv.cv_results_,5)
    report(new_lgb_gewicht_reg_search_cv.cv_results_,5)
else:
    new_pipeLGB_opening_regBEST=pd.read_pickle(os.path.join(modelsdir,file_suffix+'lgb_opening_reg_best_estimator.pkl'))
    new_pipeLGB_gewicht_regBEST=pd.read_pickle(os.path.join(modelsdir,file_suffix+'lgb_gewicht_reg_best_estimator.pkl'))
        
# retrain and evaluate on validation set
new_pipeLGB_opening_regBEST.fit(X_trainCV, y_openingtrainCV)
new_pipeLGB_gewicht_regBEST.fit(X_trainCV, y_gewichttrainCV)

          
y_opening_reg_predLGB=new_pipeLGB_opening_regBEST.predict(X_val)
y_gewicht_reg_predLGB=new_pipeLGB_gewicht_regBEST.predict(X_val)

# evaluate prediction for opening
modelnaam='LGBRegOpening'
modelnaam_naief='NaiveRegOpening'
y_opening_gt_and_pred_lgb=extract_gt_and_prediction(y_val,y_opening_reg_predLGB, "opening", df_Y)
y_opening_gt_and_pred_naive=np.stack((y_opening_gt_and_pred_lgb[:,0],y_opening_gt_and_pred_lgb[:,0]*0+y_opening_mu_trainCV),axis=1)
line1=(file_suffix+" - MAE "+ modelnaam+": "+str(np.round(mean_absolute_error(y_opening_gt_and_pred_lgb[:,0],y_opening_gt_and_pred_lgb[:,1]),4)))
line2=(file_suffix+" - MAE opening Naive: "+str(np.round(mean_absolute_error(y_opening_gt_and_pred_naive[:,0],y_opening_gt_and_pred_naive[:,1]),4)))
plot_prediction_errors(y_opening_gt_and_pred_lgb, modelnaam, "Ground truth opening [mm]", "Prediction error [mm]", "Average opening",file_suffix, resultsdir)
plot_prediction_errors(y_opening_gt_and_pred_naive, modelnaam_naief, "Ground truth opening [mm]", "Prediction error [mm]", "Average opening",file_suffix, resultsdir)
opening_valtbinnendegrenzen_voorspelling=(y_opening_reg_predLGB<y_opening_mu_trainCV+2*y_opening_stdev_trainCV)*(y_opening_reg_predLGB>y_opening_mu_trainCV-2*y_opening_stdev_trainCV).astype(int)
opening_valtbinnendegrenzen_groundtruth=df_Y_val['valid_opening_2sigma'].values
opening_modelcm = confusion_matrix(opening_valtbinnendegrenzen_groundtruth,opening_valtbinnendegrenzen_voorspelling)
opening_naivecm=confusion_matrix(np.ones(opening_valtbinnendegrenzen_groundtruth.shape), opening_valtbinnendegrenzen_groundtruth)
opening_modelacc,opening_modelspec,opening_modelsens=calculate_spec_sens_acc_from_confusionmatrix(opening_modelcm)
opening_naiveacc,opening_naivespec,opening_naivesens=calculate_spec_sens_acc_from_confusionmatrix(opening_naivecm)
rounding_number=3
line3=(file_suffix+" - "+ modelnaam+ "- "+"accuracy:"+str(np.round(opening_modelacc,rounding_number))+", specificity: "+str(0 if np.isnan(opening_modelspec) else np.round(opening_modelspec,rounding_number))+", sensitivity: "+str(np.round(opening_modelsens,rounding_number)))
line4=(file_suffix+" - Naive model for opening - ""accuracy:"+str(np.round(opening_naiveacc,rounding_number))+", specificity: "+str(0 if np.isnan(opening_naivespec) else np.round(opening_naivespec,rounding_number))+", sensitivity: "+str(np.round(opening_naivesens,rounding_number)))
lines = [line1,line2,line3,line4]
print(line1 + '\n ' + line2 + '\n ' + line3 + '\n ' + line4)
with open(os.path.join(resultsdir,modelnaam +'_' + file_suffix+'.txt'), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')

# evaluate prediction for weight
modelnaam='LGBRegGewicht'
modelnaam_naief='NaiveRegGewicht'
y_gewicht_gt_and_pred_lgb=extract_gt_and_prediction(y_val,y_gewicht_reg_predLGB, "gewicht", df_Y)
y_gewicht_gt_and_pred_naive=np.stack((y_gewicht_gt_and_pred_lgb[:,0],y_gewicht_gt_and_pred_lgb[:,0]*0+y_gewicht_mu_trainCV),axis=1)
line1=(file_suffix+" - MAE "+ modelnaam +": "+str(np.round(mean_absolute_error(y_gewicht_gt_and_pred_lgb[:,0],y_gewicht_gt_and_pred_lgb[:,1]),4)))
line2=(file_suffix+" - MAE gewicht Naive: "+str(np.round(mean_absolute_error(y_gewicht_gt_and_pred_naive[:,0],y_gewicht_gt_and_pred_naive[:,1]),4)))
plot_prediction_errors(y_gewicht_gt_and_pred_lgb, modelnaam, "Ground truth weight [g]", "Prediction error [g]", "Average weight",file_suffix, resultsdir)
plot_prediction_errors(y_gewicht_gt_and_pred_naive, modelnaam_naief, "Ground truth weight [g]", "Prediction error [g]", "Average weight",file_suffix, resultsdir)
gewicht_valtbinnendegrenzen_voorspelling=(y_gewicht_reg_predLGB<y_gewicht_mu_trainCV+2*y_gewicht_stdev_trainCV)*(y_gewicht_reg_predLGB>y_gewicht_mu_trainCV-2*y_gewicht_stdev_trainCV).astype(int)
gewicht_valtbinnendegrenzen_groundtruth=df_Y_val['valid_gewicht_2sigma'].values
gewicht_modelcm = confusion_matrix(gewicht_valtbinnendegrenzen_voorspelling, gewicht_valtbinnendegrenzen_groundtruth)
gewicht_naivecm=confusion_matrix(np.ones(gewicht_valtbinnendegrenzen_groundtruth.shape), gewicht_valtbinnendegrenzen_groundtruth)
gewicht_modelacc,gewicht_modelspec,gewicht_modelsens=calculate_spec_sens_acc_from_confusionmatrix(gewicht_modelcm)
gewicht_naiveacc,gewicht_naivespec,gewicht_naivesens=calculate_spec_sens_acc_from_confusionmatrix(gewicht_naivecm)
line3=(file_suffix+" - "+ modelnaam+ "- "+"accuracy:"+str(np.round(gewicht_modelacc,rounding_number))+", specificity: "+str(0 if np.isnan(gewicht_modelspec) else np.round(gewicht_modelspec,rounding_number))+", sensitivity: "+str(np.round(gewicht_modelsens,rounding_number)))
line4=(file_suffix+" - Naive model for weight - accuracy:"+str(np.round(gewicht_naiveacc,rounding_number))+", specificity: "+str(0 if np.isnan(gewicht_naivespec) else np.round(gewicht_naivespec,rounding_number))+", sensitivity: "+str(np.round(gewicht_naivesens,rounding_number)))
lines = [line1,line2,line3,line4]
print(line1 + '\n ' + line2 + '\n ' + line3 + '\n ' + line4)
with open(os.path.join(resultsdir,modelnaam +'_' + file_suffix+'.txt'), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')

# ####################################################################################
# ### pipelines voor regression en classification
# scaler = StandardScaler()
# lgb_opening_class=lgbm.LGBMClassifier()
# lgb_gewicht_class=lgbm.LGBMClassifier()

# # #################################################
# # lgbm regression for opening en gewicht
# lgb_opening_reg=lgbm.LGBMRegressor()
# lgb_gewicht_reg=lgbm.LGBMRegressor()

# pipeLGB_opening = Pipeline([( "scaler" , StandardScaler()),("lgb",lgb_opening_reg)])
# pipeLGB_gewicht = Pipeline([( "scaler" , StandardScaler()),("lgb",lgb_gewicht_reg)])
# sorted(pipeLGB_opening.get_params().keys())
# sorted(pipeLGB_gewicht.get_params().keys())
# lgbn_estimators=[100,250,500,1000,2000]
# lgbreg_alpha=[0, 1e-1, 1, 2, 5, 7, 10, 50, 100]
# lgbreg_lambda=[0, 1e-1, 1, 5, 10, 20, 50, 100]
# lgbsubsample=[0.1,0.5,0.8,1]
# lgbmin_child_weight = [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
# lgbmin_child_samples = [100,250,500]
# lgbnum_leaves = [5,10,25,50]

# paramsLGB_distributions = {"lgb__n_estimators":lgbn_estimators, "lgb__reg_alpha":lgbreg_alpha,"lgb__reg_lambda":lgbreg_lambda,"lgb__subsample":lgbsubsample,"lgb__min_child_weight": lgbmin_child_weight, "lgb__min_child_samples":lgbmin_child_samples, "lgb__num_leaves":lgbnum_leaves }


# if search_grid==0:
#     lgb_opening_reg_search_cv = RandomizedSearchCV(pipeLGB_opening, paramsLGB_distributions, **random_fit_reg_params)                       
#     lgb_gewicht_reg_search_cv = RandomizedSearchCV(pipeLGB_gewicht, paramsLGB_distributions, **random_fit_reg_params)
#     searchname='Random_'     
# else:
#     lgb_opening_reg_search_cv = GridSearchCV(pipeLGB_opening, paramsLGB_distributions, **grid_fit_reg_params)                       
#     lgb_gewicht_reg_search_cv = GridSearchCV(pipeLGB_opening, paramsLGB_distributions, **grid_fit_reg_params)
#     searchname='Grid_'   

# if tune_hyperparams_lgb_reg==1:
#     lgb_opening_reg_search_cv.fit(X_trainCV, y_openingtrainCV)
#     lgb_gewicht_reg_search_cv.fit(X_trainCV, y_gewichttrainCV)
#     pipeLGB_opening_regBEST=lgb_opening_reg_search_cv.best_estimator_
#     pipeLGB_gewicht_regBEST=lgb_gewicht_reg_search_cv.best_estimator_
#     pickle.dump(pipeLGB_opening_regBEST, open(searchname+'lgb_opening_reg_best_estimator.pkl','wb'))
#     pickle.dump(pipeLGB_gewicht_regBEST, open(searchname+'lgb_gewicht_reg_best_estimator.pkl','wb'))
#     report(lgb_opening_reg_search_cv.cv_results_,5)
#     report(lgb_gewicht_reg_search_cv.cv_results_,5)
# else:
#     pipeLGB_opening_regBEST=pd.read_pickle(searchname+'lgb_opening_reg_best_estimator.pkl')
#     pipeLGB_gewicht_regBEST=pd.read_pickle(searchname+'lgb_gewicht_reg_best_estimator.pkl')
        
# # retrain and evaluate on validation set
# pipeLGB_opening_regBEST.fit(X_trainCV, y_openingtrainCV)
# pipeLGB_gewicht_regBEST.fit(X_trainCV, y_gewichttrainCV)

          
# y_opening_reg_predLGB=pipeLGB_opening_regBEST.predict(X_val)
# y_gewicht_reg_predLGB=pipeLGB_gewicht_regBEST.predict(X_val)

# # evaluate prediction for opening
# y_opening_gt_and_pred_lgb=extract_gt_and_prediction(y_val,y_opening_reg_predLGB, "opening", df_Y)
# y_opening_gt_and_pred_naive=np.stack((y_opening_gt_and_pred_lgb[:,0],y_opening_gt_and_pred_lgb[:,0]*0+y_opening_mu_trainCV),axis=1)
# print("MAE opening LGB: " +str(np.round(mean_absolute_error(y_opening_gt_and_pred_lgb[:,0],y_opening_gt_and_pred_lgb[:,1]),4)))
# print("MAE opening Naive: " +str(np.round(mean_absolute_error(y_opening_gt_and_pred_naive[:,0],y_opening_gt_and_pred_naive[:,1]),4)))
# plot_prediction_errors(y_opening_gt_and_pred_lgb, 'LGBM prediction for opening', "Ground truth opening [mm]", "Prediction error [mm]", "Average opening")
# plot_prediction_errors(y_opening_gt_and_pred_naive, 'Naive prediction for opening', "Ground truth opening [mm]", "Prediction error [mm]", "Average opening")
# opening_valtbinnendegrenzen_voorspelling=(y_opening_reg_predLGB<y_opening_mu_trainCV+2*y_opening_stdev_trainCV)*(y_opening_reg_predLGB>y_opening_mu_trainCV-2*y_opening_stdev_trainCV).astype(int)
# opening_valtbinnendegrenzen_groundtruth=df_Y_val['valid_opening_2sigma'].values
# opening_modelcm = confusion_matrix(opening_valtbinnendegrenzen_groundtruth,opening_valtbinnendegrenzen_voorspelling)
# opening_naivecm=confusion_matrix(np.ones(opening_valtbinnendegrenzen_groundtruth.shape), opening_valtbinnendegrenzen_groundtruth)
# opening_modelacc,opening_modelspec,opening_modelsens=calculate_spec_sens_acc_from_confusionmatrix(opening_modelcm)
# opening_naiveacc,opening_naivespec,opening_naivesens=calculate_spec_sens_acc_from_confusionmatrix(opening_naivecm)
# rounding_number=3
# print("Naive model for opening - "+"accuracy:"+str(np.round(opening_naiveacc,rounding_number))+", specificity: "+str(0 if np.isnan(opening_naivespec) else np.round(opening_naivespec,rounding_number))+", sensitivity: "+str(np.round(opening_naivesens,rounding_number)))
# print("LGBM model for opening - "+"accuracy:"+str(np.round(opening_modelacc,rounding_number))+", specificity: "+str(0 if np.isnan(opening_modelspec) else np.round(opening_modelspec,rounding_number))+", sensitivity: "+str(np.round(opening_modelsens,rounding_number)))

# # evaluate prediction for weight
# y_gewicht_gt_and_pred_lgb=extract_gt_and_prediction(y_val,y_gewicht_reg_predLGB, "gewicht", df_Y)
# y_gewicht_gt_and_pred_naive=np.stack((y_gewicht_gt_and_pred_lgb[:,0],y_gewicht_gt_and_pred_lgb[:,0]*0+y_gewicht_mu_trainCV),axis=1)
# print("MAE gewicht LGB: " +str(np.round(mean_absolute_error(y_gewicht_gt_and_pred_lgb[:,0],y_gewicht_gt_and_pred_lgb[:,1]),4)))
# print("MAE gewicht Naive: " +str(np.round(mean_absolute_error(y_gewicht_gt_and_pred_naive[:,0],y_gewicht_gt_and_pred_naive[:,1]),4)))
# plot_prediction_errors(y_gewicht_gt_and_pred_lgb, 'LGBM prediction for weight', "Ground truth weight [g]", "Prediction error [g]", "Average weight")
# plot_prediction_errors(y_gewicht_gt_and_pred_naive, 'Naive prediction for weight', "Ground truth weight [g]", "Prediction error [g]", "Average weight")
# gewicht_valtbinnendegrenzen_voorspelling=(y_gewicht_reg_predLGB<y_gewicht_mu_trainCV+2*y_gewicht_stdev_trainCV)*(y_gewicht_reg_predLGB>y_gewicht_mu_trainCV-2*y_gewicht_stdev_trainCV).astype(int)
# gewicht_valtbinnendegrenzen_groundtruth=df_Y_val['valid_gewicht_2sigma'].values
# gewicht_modelcm = confusion_matrix(gewicht_valtbinnendegrenzen_voorspelling, gewicht_valtbinnendegrenzen_groundtruth)
# gewicht_naivecm=confusion_matrix(np.ones(gewicht_valtbinnendegrenzen_groundtruth.shape), gewicht_valtbinnendegrenzen_groundtruth)
# gewicht_modelacc,gewicht_modelspec,gewicht_modelsens=calculate_spec_sens_acc_from_confusionmatrix(gewicht_modelcm)
# gewicht_naiveacc,gewicht_naivespec,gewicht_naivesens=calculate_spec_sens_acc_from_confusionmatrix(gewicht_naivecm)
# print("Naive model for weight - "+"accuracy:"+str(np.round(gewicht_naiveacc,rounding_number))+", specificity: "+str(0 if np.isnan(gewicht_naivespec) else np.round(gewicht_naivespec,rounding_number))+", sensitivity: "+str(np.round(gewicht_naivesens,rounding_number)))
# print("LGBM model for weight - "+"accuracy:"+str(np.round(gewicht_modelacc,rounding_number))+", specificity: "+str(0 if np.isnan(gewicht_modelspec) else np.round(gewicht_modelspec,rounding_number))+", sensitivity: "+str(np.round(gewicht_modelsens,rounding_number)))

# # random forest regressor using randomized hyperparameter search
# # utility function instead of loss function, so the greater the better!
# rf_opening_reg = RandomForestRegressor()
# rf_gewicht_reg = RandomForestRegressor()

# pipeRFRopening = Pipeline([( "scaler" , StandardScaler()),("rf",rf_opening_reg)])
# pipeRFRgewicht = Pipeline([( "scaler" , StandardScaler()),("rf",rf_gewicht_reg)])
# sorted(pipeRFRopening.get_params().keys())
# sorted(pipeRFRgewicht.get_params().keys())

# rf__n_estimators=[500,2000]
# rf__max_depth=[5,10,25,50]
# rf__max_leaf_nodes=[5,10,25,50]
# rf__min_samples_split=[5,10,25,50]
# rf__min_samples_leaf=[5,10,25,50]
# rf__max_features=[10,20,30,40,50]
# paramsRFR_distributions = {"rf__n_estimators":rf__n_estimators, "rf__max_depth":rf__max_depth, 'rf__max_leaf_nodes':rf__max_leaf_nodes, 'rf__min_samples_split':rf__min_samples_split, 'rf__min_samples_leaf': rf__min_samples_leaf, 'rf__max_features': rf__max_features}

# if search_grid==0:
#     rfr_opening_reg_search_cv = RandomizedSearchCV(pipeRFRopening, paramsRFR_distributions, **random_fit_reg_params)                       
#     rfr_gewicht_reg_search_cv = RandomizedSearchCV(pipeRFRgewicht, paramsRFR_distributions, **random_fit_reg_params)
#     searchname='Random_'     
# else:
#     rfr_opening_reg_search_cv = GridSearchCV(pipeRFRopening, paramsRFR_distributions, **grid_fit_reg_params)                       
#     rfr_gewicht_reg_search_cv = GridSearchCV(pipeRFRgewicht, paramsRFR_distributions, **grid_fit_reg_params)
#     searchname='Grid_' 

# if tune_hyperparams_rfr_reg==1:
#     rfr_opening_reg_search_cv.fit(X_trainCV, y_openingtrainCV)
#     rfr_gewicht_reg_search_cv.fit(X_trainCV, y_gewichttrainCV)
#     pipeRFR_opening_regBEST=rfr_opening_reg_search_cv.best_estimator_
#     pipeRFR_gewicht_regBEST=rfr_gewicht_reg_search_cv.best_estimator_
#     pickle.dump(pipeRFR_opening_regBEST, open(searchname+'rfr_opening_reg_best_estimator.pkl','wb'))
#     pickle.dump(pipeRFR_gewicht_regBEST, open(searchname+'rfr_gewicht_reg_best_estimator.pkl','wb'))
#     report(rfr_opening_reg_search_cv.cv_results_,5)
#     report(rfr_gewicht_reg_search_cv.cv_results_,5)
# else:
#     pipeRFR_opening_regBEST=pd.read_pickle(searchname+'rfr_opening_reg_best_estimator.pkl')
#     pipeRFR_gewicht_regBEST=pd.read_pickle(searchname+'rfr_gewicht_reg_best_estimator.pkl')
        
# # retrain and evaluate on validation set
# pipeRFR_opening_regBEST.fit(X_trainCV, y_openingtrainCV)
# pipeRFR_gewicht_regBEST.fit(X_trainCV, y_gewichttrainCV)

          
# y_opening_reg_predRFR=pipeRFR_opening_regBEST.predict(X_val)
# y_gewicht_reg_predRFR=pipeRFR_gewicht_regBEST.predict(X_val)

# # evaluate prediction for opening
# y_opening_gt_and_pred_rfr=extract_gt_and_prediction(y_val,y_opening_reg_predRFR, "opening", df_Y)
# y_opening_gt_and_pred_naive=np.stack((y_opening_gt_and_pred_rfr[:,0],y_opening_gt_and_pred_rfr[:,0]*0+y_opening_mu_trainCV),axis=1)
# print("MAE opening Random Forest: " +str(np.round(mean_absolute_error(y_opening_gt_and_pred_rfr[:,0],y_opening_gt_and_pred_rfr[:,1]),4)))
# print("MAE opening Naive: " +str(np.round(mean_absolute_error(y_opening_gt_and_pred_naive[:,0],y_opening_gt_and_pred_naive[:,1]),4)))
# plot_prediction_errors(y_opening_gt_and_pred_rfr, 'RFR prediction for opening', "Ground truth opening [mm]", "Prediction error [mm]", "Average opening")
# plot_prediction_errors(y_opening_gt_and_pred_naive, 'Naive prediction for opening', "Ground truth opening [mm]", "Prediction error [mm]", "Average opening")
# opening_valtbinnendegrenzen_voorspelling=(y_opening_reg_predRFR<y_opening_mu_trainCV+2*y_opening_stdev_trainCV)*(y_opening_reg_predRFR>y_opening_mu_trainCV-2*y_opening_stdev_trainCV).astype(int)
# opening_valtbinnendegrenzen_groundtruth=df_Y_val['valid_opening_2sigma'].values
# opening_modelcm = confusion_matrix(opening_valtbinnendegrenzen_groundtruth,opening_valtbinnendegrenzen_voorspelling)
# opening_naivecm=confusion_matrix(np.ones(opening_valtbinnendegrenzen_groundtruth.shape), opening_valtbinnendegrenzen_groundtruth)
# opening_modelacc,opening_modelspec,opening_modelsens=calculate_spec_sens_acc_from_confusionmatrix(opening_modelcm)
# opening_naiveacc,opening_naivespec,opening_naivesens=calculate_spec_sens_acc_from_confusionmatrix(opening_naivecm)
# rounding_number=3
# print("Naive model for opening - "+"accuracy:"+str(np.round(opening_naiveacc,rounding_number))+", specificity: "+str(0 if np.isnan(opening_naivespec) else np.round(opening_naivespec,rounding_number))+", sensitivity: "+str(np.round(opening_naivesens,rounding_number)))
# print("RFR model for opening - "+"accuracy:"+str(np.round(opening_modelacc,rounding_number))+", specificity: "+str(0 if np.isnan(opening_modelspec) else np.round(opening_modelspec,rounding_number))+", sensitivity: "+str(np.round(opening_modelsens,rounding_number)))

# # evaluate prediction for weight
# y_gewicht_gt_and_pred_rfr=extract_gt_and_prediction(y_val,y_gewicht_reg_predRFR, "gewicht", df_Y)
# y_gewicht_gt_and_pred_naive=np.stack((y_gewicht_gt_and_pred_rfr[:,0],y_gewicht_gt_and_pred_rfr[:,0]*0+y_gewicht_mu_trainCV),axis=1)
# print("MAE gewicht RFR: " +str(np.round(mean_absolute_error(y_gewicht_gt_and_pred_rfr[:,0],y_gewicht_gt_and_pred_rfr[:,1]),4)))
# print("MAE gewicht Naive: " +str(np.round(mean_absolute_error(y_gewicht_gt_and_pred_naive[:,0],y_gewicht_gt_and_pred_naive[:,1]),4)))
# plot_prediction_errors(y_gewicht_gt_and_pred_rfr, 'RFR prediction for weight', "Ground truth weight [g]", "Prediction error [g]", "Average weight")
# plot_prediction_errors(y_gewicht_gt_and_pred_naive, 'Naive prediction for weight', "Ground truth weight [g]", "Prediction error [g]", "Average weight")
# gewicht_valtbinnendegrenzen_voorspelling=(y_gewicht_reg_predRFR<y_gewicht_mu_trainCV+2*y_gewicht_stdev_trainCV)*(y_gewicht_reg_predRFR>y_gewicht_mu_trainCV-2*y_gewicht_stdev_trainCV).astype(int)
# gewicht_valtbinnendegrenzen_groundtruth=df_Y_val['valid_gewicht_2sigma'].values
# gewicht_modelcm = confusion_matrix(gewicht_valtbinnendegrenzen_voorspelling, gewicht_valtbinnendegrenzen_groundtruth)
# gewicht_naivecm=confusion_matrix(np.ones(gewicht_valtbinnendegrenzen_groundtruth.shape), gewicht_valtbinnendegrenzen_groundtruth)
# gewicht_modelacc,gewicht_modelspec,gewicht_modelsens=calculate_spec_sens_acc_from_confusionmatrix(gewicht_modelcm)
# gewicht_naiveacc,gewicht_naivespec,gewicht_naivesens=calculate_spec_sens_acc_from_confusionmatrix(gewicht_naivecm)
# print("Naive model for weight - "+"accuracy:"+str(np.round(gewicht_naiveacc,rounding_number))+", specificity: "+str(0 if np.isnan(gewicht_naivespec) else np.round(gewicht_naivespec,rounding_number))+", sensitivity: "+str(np.round(gewicht_naivesens,rounding_number)))
# print("RFR model for weight - "+"accuracy:"+str(np.round(gewicht_modelacc,rounding_number))+", specificity: "+str(0 if np.isnan(gewicht_modelspec) else np.round(gewicht_modelspec,rounding_number))+", sensitivity: "+str(np.round(gewicht_modelsens,rounding_number)))


# opening_beste_lgb__subsample=1
# opening_beste_lgb__reg_lambda=1
# opening_beste_lgb__reg_alpha=0
# opening_beste_lgb__num_leaves=5
# opening_beste_lgb__n_estimators=2000
# opening_beste_lgb__min_child_weight=1
# opening_beste_lgb__min_child_samples=100

# gewicht_beste_lgb__subsample=0.1
# gewicht_beste_lgb__reg_lambda=5
# gewicht_beste_lgb__reg_alpha=0
# gewicht_beste_lgb__num_leaves=5
# gewicht_beste_lgb__n_estimators=2000
# gewicht_beste_lgb__min_child_weight=10
# gewicht_beste_lgb__min_child_samples=100

# opening_selected_steps=[('scaler', StandardScaler()), ('lgb', lgbm.LGBMRegressor(subsample=opening_beste_lgb__subsample, reg_lambda=opening_beste_lgb__reg_lambda, reg_alpha=opening_beste_lgb__reg_alpha, min_child_samples=opening_beste_lgb__min_child_samples, min_child_weight=opening_beste_lgb__min_child_weight,n_estimators=opening_beste_lgb__n_estimators, num_leaves=opening_beste_lgb__num_leaves))]
# gewicht_selected_steps=[('scaler', StandardScaler()), ('lgb', lgbm.LGBMRegressor(subsample=gewicht_beste_lgb__subsample, reg_lambda=gewicht_beste_lgb__reg_lambda, reg_alpha=gewicht_beste_lgb__reg_alpha, min_child_samples=gewicht_beste_lgb__min_child_samples, min_child_weight=gewicht_beste_lgb__min_child_weight,n_estimators=gewicht_beste_lgb__n_estimators, num_leaves=gewicht_beste_lgb__num_leaves))]
# opening_best_steps=[('scaler', StandardScaler()), ('lgb', lgb_opening_search_cv.best_estimator_.get_params()['lgb'])]
# gewicht_best_steps=[('scaler', StandardScaler()), ('lgb', lgb_gewicht_search_cv.best_estimator_.get_params()['lgb'])]

# opening_selected_estimator_=Pipeline(steps=opening_selected_steps)
# gewicht_selected_estimator_=Pipeline(steps=gewicht_selected_steps)

# opening_best_estimator_=Pipeline(steps=opening_best_steps)
# gewicht_best_estimator_=Pipeline(steps=gewicht_best_steps)


# # apply on test set
# trainscaler = StandardScaler()
# X_train_total=trainscaler.fit_transform(X_train)
# X_test_transformed=trainscaler.transform(X_test)

# opening_best_estimator_.fit(X_train_total,y_opening)
# opening_selected_estimator_.fit(X_train_total,y_opening)
# gewicht_best_estimator_.fit(X_train_total,y_gewicht)
# gewicht_selected_estimator_.fit(X_train_total,y_gewicht)

# y_opening_predLGB = opening_best_estimator_.predict(X_test_transformed)
# y_gewicht_predLGB = gewicht_best_estimator_.predict(X_test_transformed)

# # evaluate prediction for opening
# y_opening_gt_and_pred_lgb=extract_gt_and_prediction(y_test,y_opening_predLGB, "opening", df_Y)
# y_opening_gt_and_pred_naive=np.stack((y_opening_gt_and_pred_lgb[:,0],y_opening_gt_and_pred_lgb[:,0]*0+y_opening_mu_train),axis=1)
# print("MAE opening LGB: " +str(np.round(mean_absolute_error(y_opening_gt_and_pred_lgb[:,0],y_opening_gt_and_pred_lgb[:,1]),4)))
# print("MAE opening Naive: " +str(np.round(mean_absolute_error(y_opening_gt_and_pred_naive[:,0],y_opening_gt_and_pred_naive[:,1]),4)))
# plot_prediction_errors(y_opening_gt_and_pred_lgb, 'LGBM prediction for opening', "Ground truth opening [mm]", "Prediction error [mm]", "Average opening")
# plot_prediction_errors(y_opening_gt_and_pred_naive, 'Naive prediction for opening', "Ground truth opening [mm]", "Prediction error [mm]", "Average opening")
# opening_valtbinnendegrenzen_voorspelling=(y_opening_predLGB<y_opening_mu_train+2*y_opening_stdev_train)*(y_opening_predLGB>y_opening_mu_train-2*y_opening_stdev_train).astype(int)
# opening_valtbinnendegrenzen_groundtruth=df_Y_test['valid_opening_2sigma'].values
# opening_modelcm = confusion_matrix(opening_valtbinnendegrenzen_groundtruth,opening_valtbinnendegrenzen_voorspelling)
# opening_naivecm=confusion_matrix(np.ones(opening_valtbinnendegrenzen_groundtruth.shape), opening_valtbinnendegrenzen_groundtruth)
# opening_modelacc,opening_modelspec,opening_modelsens=calculate_spec_sens_acc_from_confusionmatrix(opening_modelcm)
# opening_naiveacc,opening_naivespec,opening_naivesens=calculate_spec_sens_acc_from_confusionmatrix(opening_naivecm)
# rounding_number=3
# print("Naive model for opening - "+"accuracy:"+str(np.round(opening_naiveacc,rounding_number))+", specificity: "+str(0 if np.isnan(opening_naivespec) else np.round(opening_naivespec,rounding_number))+", sensitivity: "+str(np.round(opening_naivesens,rounding_number)))
# print("LGBM model for opening - "+"accuracy:"+str(np.round(opening_modelacc,rounding_number))+", specificity: "+str(0 if np.isnan(opening_modelspec) else np.round(opening_modelspec,rounding_number))+", sensitivity: "+str(np.round(opening_modelsens,rounding_number)))

# # evaluate prediction for weight
# y_gewicht_gt_and_pred_lgb=extract_gt_and_prediction(y_test,y_gewicht_predLGB, "gewicht", df_Y)
# y_gewicht_gt_and_pred_naive=np.stack((y_gewicht_gt_and_pred_lgb[:,0],y_gewicht_gt_and_pred_lgb[:,0]*0+y_gewicht_mu_train),axis=1)
# print("MAE gewicht LGB: " +str(np.round(mean_absolute_error(y_gewicht_gt_and_pred_lgb[:,0],y_gewicht_gt_and_pred_lgb[:,1]),4)))
# print("MAE gewicht Naive: " +str(np.round(mean_absolute_error(y_gewicht_gt_and_pred_naive[:,0],y_gewicht_gt_and_pred_naive[:,1]),4)))
# plot_prediction_errors(y_gewicht_gt_and_pred_lgb, 'LGBM prediction for weight', "Ground truth weight [g]", "Prediction error [g]", "Average weight")
# plot_prediction_errors(y_gewicht_gt_and_pred_naive, 'Naive prediction for weight', "Ground truth weight [g]", "Prediction error [g]", "Average weight")
# gewicht_valtbinnendegrenzen_voorspelling=(y_gewicht_predLGB<y_gewicht_mu_train+2*y_gewicht_stdev_train)*(y_gewicht_predLGB>y_gewicht_mu_train-2*y_gewicht_stdev_train).astype(int)
# gewicht_valtbinnendegrenzen_groundtruth=df_Y_test['valid_gewicht_2sigma'].values
# gewicht_modelcm = confusion_matrix(gewicht_valtbinnendegrenzen_voorspelling, gewicht_valtbinnendegrenzen_groundtruth)
# gewicht_naivecm=confusion_matrix(np.ones(gewicht_valtbinnendegrenzen_groundtruth.shape), gewicht_valtbinnendegrenzen_groundtruth)
# gewicht_modelacc,gewicht_modelspec,gewicht_modelsens=calculate_spec_sens_acc_from_confusionmatrix(gewicht_modelcm)
# gewicht_naiveacc,gewicht_naivespec,gewicht_naivesens=calculate_spec_sens_acc_from_confusionmatrix(gewicht_naivecm)
# print("Naive model for weight - "+"accuracy:"+str(np.round(gewicht_naiveacc,rounding_number))+", specificity: "+str(0 if np.isnan(gewicht_naivespec) else np.round(gewicht_naivespec,rounding_number))+", sensitivity: "+str(np.round(gewicht_naivesens,rounding_number)))
# print("LGBM model for weight - "+"accuracy:"+str(np.round(gewicht_modelacc,rounding_number))+", specificity: "+str(0 if np.isnan(gewicht_modelspec) else np.round(gewicht_modelspec,rounding_number))+", sensitivity: "+str(np.round(gewicht_modelsens,rounding_number)))


# # #################################################

# # #################################################
# # # xgboost regression
# # pipeXGB = Pipeline([( "scaler" , StandardScaler()),("xgb",xgb_opening_reg)])
# # sorted(pipeXGB.get_params().keys())
# # y_opening=y_train[:,df_Y.columns.get_loc("opening")]
# # xgbn_estimators=[100,250,500,1000,2000]
# # xgbreg_alpha=[0, 1e-1, 1, 2, 5, 7, 10, 50, 100]
# # xgbreg_lambda=[0, 1e-1, 1, 5, 10, 20, 50, 100]
# # xgbsubsample=[0.1,0.5,0.8,1]
# # xgbmin_child_weight = [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
# # xgbmin_child_samples = [100,250,500]
# # xgbnum_leaves = [5,10,25,50]

# # paramsXGB_distributions = {"xgb__n_estimators":lgbn_estimators, "xgb__reg_alpha":lgbreg_alpha,"xgb__reg_lambda":lgbreg_lambda,"xgb__subsample":lgbsubsample,"xgb__min_child_weight": lgbmin_child_weight, "xgb__min_child_samples":lgbmin_child_samples, "xgb__num_leaves":lgbnum_leaves }
# # xgb_search_cv = RandomizedSearchCV(pipeXGB, paramsXGB_distributions, **random_fit_params)                       
# # tune_hyperparams_xgbm=0
# # if tune_hyperparams_xgbm==1:
# #     xgb_search_cv.fit(X_train, y_opening)
# # report(xgb_search_cv.cv_results_,5)


# # #####
# # beste_xgb__subsample=1
# # beste_xgb__reg_lambda=20
# # beste_xgb__reg_alpha=0
# # beste_xgb__num_leaves=10
# # beste_xgb__n_estimators=250
# # beste_xgb__min_child_weight=10
# # beste_xgb__min_child_samples=100
# # _steps=[('scaler', StandardScaler()), ('xgb', xgboost.XGBRegressor(subsample=beste_xgb__subsample, reg_lambda=beste_xgb__reg_lambda, reg_alpha=beste_xgb__reg_alpha, min_child_samples=beste_xgb__min_child_samples, min_child_weight=beste_xgb__min_child_weight,n_estimators=beste_xgb__n_estimators, num_leaves=beste_xgb__num_leaves))]
# # pipeXGB_final=Pipeline(steps=_steps)

# # # apply on test set
# # trainscaler = StandardScaler()
# # X_train_total=trainscaler.fit_transform(X_train)
# # xgbreg_final=pipeXGB_final.get_params()['xgb']
# # xgbreg_final.fit(X_train_total,y_opening)

# # X_test_transformed=trainscaler.transform(X_test)
# # y_predXGB = xgbreg_final.predict(X_test_transformed)

# # # evaluate prediction
# # # evaluate prediction
# # y_gt_and_pred_xgb=extract_gt_and_prediction(y_test,y_predXGB, "opening", df_Y)
# # y_gt_and_pred_naive=np.stack((y_gt_and_pred_xgb[:,0],y_gt_and_pred_xgb[:,0]*0+y_opening_mu_train),axis=1)

# # print("MAE XGB: " +str(np.round(mean_absolute_error(y_gt_and_pred_xgb[:,0],y_gt_and_pred_xgb[:,1]),4)))
# # print("MAE Naive: " +str(np.round(mean_absolute_error(y_gt_and_pred_naive[:,0],y_gt_and_pred_naive[:,1]),4)))

# # plot_prediction_errors(y_gt_and_pred_xgb, 'LGBM', "Ground truth opening ")
# # plot_prediction_errors(y_gt_and_pred_naive, 'Naive')  



# # valtbinnendegrenzen_voorspelling=(y_predXGB<y_opening_mu_train+2*y_opening_stdev_train)*(y_predXGB>y_opening_mu_train-2*y_opening_stdev_train).astype(int)
# # valtbinnendegrenzen_groundtruth=df_Y_test['valid_opening_2sigma'].values

# # modelcm = confusion_matrix(valtbinnendegrenzen_voorspelling, valtbinnendegrenzen_groundtruth)
# # naivecm=confusion_matrix(np.ones(valtbinnendegrenzen_groundtruth.shape), valtbinnendegrenzen_groundtruth)

# # modelacc,modelspec,modelsens=calculate_spec_sens_acc_from_confusionmatrix(modelcm)
# # naiveacc,naivespec,naivesens=calculate_spec_sens_acc_from_confusionmatrix(naivecm)

# # print("Naive model - "+"acc:"+str(np.round(naiveacc,3))+", spec: "+str(np.round(naivespec,3))+", sens: "+str(np.round(naivesens,3)))
# # print("LGBM model - "+"acc:"+str(np.round(modelacc,3))+", spec: "+str(np.round(modelacc,3))+", sens: "+str(np.round(modelacc,3)))


# # # random forest regressor using randomized hyperparameter search
# # # utility function instead of loss function, so the greater the better!
# # pipeRFR = Pipeline([( "scaler" , StandardScaler()),("rf",rf_opening_reg)])
# # sorted(pipeRFR.get_params().keys())

# # rf__n_estimators=[500,2000]
# # rf__max_depth=[5,10,25,50]
# # rf__max_leaf_nodes=[5,10,25,50]
# # rf__min_samples_split=[5,10,25,50]
# # rf__min_samples_leaf=[5,10,25,50]
# # rf__max_features=[10,20,30,40,50]

# # train_rf_model=0
# # if train_rf_model==1:
# #     paramsRFR_distributions = {"rf__n_estimators":rf__n_estimators, "rf__max_depth":rf__max_depth, 'rf__max_leaf_nodes':rf__max_leaf_nodes, 'rf__min_samples_split':rf__min_samples_split, 'rf__min_samples_leaf': rf__min_samples_leaf, 'rf__max_features': rf__max_features}
# #     rnd_search_cv = RandomizedSearchCV(pipeRFR, paramsRFR_distributions, n_iter=500, **random_fit_params)                     
# #     rnd_search_cv.fit(X_train, y_train[:,df_Y.columns.get_loc("opening")])

# #     report(rnd_search_cv.cv_results_,5)
# #     rnd_search_cv.best_estimator_
# #     pickle.dump(rnd_search_cv.best_estimator_, open('rfr_best_estimator.pkl','wb'))


# # # loiad settings for selected model
# # load_model=0
# # if load_model==1:
# #     loaded_model = pickle.load(open('rfr_more_trees_larger_testset_best_estimator.pkl', 'rb'))

# # beste_max_depth=10
# # beste_max_features=50
# # beste_max_leaf_nodes=50
# # beste_min_samples_leaf=5
# # beste_min_samples_split=5
# # beste_n_estimators=2500
# # rnd_search_cv.best_estimator_=Pipeline(steps=[('scaler', StandardScaler()), ('rf', RandomForestRegressor(max_depth=beste_max_depth, max_features=beste_max_features, max_leaf_nodes=beste_max_leaf_nodes, min_samples_leaf=beste_min_samples_leaf, min_samples_split=beste_min_samples_split, n_estimators=beste_n_estimators))])

# # # apply on test set
# # trainscaler = StandardScaler()
# # X_train_total=trainscaler.fit_transform(X_train)
# # rfreg_final=rnd_search_cv.best_estimator_.get_params()['rf']
# # rfreg_final.fit(X_train_total,y_train[:,df_Y.columns.get_loc("opening")])

# # X_test_transformed=trainscaler.transform(X_test)
# # y_pred = rfreg_final.predict(X_test_transformed)

# # # evaluate prediction
# # y_gt_and_pred_df=pd.DataFrame(np.stack((y_test[:,df_Y.columns.get_loc("opening")], y_pred), axis=1), columns=['gt','pred'])
# # y_gt_and_pred=y_gt_and_pred_df.sort_values(by='gt').values
# # print("MAE RF: " +str(mean_absolute_error(y_gt_and_pred[:,0],y_gt_and_pred[:,1])))
# # print("MAE Naive: " +str(mean_absolute_error(y_gt_and_pred[:,0],y_gt_and_pred[:,0]*0+y_gt_and_pred[:,0].mean())))
      
# # plt.scatter(y_gt_and_pred[:,0], y_gt_and_pred[:,0]-y_gt_and_pred[:,1], alpha=0.5)
# # plt.xlabel('Ground truth opening [mm]')
# # plt.ylabel('Prediction error [mm]')
# # plt.axvline(x=y_gt_and_pred[:,0].mean(), linestyle='dashed')
# # plt.legend(['average opening', 'prediction error'])
# # plt.title('Random Forest Regression: Prediction errors on test set')
# # plt.show()


# # # feature importance
# # # feature selection 1
# # sel = SelectFromModel(rfreg_final)
# # sel.fit(X_train_total,y_train[:,df_Y.columns.get_loc("opening")])
# # sel.get_support()
# # selected_feat= np.array(feature_names_applied_to_curves)[sel.get_support()]
# # len(selected_feat)
# # print(selected_feat)


# # # feature selection 2
# # features = feature_names_applied_to_curves
# # importances = rfreg_final.feature_importances_
# # indices = np.argsort(importances)[::-1]
# # selection=indices.shape[0]
# # selection=10
# # plt.title(str(selection) + ' most important features according to random forest')
# # plt.barh(range(selection)[::-1], importances[indices][:selection], color='b', align='center')
# # plt.yticks(range(selection)[::-1], [features[i] for i in indices[:selection]])
# # plt.xlabel('Relative Importance')
# # plt.show()

# # # visualization
# # estimator=rfreg_final.estimators_[5]
# # plot_tree(estimator, feature_names=features, filled=True)
# # dtreeviz(estimator, X_train_total, y_train[:,df_Y.columns.get_loc("opening")], target_name='opening',feature_names=features)


# # # decision tree regressor using randomized hyperparameter search
# # # utility function instead of loss function, so the greater the better!

# # pipeDTR = Pipeline([( "scaler" , StandardScaler()),("rf",DecisionTreeRegressor())])
# # sorted(pipeDTR.get_params().keys())
# # paramsDTR_distributions = {"rf__max_depth": list(range(2, 41)), 'rf__max_leaf_nodes': list(range(2, 100)), 'rf__min_samples_split': list(range(2, 41)), 'rf__min_samples_leaf': list(range(2, 41)), 'rf__max_features': list(range(5, 31))}
# # rnd_search_cv = RandomizedSearchCV(pipeDTR, paramsDTR_distributions, n_iter=1000, verbose=2, cv=5, random_state=42, scoring='neg_mean_absolute_error', n_jobs=-1)                     
# # rnd_search_cv.fit(X_train, y_train[:,df_Y.columns.get_loc("opening")])
# # report(rnd_search_cv.cv_results_,5)
# # rnd_search_cv.best_estimator_

# # # apply on test set
# # trainscaler = StandardScaler()
# # trainscaler.fit_transform(X_train)
# # X_test_scaled=trainscaler.transform(X_test)
# # y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)



# # # Looks much better than the linear model. Let's select this model and evaluate it on the test set:

# # # In[70]:


# # y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
# # mse = mean_squared_error(y_test, y_pred)
# # np.sqrt(mse)



# # # for grid search (WIP!)
# # paramsDTR = {'rf__max_leaf_nodes': list(range(2, 100)), 'rf__min_samples_split': list(range(2, 21))}
# # grid_search_cv = GridSearchCV(pipeDTR, paramsDTR, verbose=2)
# # grid_search_cv.fit(X_train, y_train[:,df_Y.columns.get_loc("opening")])








# # pipe1 = Pipeline(steps=[('transformer', scaler), ('estimator', svmlinreg)])
# # pipe2 = Pipeline(steps=[('transformer', scaler), ('estimator', svmreg)])
# # pipe3 = Pipeline([('scaler', scaler), ('dtreg', dtreg)])
# # pipe4 = Pipeline(steps=[('transformer', scaler), ('estimator', rfreg)])

# # params3 = {'rf__max_leaf_nodes': list(range(2, 100)), 'rf__min_samples_split': [2, 3, 4]}


# # pipe3=Pipeline(steps=[('transformer', StandardScaler()), ('estimator', DecisionTreeRegressor())])


# # # In this training set, the targets are tens of thousands of dollars. The RMSE gives a rough idea of the kind of error you should expect (with a higher weight for large errors): so with this model we can expect errors somewhere around $10,000. Not great. Let's see if we can do better with an RBF Kernel. We will use randomized search with cross validation to find the appropriate hyperparameter values for `C` and `gamma`:

# # # In[67]:


# # param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
# # rnd_search_cv = RandomizedSearchCV(SVR(), param_distributions, n_iter=10, verbose=2, cv=3, random_state=42)
# # rnd_search_cv.fit(X_train_scaled, y_train)


# # # In[68]:


# # rnd_search_cv.best_estimator_


# # # Now let's measure the RMSE on the training set:

# # # In[69]:


# # y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
# # mse = mean_squared_error(y_train, y_pred)
# # np.sqrt(mse)


# # Looks much better than the linear model. Let's select this model and evaluate it on the test set:

# # In[70]:


# y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
# mse = mean_squared_error(y_test, y_pred)
# np.sqrt(mse)


# # In[22]:


# grid_search_cv.best_estimator_



# # Parameters of pipelines can be set using ‘__’ separated parameter names:
# param_grid = {
#     'pca__n_components': [5, 15, 30, 45, 64],
#     'logistic__C': np.logspace(-4, 4, 4),
# }
# search = GridSearchCV(pipe, param_grid, n_jobs=-1)
# search.fit(X_digits, y_digits)
# print("Best parameter (CV score=%0.3f):" % search.best_score_)
# print(search.best_params_)


# cv = ShuffleSplit(n_splits=5, random_state=42)





# scores_opening = cross_val_score(pipeline, X_train, y_train[:,df_Y.columns.get_loc("opening")], cv = cv)


# # Lasso model using CV
# # use automatically configured the lasso regression algorithm
# X=df_X.values
# scaler=StandardScaler()
# X=scaler.fit_transform(X)
# opening=df_Y.iloc[:,0]





# tree_reg = DecisionTreeRegressor()


# lasso = Lasso(alpha=0.5, normalize=True)
# lasso.fit(X,opening)

# scores = cross_val_score(lasso, X, opening, scoring="neg_mean_squared_error", cv=5)


# cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
# model = LassoCV(alphas=arange(0, 10, 0.01), cv=cv)
# # fit model
# a=model.fit(X, opening)
# # summarize chosen configuration
# print('alpha: %f' % model.alpha_)







# # split train and test-set
# df_train, df_test = train_test_split(df_X_Y, test_size=0.1, random_state=42)
# cv = ShuffleSplit(n_splits=5, random_state=42)



# scaler=StandardScaler()
# X_train_scaled, Y_train_scaled=scaler.fit_transform(X_train),scaler.fit_transform(Y_train)


# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_index, test_index in split.split(housing, housing["income_cat"]):
#     strat_train_set = housing.loc[train_index]
#     strat_test_set = housing.loc[test_index]



# num_pipeline = Pipeline([
# ('imputer', SimpleImputer(strategy="median")),
# ('attribs_adder', CombinedAttributesAdder()),
# ('std_scaler', StandardScaler()),
# ])
# housing_num_tr = num_pipeline.fit_transform(housing_num)



# # voorbeeldje 1a
# # from numpy import arange
# # from pandas import read_csv
# # from sklearn.model_selection import GridSearchCV
# # from sklearn.model_selection import RepeatedKFold
# # from sklearn.linear_model import Lasso
# # # load the dataset
# # url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
# # dataframe = read_csv(url, header=None)
# # data = dataframe.values
# # X, y = data[:, :-1], data[:, -1]
# # # define model
# # model = Lasso()
# # # define model evaluation method
# # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# # # define grid
# # grid = dict()
# # grid['alpha'] = arange(0, 1, 0.01)
# # # define search
# # search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# # # perform the search
# # results = search.fit(X, y)
# # # summarize
# # print('MAE: %.3f' % results.best_score_)
# # print('Config: %s' % results.best_params_)
# # # voorbeeldje 1b
# # # use automatically configured the lasso regression algorithm
# # from numpy import arange
# # from pandas import read_csv
# # from sklearn.linear_model import LassoCV
# # from sklearn.model_selection import RepeatedKFold
# # # load the dataset
# # url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
# # dataframe = read_csv(url, header=None)
# # data = dataframe.values
# # X, y = data[:, :-1], data[:, -1]
# # # define model evaluation method
# # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# # # define model
# # model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
# # # fit model
# # model.fit(X, y)
# # # summarize chosen configuration
# # print('alpha: %f' % model.alpha_)


# from sklearn.feature_selection import VarianceThreshold
# X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
# selector = VarianceThreshold()
# selector.fit(X)

## feature importance
selection=20
fi_opening=new_pipeLGB_opening_regBEST['lgb']
fi_opening.fit(StandardScaler().fit_transform(X_trainCV), y_openingtrainCV)
fimportances=list(fi_opening.feature_importances_)
fimportances, fnames = zip(*sorted(zip(fimportances, X_correlated_feature_names)))
fimportances=list(fimportances/np.max(fimportances)*100)[::-1][:selection]
fnames=list(fnames)[::-1][:selection]
fiSeries=pd.Series(fimportances,index=fnames)
ax=sns.barplot(y=fiSeries.index,x=fiSeries.values)
ax.set(title='Top 20 most important features', xlabel='Relative importance', ylabel='Feature')

