from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd
import os

import FeatureExtract as FE

################################DATA PREPARATION#######################################################

def split_data(path="./scaled_pickles", train_frac = 0.8):
    ### split data into evaluation and training set
    assert train_frac < 1
    assert train_frac >= 0

    # shuffle data randomly to make different datasets
    all_files = os.listdir(path)
    # data_size = data.shape[0] # data.shape[0] changed to fit dictionary
    # indices = np.arange(data_size)  
    np.random.shuffle(all_files)
    #data_shuffled = data[indices]

    # split shuffeled data into test, train and eval according to fractions
    idx_eval = int(train_frac * len(all_files))

    files_train = all_files[:idx_eval]
    files_eval = all_files[idx_eval:]

    return files_train, files_eval

files_train_scal, files_eval_scal = split_data()
files_train, files_eval = split_data('./pickles')


def file_to_data(files):
    a = list()
    for i, idx in enumerate(files):
        a.append(FE.extract_features(f"./pickles/{idx}"))
    df = pd.DataFrame(a)
    return df

def file_to_data_scaled(files):
    a = list()
    for i, idx in enumerate(files):
        a.append(FE.extract_features_scaled(f"./scaled_pickles/{idx}"))
    df = pd.DataFrame(a)
    return df

df_eval_original = file_to_data(files_eval)
df_train_original = file_to_data(files_train)

df_eval_scaled = file_to_data_scaled(files_eval_scal)
df_train_scaled = file_to_data_scaled(files_train_scal)
df_eval_scaled = df_eval_scaled.reindex(df_train_scaled.columns, axis=1)

df_eval = pd.concat([df_eval_original, df_eval_scaled])
df_train = pd.concat([df_train_original, df_train_scaled])
df_eval = df_eval.reindex(df_train.columns, axis=1)


def split_data_kfold(df_train):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    X = df_train.drop(columns=['target'])
    y = df_train['target']
    
    for train_index, test_index in kf.split(df_train):
        X_train.append(X.iloc[train_index])
        X_test.append(X.iloc[test_index])
        y_train.append(y.iloc[train_index])
        y_test.append(y.iloc[test_index])
        
    return X_train, X_test, y_train, y_test

# Ensure the data is correct and check the type of first element
X_train, X_test, y_train, y_test = split_data_kfold(df_train)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = split_data_kfold(df_train_scaled)

################################LINEAR-REGRESSION######################################################

def Linear_Reg_KF(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test):
    # returns best fitting linear model of all trained models based on MSE
    # X_train, y_train, X_test, y_test are numpy arrays
    # returns best model, MSE, R2
    #X_train, X_test, y_train, y_test = split_data_kfold(df_train)

    MSE = []
    R2 = []
    
    for i in range(len(X_train)):      
        
        model = LinearRegression()
        model.fit(X_train[i], y_train[i])
        y_pred = model.predict(X_test[i])

        MSE.append(mean_squared_error(y_test[i], y_pred))
        R2.append(r2_score(y_test[i], y_pred))

    return np.mean(MSE), np.mean(R2)

def Linear_Reg(df_train=df_train, df_eval=df_eval):
    lr = LinearRegression()
    lr.fit(df_train.drop(columns='target'), df_train['target'])

    y_pred = lr.predict(df_eval.drop(columns='target'))
    
    MSE_eval = mean_squared_error(df_eval['target'], y_pred)
    R2_eval = r2_score(df_eval['target'], y_pred)

    return MSE_eval, R2_eval, lr


################################LINEAR-REGRESSION######################################################

def Lin_Reg_Interaction(df_train=df_train, df_eval=df_eval):
    # Linear Regression with Interaction Terms
    poly = PolynomialFeatures(degree=2, interaction_only=False,include_bias = False)
    X_poly = poly.fit_transform(df_train.drop(columns=['target']))

    lri = LinearRegression()
    lri.fit(X_poly, df_train['target'])

    print(len(lri.coef_))

    y_eval = df_eval['target']

    X_poly_eval = poly.fit_transform(df_eval.drop(columns=['target']))
    y_pred_lri = lri.predict(X_poly_eval)
    MSE_lri = mean_squared_error(y_eval, y_pred_lri)
    R2_lri = r2_score(y_eval, y_pred_lri)

    return MSE_lri, R2_lri, lri


################################RANDOM-FOREST######################################################

def Random_Forest_KF(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):
    # returns best fitting linear model of all trained models based on MSE
    # X_train, y_train, X_test, y_test are numpy arrays
    # returns best model, MSE, R2
    MSE = []
    R2 = []
    
    for i in range(len(X_train)): 

        rf = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=5)
        rf.fit(X_train[i], y_train[i])
        y_pred = rf.predict(X_test[i])

        MSE.append(mean_squared_error(y_test[i], y_pred))
        R2.append(r2_score(y_test[i], y_pred))

    return np.mean(MSE), np.mean(R2)

def Random_Forest(df_train=df_train, df_eval=df_eval):
    rf = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=5)
    rf.fit(df_train.drop(columns='target'), df_train['target'])

    y_pred = rf.predict(df_eval.drop(columns='target'))
    
    MSE_eval = mean_squared_error(df_eval['target'], y_pred)
    R2_eval = r2_score(df_eval['target'], y_pred)

    return MSE_eval, R2_eval, rf


################################RANDOM-FOREST-SCALED######################################################

def Random_Forest_Scaled_KF(X_train_scaled=X_train_scaled, y_train_scaled=y_train_scaled, X_test_scaled=X_test_scaled, y_test_scaled=y_test_scaled):
    # returns best fitting linear model of all trained models based on MSE
    # X_train, y_train, X_test, y_test are numpy arrays
    # returns best model, MSE, R2
    MSE = []
    R2 = []
    
    for i in range(len(X_train_scaled)): 

        rf = RandomForestRegressor()
        rf.fit(X_train_scaled[i], y_train_scaled[i])
        y_pred = rf.predict(X_test_scaled[i])

        MSE.append(mean_squared_error(y_test_scaled[i], y_pred))
        R2.append(r2_score(y_test_scaled[i], y_pred))

    return np.mean(MSE), np.mean(R2)

def Random_Forest_scaled(df_train_scaled=df_train_scaled, df_eval_scaled=df_eval_scaled):
    rf = RandomForestRegressor()
    rf.fit(df_train_scaled.drop(columns='target'), df_train_scaled['target'])

    y_pred = rf.predict(df_eval_scaled.drop(columns='target'))
    
    MSE_eval = mean_squared_error(df_eval_scaled['target'], y_pred)
    R2_eval = r2_score(df_eval_scaled['target'], y_pred)

    return MSE_eval, R2_eval, rf


################################RANDOM-FOREST-HYPERPARAM_TUNING######################################################

def Random_Forest_hyperparam(df_train=df_train, df_eval=df_eval):
    max_depth = [int(x) for x in np.linspace(1, 25, num = 5)]
    max_depth.append(None)
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 110, num = 5)]

    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True, False],
        'max_depth': max_depth,
        'min_samples_leaf': [2, 3, 4],
        'min_samples_split': [2, 4, 8],
        'n_estimators': n_estimators,
        'criterion': ['squared_error', 'absolute_error']
    }
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    random_search = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2, n_iter=75, random_state=42)

    random_search.fit(df_train.drop(columns='target'), df_train['target'])
    y_pred = random_search.predict(df_eval.drop(columns='target'))

    MSE_rf_hyperparam = mean_squared_error(df_eval['target'], y_pred)
    R2_rf_hyperparam = r2_score(df_eval['target'], y_pred)

    print(random_search.best_params_)

    return MSE_rf_hyperparam, R2_rf_hyperparam, random_search