from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

################################LINEAR-REGRESSION######################################################

def Linear_Reg_KF(X_train, y_train, X_test, y_test):
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

def Linear_Reg(df_train, df_eval):
    lr = LinearRegression()
    lr.fit(df_train.drop(columns='target'), df_train['target'])

    y_pred = lr.predict(df_eval.drop(columns='target'))
    
    MSE_eval = mean_squared_error(df_eval['target'], y_pred)
    R2_eval = r2_score(df_eval['target'], y_pred)

    return MSE_eval, R2_eval


################################LINEAR-REGRESSION######################################################

def Lin_Reg_Interaction(df_train, df_eval):
    # Linear Regression with Interaction Terms
    poly = PolynomialFeatures(interaction_only=True,include_bias = False)
    X_poly = poly.fit_transform(df_train.drop(columns=['target']))

    lri = LinearRegression()
    lri.fit(X_poly, df_train['target'])

    y_eval = df_eval['target']

    X_poly_eval = poly.fit_transform(df_eval.drop(columns=['target']))
    y_pred_lri = lri.predict(X_poly_eval)
    MSE_lri = mean_squared_error(y_eval, y_pred_lri)
    R2_lri = r2_score(y_eval, y_pred_lri)

    return MSE_lri, R2_lri


################################RANDOM-FOREST######################################################

def Random_Forest_KF(X_train, y_train, X_test, y_test):
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

def Random_Forest(df_train, df_eval):
    rf = RandomForestRegressor()
    rf.fit(df_train.drop(columns='target'), df_train['target'])

    y_pred = rf.predict(df_eval.drop(columns='target'))
    
    MSE_eval = mean_squared_error(df_eval['target'], y_pred)
    R2_eval = r2_score(df_eval['target'], y_pred)

    return MSE_eval, R2_eval


################################RANDOM-FOREST-SCALED######################################################

def Random_Forest_Scaled_KF(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled):
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

def Random_Forest_scaled(df_train_scaled, df_eval_scaled):
    rf = RandomForestRegressor()
    rf.fit(df_train_scaled.drop(columns='target'), df_train_scaled['target'])

    y_pred = rf.predict(df_eval_scaled.drop(columns='target'))
    
    MSE_eval = mean_squared_error(df_eval_scaled['target'], y_pred)
    R2_eval = r2_score(df_eval_scaled['target'], y_pred)

    return MSE_eval, R2_eval


################################RANDOM-FOREST-HYPERPARAM_TUNING######################################################

def Random_Forest_hyperparam(df_train, df_eval):
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

    return MSE_rf_hyperparam, R2_rf_hyperparam