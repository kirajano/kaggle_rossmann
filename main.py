# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:03:17 2021

@author: Kiril, Bahar, Abeer
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
# Custom Functions
import functions as funct
# Custom Settings
# Making notna() recognize also empty strings and numpy.inf
pd.options.mode.use_inf_as_na = True
import warnings
warnings.filterwarnings('ignore')


### DATA LOADING AND CLEANING ###

# Loading both data sets
train = pd.read_csv("./train.csv")
store = pd.read_csv("./store.csv")

# Check missing values in train and store
train.isnull().sum() / train.shape[0]
store.isnull().sum() / store.shape[0]

# Remove stores with no IDs since predictions will be based on stores
train.dropna(axis=0, inplace=True, subset=["Store", "Sales"])

# Joining datasets together
dataset = train.join(store.set_index("Store"), on="Store").reset_index(drop=True)

# Inspecting nulls after joining
dataset.isnull().sum() / dataset.shape[0]

# Replacing DayOfWeek with proper data based on Date column
dataset = dataset.astype({"Date": "datetime64[ns]"})
dataset["DayOfWeek"] = dataset["Date"].dt.weekday

# Removing rows with many nulls
nans_to_drop =["DayOfWeek", "Customers", "Open", "Promo", 
                "StateHoliday", "SchoolHoliday","CompetitionOpenSinceMonth", 
                "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear", 
                "PromoInterval"]
dataset.dropna(axis=0, inplace=True, subset=nans_to_drop)

# Drop rows with no sales
dataset = dataset.loc[(dataset["Sales"] != 0)]

# Adjusting datatypes
col_types = {"Date": "datetime64[ns]", "DayOfWeek": int, "Store": int, "Customers": int, 
             "Open": int, "Promo": int, "SchoolHoliday": str, "CompetitionOpenSinceMonth": int,
             "CompetitionOpenSinceYear": int, "Promo2SinceWeek": int, "Promo2SinceYear": int}
dataset = dataset.astype(col_types, copy=False)


### ENCODING ###

# Target Encoding for Store
te = TargetEncoder(cols=["Store"])
te.fit(dataset["Store"], dataset["Customers"])
dataset["Store_tgt_enc"] = te.transform(dataset["Store"])

# Target Encoding for DayOfWeek
te = TargetEncoder(cols=["DayOfWeek"])
te.fit(dataset["DayOfWeek"], dataset["Customers"])
dataset["DayOfWeek_tgt_enc"] = te.transform(dataset["DayOfWeek"])



### TRAIN / TEST SPLIT ###

# Resetting index prior split
dataset.reset_index(drop=True, inplace=True)

# Spliting data
x = dataset.loc[:, (dataset.columns != "Sales")]
y = dataset["Sales"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None, shuffle=False)



###  BASELINE MODEL based on simple average ###
print("**** Calculating Baseline model using Average ****")
funct.baseline_model(y_train, y_test)



### FEATURE SELECTION AND ENGINEERING ###

# Check correlations visuallly
plt.subplots(figsize=(11,9))
cols_to_exl = ["Open", "Promo2"]
sns.heatmap(dataset.drop(cols_to_exl, axis=1).corr(), cbar=False, annot=True)

# Select features based on strong correlation
features = ["Customers", "Promo", "DayOfWeek_tgt_enc", "Store_tgt_enc"]



### FIRST MODEL APPLICATION ###
rfr = RandomForestRegressor(max_depth=20, n_estimators=50)
rfr.fit(x_train[features], y_train)
y_pred = rfr.predict(x_test[features])
print(f"Random Forest Regresor with features {features} yield {funct.metric(y_test, y_pred):.2f}% RMSPE as error")



### CROSS VALIDATION (to check if overfitting) ###

# With cross_val_score
print("**** Running Cross-Validation via cross_val_score on training set ****")
rfr_cv = RandomForestRegressor(max_depth=20, n_estimators=50)
cv_accy = cross_val_score(rfr_cv, x_train[features], y_train, cv=5)
print(f"R2 Score for CV on training set: {cv_accy.mean():.2f}")

# Custom Cross-Validation
print("**** Running Cross-Validation via train-validate-test split ****")
funct.custom_cv(dataset, features, train_cut=0.6, val_cut=0.8)



### PIPELINING MODELS WITH SCALER ###
print("**** Building Pipeline with Standard Scaler and RandomForest, KNN and XGBoost Regressors ****")
models = [RandomForestRegressor(max_depth=20, n_estimators=50),
            KNeighborsRegressor(weights="distance"),
         #XGBRegressor(n_estimators=500, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=0.8)
         ]
funct.pipeline_models(x_train, x_test, y_train, y_test, features, *models)


### HYPER PARAMETER TUNING ###
# Model selection based on above comparison

#print("**** Picking XGBoost Regression as best model and tune hyperparameters ****")
#model = XGBRegressor()
#params = {"max_depth": [10, 15, 20], "n_estimators": [400, 500, 600],  "eta": [0.1], "subsample": [0.7], "colsample_bytree": [0.8]}
#final_model = funct.model_tuning(model, params, x, y, features, n_splits=3)
# Exporting Final Model
#print("**** Exporting Model to Pickle ****")
#save_model = "rossman_model.sav"
#pickle.dump(final_model, open(save_model, "wb"))


### FINAL TRAIN RUN ### 
print("++++++ MODEL BULDING DONE ++++++")
save_model = "rossman_model.sav"
load_model = pickle.load(open(save_model, "rb"))
print("*** Fetching Hold-Out Set ***")
holdout_data = funct.holdout_data(store)
y_test = holdout_data["Sales"]
print("*** Predicting for Hold-out Set ***")
y_pred = load_model.predict(holdout_data[features])
print("######## FINAL SCORE ########")
print(f"Prediction on Hold-Out Set: {funct.metric(y_test, y_pred):.2f}% ")