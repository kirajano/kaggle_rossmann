# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:46:09 2021

@author: Kiril Kasjanov
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder





def metric(actuals, preds):
    assert preds.shape == actuals.shape
    return 100 * (np.sqrt(np.mean(np.square((actuals - preds) / actuals))))


def baseline_model(y_train, y_test):
	""" Baseline Model based on Average Sales"""
	y_pred = np.array([y_train.mean() for _ in range(y_test.shape[0])])
	print(f"Baseline yield {metric(y_pred, y_test):.2f}% RMSPE as error")


def custom_cv(dataset, features, train_cut=0.6, val_cut=0.8):
	"""
	Function for custom train, validate and test split. The evaluation is used with RandomForestRegressor.
	Parameters:
		dataset (dtype = pandas.DataFrame): the dataset that will be split
		train_cut (float): percentage of total dataset to be cut at as train set
		val_cut (float): percentage of total dataset to be cut at as validation set
	Output:
		Prints the RMSPE for train vs validate and train+validate vs test
	"""
	train, validate, test = np.split(dataset, [int(train_cut * len(dataset)) -1, int(val_cut * len(dataset))] )
	rfr_cv = RandomForestRegressor(max_depth=20, n_estimators=50)

	cv_x_train = train.loc[:, (train.columns != "Sales")]
	cv_y_train = train["Sales"]

	cv_x_validate = validate.loc[:, (validate.columns != "Sales")]
	cv_y_validate = validate["Sales"]

	cv_x_test = test.loc[:, (test.columns != "Sales")]
	cv_y_test = test["Sales"]

	cv_data = dict(validate_sample=[cv_x_train, cv_y_train, cv_x_validate, cv_y_validate],
			test_sample=[pd.concat([cv_x_train, cv_x_validate]), pd.concat([cv_y_train, cv_y_validate]), cv_x_test, cv_y_test])

	for name, sample in cv_data.items():
		rfr_cv.fit(sample[0][features], sample[1])
		pred = rfr_cv.predict(sample[2][features])
		print(f"{name} RMSPE: {metric(sample[3], pred):.2f}%")


def pipeline_models(x_train, x_test, y_train, y_test, features, *models):
	""" Pipeline models and apply scaling"""
	for model in models:
		pipe = Pipeline(steps=[
			("scaler", StandardScaler()),
			("regressor", model)])
		pipe.fit(x_train[features], y_train)
		y_pred = pipe.predict(x_test[features])
		print(str(model).split("(")[0], f"{metric(y_test, y_pred):.2f}%")


def model_tuning(model, params, x, y, features, n_splits=5):
	"""Model tuning based on KFold split and GridSearch"""
	cv = KFold(n_splits=n_splits, random_state=None, shuffle=False)
	gscv = GridSearchCV(model, params, scoring="r2", n_jobs=-1, cv=cv)
	gscv.fit(x[features], y)
	print(f'Best Score: {gscv.best_score_}')
	print(f'Best Hyperparameters: {gscv.best_params_}')
	return gscv


def holdout_data(store):
	""" Run external with final model """
	ext_data = pd.read_csv("./holdout.csv")
	ext_data.dropna(axis=0, inplace=True, subset=["Store", "Sales"])
	ext_data = ext_data.join(store.set_index("Store"), on="Store").reset_index(drop=True)
	ext_data = ext_data.astype({"Date": "datetime64[ns]"})
	ext_data["DayOfWeek"] = ext_data["Date"].dt.weekday
	ext_data = ext_data.loc[(ext_data["Sales"] != 0)]
	ext_data.dropna(axis=0, inplace=True)
	col_types = {"Date": "datetime64[ns]", "DayOfWeek": int, "Store": int, "Customers": int, 
             "Open": int, "Promo": int, "SchoolHoliday": str, "CompetitionOpenSinceMonth": int,
             "CompetitionOpenSinceYear": int, "Promo2SinceWeek": int, "Promo2SinceYear": int}
	ext_data = ext_data.astype(col_types, copy=False)
	# Target Encoding for Store
	te = TargetEncoder(cols=["Store"])
	te.fit(ext_data["Store"], ext_data["Customers"])
	ext_data["Store_tgt_enc"] = te.transform(ext_data["Store"])

	# Target Encoding for DayOfWeek
	te = TargetEncoder(cols=["DayOfWeek"])
	te.fit(ext_data["DayOfWeek"], ext_data["Customers"])
	ext_data["DayOfWeek_tgt_enc"] = te.transform(ext_data["DayOfWeek"])
	ext_data.reset_index(drop=True, inplace=True)

	return ext_data





