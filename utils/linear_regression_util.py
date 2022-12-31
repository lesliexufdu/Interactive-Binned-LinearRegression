#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: linear_regression.py
# Created Date: 2022-12-19
# Author: Leslie Xu
# Contact: <lesliexufdu@163.com>
#
# Last Modified: 2022-12-19 06:59:18
#
# Eutopias for Euphoria.
# -----------------------------------
# HISTORY:
###


import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp


def categorical_feature_str(x, bins):
    '''将分类变量的值映射成字符串'''
    for i in bins:
        if x in i:
            return tuple(i)


def feature2woe(feature, bins, woe_table, if_categorical=True):
	'''将特征series转换成WOE值'''
	woe_map = {tuple(i["值域"]):i["WOE"] for i in woe_table}
	if if_categorical: # 分类型变量
		woe_data = feature.apply(categorical_feature_str, bins=bins).map(woe_map).to_numpy()
	else:
		cut_bins = [-np.inf,*bins,np.inf]
		cut_labels = [("-",bins[0])] + [(i,j) for i,j in zip(bins[:-1],bins[1:])] + [(bins[-1],"+")]
		woe_data = pd.cut(feature, bins=cut_bins, labels=cut_labels, duplicates='drop').map(woe_map).to_numpy()
	return woe_data


def data_prepare(project_path, project_config, used_features, features_config, mode="train"):
	'''读取数据并处理成WOE形式'''
	data_path = os.path.join(project_path,f"{mode}_data.csv")
	df = pd.read_csv(data_path, sep="\t", usecols=[*used_features,project_config['target']])
	df_return = {project_config['target']:df[project_config['target']].to_numpy()}
	for f in used_features:
		if_categorical = f in project_config["categorical features"]
		df_return[f] = feature2woe(df[f], features_config[f]['bins'], features_config[f]['woe'], if_categorical=if_categorical)
	return pd.DataFrame(df_return)


def linear_regression_build(project_path, project_config, used_features, features_config):
	'''根据分组数据逻辑回归,返回LR类'''
	data_to_train = data_prepare(project_path, project_config, used_features, features_config, mode="train")
	clf = LogisticRegression(solver='saga', n_jobs=10)
	clf.fit(data_to_train[used_features], data_to_train[project_config['target']])
	return clf


def model_evaluate(clf, project_path, project_config, used_features, features_config):
	'''根据训练的LR类评估各数据集上的表现'''
	if "oot_data.csv" in os.listdir(project_path):
		datasets= ["train","test","oot"]
	else:
		datasets= ["train","test"]
	evaluate_result = []
	for data_mode in datasets:
		data_to_evaluate = data_prepare(project_path, project_config, used_features, features_config, mode=data_mode)
		target = data_to_evaluate[project_config['target']].to_numpy()
		pred = clf.predict_proba(data_to_evaluate[used_features])[:,1]
		KS = ks_2samp(pred[target==0], pred[target==1])[0] * 100
		auc = roc_auc_score(target, pred)
		evaluate_result.append({"数据集":data_mode,"KS":KS,"AUC":auc})
	return evaluate_result