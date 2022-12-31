#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: features_process.py
# Created Date: 2022-10-28
# Author: Leslie Xu
# Contact: <lesliexufdu@163.com>
#
# Last Modified: 2022-12-15 05:37:15
#
# preprocess the features that will be used in the model
# -----------------------------------
# HISTORY:
###


import numpy as np
import pandas as pd
import re
from sklearn import tree


def feature_split_bins(
    feature,
    target,
    if_categorical,
    split_config={"split_method":"决策树","split_parameter":0.01}
):
    '''分裂自变量并找到自变量的分裂节点'''
    if split_config is None:
        split_config = {"split_method":"决策树","split_parameter":0.01}
    if if_categorical:
        return categorical_values(feature)
    elif split_config['split_method']=='决策树':
        return tree_split(feature, target, min_samples_leaf=split_config['split_parameter'])
    elif split_config['split_method']=='等距':
        return equal_width_split(feature, nbins=split_config['split_parameter'])
    elif split_config['split_method']=='分位数':
        return quantile_split(feature, nbins=split_config['split_parameter'])
    else:
        return None


def tree_parser(clf):
    '''parse the split thresholds by DecisionTreeClassifier instance'''
    tree_info = tree.export_graphviz(clf, out_file=None)
    tree_info = tree_info.split('\n')
    split_values = []
    pattern = re.compile(r'X\[0\] <= (-?\d+\.?\d*)')
    for i in tree_info:
        mm = pattern.search(i)
        if mm:
            split_values.append(float(mm.groups()[0]))
    return sorted(list(set(split_values)))


def tree_split(
    data,
    target,
    min_samples_leaf = 0.01,
    max_depth = None,
    criterion = 'gini'
):
    '''
    返回决策树分裂的节点列表
    Args:
        data (pandas.Series, 1-D numpy.ndarray):
            输入的单特征数据
        target (pandas.Series, 1-D numpy.ndarray):
            The 1-D target values. Each value should be either 0 or 1.
        min_samples_leaf (int or float or dict):
            The min_samples_leaf parameter of sklearn.tree.DecisionTreeClassifier.
            If dict, each feature is splitted with a unique min_samples_leaf.
        max_depth (int):
            The max_depth parameter of sklearn.tree.DecisionTreeClassifier.
        criterion (str):
            The criterion parameter of sklearn.tree.DecisionTreeClassifier.
    Returns:
        The split thresholds, like `[threshold00, threshold01...]`
    '''
    clf = tree.DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
    )
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    clf.fit(data.to_frame(), target)
    return tree_parser(clf)


def quantile_split(data, nbins = 10):
    '''
    返回分位数分裂的节点列表
    Args:
        data (pandas.Series, numpy.ndarray):
            输入的单特征数据
        nbins (int):
            分位数分段数,如果有重复值则合并
    Returns:
        The split thresholds, like `[threshold00, threshold01...]`
    '''
    _,bins = pd.qcut(data, nbins, retbins=True, duplicates='drop')
    if bins.shape[0]<=2:
        return []
    else:
        return bins[1:-1].round(4).tolist()


def equal_width_split(data, nbins = 10):
    '''
    返回等间距分裂的节点列表
    Args:
        data (pandas.Series, numpy.ndarray):
            输入的单特征数据
        nbins (int):
            分位数分段数,如果有重复值则合并
    Returns:
        The split thresholds, like `[threshold00, threshold01...]`
    '''
    _,bins = pd.cut(data, nbins, retbins=True, duplicates='drop')
    if bins.shape[0]<=2:
        return []
    else:
        return bins[1:-1].round(4).tolist()


def categorical_values(data):
    '''为分类变量寻找分立值'''
    return [ [i] for i in data.unique() ]