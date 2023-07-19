#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: utils.py
# Created Date: 2022-12-06
# Author: Leslie Xu
# Contact: <lesliexufdu@163.com>
#
# Last Modified: 2023-07-19 11:03:35
#
# 综合utils中的模块后的辅助函数
# -----------------------------------
# HISTORY:
###


import os
import pandas as pd
from utils.features_split_process import feature_split_bins
from utils.features_calculate_process import bins2woe


def feature_split_woe(project_path, project_config, selected_feature, if_woe=True):
    '''读入输入训练数据、特征分裂后根据分裂节点计算WOE表格'''
    train_data_path = os.path.join(project_path,"train_data.csv")
    if_categorical = selected_feature in project_config["categorical features"]
    df = pd.read_csv(
        train_data_path,
        sep="\t",
        usecols=[selected_feature,project_config['target']],
        dtype={selected_feature:str} if if_categorical else None
    )
    split_config = project_config["split config"].get(selected_feature, None)
    split_result = feature_split_bins(
        df[selected_feature],
        df[project_config['target']],
        if_categorical,
        split_config
    )
    if if_woe:
        woe_data = bins2woe(
            df[selected_feature],
            df[project_config['target']],
            if_categorical,
            split_result
        )
        return split_result,woe_data
    return split_result


def feature_woe(project_path, project_config, selected_feature, bins):
    '''读入输入训练数据并根据分裂节点计算WOE表格'''
    train_data_path = os.path.join(project_path,"train_data.csv")
    if_categorical = selected_feature in project_config["categorical features"]
    df = pd.read_csv(
        train_data_path,
        sep="\t",
        usecols=[selected_feature,project_config['target']]
    )
    woe_data = bins2woe(
        df[selected_feature],
        df[project_config['target']],
        if_categorical,
        bins
    )
    return woe_data