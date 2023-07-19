#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: data_util.py
# Created Date: 2022-11-10
# Author: Leslie Xu
# Contact: <lesliexufdu@163.com>
#
# Last Modified: 2023-07-19 03:06:43
#
# 项目创建页面的数据处理工具
# -----------------------------------
# HISTORY:
###

import base64
import io
import os
import hjson
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_DIR = "./datasets"
PROJECT_DIR = "./projects"


def parse_contents(content):
    '''解析上传的文件'''
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    return decoded.decode('utf-8')


def save_dataset(dataset_contents, delimeter, dataset_path):
    '''根据上传的文件和分隔符和指定路径保留数据集'''
    data_content = pd.read_csv(io.StringIO(parse_contents(dataset_contents)), sep=delimeter)
    for col in data_content.columns:
        if data_content[col].dtype=="O": ## 数据中的tab字符被转换以避免错乱
            data_content[col] = data_content[col].fillna("").str.replace("\t+", " ", regex=True)
    data_content.to_csv(dataset_path, sep="\t", index=False)


def project_write(df, y, columns, categorical_columns, project_path, df_oot=None, missing_value_config=None, split_config=None, test_size=0.2):
    '''根据指定信息生成写入project'''
    if missing_value_config:
        df = df.fillna(missing_value_config)
        if df_oot is not None:
            df_oot = df_oot.fillna(missing_value_config)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    project_traindata_path = os.path.join(project_path,"train_data.csv")
    project_testdata_path = os.path.join(project_path,"test_data.csv")
    train_data,test_data = train_test_split(df, test_size=test_size, stratify=df[y])
    ## 写入项目数据
    train_data.to_csv(project_traindata_path, sep="\t", index=None)
    test_data.to_csv(project_testdata_path, sep="\t", index=None)
    ## 写入项目配置
    project_config_path = os.path.join(project_path,"config.hjson")
    with open(project_config_path,'w',encoding="utf-8") as f:
        hjson.dump({
            "target":y,
            "train data samples":train_data.shape[0],
            "train data positive samples":int(train_data[y].sum()),
            "train data positive rate":float(train_data[y].mean()),
            "features":columns,
            "categorical features":categorical_columns,
            "split config":split_config
        }, f)
    ## 如果存在,写入样本外数据
    if df_oot is not None:
        project_ootdata_path = os.path.join(project_path,"oot_data.csv")
        df_oot.to_csv(project_ootdata_path, sep="\t", index=None)