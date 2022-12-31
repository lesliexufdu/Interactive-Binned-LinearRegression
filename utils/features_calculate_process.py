#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: features_calculate_process.py
# Created Date: 2022-12-03
# Author: Leslie Xu
# Contact: <lesliexufdu@163.com>
#
# Last Modified: 2022-12-19 06:58:38
#
# Eutopias for Euphoria.
# -----------------------------------
# HISTORY:
###


import numpy as np
import pandas as pd


def categorical_feature_str(x, bins):
    '''将分类变量的值映射成字符串'''
    for i in bins:
        if x in i:
            return tuple(i)


def woe_value(total_positive, total_negative, positive, negative):
    '''计算单一分组的WOE值'''
    return np.log((negative+np.finfo(float).eps)/(positive+np.finfo(float).eps))+\
        np.log(total_positive/total_negative)


def bins2woe(
    feature,
    target,
    if_categorical,
    bins
):
    '''根据bins计算分组的WOE值'''
    positive_total = target.sum()
    negative_total = target.shape[0]-positive_total
    if if_categorical: # 分类型变量
        df = pd.DataFrame({
            "值域":feature.apply(categorical_feature_str, bins=bins),
            "target":target
        })
    else:
        cut_bins = [-np.inf,*bins,np.inf]
        cut_labels = [("-",bins[0])] + [(i,j) for i,j in zip(bins[:-1],bins[1:])] + [(bins[-1],"+")]
        df = pd.DataFrame({
            "值域":pd.cut(feature, bins=cut_bins, labels=cut_labels, duplicates='drop'),
            "target":target
        })
    # 计算各统计指标
    grouped_data = df.groupby('值域')['target'].agg(['count','sum','mean']).rename(columns={'count':'样本数','sum':'正样本数','mean':'正样本率'}).reset_index()
    grouped_data['分组占比'] = grouped_data['样本数']/target.shape[0]
    grouped_data['WOE'] = (
        np.log((grouped_data['样本数']-grouped_data['正样本数']+np.finfo(float).eps)/(grouped_data['正样本数']+np.finfo(float).eps))
        +
        np.log(positive_total/negative_total)
    )
    # 子分组
    grouped_data['子分组'] = [[i] for i in grouped_data['值域'].tolist()]
    grouped_data['子分组数'] = 1
    # 用于展示的range
    if if_categorical:
        grouped_data['range'] = grouped_data['值域'].map(lambda x:",".join(x))
        grouped_data = grouped_data.sort_values('WOE')
    else:
        grouped_data['range'] = grouped_data['值域'].map(lambda x:f"({x[0]},{x[1]})" if x[1]=="+" else f"({x[0]},{x[1]}]")
    return grouped_data.to_dict('records')


def woe_table_merge(woe_table, rows_to_merge, total_positive, total_negative, if_categorical=False):
    '''根据原有WOE表格和要合并的组生成新的WOE表格'''
    merged_row = {'样本数':0,'正样本数':0,'值域':[],'子分组':[],'子分组数':0}
    unmerged_rows = []
    if not if_categorical:
        min_row = min(rows_to_merge)
        max_row = max(rows_to_merge)
        rows_to_merge = list(range(min_row,max_row+1))
    # 归并数据
    for idx,row in enumerate(woe_table):
        if idx in rows_to_merge:
            merged_row['样本数'] += row['样本数']
            merged_row['正样本数'] += row['正样本数']
            merged_row['值域'] += list(row['值域'])
            merged_row['子分组'] += list(row['子分组'])
            merged_row['子分组数'] += row['子分组数']
        else:
            unmerged_rows.append(row)
    # 处理归并后的值域
    if if_categorical:
        merged_row['range'] = ",".join(merged_row['值域'])
        merged_row['值域'] = tuple(merged_row['值域'])
    else:
        merged_row['值域'] = (merged_row['值域'][0],merged_row['值域'][-1])
        if merged_row['值域'][-1]=="+":
            merged_row['range'] = f"({merged_row['值域'][0]},{merged_row['值域'][-1]})"
        else:
            merged_row['range'] = f"({merged_row['值域'][0]},{merged_row['值域'][-1]}]"
    # 计算其他指标
    merged_row['正样本率'] = merged_row['正样本数']/merged_row['样本数']
    merged_row['分组占比'] = merged_row['样本数']/(total_positive+total_negative)
    merged_row['WOE'] = woe_value(
        total_positive,
        total_negative,
        merged_row['正样本数'],
        merged_row['样本数'] - merged_row['正样本数']
    )
    # 返回
    if if_categorical:
        woe_return = sorted(unmerged_rows+[merged_row], key=lambda x:x['WOE'])
        merged_bins = [list(i['值域']) for i in woe_return]
    else:
        woe_return = sorted(unmerged_rows+[merged_row], key=lambda x:-np.inf if x['值域'][0]=='-' else x['值域'][0])
        merged_bins = sorted([i['值域'][0] for i in woe_return if i['值域'][0]!="-"])
    return woe_return,merged_bins


def woe_table_unmerge(woe_table, rows_to_unmerge, original_woe_table, if_categorical=False):
    '''根据原有WOE表格和要展开的组生成新的WOE表格'''
    unmerged_rows = []
    for idx,row in enumerate(woe_table):
        if idx in rows_to_unmerge and row["子分组数"]>1:
            child_groups = [list(i) for i in row['子分组']]
            unmerged_rows += [i for i in original_woe_table if i['值域'] in child_groups]
        else:
            unmerged_rows.append(row)
    # 返回
    if if_categorical:
        woe_return = sorted(unmerged_rows, key=lambda x:x['WOE'])
        unmerged_bins = [list(i['值域']) for i in woe_return]
    else:
        woe_return = sorted(unmerged_rows, key=lambda x:-np.inf if x['值域'][0]=='-' else x['值域'][0])
        unmerged_bins = sorted([j for i in woe_return for j in i['值域'] if j not in ("-","+")])
    return woe_return,unmerged_bins


def check_bin_if_merged(bin, original_bins, if_categorical=True):
    '''检测当前分组是否是合并过的分组'''
    bins_return = []
    if if_categorical:
        for i in original_bins:
            if set(i)&set(bin)==set(i):
                bins_return.append(i)
                bin = [j for j in bin if j not in i]
                if bin==[]:
                    break
    else:
        original_bins_extend = [-np.inf,*original_bins,np.inf]
        left_bin = bin[0] if bin[0]!='-' else -np.inf
        right_bin = bin[1] if bin[1]!='+' else np.inf
        for i,j in zip(original_bins_extend[:-1],original_bins_extend[1:]):
            if i<left_bin<j or i<right_bin<j:
                break
            if left_bin<=i and right_bin>=j:
                i_map = i if i!=-np.inf else '-'
                j_map = j if j!=np.inf else '+'
                bins_return.append((i_map,j_map))
    return tuple(bins_return),len(bins_return)


def woe_subbins_calculate(woe_data, original_bins, if_categorical=True):
    '''根据original_bins重新计算当前WOE数据的子分组'''
    woe_data_return = []
    for i in woe_data:
        sub_bins,sub_bins_count = check_bin_if_merged(i['值域'],original_bins,if_categorical)
        woe_data_return.append({**i,"子分组":sub_bins,"子分组数":sub_bins_count})
    return woe_data_return


def bins_merge_indices_calculate(bins, original_bins):
    '''根据bins和original bins计算出哪些组需要合并,仅仅对数值型变量的分组有效'''
    start_position = 0
    merged_indices = []
    for idx,b in enumerate(original_bins):
        if b in bins:
            if idx!=start_position:
                merged_indices.append((start_position,idx))
            start_position = idx+1
    if b not in bins:
        merged_indices.append((start_position,idx+1))
    return merged_indices