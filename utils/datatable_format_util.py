#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: datatable_format_util.py
# Created Date: 2022-11-13
# Author: Leslie Xu
# Contact: <lesliexufdu@163.com>
#
# Last Modified: 2023-07-19 11:26:09
#
# 输出合适的数据格式
# -----------------------------------
# HISTORY:
###


import numpy as np


def data_bars_diverging(column, column_max=1, column_min=-1, n_bins=100, color_above='#3D9970', color_below='#FF4136'):
    '''生成条件格式databar'''
    column_max = np.ceil(column_max*10**6)/10**6
    column_min = np.floor(column_min*10**6)/10**6
    ranges = [
        (
            (column_max-column_min)*i/n_bins+column_min,
            (column_max-column_min)*(i+1)/n_bins+column_min,
            i,
            i+1
        ) for i in range(n_bins)
    ] # 将列的值从最小值到最大值等距离分成100份,计算上下限和上下限百分比
    styles = []
    mid_line = np.clip(1-column_max/(column_max-column_min),0,1) # data bar的中线位置,随上下限而变动
    for min_bound,max_bound,min_bound_percentage,max_bound_percentage in ranges:
        if min_bound==column_min:
            filter_query = f'{{{column}}} < {max_bound}'
        elif max_bound==column_max:
            filter_query = f'{{{column}}} >= {min_bound}'
        else:
            filter_query = f'{{{column}}} >= {min_bound}' + " && " + f'{{{column}}} < {max_bound}'
        style = {
            'if': {
                'filter_query': filter_query,
                'column_id': column
            },
            'paddingBottom': 2,
            'paddingTop': 2
        }
        if max_bound > 0: # midpoint之上的值,从中线以后填充颜色
            if mid_line>0:
                background = f"""
                    linear-gradient(90deg,
                    white 0%,
                    white {mid_line:.0%},
                    {color_above} {mid_line:.0%},
                    {color_above} {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """
            else:
                background = f"""
                    linear-gradient(90deg,
                    {color_above} 0%,
                    {color_above} {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """
        else: # midpoint之下的值,从中线往左填充颜色
            if mid_line<1:
                background = f"""
                    linear-gradient(90deg,
                    white 0%,
                    white {min_bound_percentage}%,
                    {color_below} {min_bound_percentage}%,
                    {color_below} {mid_line:.0%},
                    white {mid_line:.0%},
                    white 100%)
                """
            else:
                background = f"""
                    linear-gradient(90deg,
                    white 0%,
                    white {min_bound_percentage}%,
                    {color_below} {min_bound_percentage}%,
                    {color_below} 100%)
                """
        style['background'] = background
        styles.append(style)
    return styles