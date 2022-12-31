#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: model_report_util.py
# Created Date: 2022-12-22
# Author: Leslie Xu
# Contact: <lesliexufdu@163.com>
#
# Last Modified: 2022-12-30 05:49:57
#
# Eutopias for Euphoria.
# -----------------------------------
# HISTORY:
###


import os
import hjson
from scipy.stats import ks_2samp
from scipy.special import expit
import numpy as np,pandas as pd
from sklearn.metrics import roc_curve,auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def modelconfig2sqlscore(
    features_config,
    model_parameters,
    P,
    Q,
    PDO,
    negative_weight,
    min_score,
    max_score
):
    '''根据指定的模型配置输出SQL字符串'''
    B = -PDO/np.log(2)
    A = P + B*np.log(Q) - B*np.log(negative_weight)
    intercept = (A + B*model_parameters['intercept_'][0])/model_parameters['n_features_in_']
    sql_strs_result = []
    for feature,feature_coef in zip(model_parameters['feature_names_in_'],model_parameters['coef_'][0]):
        sql_str = "case "
        if isinstance(features_config[feature]['bins'][0],list):
            for woe_config in features_config[feature]['woe']:
                score_increment = round(intercept + B*woe_config['WOE']*feature_coef,2)
                feature_values_str = "','".join(woe_config['值域'])
                sql_str += f"when {feature} in ('{feature_values_str}') then {score_increment}\n"
        else:
            for woe_config in features_config[feature]['woe']:
                score_increment = round(intercept + B*woe_config['WOE']*feature_coef,2)
                if woe_config['值域'][1]=='+':
                    sql_str += f"when {feature}>{woe_config['值域'][0]} then {score_increment}\n"
                else:
                    sql_str += f"when {feature}<={woe_config['值域'][1]} then {score_increment}\n"
        sql_str += "else 0 end"
        sql_strs_result.append(sql_str)
    return "greatest(least(" + "\n+\n".join(sql_strs_result)+f",{max_score}),{min_score})"


def categorical_feature_str(x, bins):
    '''将分类变量的值映射成字符串'''
    for i in bins:
        if x in i:
            return ",".join(i)


def feature_transform(df, model_config, if_woe_calculate=True):
    '''将自变量转换成分组自变量和WOE值'''
    features_grouped = {}
    if if_woe_calculate: features_woe = {}
    for i in model_config['model parameters']['feature_names_in_']:
        bins = model_config['used features'][i]['bins']
        woe_map = {i['range']:i['WOE'] for i in model_config['used features'][i]['woe']}
        if isinstance(bins[0],list):
            categorical_order = pd.Series(woe_map).sort_values().index.tolist()
            features_grouped[i] = df[i].apply(categorical_feature_str, bins=bins).astype("category").cat.set_categories(categorical_order)
        else:
            cut_bins = [-np.inf,*bins,np.inf]
            cut_labels = [f"(-,{bins[0]}]"] + [f"({i},{j}]" for i,j in zip(bins[:-1],bins[1:])] + [f"({bins[-1]},+)"]
            features_grouped[i] = pd.cut(df[i], bins=cut_bins, labels=cut_labels, duplicates='drop')
        if if_woe_calculate: features_woe[i] = features_grouped[i].map(woe_map)
    if if_woe_calculate:
        return features_grouped,features_woe
    return features_grouped


def probability_calculate(input_features_array, model_parameters):
    '''根据输入特征array和模型参数,计算logit和probability'''
    coef_array = np.array(model_parameters['coef_'])
    logits = (input_features_array*coef_array).sum(axis=1) + model_parameters['intercept_']
    proba = expit(logits)
    return logits,proba


def card_score(pred, P=660, Q=20, PDO=50, negative_weight=1, min_score=200, max_score=1000):
    '''
    根据概率1返回卡评分
        pred:线性预测值logit(array或scalar)
        Q:设定的Odds比率
        P:比率Q时的分数
        PDO:比率为2Q时分值增加量
        negative_weight:正样本抽样比例
    '''
    B = -PDO/np.log(2)
    A = P + B*np.log(Q)
    Odds = -pred
    score = A - B*(Odds + np.log(negative_weight))
    return np.clip(score.round(2), a_min=min_score, a_max=max_score)


def read_raw_data(data_path, target_name, features_name):
    '''读取需要的数据集'''
    df = pd.read_csv(data_path, sep="\t", usecols=[target_name,*features_name])
    return df


def read_transform_score(
    data_path,
    features,
    target,
    model_config,
    P=660,
    Q=20,
    PDO=50,
    negative_weight=1,
    min_score=0,
    max_score=1000,
    if_score=True
):
    '''
    读取数据、特征转换、打分的全过程
    入参:
        data_path: csv文件所在路径
        features: 特征列表
        target: Y变量名称
        model_config: 模型配置
        P: 评分卡配置
        Q: 评分卡配置
        PDO: 评分卡配置
        negative_weight: 评分卡配置
        min_score: 评分卡配置
        max_score: 评分卡配置
        if_score: True时返回(分组特征字典,y值array,预测概率,预测评分),False则返回(分组特征字典,y值array)
    '''
    df = read_raw_data(data_path, target, features)
    if if_score:
        features_grouped,features_woe = feature_transform(df, model_config)
        logits,proba = probability_calculate(
            pd.DataFrame(features_woe).values,
            model_config['model parameters']
        )
        scores = card_score(
            logits,
            P=P,
            Q=Q,
            PDO=PDO,
            negative_weight=negative_weight,
            min_score=min_score,
            max_score=max_score
        )
        return features_grouped,df[target].values,proba,scores
    else:
        features_grouped = feature_transform(df, model_config, if_woe_calculate=False)
        return features_grouped,df[target].values


def grouped_variable_summary(feature_group, target):
    '''按分段变量汇总成统计表'''
    df_group = pd.DataFrame({'分数区间':feature_group,'target':target})
    # 计算分组0/1数量
    result = df_group.groupby('分数区间')['target'].agg(['count','sum','mean']).rename(columns={'count':'样本数','sum':'正样本数','mean':'正样本率'})
    result['负样本数'] = result['样本数'] - result['正样本数']
    # 总的0/1数量
    result['总正样本数'] = result['正样本数'].sum()
    result['总负样本数'] = result['负样本数'].sum()
    # 分组0/1数量占总的0/1数量比例
    result['正样本比例'] = result['正样本数']/result['总正样本数']
    result['负样本比例'] = result['负样本数']/result['总负样本数']
    # 分组0/1数量累计占总的0/1数量比例
    result['累计正样本比例'] = result['正样本数'].cumsum()/result['总正样本数']
    result['累计负样本比例'] = result['负样本数'].cumsum()/result['总负样本数']
    # 分组占比
    result['样本占比'] = result['样本数']/result['样本数'].sum()
    return result.reset_index()


def generate_statistic_report(stats_result):
    stats_result = pd.DataFrame(stats_result)
    # 生成表格
    fig = go.Figure(
        data=go.Table(
            header={
                "values":["<b>数据集</b>",'<b>样本数</b>','<b>正样本数</b>','<b>正样本率</b>'],
                "fill_color":'#70AD47',
                "align":'left',
                "font":{"color":'white'},
                "height":30
            },
            cells={
                "values":[
                    stats_result['dataset'],
                    stats_result['样本数'],
                    stats_result['正样本数'],
                    stats_result['正样本率']
                ],
                "align":'left',
                "height":30
            }
        ),
        layout=go.Layout(
            height=30*stats_result.shape[0]+50,
            margin=dict(l=10,r=10,b=10,t=10)
        )
    )
    return fig


def generate_auc_plot(auc_figure_data):
    fig = make_subplots(
        rows=1, cols=len(auc_figure_data),
        subplot_titles=[
            "AUC {auc:.4f} for {data} set".format(auc=auc(i['FPR'],i['TPR']),data=i['data name'])
            for i in auc_figure_data
        ],
    )
    for idx,figure_data in enumerate(auc_figure_data, start=1):
        fig.add_trace(
            go.Scatter(
                x=figure_data['FPR'],
                y=figure_data['TPR'],
                marker={"color":'#0099ff',"symbol":"square"},
                mode="lines+markers"
            ),
            row=1, col=idx
        )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="False Positive Rate")
    fig.update_yaxes(title_text="True Positive Rate")
    return fig


def generate_pr_plot(pr_figure_data):
    table_fig = make_subplots(
        rows=1, cols=len(pr_figure_data),
        specs=[[{"type": "table"}]*len(pr_figure_data)],
        subplot_titles=[i['data name'] for i in pr_figure_data],
        horizontal_spacing=0.1/len(pr_figure_data)
    )
    plot_fig = make_subplots(
        rows=1, cols=len(pr_figure_data),
        subplot_titles=[i['data name'] for i in pr_figure_data]
    )
    for idx,figure_data in enumerate(pr_figure_data, start=1):
        table_fig.add_trace(
            go.Table(
                header={
                    "values":["<b>分数</b>",'<b>样本数</b>','<b>正样本数</b>','<b>正样本率</b>','<b>累计正样本比例</b>','<b>累计负样本比例</b>'],
                    "fill_color":'#70AD47',
                    "align":'left',
                    "font":{"color":'white'},
                    "height":30
                },
                cells={
                    "values":[
                        figure_data['data']['分数区间'],
                        figure_data['data']['样本数'],
                        figure_data['data']['正样本数'],
                        figure_data['data']['正样本率'].map(lambda x:f"{x:.2%}"),
                        figure_data['data']['累计正样本比例'].map(lambda x:f"{x:.2%}"),
                        figure_data['data']['累计负样本比例'].map(lambda x:f"{x:.2%}")
                    ],
                    "align":'left',
                    "height":30
                }
            ),
            row=1, col=idx
        )
        plot_fig.add_trace(
            go.Scatter(
                x=figure_data['data']['分数区间'],
                y=figure_data['data']['正样本率'],
                marker={"color":'#0099ff',"symbol":"square"},
                mode="lines+markers"
            ),
            row=1, col=idx
        )
    plot_fig.update_layout(
        yaxis={'tickformat':".0%"},
        showlegend=False,
        margin=dict(l=10,r=10,b=10,t=30)
    )
    plot_fig.update_xaxes(title_text="分数区间")
    plot_fig.update_yaxes(title_text="正样本率")
    table_fig.update_layout(
        height=30*figure_data['data'].shape[0]+70,
        margin=dict(l=10,r=10,b=10,t=30)
    )
    return [table_fig,plot_fig]


def generate_KS_plot(KS_figure_data):
    fig = make_subplots(
        rows=1, cols=len(KS_figure_data),
        subplot_titles=[
            "KS {KS:.2f} for {data} set".format(KS=i['KS']*100,data=i['data name'])
            for i in KS_figure_data
        ]
    )
    for idx,figure_data in enumerate(KS_figure_data, start=1):
        if_showlegend = True if idx==1 else False
        fig.add_trace(
            go.Scatter(
                x=figure_data['data']['分数区间'],
                y=figure_data['data']['累计正样本比例'],
                marker={"color":'#0099ff',"symbol":"square"},
                mode="lines+markers",
                name='正样本比例',
                legendgroup="正样本比例",
                showlegend=if_showlegend
            ),
            row=1, col=idx
        )
        fig.add_trace(
            go.Scatter(
                x=figure_data['data']['分数区间'],
                y=figure_data['data']['累计负样本比例'],
                marker={"color":'#404040',"symbol":"square"},
                mode="lines+markers",
                name='负样本比例',
                legendgroup="负样本比例",
                showlegend=if_showlegend
            ),
            row=1, col=idx
        )
    fig.update_layout(yaxis={'tickformat':".0%"},showlegend=True,margin=dict(l=10,r=10,b=10,t=30))
    fig.update_xaxes(title_text="分数区间")
    fig.update_yaxes(title_text="累计占比")
    return fig


def generate_sample_distribution_plot(sample_distribution_data):
    table_fig = make_subplots(
        rows=1,
        cols=len(sample_distribution_data),
        specs=[[{"type":"table"}]*len(sample_distribution_data)],
        subplot_titles=[i['data name'] for i in sample_distribution_data],
        horizontal_spacing=0.1/len(sample_distribution_data)
    )
    plot_fig = make_subplots(
        rows=1,
        cols=len(sample_distribution_data),
        subplot_titles=[i['data name'] for i in sample_distribution_data]
    )
    for idx,figure_data in enumerate(sample_distribution_data, start=1):
        if_showlegend = True if idx==1 else False
        table_fig.add_trace(
            go.Table(
                header={
                    "values":["<b>分数</b>",'<b>样本数</b>','<b>正样本数</b>','<b>正样本率</b>','<b>累计正样本比例</b>','<b>累计负样本比例</b>'],
                    "fill_color":'#70AD47',
                    "align":'left',
                    "font":{"color":'white'},
                    "height":30
                },
                cells={
                    "values":[
                        figure_data['data']['分数区间'],
                        figure_data['data']['样本数'],
                        figure_data['data']['正样本数'],
                        figure_data['data']['正样本率'].map(lambda x:f"{x:.2%}"),
                        figure_data['data']['累计正样本比例'].map(lambda x:f"{x:.2%}"),
                        figure_data['data']['累计负样本比例'].map(lambda x:f"{x:.2%}")
                    ],
                    "height":30,
                    "align":'left'
                }
            ),
            row=1, col=idx
        )
        plot_fig.add_trace(
            go.Bar(
                x=figure_data['data']['分数区间'],
                y=figure_data['data']['正样本比例'],
                name='正样本',
                legendgroup="正样本",
                marker_color='rgb(26, 118, 255)',
                showlegend=if_showlegend
            ),
            row=1, col=idx
        )
        plot_fig.add_trace(
            go.Bar(
                x=figure_data['data']['分数区间'],
                y=figure_data['data']['负样本比例'],
                name='负样本',
                legendgroup="负样本",
                marker_color='crimson',
                showlegend=if_showlegend
            ),
            row=1, col=idx
        )
        plot_fig.add_trace(
            go.Scatter(
                x=figure_data['data']['分数区间'],
                y=figure_data['data']['样本占比'],
                marker={"color":'#404040',"symbol":"square"},
                mode="lines+markers",
                name='样本占比',
                legendgroup="样本占比",
                showlegend=if_showlegend
            ),
            row=1, col=idx
        )
    plot_fig.update_layout(
        barmode='group',
        yaxis={'tickformat':".0%"},
        margin=dict(l=10,r=10,b=10,t=30)
    )
    plot_fig.update_xaxes(title_text="分数区间")
    plot_fig.update_yaxes(title_text="样本占比")
    table_fig.update_layout(
        height=30*sample_distribution_data[0]['data'].shape[0]+70,
        margin=dict(l=10,r=10,b=10,t=30)
    )
    return [table_fig,plot_fig]


def generate_features_distribution_plot(features_distribution_data):
    features_distribution_data_trans = [i for i in zip(*features_distribution_data)] # 转换成[feature1列表,feature2列表...]
    fig = make_subplots(
        rows=len(features_distribution_data_trans),
        cols=len(features_distribution_data_trans[0]),
        subplot_titles=[
            "{feature} {data}".format(feature=j['feature'],data=j['data name'])
            for i in features_distribution_data_trans for j in i
        ],
        specs=[[{"secondary_y":True} for j in i] for i in features_distribution_data_trans]
    )
    for row_idx,feature_data_row in enumerate(features_distribution_data_trans, start=1):
        for col_idx,feature_data_col in enumerate(feature_data_row, start=1):
            if_showlegend = True if row_idx+col_idx==2 else False
            fig.add_trace(
                go.Bar(
                    name='样本占比(左轴)',
                    x=feature_data_col['statistics']['分数区间'],
                    y=feature_data_col['statistics']['样本占比'],
                    legendgroup="样本占比",
                    marker_color='rgb(26, 118, 255)',
                    showlegend=if_showlegend
                ),
                secondary_y=False,
                row=row_idx, col=col_idx
            )
            fig.add_trace(
                go.Scatter(
                    x=feature_data_col['statistics']['分数区间'],
                    y=feature_data_col['statistics']['正样本率'],
                    marker={"color":'#0099ff',"symbol":"square"},
                    mode="lines+markers",
                    name='正样本率(右轴)',
                    legendgroup="正样本率",
                    showlegend=if_showlegend
                ),
                secondary_y=True,
                row=row_idx, col=col_idx
            )
    fig.update_layout(height=400*len(features_distribution_data_trans)+20)
    fig.update_yaxes(tickformat=".0%", secondary_y=True)
    fig.update_yaxes(tickformat=".0%", secondary_y=False)
    return fig


def generate_model_evaluation_report(
    project_path,
    model_config,
    P=660,
    Q=20,
    PDO=50,
    negative_weight=1,
    min_score=0,
    max_score=1000,
    quantile_bins_num=10,
    equalscore_bins_num=10,
    if_features_distribution=True,
    if_score_evaluation=True
):
    '''
    生成模型评价图标
    if_feature_distribution: True时会返回变量分布,False则不会
    if_score_evaluation: True时会返回模型整体评价,False则不会
    '''

    # 项目配置
    with open(os.path.join(project_path,"config.hjson"),"r") as f:
        project_config = hjson.load(f)
    target = project_config['target']
    features = model_config['model parameters']['feature_names_in_']

    # 读取可用数据
    all_datas = [i for i in os.listdir(project_path) if i.endswith('_data.csv')]
    all_datas = [i for i in ["train_data.csv","test_data.csv","oot_data.csv"] if i in all_datas]
    # 待返回数据
    if if_score_evaluation: # 返回整体分数评价
        statistic_report_data = []
        auc_figure_data = []
        KS_figure_data = []
        sample_distribution_data = []
    if if_features_distribution: # 返回变量分布
        features_distribution_data = []

    # 计算制图数据
    for idx,data_name in enumerate(all_datas):
        data_path = os.path.join(project_path, data_name)
        if if_score_evaluation: # 返回整体分数评价
            features_grouped,target_array,proba,scores = read_transform_score(
                data_path,
                features,
                target,
                model_config,
                P=P,
                Q=Q,
                PDO=PDO,
                negative_weight=negative_weight,
                min_score=min_score,
                max_score=max_score,
                if_score=True
            )
            ## 描述性统计
            stats = {
                "dataset":data_name.split("_",maxsplit=1)[0],
                "样本数":target_array.shape[0],
                "正样本数":target_array.sum(),
            }
            stats["正样本率"] = f'{stats["正样本数"]/stats["样本数"]:.2%}'
            statistic_report_data.append(stats)
            ## auc
            fpr,tpr,thresholds = roc_curve(target_array, proba)
            auc_figure_data.append({
                "data name":data_name.split("_",maxsplit=1)[0],
                "FPR":fpr,
                "TPR":tpr
            })
            ## KS和PR数据
            if data_name=="train_data.csv":
                _,bins = pd.qcut(scores, q=quantile_bins_num, duplicates='drop', retbins=True)
                bins = bins.round(2)
                quantile_cut_parameter = {
                    "bins":[-np.inf]+bins[1:-1].tolist()+[np.inf],
                    "labels":[f"(-,{bins[1]})"] + [f"({i},{j})" for i,j in zip(bins[1:-2],bins[2:-1])] + [f"({bins[-2]},+)"]
                }
            ks_scores_bins = pd.cut(scores, duplicates='drop', precision=2, **quantile_cut_parameter)
            ks_scores_summary = grouped_variable_summary(ks_scores_bins, target_array)
            KS_figure_data.append({
                "data name":data_name.split("_",maxsplit=1)[0],
                "data":ks_scores_summary,
                "KS":ks_2samp(scores[target_array==0],scores[target_array==1])[0]
            })
            ## 样本分布
            if data_name=="train_data.csv":
                _,bins = pd.cut(scores, bins=equalscore_bins_num, duplicates='drop', retbins=True)
                bins = bins.round(2)
                equalscore_cut_parameter = {
                    "bins":[-np.inf]+bins[1:-1].tolist()+[np.inf],
                    "labels":[f"(-,{bins[1]})"] + [f"({i},{j})" for i,j in zip(bins[1:-2],bins[2:-1])] + [f"({bins[-2]},+)"]
                }
            equalscore_scores_bins = pd.cut(scores, duplicates='drop', precision=2, **equalscore_cut_parameter)
            equalscore_scores_summary = grouped_variable_summary(equalscore_scores_bins, target_array)
            sample_distribution_data.append({
                "data name":data_name.split("_",maxsplit=1)[0],
                "data":equalscore_scores_summary
            })
            ## 特征分布
            if if_features_distribution: # 返回变量分布
                features_distribution_data.append([
                    {
                        "feature":feature_name,
                        "data name": data_name.split("_",maxsplit=1)[0],
                        "statistics": grouped_variable_summary(feature_value,target_array)
                    }
                    for feature_name,feature_value in features_grouped.items()
                ])
        elif if_features_distribution: # 返回变量分布
            features_grouped,target_array = read_transform_score(
                data_path,
                features,
                target,
                model_config,
                P=P,
                Q=Q,
                PDO=PDO,
                negative_weight=negative_weight,
                min_score=min_score,
                max_score=max_score,
                if_score=False
            )
            features_distribution_data.append([
                {
                    "feature":feature_name,
                    "data name": data_name.split("_",maxsplit=1)[0],
                    "statistics": grouped_variable_summary(feature_value,target_array)
                }
                for feature_name,feature_value in features_grouped.items()
            ])

    # 返回图表
    if if_score_evaluation:
        statistic_report_return = generate_statistic_report(statistic_report_data)
        auc_figure_return = generate_auc_plot(auc_figure_data)
        pr_figure_return = generate_pr_plot(KS_figure_data)
        KS_figure_return = generate_KS_plot(KS_figure_data)
        sample_distribution_figure_return = generate_sample_distribution_plot(sample_distribution_data)
        if if_features_distribution:
            features_distribution_figure_return = generate_features_distribution_plot(features_distribution_data)
            return statistic_report_return,auc_figure_return,pr_figure_return,KS_figure_return,sample_distribution_figure_return,features_distribution_figure_return
        return statistic_report_return,auc_figure_return,pr_figure_return,KS_figure_return,sample_distribution_figure_return
    elif if_features_distribution:
        features_distribution_figure_return = generate_features_distribution_plot(features_distribution_data)
        return features_distribution_figure_return
