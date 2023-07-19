#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: linear_regression.py
# Created Date: 2022-10-28
# Author: Leslie Xu
# Contact: <lesliexufdu@163.com>
#
# Last Modified: 2023-07-19 03:17:33
#
# linear regression page
# -----------------------------------
# HISTORY:
###


import io
import os
import hjson
import shutil
import dash
from dash import dcc, html, dash_table, callback, no_update, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Scheme
import pandas as pd
from utils.datatable_format_util import data_bars_diverging
from utils.data_util import parse_contents
from utils.features_calculate_process import woe_table_merge,woe_subbins_calculate,woe_table_unmerge,bins_merge_indices_calculate
from utils.linear_regression_util import linear_regression_build,model_evaluate
from comprehensive_utils import feature_split_woe,feature_woe


PROJECT_DIR = "./projects"


dash.register_page(__name__, order=1, name="逻辑回归")


layout = html.Div(children=[

    # 模型训练
    dbc.Row([
        dbc.Col(html.Hr(style={"border":"3px"}),width=5),
        dbc.Col(html.H3("模型训练",style={"width":"100%","text-align":"center"}),width=2),
        dbc.Col(html.Hr(style={"border":"3px"}),width=5)
    ], align="center", style={"margin-bottom":"10px"}),
    dbc.Row([
        dbc.Col(html.Label("选择项目:"), width="auto"),
        dbc.Col(dcc.Dropdown(id='lr-project-options',style={"width":"100%"}), width=3),
        dbc.Col(html.Button(
            '刷新',
            id="refresh-projects",
            className="button-primary"
        ), width="auto"),
        dcc.Interval(
            id='project-interval-component',
            interval=10*1000,
            n_intervals=0
        ),
        dcc.Store(id='intermediate-project-config'),
        dbc.Col(
            dcc.Markdown(id='project-description',style={"width":"100%"}),
            width={"size":"auto","offset":2}
        )],
        align="center",
        style={"margin-bottom":"10px"}
    ),
    dbc.Row([
        dbc.Col(html.Label("选择或导入模型配置:"), width="auto"),
        dbc.Col(dcc.Dropdown(id='project-model-options',style={"width":"100%"}), width=3),
        dbc.Col(dcc.Upload(
            id='upload-model-config',
            children=['拖曳或',html.A('选择文件')],
            style={
                'textAlign':'center',
                "width":"100%",
                'borderStyle':'dashed',
                'borderWidth': '1px'
            }
        ), width=3),
        dbc.Col(html.Button(
            '确定',
            id="choose-upload-model-button",
            className="button-primary",
            title="优先使用上传的配置"
        ), width="auto")],
        align="center",
        style={"margin-bottom":"10px"}
    ),
    dbc.Row(id="model-config-info"),
    dbc.Row([
        dbc.Col(html.Label("选择自变量:"), width="auto"),
        dbc.Col(dcc.Dropdown(id='features-options',style={"width":"100%"}), width=2),
        dbc.Col(dcc.Markdown(id='iv-value',style={"width":"100%"}), width="auto"),
        dbc.Col(
            html.Button('删除自变量',id="delete-feature",style={'background-color':'#ff0000','color':'#ffffff'}),
            width="auto"
        ),
        dcc.Store(id='intermediate-used-features'),
        dcc.Store(id='initial-model-config'),
        dcc.Store(id='intermediate-features-config'),
        dbc.Col(html.Label("已删除自变量:"), width={"size":"auto","offset":1}),
        dbc.Col(dcc.Dropdown(id='deleted-features-options',style={"width":"100%"}), width=2),
        dbc.Col(
            html.Button('恢复变量',id="undelete-feature",className="button-primary"),
            width="auto"
        ),
        dbc.Col(
            html.Button('恢复所有变量',id="undelete-all-features",className="button-primary"),
            width="auto"
        )],
        align="center",
        style={"margin-bottom":"10px"}
    ),
    dbc.Row([
        dbc.Col(
            html.Button(
                '合并',
                id="merge-bins",
                className="button-primary",
                title="分类变量可以合并非连续的独立值,普通变量合并所选行的最大最小值所在的区间"
            ),
            width="auto"
        ),
        dbc.Col(
            html.Button('展开',id="unmerge-bins",className="button-primary"),
            width="auto"
        ),
        dbc.Col(
            dcc.RadioItems(id='feature-split-method',options=["决策树","等距","分位数"],value='决策树',inline=True),
            width={"size":"auto","offset":1}
        ),
        dbc.Col(html.Label("分裂参数:",title="决策树则为最小叶子占比,等距和分位数则为分段数"), width="auto"),
        dbc.Col(
            dcc.Input(id='feature-split-parameter',type='number',value=0.01,style={"width":"100%"}),
            width=1
        ),
        dbc.Col(
            html.Button(
                "重初始化",
                id="feature-reinitialize",
                className="button-primary",
                title="连续变量按指定方式初始化,分类变量初始化为分立值"
            ),
            width="auto"
        ),
        dbc.Col(html.Label("自定义分裂点:",title="仅对数值变量有效"), width={"size":"auto","offset":1}),
        dbc.Col(dcc.Input(id='selfdefined-split-value', type='number', style={"width":"100%"}), width=1),
        dbc.Col(html.Button(
            '添加',
            id="add-split-value",
            className="button-primary"
        ), width="auto")],
        align="center",
        style={"margin-bottom":"10px"}
    ),
    dash_table.DataTable(
        id="woe-detail-table",
        columns=[
            {"name":"range","id":"range"},
            {"name":"子分组数","id":"子分组数"},
            {"name":"样本数","id":"样本数"},
            {"name":"分组占比","id":"分组占比","type":'numeric',"format":Format(precision=2,scheme=Scheme.percentage)},
            {"name":"正样本数","id":"正样本数"},
            {"name":"正样本率","id":"正样本率","type":'numeric',"format":Format(precision=2,scheme=Scheme.percentage)},
            {"name":"WOE","id":"WOE"}
        ],
        row_selectable='multi',
        style_table={'height': '300px','overflowY': 'auto'},
        style_cell={
            'textAlign':'left',
            'overflow': 'hidden',
            'textOverflow': 'clip',
            'maxWidth': 0
        },
        style_cell_conditional=[
            {'if': {'column_id': 'range'}, 'width': '13%'},
            {'if': {'column_id': '子分组数'}, 'width': '13%'},
            {'if': {'column_id': '样本数'}, 'width': '13%'},
            {'if': {'column_id': '分组占比'}, 'width': '13%'},
            {'if': {'column_id': '正样本数'}, 'width': '13%'},
            {'if': {'column_id': '正样本率'}, 'width': '13%'},
            {'if': {'column_id': 'WOE'}, 'width': '22%'},
        ],
        style_header={
            'backgroundColor': '#70AD47',
            'color': 'white',
            'fontWeight': 'bold'
        },
        fixed_rows={'headers': True},
        tooltip_duration=None,
        style_as_list_view=True
    ),
    dbc.Row([
        dbc.Col(html.Button('开始建模',id="start-modeling",className="button-primary"), width="auto"),
        dcc.Store(id='model-object'),
        dcc.Store(id='model-performance')
    ], align="center", style={"margin-bottom":"10px"}),
    dbc.Row(id="model-building-info"),
    html.Div(id="model-performance-preview"),

    # 模型保存与导出
    dbc.Row([
        dbc.Col(html.Hr(style={"border":"3px"}),width=5),
        dbc.Col(html.H3("模型保存",style={"width":"100%","text-align":"center"}),width=2),
        dbc.Col(html.Hr(style={"border":"3px"}),width=5)
    ], align="center", style={"margin-bottom":"10px"}),
    dbc.Row([
        dbc.Col(html.Label("模型名称:",title="不可缺失;将覆盖已存在的模型配置"), width={"size":"auto"}),
        dbc.Col(
            dcc.Input(id='model-name-saved',type='text',value="",style={"width":"100%"}),
            width=2
        ),
        dbc.Col(html.Button('保存当前模型',id="btn-save-model",className="button-primary"), width="auto")
    ], align="center", style={"margin-bottom":"10px"}),
    dbc.Row(id="model-save-info"),
    dbc.Row([
        dbc.Col(html.Label("选择模型:"), width={"size":"auto"}),
        dbc.Col(dcc.Dropdown(id='model-to-output',style={"width":"100%"}), width=3),
        dbc.Col([
            html.Button('模型导出',id="btn-output-model",className="button-primary"),
            dcc.Download(id="download-model-config")
        ], width="auto"),
        dbc.Col(html.Button('删除模型',id="btn-delete-model",style={'background-color':'#ff0000','color':'#ffffff'}), width="auto")
    ], align="center", style={"margin-bottom":"10px"})

])


@callback(
    Output('lr-project-options', 'options'),
    Input('refresh-projects', 'n_clicks'),
    Input('project-interval-component', 'n_intervals'),
    State('lr-project-options', 'options')
)
def update_project_options(n_clicks, n_intervals, project_options):
    '''定时更新project列表'''
    project_valid_return = [
        {"label":i,"value":os.path.join(PROJECT_DIR,i)}
        for i in os.listdir(PROJECT_DIR)
        if os.path.isdir(os.path.join(PROJECT_DIR,i))
    ]
    if project_options and set([i['value'] for i in project_valid_return])==set([i['value'] for i in project_options]):
        return no_update
    return project_valid_return


@callback(
    Output('intermediate-project-config', 'data'),
    Input('lr-project-options', 'value'),
    prevent_initial_call=True
)
def read_project_config(project):
    '''更新project配置'''
    if project:
        project_config_path = os.path.join(project,"config.hjson")
        with open(project_config_path,"r",encoding="utf-8") as f:
            project_config = hjson.load(f)
        return project_config
    else:
        return None


@callback(
    Output('project-description', 'children'),
    Input('intermediate-project-config', 'data'),
    prevent_initial_call=True
)
def update_project_desription(project_config):
    '''展示project概要'''
    if project_config:
        samples = project_config['train data samples']
        positive_samples = project_config['train data positive samples']
        positive_rate = project_config['train data positive rate']
        description = f"训练样本总数:{samples}, 正样本数:{positive_samples}, 正样本率:{positive_rate:.2%}"
        return description
    else:
        return None


@callback(
    Output('intermediate-used-features', 'data'),
    [
        Input('intermediate-project-config', 'data'),
        Input('initial-model-config', 'data'),
        Input('delete-feature', 'n_clicks'),
        Input('undelete-feature', 'n_clicks'),
        Input('undelete-all-features', 'n_clicks')
    ],
    [
        State('features-options', 'value'),
        State('deleted-features-options', 'value'),
        State('intermediate-used-features', 'data')
    ],
    prevent_initial_call=True
)
def update_used_features(
    project_config,
    model_config,
    delete_feature_button,
    undelete_feature_button,
    undelete_allfeatures_button,
    current_feature,
    current_deleted_feature,
    current_features_config
):
    '''
    依据初始化信息或增删命令更新当前在用的变量列表
    输入:
        project_config: 项目配置
        model_config: 初始化模型配置
        delete_feature_button: 删除变量按钮
        undelete_feature_button: 恢复变量按钮
        undelete_allfeatures_button: 删除所有变量按钮
        current_feature: 当前被选择变量值
        current_deleted_feature: 当前被选择的已删除变量值
        current_features_config: 当前在使用的变量和被删除的变量列表
    输出:
        current_features_config: 当前在使用的变量和被删除的变量列表
    '''

    if callback_context.triggered_id=="intermediate-project-config" and project_config:
        # 初始化项目
        current_features_config_return = {
            "used features":project_config["features"],
            "deprecated features":[]
        }

    elif callback_context.triggered_id=="initial-model-config" and model_config:
        # 初始化模型配置
        current_features_config_return = {
            "used features":list(model_config["used features"].keys()),
            "deprecated features":list(model_config["deprecated features"].keys())
        }

    elif callback_context.triggered_id=="delete-feature" and current_feature:
        # 删除变量
        current_features_config_return = {
            "used features":[i for i in current_features_config["used features"] if i!=current_feature],
            "deprecated features":current_features_config["deprecated features"]+[current_feature]
        }

    elif callback_context.triggered_id=="undelete-feature" and current_deleted_feature:
        # 恢复变量
        current_features_config_return = {
            "used features":current_features_config["used features"]+[current_deleted_feature],
            "deprecated features":[i for i in current_features_config["deprecated features"] if i!=current_deleted_feature]
        }

    elif callback_context.triggered_id=="undelete-all-features" and project_config and current_features_config["deprecated features"]:
        # 恢复变量
        current_features_config_return = {
            "used features":project_config["features"],
            "deprecated features":[]
        }

    else:
        current_features_config_return = None

    return current_features_config_return


@callback(
    Output('features-options', 'options'),
    Output('deleted-features-options', 'options'),
    Input('intermediate-used-features', 'data'),
    prevent_initial_call=True
)
def update_used_deprecated_features_options(used_features):
    '''更新可删除和可恢复的变量列表'''
    if used_features:
        return used_features["used features"],used_features["deprecated features"]
    else:
        return [],[]


@callback(
    [
        Output('initial-model-config','data'),
        Output('model-config-info', 'children'),
        Output('project-model-options', 'value'),
        Output('upload-model-config', 'contents')
    ],
    [
        Input('choose-upload-model-button', 'n_clicks')
    ],
    [
        State('project-model-options', 'value'),
        State('upload-model-config', 'contents'),
        State('intermediate-project-config', 'data')
    ],
    prevent_initial_call=True
)
def initialize_model_config(
    model_button,
    model_path,
    model_config,
    project_config
):
    '''
    上传一个模型配置作为初始化模型
    入参:
        model_button: 上传或选择模型的按钮
        model_path: 模型路径
        model_config: 上传的模型配置内容
        project_config: 项目原始配置数据
    返回:
        initial-model-config: 初始化的模型配置信息
        model-config-info: 模型配置上传后的提示信息
        model-path: 当前选中模型,完成后刷新
        model-config: 当前上传的模型配置,完成后刷新
    '''
    # 返回信息
    info_return = None
    model_config_return = None

    if project_config is None:
        info_return = dbc.Alert("请先选择项目!", color="danger", is_open=True, duration=4000)

    elif model_config is None and model_path is None:
        info_return = dbc.Alert("选择或上传一个模型配置!", color="danger", is_open=True, duration=4000)
        model_config_return = no_update

    else:
        if model_config is not None:
            model_path = io.StringIO(parse_contents(model_config))
        with open(os.path.join(model_path,"config.hjson"),"r",encoding="utf-8") as f:
            model_config_return = hjson.load(f)
        model_config_return["used features"] = {
            k:v
            for k,v in model_config_return["used features"].items()
            if k in project_config["features"]
        }
        model_config_return["deprecated features"] = {
            k:v
            for k,v in model_config_return["deprecated features"].items()
            if k in project_config["features"] and k not in model_config_return["used features"].keys()
        }
    return model_config_return,info_return,None,None


@callback(
    [
        Output('intermediate-features-config','data'),
        Output('woe-detail-table','selected_rows')
    ],
    [
        Input('lr-project-options', 'value'),
        Input('initial-model-config','data'),
        Input('features-options', 'value'),
        Input('merge-bins', 'n_clicks'),
        Input('unmerge-bins', 'n_clicks'),
        Input('feature-reinitialize', 'n_clicks'),
        Input('add-split-value', 'n_clicks')
    ],
    [
        State('intermediate-features-config','data'),
        State('intermediate-project-config', 'data'),
        State('woe-detail-table', 'selected_rows'),
        State('feature-split-method', 'value'),
        State('feature-split-parameter', 'value'),
        State('selfdefined-split-value', 'value')
    ],
    prevent_initial_call=True
)
def update_model_config(
    project_path,
    initial_model_config,
    selected_feature,
    merge_button,
    unmerge_button,
    feature_reinitialize_button,
    add_split_value_button,
    current_model_config,
    project_config,
    woe_table_selected_rows,
    feature_split_method,
    feature_split_parameter,
    selfdefined_split_value
):
    '''
    初始化模型或删改变量后更新当前模型的变量分段配置
    入参:
        project_path: 模型所在文件夹
        initial_model_config: 初始化的模型配置
        selected_feature: 当前选中特征
        merge_button: 合并分段按钮
        unmerge_button: 展开分段按钮
        feature_reinitialize_button: 特征初始化按钮
        add_split_value_button: 分裂节点增添按钮
        current_model_config: 当前变量分段配置
        project_config: 模型配置
        woe_table_selected_rows: WOE表格被选中的行
        feature_split_method: 特征分裂方式
        feature_split_parameter: 特征分裂参数
        selfdefined_split_value: 自定义分裂参数
    返回:
        current_features_config_return: 当前模型配置
        selected_rows_return: 刷新被选中列表
    '''
    current_features_config_return = no_update
    selected_rows_return = no_update

    if callback_context.triggered_id=="lr-project-options":
        return None,[]

    if callback_context.triggered_id=="initial-model-config" and initial_model_config:
        initial_features_config = {
            **initial_model_config["used features"],
            **initial_model_config["deprecated features"]
        }
        current_features_config_return = {k:{} for k in initial_features_config.keys()}
        for k,v in initial_features_config.items():
            if "original bins" not in v.keys():
                split_result,woe_data = feature_split_woe(project_path, project_config, k)
                current_features_config_return[k]["original bins"] = split_result
                current_features_config_return[k]["original woe"] = woe_data
            else: # 初始化上传不论是否传了WOE值都要重新计算
                current_features_config_return[k]["original bins"] = v["original bins"]
                current_features_config_return[k]["original woe"] = feature_woe(
                    project_path,
                    project_config,
                    k,
                    v["original bins"]
                )
            if "bins" not in v.keys():
                current_features_config_return[k]["bins"] = current_features_config_return[k]["original bins"]
                current_features_config_return[k]["woe"] = current_features_config_return[k]["original woe"]
            else: # 这里不判断bins是否是original bins的合理子分段,如果不规范会导致后续报错
                # 初始化上传不论是否传了WOE值都要重新计算
                current_features_config_return[k]["bins"] = v["bins"]
                if_categorical = k in project_config["categorical features"]
                woe_temp = feature_woe(
                    project_path,
                    project_config,
                    k,
                    v["bins"]
                )
                current_features_config_return[k]["woe"] = woe_subbins_calculate(
                    woe_temp,
                    current_features_config_return[k]["original bins"],
                    if_categorical
                )

    elif callback_context.triggered_id=="features-options" and selected_feature:
        if not current_model_config: # 还没有配置文件
            split_result,woe_data = feature_split_woe(project_path, project_config, selected_feature)
            current_model_config = {
                selected_feature: {
                    "original bins":split_result,
                    "bins":split_result,
                    "original woe":woe_data,
                    "woe":woe_data,
                }
            }
        elif selected_feature not in current_model_config.keys(): # 当前变量不在配置文件中
            split_result,woe_data = feature_split_woe(project_path, project_config, selected_feature)
            current_model_config[selected_feature] = {
                "original bins":split_result,
                "bins":split_result,
                "original woe":woe_data,
                "woe":woe_data,
            }
        current_features_config_return = current_model_config

    elif callback_context.triggered_id=="features-options" and not selected_feature:
        # 取消特征选择的情况
        current_features_config_return = current_model_config

    elif callback_context.triggered_id=="merge-bins" and woe_table_selected_rows:
        if len(woe_table_selected_rows)>1:
            if_categorical = selected_feature in project_config["categorical features"]
            woe_update,bins_update = woe_table_merge(
                current_model_config[selected_feature]["woe"],
                woe_table_selected_rows,
                project_config['train data positive samples'],
                project_config['train data samples']-project_config['train data positive samples'],
                if_categorical
            )
            current_model_config[selected_feature]["bins"] = bins_update
            current_model_config[selected_feature]["woe"] = woe_update
            current_features_config_return = current_model_config
            selected_rows_return = []

    elif callback_context.triggered_id=="unmerge-bins" and woe_table_selected_rows:
        if_categorical = selected_feature in project_config["categorical features"]
        woe_update,bins_update = woe_table_unmerge(
            current_model_config[selected_feature]["woe"],
            woe_table_selected_rows,
            current_model_config[selected_feature]["original woe"],
            if_categorical=if_categorical
        )
        current_model_config[selected_feature]["bins"] = bins_update
        current_model_config[selected_feature]["woe"] = woe_update
        current_features_config_return = current_model_config
        selected_rows_return = []

    elif callback_context.triggered_id=="feature-reinitialize" and selected_feature:
        project_config_copy = project_config.copy()
        project_config_copy["split config"][selected_feature] = {
            "split_method": feature_split_method,
            "split_parameter": feature_split_parameter
        }
        split_result,woe_data = feature_split_woe(project_path, project_config_copy, selected_feature)
        current_model_config[selected_feature] = {
            "original bins":split_result,
            "bins":split_result,
            "original woe":woe_data,
            "woe":woe_data,
        }
        current_features_config_return = current_model_config

    elif callback_context.triggered_id=="add-split-value" and selected_feature:
        if_categorical = selected_feature in project_config["categorical features"]
        if not if_categorical and selfdefined_split_value not in current_model_config[selected_feature]["original bins"]:
            project_config_copy = project_config.copy()
            original_bins = sorted(current_model_config[selected_feature]["original bins"] + [selfdefined_split_value])
            current_model_config[selected_feature]["original bins"] = original_bins
            current_model_config[selected_feature]["original woe"] = feature_woe(
                project_path,
                project_config,
                selected_feature,
                original_bins
            )
            bins = sorted(current_model_config[selected_feature]["bins"] + [selfdefined_split_value])
            bins_to_merged = bins_merge_indices_calculate(bins, original_bins)
            woe = current_model_config[selected_feature]["original woe"]
            for selected_rows in bins_to_merged:
                woe,_ = woe_table_merge(
                    woe,
                    selected_rows,
                    project_config['train data positive samples'],
                    project_config['train data samples']-project_config['train data positive samples'],
                    False
                )
            current_model_config[selected_feature]["bins"] = bins
            current_model_config[selected_feature]["woe"] = woe
            current_features_config_return = current_model_config

    return current_features_config_return,selected_rows_return


@callback(
    Output('feature-split-method', 'value'),
    Input('features-options', 'value'),
    State('intermediate-project-config', 'data'),
    prevent_initial_call=True
)
def update_split_method(selected_feature, project_config):
    '''更新默认的分裂方法'''
    if selected_feature and project_config and selected_feature not in project_config["categorical features"]:
        return project_config['split config'][selected_feature]['split_method']
    else:
        return "决策树"


@callback(
    Output('feature-split-parameter', 'value'),
    Input('feature-split-method', 'value'),
    State('features-options', 'value'),
    State('intermediate-project-config', 'data'),
    prevent_initial_call=True
)
def update_split_parameter(split_method, selected_feature, project_config):
    '''更新默认的分裂参数'''
    if selected_feature and project_config and selected_feature not in project_config["categorical features"] and project_config['split config'][selected_feature]['split_method']==split_method:
        return project_config['split config'][selected_feature]['split_parameter']
    else:
        parameter_map = {"决策树":0.01,"等距":10,"分位数":10}
        return parameter_map[split_method]


@callback(
    [
        Output('woe-detail-table', 'data'),
        Output('woe-detail-table', 'style_data_conditional'),
        Output('woe-detail-table', 'tooltip_data'),
        Output('iv-value', 'children')
    ],
    [
        Input('intermediate-features-config','data')
    ],
    [
        State('features-options', 'value'),
        State('intermediate-project-config', 'data')
    ],
    prevent_initial_call=True
)
def update_woe_table(features_config, feature, project_config):
    '''更新展示的WOE表格'''
    if (
        feature is not None and
        features_config is not None and
        feature in features_config.keys() and
        "woe" in features_config[feature].keys()
    ):
        woe_data = pd.DataFrame(
            features_config[feature]['woe'],
            columns=['range','子分组数','样本数','分组占比','正样本数','正样本率','WOE']
        )
        # 计算IV
        negative_total = project_config['train data samples']-project_config['train data positive samples']
        negative_group_count = woe_data['样本数'] - woe_data['正样本数']
        positive_total = project_config['train data positive samples']
        positive_group_count = woe_data['正样本数']
        IV = (negative_group_count/negative_total - positive_group_count/positive_total)*woe_data['WOE']
        style_woe_bar = data_bars_diverging('WOE', column_max=woe_data.WOE.max(), column_min=woe_data.WOE.min())
        return (
            woe_data.to_dict('records'),
            style_woe_bar,
            [{'range':i} for i in woe_data['range']],
            f"IV值:{IV.sum():.2f}"
        )
    return None,None,None,None


@callback(
    [
        Output('model-object', 'data'),
        Output('model-performance', 'data'),
        Output('model-building-info', 'children')
    ],
    [
        Input('start-modeling', 'n_clicks'),
        Input('lr-project-options', 'value')
    ],
    [
        State('intermediate-features-config','data'),
        State('features-options', 'options'),
        State('intermediate-project-config', 'data')
    ],
    prevent_initial_call=True
)
def model_building(start_model_button, project_path, features_config, features, project_config):
    '''根据现有配置拟合模型并返回模型初步表现'''
    if callback_context.triggered_id=="lr-project-options":
        return None,None,None
    if project_config and features and features_config:
        if any([i not in features_config.keys() for i in features]):
            return None,None,dbc.Alert("请手动确认所有变量的分组!", color="danger", is_open=True, duration=4000)
        clf = linear_regression_build(project_path, project_config, features, features_config)
        evaluate_result = model_evaluate(clf, project_path, project_config, features, features_config)
        return (
            {
                'classes_':clf.classes_,
                'coef_':clf.coef_,
                'intercept_':clf.intercept_,
                'n_features_in_':clf.n_features_in_,
                'feature_names_in_':clf.feature_names_in_
            },
            evaluate_result,
            dbc.Alert("建模完成!", color="success", is_open=True, duration=4000)
        )
    return None,None,None


@callback(
    Output('model-performance-preview', 'children'),
    [
        Input('model-performance', 'data')
    ],
    prevent_initial_call=True
)
def model_performance_preview(model_performance_data):
    '''展示模型表现'''
    if model_performance_data:
        return dash_table.DataTable(
            data=model_performance_data,
            columns=[
                {"name":"数据集","id":"数据集"},
                {"name":"KS","id":"KS","type":'numeric',"format":Format(precision=2,scheme=Scheme.fixed)},
                {"name":"AUC","id":"AUC","type":'numeric',"format":Format(precision=2,scheme=Scheme.fixed)}
            ],
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#70AD47',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_cell={'textAlign': 'left'},
            style_cell_conditional=[
                {'if': {'column_id': '数据集'}, 'width': '33%'},
                {'if': {'column_id': 'KS'}, 'width': '33%'},
                {'if': {'column_id': 'AUC'}, 'width': '33%'}
            ],
            style_data_conditional=[{
                'if': {'row_index': 'odd'},
                'backgroundColor': '#E2EFDA',
            }]
        )
    return None


@callback(
    Output('model-name-saved', 'value'),
    Input('project-model-options', 'options'),
    prevent_initial_call=True
)
def update_model_name(current_model_options):
    '''更新预设置的模型名称'''
    if current_model_options:
        idx = 0
        model_list = [i['label'] for i in current_model_options]
        while f"model-{idx}" in model_list:
            idx += 1
        return f"model-{idx}"
    return ""


@callback(
    [
        Output('model-save-info', 'children'),
        Output('project-model-options', 'options')
    ],
    [
        Input('btn-save-model', 'n_clicks'),
        Input('btn-delete-model', 'n_clicks'),
        Input('lr-project-options', 'value'),
    ],
    [
        State('model-name-saved', 'value'),
        State('intermediate-used-features', 'data'),
        State('intermediate-features-config','data'),
        State('model-object', 'data'),
        State('model-to-output', 'value'),
        State('project-model-options', 'options')
    ],
    prevent_initial_call=True
)
def save_model_config(
    save_model_button,
    delete_model_button,
    project_path,
    model_name,
    used_deprecated_features,
    features_config,
    model_object,
    model_to_output,
    current_model_list
):
    '''
    功能:
        1. 选中项目时,更新可选模型列表
        2. 将模型配置文件保存到服务器,并更新可选模型列表
        3. 删除指定模型,并更新可选模型列表
    入参:
        save_model_button: 保存模型按钮
        model_name: 欲保存的模型名称
        project_path: 项目地址
        used_deprecated_features: 正在使用和废弃的变量名称
        features_config: 变量分段信息
        model_object: 已训练的模型
        current_model_list: 当前已保存的模型列表
    返回:
        model_save_info_return: 提示信息
        models_return: 更新当前可选模型列表
    '''
    model_config_tosave = {}

    if callback_context.triggered_id=="lr-project-options":
        if project_path:
            models_path = os.path.join(project_path,"models") # 承载模型配置的文件夹
            if not os.path.exists(models_path):
                os.makedirs(models_path)
            models_list = [{"label":i,"value":os.path.join(models_path,i)}
                for i in os.listdir(models_path)
                if os.path.isdir(os.path.join(models_path,i))
            ]
            return None,models_list
        else:
            return None,[]

    elif callback_context.triggered_id=="btn-delete-model":
        if project_path and model_to_output:
            shutil.rmtree(model_to_output)
            models_list = [i for i in current_model_list if i['value']!=model_to_output ]
            return dbc.Alert("模型已删除!",color="success",is_open=True,duration=4000),models_list
        else:
            return dbc.Alert("没有待删除的模型!",color="danger",is_open=True,duration=4000),current_model_list

    elif callback_context.triggered_id=="btn-save-model":
        if model_name and project_path and used_deprecated_features:
            if all([i in features_config.keys() for i in used_deprecated_features["used features"]]): # 所有变量都有确定分箱
                model_config_tosave["used features"] = {
                    k:v for k,v in features_config.items() if k in used_deprecated_features["used features"]
                }
                model_config_tosave["deprecated features"] = {
                    k:v for k,v in features_config.items() if k in used_deprecated_features["deprecated features"]
                }
            else:
                return dbc.Alert("部分变量未被分组!",color="danger",is_open=True,duration=4000),current_model_list
            if model_object:
                model_config_tosave["model parameters"] = {**model_object}
            ## 保存配置
            model_path = os.path.join(project_path,"models",model_name)
            if not os.path.exists(model_path):
                model_save_info_return = dbc.Alert("创建新模型!", color="success", is_open=True, duration=4000)
                os.makedirs(model_path)
                models_list = current_model_list+[{'label':model_name,'value':model_path}]
            else:
                model_save_info_return = dbc.Alert("模型被覆盖!", color="danger", is_open=True, duration=4000)
                models_list = current_model_list
            with open(os.path.join(model_path,"config.hjson"),'w',encoding="utf-8") as f:
                hjson.dump(model_config_tosave, f)
            return model_save_info_return,models_list
        else:
            return dbc.Alert("没有可保存的模型配置!",color="danger",is_open=True,duration=4000),current_model_list

    return None,no_update


@callback(
    Output('model-to-output', 'options'),
    Output('model-to-output', 'value'),
    Input('project-model-options', 'options'),
    prevent_initial_call=True
)
def update_model_options(model_options):
    '''更新已存在的模型列表至model-to-output'''
    return model_options,None


@callback(
    Output("download-model-config", "data"),
    Input("btn-output-model", "n_clicks"),
    State('model-to-output', 'value'),
    prevent_initial_call=True,
)
def output_model_config(output_model_button, model_to_output):
    if model_to_output:
        return dcc.send_file(os.path.join(model_to_output,"config.hjson"))
    else:
        return no_update