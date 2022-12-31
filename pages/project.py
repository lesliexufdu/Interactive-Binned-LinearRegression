#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: tab1.py
# Created Date: 2022-10-29
# Author: Leslie Xu
# Contact: <lesliexufdu@163.com>
#
# Last Modified: 2022-12-22 05:29:28
#
# This page contains the logic of data upload and project config.
# -----------------------------------
# HISTORY:
###


import os
import shutil
import io
import hjson
import pandas as pd
import dash
from dash import dcc, html, dash_table, callback, no_update, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from utils.data_util import parse_contents,save_dataset,project_write


DATASET_DIR = "./datasets"
PROJECT_DIR = "./projects"


dash.register_page(__name__, order=0, name="项目管理", path='/')


layout = dbc.Container(
    children=[

        # 创建数据集
        dbc.Row([
            dbc.Col(html.Hr(style={"border":"3px"}),width=5),
            dbc.Col(html.H3("创建数据集",style={"width":"100%","text-align":"center"}),width=2),
            dbc.Col(html.Hr(style={"border":"3px"}),width=5)
        ], align="center", style={"margin-bottom":"10px"}),
        dcc.Upload(
            id='upload-data',
            children=['拖曳或',html.A('选择文件')],
            style={
                'margin': '5px 0',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'textAlign': 'center'
            }
        ),
        dbc.Row([
            ## 定义文件分隔符
            dbc.Col(html.Label("分隔符:"), width='auto'),
            dbc.Col(dcc.RadioItems(
                id='delimeter-options',
                options=["tab","comma","space","user defined"],
                value='comma',
                inline=True,
                inputStyle={"margin":"5px"}
            ), width="auto"),
            dbc.Col(dcc.Input(
                id='delimeter-input',
                type='text',
                value="",
                placeholder="自定义分隔符"
            ), width="auto")],
            align="center",
            style={"margin-bottom":"10px"}
        ),
        dbc.Row([
            ## 定义数据集名称
            dbc.Col(html.Label("数据集名称:", title="如果不自定义,将使用文件原本的名称"), width="auto"),
            dbc.Col(dcc.Input(
                id='dataset-name',
                type='text',
                value="",
                style={"width":"100%"},
                placeholder="自定义数据集名称"
            ), width=2),
            dbc.Col(html.Button(
                '创建',
                id="create-dataset",
                title="创建行为将覆盖已存在的同名数据集",
                className="button-primary"
            ), width="auto")],
            align="center",
            style={"margin-bottom":"10px"}
        ),
        dbc.Row(id="dataset-create-info"),
        dbc.Row([
            dbc.Col(html.Label("现有数据集:"), width="auto"),
            dbc.Col(dcc.Dropdown(id='dataset-options',style={"width":"100%"}), width=3),
            dbc.Col(html.Button(
                '刷新',
                id="refresh-dataset",
                className="button-primary"
            ), width="auto"),
            dbc.Col(html.Button(
                '预览',
                id="preview-dataset",
                className="button-primary"
            ), width="auto"),
            dbc.Col(html.Button(
                '删除',
                id="delete-dataset",
                style={'background-color':'#ff0000','color':'#ffffff'}
            ), width="auto")],
            align="center",
            style={"margin-bottom":"10px"}
        ),
        html.Div(id="dataset-preview-content"),

        # 创建项目
        dbc.Row([
            dbc.Col(html.Hr(style={"border":"3px"}),width=5),
            dbc.Col(html.H3("创建项目",style={"width":"100%","text-align":"center"}),width=2),
            dbc.Col(html.Hr(style={"border":"3px"}),width=5)
        ], align="center", style={"margin-bottom":"10px"}),
        ## 定义基础数据
        dbc.Row([
            dbc.Col(html.Label("数据集:", title="训练和验证数据将按照比例拆分"), width="auto"),
            dbc.Col(dcc.Dropdown(id='dataset-options-2',style={"width":"100%"}), width=3),
            dbc.Col(html.Label("样本外数据:", title="可以不存在"), width={"size":"auto","offset":2}),
            dbc.Col(dcc.Dropdown(id='dataset-oot-options',style={"width":"100%"}), width=3),
            dcc.Store(id='intermediate-dataframe')
        ], align="center", style={"margin-bottom":"10px"}),
        ## 定义变量
        dbc.Row([
            dbc.Col(html.Label("目标变量:", title="该列只包含数值0和1", style={"width":"100%"}), width="auto"),
            dbc.Col(dcc.Dropdown(id='y-input', style={"width":"100%"}), width=3),
            dbc.Col(html.Label("自变量:"), width="auto"),
            dbc.Col(dcc.Dropdown(id='columns-input',multi=True, style={"width":"100%"}), width=3),
            dbc.Col(html.Label("分类自变量:"), width="auto"),
            dbc.Col(dcc.Dropdown(id='categorical-columns-input', multi=True, style={"width":"100%"}), width=3)],
            align="center",
            style={"margin-bottom":"10px"}
        ),
        ## 定义缺失值填充和分裂方式
        dbc.Row([
            dbc.Col(html.Label("缺失值填充:", title="如果上传配置文件,配置文件中涉及的列的填充值将被覆盖"), width="auto"),
            dbc.Col(
                dcc.Input(id='missing-value-fill',type='number',value=0,style={"width":"100%"}),
                width=1
            ),
            dbc.Col(dcc.Upload( # 标准JSON格式的配置文件,列名:填充值
                id='upload-missing-value-fill-config',
                children='上传缺失值填充配置文件',
                style={
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'textAlign': 'center',
                    "width":"100%"
                }
            ), width=2),
            dbc.Col(html.Label("变量分裂方式:", title="如果上传配置文件,配置文件中涉及的列的填充值将被覆盖"), width={"size":"auto","offset":1}),
            dbc.Col(dcc.RadioItems(
                id='column-split-method',
                options=["决策树","等距","分位数"],
                value='决策树',
                inline=True
            ), width='auto'),
            dbc.Col(
                html.Label("分裂参数:", title="决策树则为最小叶子占比,等距和分位数则为分段数"),
                width="auto"
            ),
            dbc.Col(
                dcc.Input(id='split-parameter',type='number',value=0.01, style={"width":"100%"}),
                width=1
            ),
            dbc.Col(dcc.Upload( # 标准JSON格式的配置文件,列名:{"method":XXX,"parameter":参数}
                id='upload-split-config',
                children='上传特征分裂配置文件',
                style={
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'textAlign': 'center',
                    "width":"100%"
                }
            ), width=2)],
            align="center",
            style={"margin-bottom":"10px"}
        ),
        dbc.Row([
            dbc.Col(html.Label("项目名称:", title="如果忽略,将使用数据集名称"), width="auto"),
            dbc.Col(dcc.Input(
                id='project-name',
                type='text',
                value="",
                placeholder="项目名称"
            ), width=2),
            dbc.Col(html.Label("测试集比例:"), width={"size":"auto","offset":2}),
            dbc.Col(dcc.Input(
                id='test-proportion',
                type='number',
                value=0.2,
                style={"width":"100%"}
            ), width=1),
            dbc.Col(html.Button(
                '创建',
                id="create-project",
                title="已存在的项目将被覆盖",
                className="button-primary"
            ), width="auto")],
            align="center",
            style={"margin-bottom":"10px"}
        ),
        dbc.Row(id="project-create-info"),
        dbc.Row([
            dbc.Col(html.Label("所有项目:"), width="auto"),
            dbc.Col(dcc.Dropdown(id='project-options',style={"width":"100%"}), width=3),
            dbc.Col(html.Button(
                '刷新',
                id="refresh-project",
                className="button-primary"
            ), width="auto"),
            dbc.Col(html.Button(
                '预览',
                id="preview-project",
                className="button-primary"
            ), width="auto"),
            dbc.Col(html.Button(
                '删除',
                id="delete-project",
                style={'background-color':'#ff0000','color':'#ffffff'}
            ), width="auto")],
            align="center",
            style={"margin-bottom":"10px"}
        ),
        html.Div(id="project-preview-content")

    ], fluid=True
)


@callback(
    [
        Output('dataset-create-info', 'children'),
        Output('dataset-options', 'options'),
        Output('upload-data', 'contents'),
        Output('delimeter-options', 'value'),
        Output('delimeter-input', 'value'),
        Output('dataset-name', 'value')
    ],
    [
        Input('create-dataset', 'n_clicks'),
        Input('refresh-dataset', 'n_clicks'),
        Input('delete-dataset', 'n_clicks')
    ],
    [
        State('upload-data', 'contents'),
        State('upload-data', 'filename'),
        State('delimeter-options', 'value'),
        State('delimeter-input', 'value'),
        State('dataset-name', 'value'),
        State('dataset-options', 'value')
    ]
)
def create_delete_dataset(
    n_clicks_create,
    n_clicks_refresh,
    n_clicks_delete,
    contents,
    filename,
    delimeter_option,
    delimeter_input,
    dataset_name,
    dataset_option
):
    '''
    数据集管理,
    入参:
        n_clicks_create: 创建按钮
        n_clicks_refresh: 刷新按钮
        n_clicks_delete: 删除按钮
        contents: 上传文件的内容
        filename: 文件名
        delimeter_option: 选中的分隔符
        delimeter_input: 用户自定义的分隔符
        dataset_name: 数据集名称
        dataset_option: 当前被选中的数据集名称
    输出:
        dataset create info: 提示信息
        datasets' list: 有效数据集
        contents: 上传文件按内容,创建数据集后被清空
        delimeter-options: 分隔符选项,创建数据集后初始化
        delimeter input: 自定义分隔符,创建数据集后初始化
        dataset name input: 输入的数据集名称,创建数据集后初始化
    '''

    # 返回信息
    info_return = no_update
    dataset_valid_return = no_update
    upload_contents_return = no_update
    delimeter_options_return = no_update
    delimeter_input_return = no_update
    dataname_input_return = no_update

    if callback_context.triggered_id is None or callback_context.triggered_id=="refresh-dataset": # 初始化或刷新
        dataset_valid = [os.path.splitext(i)[0] for i in os.listdir(DATASET_DIR) if i.endswith("dataset")]
        dataset_valid_return = dataset_valid

    elif callback_context.triggered_id=="delete-dataset": # 删除数据集
        dataset_valid = [os.path.splitext(i)[0] for i in os.listdir(DATASET_DIR) if i.endswith("dataset")]
        if dataset_option is None or dataset_option not in dataset_valid:
            dataset_valid_return = dataset_valid
            info_return = dbc.Alert(
                "数据集并不存在!",
                color="danger",
                is_open=True,
                duration=4000
            )
        else:
            dataset_valid.remove(dataset_option)
            dataset_path = os.path.join(DATASET_DIR,f"{dataset_option}.dataset")
            os.remove(dataset_path)
            info_return = dbc.Alert(
                f"数据集{dataset_option}已被删除!",
                color="success",
                is_open=True,
                duration=4000
            )
            dataset_valid_return = dataset_valid

    else: # 创建
        if contents is None:
            info_return = dbc.Alert(
                f"请先上传文件!",
                color="danger",
                is_open=True,
                duration=4000
            )
        elif delimeter_option=="user defined" and (delimeter_input is None or delimeter_input==""):
            info_return = dbc.Alert(
                f"请输入正确的分隔符!",
                color="danger",
                is_open=True,
                duration=4000
            )
        else:
            dataset_valid = [os.path.splitext(i)[0] for i in os.listdir(DATASET_DIR) if i.endswith("dataset")]
            # delimeter define
            if delimeter_option=="user defined":
                delimeter = delimeter_input
            elif delimeter_option=="comma":
                delimeter = ","
            elif delimeter_option=="space":
                delimeter = " "
            elif delimeter_option=="tab":
                delimeter = "\t"
            # dataset name
            if dataset_name is None or dataset_name=="":
                dataset_name = os.path.splitext(filename)[0]
            else:
                dataset_name = dataset_name.strip().replace(" ","_")
            # 保存数据集
            dataset_path = os.path.join(DATASET_DIR,f"{dataset_name}.dataset")
            save_dataset(contents, delimeter, dataset_path)
            if dataset_name not in dataset_valid:
                dataset_valid.append(dataset_name)
            # return info
            info_return = dbc.Alert(
                f"数据集{dataset_name}成功创建!",
                color="success",
                is_open=True,
                duration=4000
            )
            dataset_valid_return = dataset_valid
            upload_contents_return = None
            delimeter_options_return = "comma"
            delimeter_input_return = ""
            dataname_input_return = ""

    return info_return,dataset_valid_return,upload_contents_return,delimeter_options_return,delimeter_input_return,dataname_input_return


@callback(
    Output('dataset-preview-content', 'children'),
    Input('preview-dataset', 'n_clicks'),
    Input('dataset-options', 'value'),
    prevent_initial_call=True
)
def dataset_preview(preview_click, dataset):
    '''数据集预览'''
    if callback_context.triggered_id=="dataset-options" and dataset is None:
        return None
    elif callback_context.triggered_id=="preview-dataset" and dataset is not None:
        dataset_path = os.path.join(DATASET_DIR,f"{dataset}.dataset")
        df = pd.read_csv(dataset_path, sep="\t", nrows=5)
        return dash_table.DataTable(
            data=df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#70AD47',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_cell={'textAlign': 'left'},
            style_data_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#E2EFDA',
            }]
        )
    else:
        return no_update


@callback(
    Output('dataset-options-2', 'options'),
    Output('dataset-oot-options', 'options'),
    Input('dataset-options', 'options')
)
def update_dataset_options2(options):
    '''项目面板的数据集选项同步'''
    return options,options


@callback(
    [
        Output('columns-input', 'options'),
        Output('columns-input', 'value'),
        Output('intermediate-dataframe', 'data'),
        Output('y-input', 'options')
    ],
    [
        Input('dataset-options-2', 'value'),
        Input('y-input', 'value')
    ],
    prevent_initial_call=True
)
def update_columns_for_project(dataset, y):
    '''
    根据选中数据集产生目标变量候选
    参数:
        dataset: 选中的数据集
        y: 选中的y列名
    输出:
        columns_input_options: 可选自变量列表
        columns_input_value: 当前选中的自变量
        intermediate-dataframe: 数据集采样,用于确定列和列类型
        y_input: y列的备选项
    '''

    if dataset is not None:
        dataset_path = os.path.join(DATASET_DIR,f"{dataset}.dataset")
        df = pd.read_csv(dataset_path, sep="\t", nrows=10)
        y_input = df.columns.tolist()
        if y is not None:
            columns_input_options = [i for i in df.columns if i!=y]
        else:
            columns_input_options = y_input
        return columns_input_options,columns_input_options,df.to_dict('records'),y_input
    else:
        return [],[],None,[]


@callback(
    [
        Output('categorical-columns-input', 'options'),
        Output('categorical-columns-input', 'value')
    ],
    [
        Input('columns-input', 'value')
    ],
    [
        State('intermediate-dataframe', 'data')
    ],
    prevent_initial_call=True
)
def update_categorical_columns_for_project(columns, datasample):
    '''
    更新分类变量列表
    parameters:
        columns: 数据集的自变量
        datasample: 数据集样本
    output:
        categorical_columns_input_options: 分类变量选项
        categorical_columns_input_value: 分类变量选中值
    '''

    if columns is not None:
        df = pd.DataFrame(datasample)
        categorical_columns_input_options = columns
        categorical_columns_input_value = [i for i in df.columns if df[i].dtype=="O" and i in columns]
        return categorical_columns_input_options,categorical_columns_input_value
    else:
        return [],[]


@callback(
    [
        Output('project-create-info', 'children'),
        Output('project-options', 'options'),
        Output('dataset-options-2', 'value'),
        Output('project-name', 'value'),
        Output('missing-value-fill', 'value'),
        Output('upload-missing-value-fill-config', 'contents'),
        Output('column-split-method', 'value'),
        Output('upload-split-config', 'contents')
    ],
    [
        Input('create-project', 'n_clicks'),
        Input('refresh-project', 'n_clicks'),
        Input('delete-project', 'n_clicks')
    ],
    [
        State('dataset-options-2', 'value'),
        State('columns-input', 'value'),
        State('categorical-columns-input', 'value'),
        State('y-input', 'value'),
        State('missing-value-fill', 'value'),
        State('upload-missing-value-fill-config', 'contents'),
        State('column-split-method', 'value'),
        State('split-parameter', 'value'),
        State('upload-split-config', 'contents'),
        State('project-name', 'value'),
        State('project-options', 'value'),
        State('test-proportion', 'value'),
        State('dataset-oot-options', 'value')
    ]
)
def create_delete_project(
    n_clicks_create,
    n_clicks_refresh,
    n_clicks_delete,
    dataset,
    columns,
    categorical_columns,
    y,
    missing_value,
    missing_value_config,
    split_method,
    split_parameter,
    split_config,
    project_name,
    project_option,
    train_test_split_proportion,
    oot_dataset
):
    '''
    从数据集创建项目
    输入:
        n_clicks_create: 创建按钮
        n_clicks_refresh: 刷新按钮
        n_clicks_delete: 删除按钮
        dataset: 数据集名称
        columns: 所有的自变量
        categorical_columns: 所有分类自变量,是`columns`的子集
        y: 仅包含0/1的分类变量
        missing_value: 缺失值填充值
        missing_value_config: 缺失值填充配置
        split_method: 特征分裂方式
        split_parameter: 特征分裂参数
        split_config: 特征分裂配置
        project_name: 项目名称
        project_option: 当前选中的项目
        oot_dataset: 样本外数据集名称
    输出:
        project create info: callback的反馈信息
        projects' list: 有效的projects名称
        dataset option value: 被选中的数据集名称,创建后被刷新
        project name input: 项目名称,创建后被刷新
        missing value: 创建后被刷新
        missing value config: 创建后被刷新
        split method: 创建后被刷新
        split config: 创建后被刷新
    '''

    # return information
    info_return = no_update
    project_valid_return = no_update
    dataset_return = no_update
    projectname_input_return = no_update
    missing_value_return = no_update
    missing_value_config_return = no_update
    split_method_return = no_update
    split_config_return = no_update


    if callback_context.triggered_id is None or callback_context.triggered_id=="refresh-project": # initialize or refresh
        project_valid_return = [
            i
            for i in os.listdir(PROJECT_DIR)
            if os.path.isdir(os.path.join(PROJECT_DIR,i))
        ]

    elif callback_context.triggered_id=="delete-project": # delete
        project_valid = [
            i
            for i in os.listdir(PROJECT_DIR)
            if os.path.isdir(os.path.join(PROJECT_DIR,i))
        ]
        if project_option is None or project_option not in project_valid:
            project_valid_return = project_valid
            info_return = dbc.Alert(
                "目标项目不存在!",
                color="danger",
                is_open=True,
                duration=4000
            )
        else:
            project_valid.remove(project_option)
            shutil.rmtree(os.path.join(PROJECT_DIR,f"{project_option}"))
            info_return = dbc.Alert(
                f"项目{project_option}已被删除!",
                color="success",
                is_open=True,
                duration=4000
            )
            project_valid_return = project_valid

    else: # create
        if dataset is None:
            info_return = dbc.Alert(
                f"请选择您的项目所需要的数据!",
                color="danger",
                is_open=True,
                duration=4000
            )
        elif columns is None or columns==[] or y is None:
            info_return = dbc.Alert(
                f"请选择目标变量和自变量!",
                color="danger",
                is_open=True,
                duration=4000
            )
        else:
            # 读取数据集
            dataset_path = os.path.join(DATASET_DIR,f"{dataset}.dataset")
            df = pd.read_csv(dataset_path, sep="\t", usecols=[y,*columns])
            if oot_dataset is not None: # 样本外数据
                dataset_oot_path = os.path.join(DATASET_DIR,f"{oot_dataset}.dataset")
                df_oot = pd.read_csv(dataset_oot_path, sep="\t", usecols=[y,*columns])
            else:
                df_oot = None
            # 因变量必须是二元变量
            if (~df[y].isin([1,0])).sum()>0 or (oot_dataset is not None and (~df_oot[y].isin([1,0])).sum()>0):
                info_return = dbc.Alert(
                    f"目标变量只允许包含0和1!",
                    color="danger",
                    is_open=True,
                    duration=4000
                )
            else:
                # 项目名
                if project_name is None or project_name=="": # 不填则直接使用数据集名称
                    project_name = dataset
                else:
                    project_name = project_name.strip().replace(" ","_")
                # 缺失值配置
                missing_value_all_config = {} if missing_value is None else {i:missing_value for i in columns}
                if missing_value_config:
                    with open(io.StringIO(parse_contents(missing_value_config)),"r") as f:
                        missing_value_upload_config = hjson.load(f)
                    missing_value_all_config.update(missing_value_upload_config)
                # 分裂方式配置
                split_all_config = {
                    i:{"split_method":split_method,"split_parameter":split_parameter}
                    for i in columns
                    if categorical_columns is None or i not in categorical_columns
                }
                if split_config:
                    with open(io.StringIO(parse_contents(split_config)),"r") as f:
                        split_upload_config = hjson.load(f)
                    split_all_config.update(split_upload_config)
                # 写入项目内容
                project_path = os.path.join(PROJECT_DIR,f"{project_name}")
                project_write(df, y, columns, categorical_columns, project_path,
                    df_oot=df_oot,
                    missing_value_config=missing_value_all_config,
                    split_config=split_all_config,
                    test_size=train_test_split_proportion
                )
                # 更新项目列表
                project_valid = [
                    i
                    for i in os.listdir(PROJECT_DIR)
                    if os.path.isdir(os.path.join(PROJECT_DIR,i))
                ]
                if project_name not in project_valid:
                    project_valid.append(project_name)
                # 更新返回信息
                info_return = dbc.Alert(
                    f"项目{project_name}已被创建!",
                    color="success",
                    is_open=True,
                    duration=4000
                )
                project_valid_return = project_valid
                dataset_return = None
                projectname_input_return = ""
                missing_value_return = 0
                missing_value_config_return = None
                split_method_return = "tree"
                split_config_return = None

    return (
        info_return,
        project_valid_return,
        dataset_return,
        projectname_input_return,
        missing_value_return,
        missing_value_config_return,
        split_method_return,
        split_config_return
    )


@callback(
    Output('split-parameter', 'value'),
    Input('column-split-method', 'value'),
    prevent_initial_call=True
)
def update_split_parameter(split_method):
    '''更新默认的分裂参数'''
    parameter_map = {"决策树":0.01,"等距":10,"分位数":10}
    return parameter_map[split_method]


@callback(
    Output('project-preview-content', 'children'),
    Input('preview-project', 'n_clicks'),
    Input('project-options', 'value'),
    prevent_initial_call=True
)
def project_preview(preview_click, project):
    '''项目预览'''
    if callback_context.triggered_id=="project-options" and project is None:
        return None
    elif callback_context.triggered_id=="preview-project" and project is not None:
        project_path = os.path.join(PROJECT_DIR,project)
        project_data_path = os.path.join(project_path,"train_data.csv")
        df = pd.read_csv(project_data_path, sep="\t", nrows=5)
        return dash_table.DataTable(
            data=df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#70AD47',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_cell={'textAlign': 'left'},
            style_data_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#E2EFDA',
            }]
        )
    else:
        return no_update