#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: linear_regression.py
# Created Date: 2022-10-28
# Author: Leslie Xu
# Contact: <lesliexufdu@163.com>
#
# Last Modified: 2022-12-30 11:39:14
#
# linear regression page
# -----------------------------------
# HISTORY:
###


import io
import os
import base64
import hjson
from xhtml2pdf import pisa
import plotly.graph_objects as go
import dash
from dash import dcc, html, callback, no_update, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from utils.data_util import parse_contents
from utils.model_report_util import modelconfig2sqlscore,generate_model_evaluation_report


PROJECT_DIR = "./projects"


dash.register_page(__name__, order=2, name="模型报告")


layout = html.Div(children=[

    # 模型选择
    dbc.Row([
        dbc.Col(html.Hr(style={"border":"3px"}),width=5),
        dbc.Col(html.H3("导入模型",style={"width":"100%","text-align":"center"}),width=2),
        dbc.Col(html.Hr(style={"border":"3px"}),width=5)
    ], align="center", style={"margin-bottom":"10px"}),
    dbc.Row([
        dbc.Col(html.Label("选择项目:"), width="auto"),
        dbc.Col(dcc.Dropdown(id='modelreport-project-options',style={"width":"100%"}), width=2),
        dbc.Col(html.Button(
            '刷新',
            id="modelreport-refresh-projects",
            className="button-primary"
        ), width="auto"),
        dcc.Interval(
            id='modelreport-project-interval-component',
            interval=10*1000,
            n_intervals=0
        ),
        dbc.Col(html.Label("选择或导入模型:"), width={"size":"auto","offset":1}),
        dbc.Col(dcc.Dropdown(id='modelreport-model-options',style={"width":"100%"}), width=2),
        dbc.Col(dcc.Upload(
            id='modelreport-upload-model-config',
            children=['拖曳或',html.A('选择文件')],
            style={
                'textAlign':'center',
                "width":"100%",
                'borderStyle':'dashed',
                'borderWidth': '1px'
            }
        ), width=2),
        dbc.Col(html.Button(
            '确定',
            id="modelreport-choose-upload-model-btn",
            className="button-primary",
            title="优先使用上传的配置"
        ), width="auto")
    ], align="center", style={"margin-bottom":"10px"}),
    dbc.Row(id="modelreport-model-upload-info"),
    dcc.Store(id="modelreport-intermediate-model-config"),
    dbc.Row([
        dbc.Col([
            html.Button(
                '导出评分卡',
                id="modelreport-scorecard-output-btn",
                className="button-primary"
            ),
            dcc.Download(id="modelreport-scorecard-download")
        ], width="auto"),
        dbc.Col(html.Label("P",title="比率Q时的分数"),width="auto"),
        dbc.Col(dcc.Input(id='scorecard-parameter-p',type='number',value=660,style={"width":"100%"}),width=1),
        dbc.Col(html.Label("Q",title="Odds比率"),width="auto"),
        dbc.Col(dcc.Input(id='scorecard-parameter-q',type='number',value=20,style={"width":"100%"}),width=1),
        dbc.Col(html.Label("PDO",title="比率为2Q时分值增加量"),width="auto"),
        dbc.Col(dcc.Input(id='scorecard-parameter-pdo',type='number',value=50,style={"width":"100%"}),width=1),
        dbc.Col(html.Label("negative weight",title="建模时负样本的抽样倍数,仅影响转化后的分数区间;如果只用了一半的负样本,该值设定为2"),width="auto"),
        dbc.Col(dcc.Input(id='scorecard-parameter-nw',type='number',value=1,style={"width":"100%"}),width=1),
        dbc.Col(html.Label("min score",title="分数最小值"),width="auto"),
        dbc.Col(dcc.Input(id='scorecard-parameter-minscore',type='number',value=200,style={"width":"100%"}),width=1),
        dbc.Col(html.Label("max score",title="分数最大值"),width="auto"),
        dbc.Col(dcc.Input(id='scorecard-parameter-maxscore',type='number',value=1000,style={"width":"100%"}),width=1)
    ], align="center", style={"margin-bottom":"10px"}),

    # 模型报告
    dbc.Row([
        dbc.Col(html.Hr(style={"border":"3px"}),width=5),
        dbc.Col(html.H3("模型报告",style={"width":"100%","text-align":"center"}),width=2),
        dbc.Col(html.Hr(style={"border":"3px"}),width=5)
    ], align="center", style={"margin-bottom":"10px"}),
    dbc.Row([
        dbc.Col(html.Button(
            '一键生成',
            id="modelreport-onekey-btn",
            className="button-primary"
        ), width="auto"),
        dbc.Col(html.Button(
            '报告导出',
            id="modelreport-output-btn",
            className="button-primary"
        ), width="auto"),
        dbc.Col(
            dcc.RadioItems(id='modelreport-output-format',options=["HTML","PDF"],value='HTML',inline=True),
            width="auto"
        ),
        dcc.Download(id="modelreport-model-report-download")
    ], align="center", style={"margin-bottom":"10px"}),
    ## 模型整体评价
    dbc.Row([
        dbc.Col(html.H4("模型整体评价",style={"width":"100%","text-align":"center"}),width='auto'),
        dbc.Col(html.Label("评分等频分段数",title="用于KS、PR曲线"),width="auto"),
        dbc.Col(dcc.Input(id='scorecard-cuts-quantile-bins',type='number',value=10,style={"width":"100%"}),width=1),
        dbc.Col(html.Label("评分等距分段数",title="用于查看分数分布"),width="auto"),
        dbc.Col(dcc.Input(id='scorecard-cuts-equalscore-bins',type='number',value=10,style={"width":"100%"}),width=1),
        dbc.Col(html.Button('生成',id="modelreport-model-evaluation-btn",className="button-primary"),width="auto")
    ], align="center", style={"margin-bottom":"10px"}),
    ### 1. 描述性样本统计
    dbc.Row([
        dbc.Col(html.H5("描述性样本统计",style={"width":"100%"}),width="auto"),
    ], align="center", style={"margin-bottom":"10px","margin-left":"2%"}),
    html.Div(id="modelreport-statistic-sample-report-figure"),
    ### 2. AUC曲线
    dbc.Row([
        dbc.Col(html.H5("AUC曲线",style={"width":"100%"}),width="auto")
    ], align="center", style={"margin-bottom":"10px","margin-left":"2%"}),
    html.Div(id="modelreport-auc-figure"),
    ### 3. PR曲线
    dbc.Row([
        dbc.Col(html.H5("PR曲线",style={"width":"100%"}),width="auto")
    ], align="center", style={"margin-bottom":"10px","margin-left":"2%"}),
    html.Div(id="modelreport-pr-figure"),
    ### 4. KS曲线
    dbc.Row([
        dbc.Col(html.H5("KS曲线",style={"width":"100%"}),width="auto")
    ], align="center", style={"margin-bottom":"10px","margin-left":"2%"}),
    html.Div(id="modelreport-KS-figure"),
    ### 4. 样本分布
    dbc.Row([
        dbc.Col(html.H5("样本分布",style={"width":"100%"}),width="auto")
    ], align="center", style={"margin-bottom":"10px","margin-left":"2%"}),
    html.Div(id="modelreport-sample-distribution-figure"),
    ## 变量分布
    dbc.Row([
        dbc.Col(html.H4("变量分布",style={"width":"100%","text-align":"center"}),width='auto'),
        dbc.Col(html.Button('生成',id="modelreport-features-distribution-plot-btn",className="button-primary"),width="auto")
    ], align="center", style={"margin-bottom":"10px"}),
    html.Div(id="modelreport-features-distribution-figure"),

])


@callback(
    Output('modelreport-project-options', 'options'),
    Input('modelreport-refresh-projects', 'n_clicks'),
    Input('modelreport-project-interval-component', 'n_intervals'),
    State('modelreport-project-options', 'options')
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
    Output('modelreport-model-options', 'options'),
    Input('modelreport-project-options', 'value'),
    prevent_initial_call=True
)
def update_model_options(project_path):
    '''更新模型列表'''
    if project_path:
        models_path = os.path.join(project_path,"models") # 承载模型配置的文件夹
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        models_list = [{"label":i,"value":os.path.join(models_path,i)}
            for i in os.listdir(models_path)
            if os.path.isdir(os.path.join(models_path,i))
        ]
        return models_list
    return []


@callback(
    [
        Output('modelreport-intermediate-model-config', 'data'),
        Output('modelreport-model-upload-info', 'children'),
        Output('modelreport-model-options', 'value'),
        Output('modelreport-upload-model-config', 'contents')
    ],
    [
        Input('modelreport-choose-upload-model-btn', 'n_clicks'),
        Input('modelreport-project-options', 'value')
    ],
    [
        State('modelreport-model-options', 'value'),
        State('modelreport-upload-model-config', 'contents')
    ],
    prevent_initial_call=True
)
def initialize_model_config(model_button, project_path, model_path, model_config):
    '''
    上传或选择模型配置
    入参:
        model_button: 上传或选择模型的按钮
        project_path: project路径
        model_path: 模型路径
        model_config: 上传的模型配置内容
    返回:
        intermediate-model-config: 模型配置信息
        model-config-info: 模型配置上传后的提示信息
    '''
    # 返回信息
    info_return = None
    model_config_return = None

    if callback_context.triggered_id=="modelreport-choose-upload-model-btn":
        if project_path is None:
            info_return = dbc.Alert("请先选择项目!", color="danger", is_open=True, duration=4000)
        elif model_config is None and model_path is None:
            info_return = dbc.Alert("选择或上传一个模型配置!", color="danger", is_open=True, duration=4000)
            model_config_return = no_update
        else:
            if model_config is not None:
                model_path = io.StringIO(parse_contents(model_config))
            with open(os.path.join(model_path,"config.hjson"),"r") as f:
                model_config_read = hjson.load(f)
            if "used features" not in model_config_read.keys():
                info_return = dbc.Alert("没有used features!", color="danger", is_open=True, duration=4000)
            elif "model parameters" not in model_config_read.keys():
                info_return = dbc.Alert("没有model parameters!", color="danger", is_open=True, duration=4000)
            elif model_config_read["model parameters"]["n_features_in_"]!=len(model_config_read["used features"].keys()):
                info_return = dbc.Alert("used features数量和模型入参个数不一致!", color="danger", is_open=True, duration=4000)
            else:
                for v in model_config_read["used features"].values():
                    if "bins" not in v.keys() or "woe" not in v.keys():
                        info_return = dbc.Alert("缺少bins或woe值!", color="danger", is_open=True, duration=4000)
                        break
                else:
                    info_return = dbc.Alert("模型配置导入成功!", color="success", is_open=True, duration=4000)
                    model_config_return = model_config_read

    return model_config_return,info_return,None,None


@callback(
    Output("modelreport-scorecard-download", "data"),
    Input("modelreport-scorecard-output-btn", "n_clicks"),
    [
        State('modelreport-intermediate-model-config', 'data'),
        State('scorecard-parameter-p', 'value'),
        State('scorecard-parameter-q', 'value'),
        State('scorecard-parameter-pdo', 'value'),
        State('scorecard-parameter-nw', 'value'),
        State('scorecard-parameter-minscore', 'value'),
        State('scorecard-parameter-maxscore', 'value')
    ],
    prevent_initial_call=True,
)
def output_scorecard(
    output_scorecard_button,
    model_config,
    P,
    Q,
    PDO,
    negative_weight,
    min_score,
    max_score
):
    '''
    导出评分卡
    入参:
        output_scorecard_button: 导出评分卡按钮
        model_config: 模型配置
        P: 评分卡参数
        Q: 评分卡参数
        PDO: 评分卡参数
        negative_weight: 负样本抽样倍数
        min_score: 输出评分的最小值
        max_score: 输出评分的最大值
    '''
    if model_config and P and Q and PDO and negative_weight:
        features_bins_woe_config = {
            k:{i:j for i,j in v.items() if i in ('bins','woe')}
            for k,v in model_config['used features'].items()
        }
        model_parameters = model_config['model parameters']
        sql_str_result = modelconfig2sqlscore(
            features_bins_woe_config,
            model_parameters,
            P,
            Q,
            PDO,
            negative_weight,
            min_score,
            max_score
        )
        return {
            "content":sql_str_result,
            "filename":"score_card.sql"
        }
    else:
        return no_update


@callback(
    [
        Output("modelreport-statistic-sample-report-figure", "children"),
        Output("modelreport-auc-figure", "children"),
        Output("modelreport-pr-figure", "children"),
        Output("modelreport-KS-figure", "children"),
        Output("modelreport-sample-distribution-figure", "children"),
        Output("modelreport-features-distribution-figure", "children")
    ],
    [
        Input("modelreport-onekey-btn", "n_clicks"),
        Input("modelreport-model-evaluation-btn", "n_clicks"),
        Input("modelreport-features-distribution-plot-btn", "n_clicks"),
        Input('modelreport-project-options', 'value'),
        Input('modelreport-intermediate-model-config', 'data')
    ],
    [
        State('scorecard-parameter-p', 'value'),
        State('scorecard-parameter-q', 'value'),
        State('scorecard-parameter-pdo', 'value'),
        State('scorecard-parameter-nw', 'value'),
        State('scorecard-parameter-minscore', 'value'),
        State('scorecard-parameter-maxscore', 'value'),
        State('scorecard-cuts-quantile-bins', 'value'),
        State('scorecard-cuts-equalscore-bins', 'value')
    ],
    prevent_initial_call=True,
)
def update_model_evaluation_report(
    onkey_report_button,
    model_evaluation_button,
    features_distribution_plot_btn,
    project_path,
    model_config,
    P,
    Q,
    PDO,
    negative_weight,
    min_score,
    max_score,
    quantile_bins,
    equalscore_bins
):
    '''
    更新描述性统计图表
    入参:
        model_evaluation_button: 生成按钮
        project_path: 项目地址
        model_config: 模型配置
        P: 评分卡参数
        Q: 评分卡参数
        PDO: 评分卡参数
        negative_weight: 评分卡参数
        min_score: 评分卡参数
        max_score: 评分卡参数
        quantile_bins: 评分分位数分段数
        equalscore_bins: 评分等距离分段数
    返回:
        statistic_report: 描述性统计
        auc_figure: AUC曲线
        pr_figure: 正样本率曲线
        KS_figure: KS曲线
        sample_distribution_figure: 样本分布曲线
        features_distribution_figure: 变量分布
    '''

    if callback_context.triggered_id=="modelreport-project-options" or callback_context.triggered_id=="modelreport-intermediate-model-config":
        return None,None,None,None,None,None # 更改项目或模型时表格清零

    elif (
        project_path and
        model_config and
        P and
        Q and
        PDO and
        negative_weight and
        min_score and
        max_score and
        quantile_bins and
        equalscore_bins
    ):
        if callback_context.triggered_id=="modelreport-onekey-btn":
            statistic_report,auc_figure,pr_figure,KS_figure,sample_distribution,features_distribution = generate_model_evaluation_report(
                project_path,
                model_config,
                P=P,
                Q=Q,
                PDO=PDO,
                negative_weight=negative_weight,
                min_score=min_score,
                max_score=max_score,
                quantile_bins_num=quantile_bins,
                equalscore_bins_num=equalscore_bins,
                if_features_distribution=True,
                if_score_evaluation=True
            )
            return (
                dcc.Graph(figure=statistic_report),
                dcc.Graph(figure=auc_figure),
                [dcc.Graph(figure=i) for i in pr_figure],
                dcc.Graph(figure=KS_figure),
                [dcc.Graph(figure=i) for i in sample_distribution],
                dcc.Graph(figure=features_distribution)
            )
        elif callback_context.triggered_id=="modelreport-model-evaluation-btn":
            statistic_report,auc_figure,pr_figure,KS_figure,sample_distribution = generate_model_evaluation_report(
                project_path,
                model_config,
                P=P,
                Q=Q,
                PDO=PDO,
                negative_weight=negative_weight,
                min_score=min_score,
                max_score=max_score,
                quantile_bins_num=quantile_bins,
                equalscore_bins_num=equalscore_bins,
                if_features_distribution=False,
                if_score_evaluation=True
            )
            return (
                dcc.Graph(figure=statistic_report),
                dcc.Graph(figure=auc_figure),
                [dcc.Graph(figure=i) for i in pr_figure],
                dcc.Graph(figure=KS_figure),
                [dcc.Graph(figure=i) for i in sample_distribution],
                no_update
            )
        elif callback_context.triggered_id=="modelreport-features-distribution-plot-btn":
            features_distribution = generate_model_evaluation_report(
                project_path,
                model_config,
                P=P,
                Q=Q,
                PDO=PDO,
                negative_weight=negative_weight,
                min_score=min_score,
                max_score=max_score,
                quantile_bins_num=quantile_bins,
                equalscore_bins_num=equalscore_bins,
                if_features_distribution=True,
                if_score_evaluation=False
            )
            return no_update,no_update,no_update,no_update,no_update,dcc.Graph(figure=features_distribution)
    else:
        return no_update,no_update,no_update,no_update,no_update,no_update


@callback(
    Output("modelreport-model-report-download", 'data'),
    [
        Input("modelreport-output-btn", "n_clicks")
    ],
    [
        State('modelreport-output-format', 'value'),
        State('modelreport-project-options', 'value'),
        State('modelreport-intermediate-model-config', 'data'),
        State('scorecard-cuts-quantile-bins', 'value'),
        State('scorecard-cuts-equalscore-bins', 'value'),
        State("modelreport-statistic-sample-report-figure", "children"),
        State("modelreport-auc-figure", "children"),
        State("modelreport-pr-figure", "children"),
        State("modelreport-KS-figure", "children"),
        State("modelreport-sample-distribution-figure", "children"),
        State("modelreport-features-distribution-figure", "children")
    ],
    prevent_initial_call=True,
)
def download_model_evaluation_report(
    output_button,
    output_format,
    project_path,
    model_config,
    quantile_bins,
    equalscore_bins,
    statistic_report,
    auc_figure,
    pr_figure,
    KS_figure,
    sample_distribution_figure,
    features_distribution_figure
):
    '''
    输出模型报告
    入参:
        output_button: 下载按钮
        output_format: 导出格式
        project_path: 项目地址
        model_config: 模型配置
        statistic_report: 描述性统计
        quantile_bins: 分位数分割段数
        equalscore_bins: 等分数分割段数
        auc_figure: AUC图
        pr_figure: PR图
        KS_figure: KS图
        sample_distribution_figure: 样本分布
        features_distribution_figure: 自变量分布
    '''
    if statistic_report and auc_figure and pr_figure and KS_figure and sample_distribution_figure and features_distribution_figure:
        n_datas = len([i for i in os.listdir(project_path) if i.endswith('_data.csv')])
        n_features = model_config['model parameters']['n_features_in_']

        img = "<html>\n<body>\n"
        img += "<h1>模型整体评价</h1>\n"
        img += "<h2>描述性样本统计</h2>\n"
        statistic_img = go.Figure(**statistic_report["props"]["figure"])
        statistic_img = statistic_img.to_image(width=1200,height=30*n_datas+50)
        img += '<img src="data:image/png;base64,'+base64.b64encode(statistic_img).decode()+'">'
        img += "<h2>AUC曲线</h2>\n"
        auc_img = go.Figure(**auc_figure["props"]["figure"]).to_image(width=1200,height=400)
        img += '<img src="data:image/png;base64,'+base64.b64encode(auc_img).decode()+'">'
        img += "<h2>PR曲线</h2>\n"
        pr_img = go.Figure(**pr_figure[0]["props"]["figure"]).to_image(width=1200,height=30*quantile_bins+70)
        img += '<img src="data:image/png;base64,'+base64.b64encode(pr_img).decode()+'">'
        pr_img = go.Figure(**pr_figure[1]["props"]["figure"]).to_image(width=1200,height=400)
        img += '<img src="data:image/png;base64,'+base64.b64encode(pr_img).decode()+'">'
        img += "<h2>KS曲线</h2>\n"
        ks_img = go.Figure(**KS_figure["props"]["figure"]).to_image(width=1200,height=400)
        img += '<img src="data:image/png;base64,'+base64.b64encode(ks_img).decode()+'">'
        img += "<h2>样本分布</h2>\n"
        sd_img = go.Figure(**sample_distribution_figure[0]["props"]["figure"]).to_image(width=1200,height=30*equalscore_bins+70)
        img += '<img src="data:image/png;base64,'+base64.b64encode(sd_img).decode()+'">'
        sd_img = go.Figure(**sample_distribution_figure[1]["props"]["figure"])
        sd_img = sd_img.to_image(width=1200,height=400)
        img += '<img src="data:image/png;base64,'+base64.b64encode(sd_img).decode()+'">'
        img += "\n<h1>变量分布</h1>\n"
        feature_img = go.Figure(**features_distribution_figure["props"]["figure"])
        feature_img = feature_img.to_image(width=1200,height=400*n_features+20)
        img += '<img src="data:image/png;base64,'+base64.b64encode(feature_img).decode()+'">'
        img += "\n</body>\n</html>"

        if output_format=="HTML":
            return {"content":img, "filename":"model_evaluation.html"}
        elif output_format=="PDF":
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.pdfbase import pdfmetrics
            from xhtml2pdf.default import DEFAULT_FONT
            pdfmetrics.registerFont(TTFont('yh', "./assets/msyh.ttc"))
            DEFAULT_FONT['helvetica'] = 'yh'
            result_file = io.BytesIO()
            pisa_status = pisa.CreatePDF(img, dest=result_file)
            result_bytes = result_file.getvalue()
            return dcc.send_bytes(result_bytes, filename="model_evaluation.pdf")
    return no_update