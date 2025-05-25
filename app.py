import os
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import dash_bootstrap_components as dbc
from dash import callback_context

if not hasattr(np, 'int'):
    np.int = int

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# 读取数据
df = pd.read_csv("bank_credit_scoring.csv")
df.columns = [
    "Debt", "Overdue_Days", "Initial_Limit", "Birth_Date", "Sex", "Education", "Income", "Loan_Term",
    "Credit_History_Rating", "Living_Area", "Settlement_Name", "Industry_Name", "Probability_of_Default",
    "Client_ID", "Scoring_Mark", "Underage_Children_Count", "Velcom_Scoring", "Family_Status"
]

# 处理年龄
df['Age'] = 2025 - pd.to_datetime(df['Birth_Date'], errors='coerce').dt.year
# 性别映射
df['Sex'] = df['Sex'].map({"Мужской": "男", "Женский": "女"})
# 教育映射
df['Education'] = df['Education'].map({
    "Среднее": "中等教育",
    "Среднее специальное": "中等专业教育",
    "Высшее": "高等教育"
})
# 逾期分组
df['Overdue_Group'] = pd.cut(
    df['Overdue_Days'],
    bins=[-1, 0, 30, np.inf],
    labels=["无逾期", "1-30天", "30天以上"]
)
# 地区排序
area_limit_stats = df.groupby('Living_Area')['Initial_Limit'].mean().sort_values(ascending=False)
all_areas = area_limit_stats.index.tolist()
# 年份动画
df['year'] = np.random.choice(range(2018, 2023), size=len(df))

# 统一配色方案
color_sequence = px.colors.qualitative.Set2
color_scale = px.colors.sequential.YlGnBu

# 图1：性别与逾期状态比例分布
df_clean = df.dropna(subset=['Sex', 'Overdue_Group'])
grouped = df_clean.groupby(['Sex', 'Overdue_Group']).size().reset_index(name='count')
total_by_sex = grouped.groupby('Sex')['count'].transform('sum')
grouped['percentage'] = grouped['count'] / total_by_sex

sex_overdue_fig = px.bar(
    grouped,
    x='Sex',
    y='percentage',
    color='Overdue_Group',
    text='count',
    barmode='stack',
    category_orders={"Overdue_Group": ["无逾期", "1-30天", "30天以上"], "Sex": ["男", "女"]},
    color_discrete_sequence=color_sequence,
)

for trace in sex_overdue_fig.data:
    trace.textposition = 'inside'
    trace.marker.line.color = 'white'
    trace.marker.line.width = 0.3

sex_overdue_fig.update_layout(
    yaxis_title="比例",
    xaxis_title="性别",
    legend_title="逾期状态",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, traceorder='reversed'),
    template='plotly_white',
    margin=dict(l=20, r=20, t=60, b=20),
    title_text='性别与逾期状态比例分布',
    bargap=0.3
)
sex_overdue_fig.update_yaxes(tickformat=".0%", range=[0, 1], showgrid=True, zeroline=True, gridcolor='#e5e5e5', constrain='domain')
sex_overdue_fig.update_xaxes(showgrid=False)

# 图2：年龄与违约概率的关系
age_prob_df = df.dropna(subset=['Age', 'Probability_of_Default'])
hex_fig = px.density_heatmap(
    age_prob_df, x="Age", y="Probability_of_Default", nbinsx=30, nbinsy=30,
    color_continuous_scale=color_scale
)
if not age_prob_df.empty:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smooth = lowess(age_prob_df['Probability_of_Default'], age_prob_df['Age'], frac=0.2)
    hex_fig.add_trace(go.Scatter(x=smooth[:, 0], y=smooth[:, 1], mode='lines', line=dict(color='#440154', width=2), name='平滑线'))
hex_fig.update_layout(
    xaxis_title="年龄",
    yaxis_title="违约概率",
    template='plotly_white',
    coloraxis_colorbar=dict(title='密度'),
    margin=dict(l=20, r=20, t=60, b=20),
    title_text=''
)

# 图3：教育水平与信用评分分布分析
edu_score_df = df.dropna(subset=['Education', 'Scoring_Mark'])
edu_score_fig = px.violin(
    edu_score_df, x="Education", y="Scoring_Mark", box=True, points="all",
    color="Education", color_discrete_sequence=color_sequence
)
edu_score_fig.update_layout(
    xaxis_title="教育水平",
    yaxis_title="信用评分",
    showlegend=False,
    template='plotly_white',
    margin=dict(l=20, r=20, t=60, b=20),
    title_text=''
)

# 图4：收入分组与逾期天数分布分析
income_group = np.where(df['Income'] < 1500, "收入<1500", "收入≥1500")
overdue_group = pd.cut(
    df['Overdue_Days'],
    bins=[-1, 0, 30, 100, 300, np.inf],
    labels=["未逾期", "1-30天", "31-100天", "101-300天", "300天以上"]
)
mirror_df = pd.DataFrame({
    "income_group": income_group,
    "overdue_group": overdue_group
})
mirror_df = mirror_df.dropna()
mirror_df = mirror_df.groupby(['income_group', 'overdue_group']).size().reset_index(name='count')
mirror_df['sym_count'] = mirror_df.apply(lambda row: -row['count'] if row['income_group'] == "收入<1500" else row['count'], axis=1)

mirror_fig = px.bar(
    mirror_df, x="overdue_group", y="sym_count", color="income_group",
    color_discrete_sequence=color_sequence,
    text="count", barmode="relative"
)
mirror_fig.update_traces(textposition='outside')
mirror_fig.update_layout(
    xaxis_title="逾期状态",
    yaxis_title="客户数量",
    legend_title="",
    barmode="relative",
    template='plotly_white',
    margin=dict(l=20, r=20, t=60, b=20),
    title_text=''
)
mirror_fig.update_yaxes(
    tickvals=[-max(mirror_df['count']), 0, max(mirror_df['count'])],
    ticktext=[str(max(mirror_df['count'])), "0", str(max(mirror_df['count']))]
)

# 图5：各地域平均信贷限额对比
bar_df = df.groupby('Living_Area')['Initial_Limit'].mean().reset_index()
bar_df = bar_df.sort_values('Initial_Limit', ascending=False)
bar_fig = px.bar(
    bar_df,
    x='Living_Area',
    y='Initial_Limit',
    color='Initial_Limit',
    color_continuous_scale=color_scale,
    labels={'Living_Area': '地区', 'Initial_Limit': '平均信贷限额'},
    category_orders={'Living_Area': bar_df['Living_Area'].tolist()}
)
bar_fig.update_layout(
    xaxis_tickangle=-45,
    xaxis_title='居住地区',
    yaxis_title='平均初始信贷限额(元)',
    showlegend=False,
    template='plotly_white',
    margin=dict(l=20, r=20, t=60, b=20),
    title_text=''
)

# -------- 新增：债务区间与逾期天数分布、阈值效应、收入债务比分析 --------
# 1. 债务区间与逾期天数分布（堆叠条形图）
df['Debt_Bin'] = pd.cut(
    df['Debt'],
    bins=[0, 2000, 4000, 6000, 8000, 10000, np.inf],
    labels=["0-2k", "2k-4k", "4k-6k", "6k-8k", "8k-10k", "10k+"],
    right=False
)
df['Delinquency_Bin'] = pd.cut(
    df['Overdue_Days'],
    bins=[-np.inf, 0, 24, 49, np.inf],
    labels=["无逾期", "0-24天", "24-49天", "49+天"],
    right=False
)
# 统计各债务区间逾期天数分布
debt_overdue_df = df.dropna(subset=['Debt_Bin', 'Delinquency_Bin'])
debt_overdue_count = debt_overdue_df.groupby(['Debt_Bin', 'Delinquency_Bin']).size().reset_index(name='count')
# 堆叠条形图

fig_debt_overdue = px.bar(
    debt_overdue_count,
    x='Debt_Bin', y='count', color='Delinquency_Bin',
    color_discrete_sequence=color_sequence,
    category_orders={"Debt_Bin": ["0-2k", "2k-4k", "4k-6k", "6k-8k", "8k-10k", "10k+"],
                     "Delinquency_Bin": ["无逾期", "0-24天", "24-49天", "49+天"]},
    barmode='stack'
    # 移除了 text='count' 参数以去掉数字标识
)

# 更新 traces 样式，但不设置 textposition 和 marker line color/width 用于文本
fig_debt_overdue.update_traces(marker_line_color='white', marker_line_width=0.3)

fig_debt_overdue.update_layout(
    xaxis_title="债务区间(元)", yaxis_title="客户数量", legend_title="逾期天数",
    template='plotly_white', bargap=0.3,
    margin=dict(l=20, r=20, t=60, b=20), title_text=''
)
fig_debt_overdue.update_xaxes(tickangle=0)

# 2. 债务阈值效应分析（逾期率趋势线）
debt_bin_delinquency = df.dropna(subset=['Debt_Bin'])
debt_bin_delinquency['is_delinquent'] = debt_bin_delinquency['Overdue_Days'] > 0
delinquency_rate_df = debt_bin_delinquency.groupby('Debt_Bin')['is_delinquent'].mean().reset_index()
fig_debt_threshold = px.line(
    delinquency_rate_df, x='Debt_Bin', y='is_delinquent',
    markers=True, line_shape='linear',
)
fig_debt_threshold.update_traces(line_color=color_sequence[0], marker_color=color_sequence[1], line_width=3)
fig_debt_threshold.update_layout(
    xaxis_title="债务区间(元)", yaxis_title="逾期率",
    template='plotly_white',
    margin=dict(l=20, r=20, t=60, b=20), title_text=''
)
fig_debt_threshold.update_yaxes(tickformat='.0%', range=[0.10, 0.16])
fig_debt_threshold.update_xaxes(categoryorder='array', categoryarray=["0-2k", "2k-4k", "4k-6k", "6k-8k", "8k-10k", "10k+"])

# 3. 收入债务比率与逾期天数分析（散点+平滑线）
ratio_df = df[(df['Debt'] > 0) & (df['Income'] > 0)].copy()
ratio_df['Overdue_Days'] = ratio_df['Overdue_Days'].clip(lower=0)
ratio_df['income_debt_ratio'] = ratio_df['Income'] / ratio_df['Debt']
ratio_df['income_debt_ratio'] = ratio_df['income_debt_ratio'].clip(0.01, 10)
fig_income_debt = px.scatter(
    ratio_df, x='income_debt_ratio', y='Overdue_Days',
    opacity=0.2, color_discrete_sequence=[color_scale[-1]],
)

import plotly.graph_objs as go
from pygam import GAM, s

# 拟合GAM模型
gam = GAM(n_splines=5).fit(np.log10(ratio_df['income_debt_ratio'].values.reshape(-1, 1)), 
                           ratio_df['Overdue_Days'].values)
XX = gam.generate_X_grid(term=0, n=300)
predicted = gam.predict(XX)

# 创建 Plotly 图形对象
fig_income_debt = go.Figure()

# 添加散点图层
fig_income_debt.add_trace(go.Scatter(
    x=ratio_df['income_debt_ratio'], y=ratio_df['Overdue_Days'],
    mode='markers', marker=dict(color="#2c7bb6", opacity=0.2), name='数据点'
))

# 添加平滑线图层
fig_income_debt.add_trace(go.Scatter(
    x=10**XX[:, 0], y=predicted, mode='lines',
    line=dict(color="#d7191c", width=3), name='平滑线'
))

# 更新布局
fig_income_debt.update_layout(
    xaxis_title="收入/债务比率 (对数坐标)", yaxis_title="逾期天数",
    template='plotly_white',
    margin=dict(l=20, r=20, t=60, b=20), title_text='',
    xaxis_type="log", xaxis_tickvals=[0.01, 0.1, 1, 10],
    xaxis_ticktext=["0.01", "0.1", "1", "10"]
)

# 新增分析文本
analysis_texts = [
    "本图展示了不同性别客户在逾期状态上的比例分布。通过对比男性与女性在'无逾期'、'1-30天逾期'以及'30天以上逾期'三类状态下的占比，可以帮助我们了解性别因素是否对信贷风险有显著影响。可以看到，不同性别中三种逾期状态的占比没有明显差别，因此认为逾期状态与性别无关。",
    "本图反映了客户年龄与其违约概率之间的关系。通过热力图和趋势线，我们可以观察不同年龄段客户的违约风险分布情况。分析该关系有助于识别高风险年龄群体，从而在信贷审批和额度分配时进行针对性调整。可以看到，随着年龄的增长，违约概率大致呈上升趋势。因此，在信贷审批时，可以适当提高年龄较大客户的信用评分门槛。",
    "本图分析了不同教育水平客户的信用评分分布情况。通过对比'中等教育'、'中等专业教育'和'高等教育'三类客户的信用评分分布，可以评估教育背景对信用状况的影响。有较高学历水平的人群的信用评分的中位数更高，说明教育水平在一定程度上反映了客户的信用风险。",
    "本图将客户按收入分组，展示了不同收入群体在各逾期天数区间的分布。通过对比'收入<1500'和'收入≥1500'两组客户在逾期天数上的差异，可以判断收入水平对逾期风险的影响。可以看到，较低收入的人群逾期天数会更多，建议在信贷政策中加强对低收入客户的风险管控。",
    "本图展示了不同居住地区客户的平均信贷限额。通过对比各地区的平均额度，可以发现哪些地区的客户获得的信贷支持更高，哪些地区相对较低。这有助于评估信贷资源的地域分布是否合理，并为后续的市场拓展或风险防控提供参考。",
    "本图通过箱线图展示了各地区信贷限额的分布情况。用户可以选择展示前XXX个地区，进一步分析不同地区客户的信贷额度分布特征。该分析有助于发现某些地区信贷额度分布的异常情况（如极端值、分布偏态等），为信贷政策优化和风险预警提供依据。",
    "本图采用堆叠条形图展示了不同债务区间客户的逾期天数分布。整体来看，0-2k和2k-4k区间客户数量最多，占总样本约63%。高债务客户（10k+）虽有较高严重逾期风险，但也有不少良好还款记录。4k-6k区间客户需重点关注，建议加强还款提醒和流动性支持。",
    "本图展示了各债务区间的逾期率趋势。整体逾期率呈倒U型，2k-6k区间最高，10k+区间反而较低。6k-8k区间为风险拐点，说明高债务客户可能因风控更严格而逾期率下降。0-2k区间逾期率高于10k+，提示低债务客户中也有高风险群体。",
    "本图分析了收入债务比率与逾期天数的关系。低比率（0.01-0.1）客户逾期天数高，随着比率升高逾期天数下降，但高比率区间又有上升趋势，提示高收入低债务客户中也存在特殊风险。建议对极端比率客户加强早期风险干预。"
]

# 结论卡片
demographic_conclusion = dbc.Card(
    dbc.CardBody([
        html.H4("人口统计特征分析结论", className="card-title text-center mb-4"),
        html.P("通过对性别、年龄、教育水平和收入等人口统计特征的分析，我们发现：", className="card-text"),
        html.Ul([
            html.Li("年龄与违约概率呈正相关，应加强对高龄客户的信用审核"),
            html.Li("教育水平与信用评分正相关，高等教育客户信用风险较低"),
            html.Li("低收入人群逾期风险显著高于高收入人群"),
            html.Li("性别因素对逾期状态无明显影响")
        ], className="mb-4"),
        html.P("建议措施：针对高龄和低收入客户群体实施更严格的风控政策，同时为高教育水平客户提供更优惠的信贷条件。", className="card-text fw-bold")
    ]),
    className="mb-5 shadow-lg border-0",
    style={"background": "linear-gradient(to right, #f8f9fa, #e9ecef)"}
)

geographic_conclusion = dbc.Card(
    dbc.CardBody([
        html.H4("地域特征分析结论", className="card-title text-center mb-4"),
        html.P("通过对不同地区信贷限额和分布的分析，我们发现：", className="card-text"),
        html.Ul([
            html.Li("地区间信贷限额存在显著差异，部分发达地区限额明显较高"),
            html.Li("某些地区的信贷限额分布呈现右偏特征，存在异常高值"),
            html.Li("地区经济发展水平与信贷限额呈正相关关系")
        ], className="mb-4"),
        html.P("建议措施：优化地区信贷资源配置，对高风险地区实施差异化风控策略，同时挖掘高潜力地区的市场机会。", className="card-text fw-bold")
    ]),
    className="mb-5 shadow-lg border-0",
    style={"background": "linear-gradient(to right, #f8f9fa, #e9ecef)"}
)

# 导航栏
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("客户债务与逾期分析", href="/credit-risk", className="fw-bold")),
        dbc.NavItem(dbc.NavLink("人口统计特征", href="/demographic", className="fw-bold")),
        dbc.NavItem(dbc.NavLink("地域特征", href="/geographic", className="fw-bold")),
        dbc.NavItem(dbc.NavLink("小组分工", href="/team", className="fw-bold text-warning")),
    ],
    brand="银行信用评分 Dashboard",
    brand_href="/credit-risk",
    color="primary",
    dark=True,
    className="mb-4 sticky-top"
)

# 页面布局
demographic_page = html.Div([
    navbar,
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("人口统计特征对信贷健康的影响", className="text-center my-4 fw-bold"), width=12)
        ]),
        dbc.Row([
            dbc.Col(demographic_conclusion, width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("图1. 性别与逾期状态比例分布", className="mb-0 fw-bold")),
                    html.Div(analysis_texts[0], className="p-3 mb-2 bg-light rounded"),
                    dbc.CardBody(dcc.Graph(figure=sex_overdue_fig, style={"height": "520px"}))
                ], className="mb-5 shadow-lg rounded-4 border-0"),
                dbc.Card([
                    dbc.CardHeader(html.H4("图2. 年龄与违约概率的关系", className="mb-0 fw-bold")),
                    html.Div(analysis_texts[1], className="p-3 mb-2 bg-light rounded"),
                    dbc.CardBody(dcc.Graph(figure=hex_fig, style={"height": "520px"}))
                ], className="mb-5 shadow-lg rounded-4 border-0"),
                dbc.Card([
                    dbc.CardHeader(html.H4("图3. 教育水平与信用评分分布分析", className="mb-0 fw-bold")),
                    html.Div(analysis_texts[2], className="p-3 mb-2 bg-light rounded"),
                    dbc.CardBody(dcc.Graph(figure=edu_score_fig, style={"height": "520px"}))
                ], className="mb-5 shadow-lg rounded-4 border-0"),
                dbc.Card([
                    dbc.CardHeader(html.H4("图4. 收入分组与逾期天数分布分析", className="mb-0 fw-bold")),
                    html.Div(analysis_texts[3], className="p-3 mb-2 bg-light rounded"),
                    dbc.CardBody(dcc.Graph(figure=mirror_fig, style={"height": "520px"}))
                ], className="mb-5 shadow-lg rounded-4 border-0"),
                # dbc.Card([
                #     dbc.CardHeader(html.H4("图5. 债务区间与逾期天数分布", className="mb-0 fw-bold")),
                #     html.Div(analysis_texts[6], className="p-3 mb-2 bg-light rounded"),
                #     dbc.CardBody(dcc.Graph(figure=fig_debt_overdue, style={"height": "520px"}))
                # ], className="mb-5 shadow-lg rounded-4 border-0"),
                # dbc.Card([
                #     dbc.CardHeader(html.H4("图6. 债务阈值效应分析", className="mb-0 fw-bold")),
                #     html.Div(analysis_texts[7], className="p-3 mb-2 bg-light rounded"),
                #     dbc.CardBody(dcc.Graph(figure=fig_debt_threshold, style={"height": "520px"}))
                # ], className="mb-5 shadow-lg rounded-4 border-0"),
                # dbc.Card([
                #     dbc.CardHeader(html.H4("图7. 收入债务比率与逾期天数分析", className="mb-0 fw-bold")),
                #     html.Div(analysis_texts[8], className="p-3 mb-2 bg-light rounded"),
                #     dbc.CardBody(dcc.Graph(figure=fig_income_debt, style={"height": "520px"}))
                # ], className="mb-5 shadow-lg rounded-4 border-0")
            ], width=12)
        ])
    ], fluid=True, style={"background": "#f8f9fa", "minHeight": "100vh", "paddingBottom": "40px"})
])

geographic_page = html.Div([
    navbar,
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("地域与信贷限额的关联", className="text-center my-4 fw-bold"), width=12)
        ]),
        dbc.Row([
            dbc.Col(geographic_conclusion, width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("图8. 各地域平均信贷限额对比", className="mb-0 fw-bold")),
                    html.Div(analysis_texts[4], className="p-3 mb-2 bg-light rounded"),
                    dbc.CardBody(dcc.Graph(figure=bar_fig, style={"height": "520px"}))
                ], className="mb-5 shadow-lg rounded-4 border-0"),
                dbc.Card([
                    dbc.CardHeader(html.H4("图9. 各地域信贷限额分布（可交互）", className="mb-0 fw-bold")),
                    html.Div(analysis_texts[5], className="p-3 mb-2 bg-light rounded"),
                    dbc.CardBody([
                        html.Label("选择展示的地区数量:", className="fw-bold mb-3"),
                        dcc.Slider(
                            id='slider-num-areas',
                            min=1,
                            max=len(all_areas),
                            value=10,
                            marks={i: f'{i}个' for i in range(1, len(all_areas)+1, 5)},
                            step=1,
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        dcc.Graph(id='box-plot', style={"height": "520px"})
                    ])
                ], className="mb-5 shadow-lg rounded-4 border-0")
            ], width=12)
        ])
    ], fluid=True, style={"background": "#f8f9fa", "minHeight": "100vh", "paddingBottom": "40px"})
])

# ========== 新增：客户债务与逾期行为分析页面 ==========
# 1. 研究背景与意义
credit_risk_intro = dbc.Card([
    dbc.CardBody([
        html.H4("基于多维特征的客户信贷风险分析与可视化报告", className="card-title text-center mb-4"),
        html.P("客户债务分布与逾期行为的关系", className="text-center mb-4 fw-bold"),
        html.P("在现代金融风险管理中，信贷风险评估是银行和金融机构的核心业务之一。随着金融科技的快速发展，传统信贷评分模型已无法完全满足风险管理的需求。根据世界银行2022年全球金融发展报告，全球范围内约有17%的贷款会出现不同程度的逾期情况，其中约5%会发展为不良贷款。因此，深入分析客户债务特征与逾期行为之间的关系，对于提高银行风险管理能力、优化信贷决策具有重要实践价值。", className="card-text"),
        html.P("本报告基于某金融机构18,420名客户的信贷数据，重点分析客户债务水平与逾期行为之间的关系，旨在回答以下关键问题：", className="card-text"),
        html.Ul([
            html.Li("不同债务水平的客户在逾期行为上是否存在显著差异？"),
            html.Li("是否存在特定的债务阈值，超过该阈值后客户逾期风险会显著增加？"),
            html.Li("如何通过可视化方法直观展示债务与逾期行为的关系模式？")
        ])
    ])
], className="mb-5 shadow-lg border-0", style={"background": "linear-gradient(to right, #f8f9fa, #e9ecef)"})

# 2. 数据概述
data_overview = dbc.Card([
    dbc.CardBody([
        html.H5("数据概述", className="card-title mb-3"),
        html.P("本分析使用的数据集包含18,420条客户记录，涉及18个变量。核心变量包括：", className="card-text"),
        html.Ul([
            html.Li("Задолженность (债务金额)：客户当前总债务水平（货币单位）"),
            html.Li("Просрочка, дни (逾期天数)：客户当前贷款逾期天数"),
            html.Li("INCOME (收入)：客户月收入"),
            html.Li("SCORINGMARK (信用评分)：客户信用评分")
        ])
    ])
], className="mb-5 shadow-lg border-0", style={"background": "linear-gradient(to right, #f8f9fa, #e9ecef)"})

# 3.1-3.3 分析与可视化（直接复用前面已生成的三个图和分析文字）
credit_analysis_cards = [
    dbc.Card([
        dbc.CardHeader(html.H4("图1. 债务区间与逾期天数分布（堆叠条形图）", className="mb-0 fw-bold")),
        html.Div("为直观展示不同债务水平客户的逾期情况分布，我们首先采用堆叠条形图呈现各债务区间内客户的逾期天数构成。这种可视化方法可以同时展示债务区间的客户数量分布和逾期严重程度的构成比例。", className="p-3 mb-2 bg-light rounded"),
        dbc.CardBody(dcc.Graph(figure=fig_debt_overdue, style={"height": "520px"})),
        html.Div(analysis_texts[6], className="p-3 mb-2 bg-light rounded")
    ], className="mb-5 shadow-lg rounded-4 border-0"),
    dbc.Card([
        dbc.CardHeader(html.H4("图2. 不同债务区间的客户逾期率（逾期率趋势图）", className="mb-0 fw-bold")),
        html.Div("为验证\"债务阈值\"假设—即是否存在某一债务水平，超过该水平后逾期概率会显著上升，我们绘制了各债务区间的逾期率趋势线图。该分析有助于识别风险管理的临界点。", className="p-3 mb-2 bg-light rounded"),
        dbc.CardBody(dcc.Graph(figure=fig_debt_threshold, style={"height": "520px"})),
        html.Div(analysis_texts[7], className="p-3 mb-2 bg-light rounded")
    ], className="mb-5 shadow-lg rounded-4 border-0"),
    dbc.Card([
        dbc.CardHeader(html.H4("图3. 收入债务比率与逾期天数分析", className="mb-0 fw-bold")),
        html.Div("为探究客户的财务健康度对逾期行为的影响，我们构建了收入债务比率（月收入/总债务）指标。该指标能够反映客户的偿债能力，理论上比率越低意味着偿债压力越大。通过散点图结合平滑曲线，可以直观展示财务压力与逾期行为的非线性关系。", className="p-3 mb-2 bg-light rounded"),
        dbc.CardBody(dcc.Graph(figure=fig_income_debt, style={"height": "520px"})),
        html.Div(analysis_texts[8], className="p-3 mb-2 bg-light rounded")
    ], className="mb-5 shadow-lg rounded-4 border-0")
]

# 4. 主要发现
credit_findings = dbc.Card([
    dbc.CardBody([
        html.H5("主要发现", className="card-title mb-3"),
        html.Ul([
            html.Li("债务-逾期关系的分析中，我们揭示了债务水平与逾期风险之间并非简单的线性关系，中等债务水平（4k-6k）的客户反而表现出最高的逾期风险。"),
            html.Li("在债务阈值效应分析中，我们确认了债务阈值的存在，特别是6k左右的债务水平可能代表了风险特征的转变点。"),
            html.Li("在收入债务比率与逾期天数分析中，我们识别出了逾期天数随着收入债务比率先下降再上升的趋势，这提示我们对还债压力小的人群也需要加强早期风险干预，建立预警机制。")
        ])
    ])
], className="mb-5 shadow-lg border-0", style={"background": "linear-gradient(to right, #f8f9fa, #e9ecef)"})

credit_risk_page = html.Div([
    navbar,
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("客户债务与逾期行为分析", className="text-center my-4 fw-bold"), width=12)
        ]),
        dbc.Row([
            dbc.Col(credit_risk_intro, width=12)
        ]),
        dbc.Row([
            dbc.Col(data_overview, width=12)
        ]),
        dbc.Row([
            dbc.Col(card, width=12) for card in credit_analysis_cards
        ]),
        dbc.Row([
            dbc.Col(credit_findings, width=12)
        ])
    ], fluid=True, style={"background": "#f8f9fa", "minHeight": "100vh", "paddingBottom": "40px"})
])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

team_page = html.Div([
    navbar,
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("小组分工", className="text-center my-4 fw-bold"), width=12)
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4("成员与任务分工", className="card-title text-center mb-4"),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Span("1. 井语若", className="fw-bold me-2 text-primary"),
                                html.Span("客户债务与逾期分析可视化代码、PPT制作")
                            ], className="py-3 px-4 border-0 bg-light rounded mb-2 shadow-sm"),
                            dbc.ListGroupItem([
                                html.Span("2. 乔一诺", className="fw-bold me-2 text-success"),
                                html.Span("人口统计特征分析可视化代码、口头汇报")
                            ], className="py-3 px-4 border-0 bg-light rounded mb-2 shadow-sm"),
                            dbc.ListGroupItem([
                                html.Span("3. 滕若萱", className="fw-bold me-2 text-info"),
                                html.Span("地域特征分析可视化代码、html报告制作、口头汇报")
                            ], className="py-3 px-4 border-0 bg-light rounded shadow-sm")
                        ], className="border-0"),
                        html.Div("感谢每位成员的辛勤付出！", className="text-center mt-4 fw-bold text-secondary")
                    ])
                ], className="shadow-lg rounded-4 border-0", style={"background": "linear-gradient(to right, #f8f9fa, #e9ecef)"}),
                width=8, className="offset-2"
            )
        ])
    ], fluid=True, style={"background": "#f8f9fa", "minHeight": "100vh", "paddingBottom": "40px"})
])                        


# 回调函数：根据滚动条值更新箱线图
@app.callback(
    Output('box-plot', 'figure'),
    [Input('slider-num-areas', 'value')]
)
def update_graph(num_areas):
    top_areas = all_areas[:num_areas]
    filtered_df = df[df['Living_Area'].isin(top_areas)]
    fig = px.box(
        filtered_df,
        x='Living_Area',
        y='Initial_Limit',
        color='Living_Area',
        labels={'Living_Area': '居住地区', 'Initial_Limit': '初始信贷限额'},
        template='plotly_white',
        color_discrete_sequence=color_sequence
    )
    fig.update_layout(
        xaxis_title='居住地区',
        yaxis_title='初始信贷限额(元)',
        height=520,
        showlegend=False,
        xaxis={'categoryorder':'total descending'},
        margin=dict(l=20, r=20, t=60, b=20),
        title_text=''
    )
    return fig

# 页面切换回调
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/geographic':
        return geographic_page
    elif pathname == '/demographic':
        return demographic_page
    elif pathname == '/credit-risk':
        return credit_risk_page
    elif pathname == '/team':
        return team_page
    else:
        return credit_risk_page


# if __name__ == '__main__':
#     app.run(debug=True)