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

# 统一配色
color_seq = px.colors.sequential.Plasma

# 图1
sex_overdue_fig = px.histogram(
    df.dropna(subset=['Sex', 'Overdue_Group']),
    x="Sex",
    color="Overdue_Group",
    barmode="relative",
    barnorm="percent",
    text_auto=True,
    color_discrete_sequence=color_seq,
    category_orders={"Overdue_Group": ["无逾期", "1-30天", "30天以上"], "Sex": ["男", "女"]},
    title="图1. 性别与逾期状态比例分布"
)
sex_overdue_fig.update_layout(
    yaxis_title="比例",
    xaxis_title="性别",
    legend_title="逾期状态",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template='plotly_white',
    margin=dict(l=20, r=20, t=60, b=20)
)
sex_overdue_fig.update_yaxes(tickformat=".0%")

# 图2
age_prob_df = df.dropna(subset=['Age', 'Probability_of_Default'])
hex_fig = px.density_heatmap(
    age_prob_df, x="Age", y="Probability_of_Default", nbinsx=30, nbinsy=30,
    color_continuous_scale=color_seq,
    title="图2. 年龄与违约概率的关系"
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
    margin=dict(l=20, r=20, t=60, b=20)
)

# 图3
edu_score_df = df.dropna(subset=['Education', 'Scoring_Mark'])
edu_score_fig = px.violin(
    edu_score_df, x="Education", y="Scoring_Mark", box=True, points="all",
    color="Education", color_discrete_sequence=color_seq,
    title="图3. 教育水平与信用评分分布分析"
)
edu_score_fig.update_layout(
    xaxis_title="教育水平",
    yaxis_title="信用评分",
    showlegend=False,
    template='plotly_white',
    margin=dict(l=20, r=20, t=60, b=20)
)

# 图4（条形图，排序+颜色随数额）
bar_df = df.groupby('Living_Area')['Initial_Limit'].mean().reset_index()
bar_df = bar_df.sort_values('Initial_Limit', ascending=False)
bar_fig = px.bar(
    bar_df,
    x='Living_Area',
    y='Initial_Limit',
    color='Initial_Limit',
    color_continuous_scale=color_seq,
    title='图4. 各地域平均信贷限额对比',
    labels={'Living_Area': '地区', 'Initial_Limit': '平均信贷限额'},
    category_orders={'Living_Area': bar_df['Living_Area'].tolist()}
)
bar_fig.update_layout(
    xaxis_tickangle=-45,
    xaxis_title='居住地区',
    yaxis_title='平均初始信贷限额(元)',
    showlegend=False,
    template='plotly_white',
    margin=dict(l=20, r=20, t=60, b=20)
)

# 顶部导航栏
navbar = dbc.NavbarSimple(
    brand="银行信用评分 Dashboard",
    color="primary",
    dark=True,
    className="mb-4"
)

# 布局设置
app.layout = html.Div([
    navbar,
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.H1("银行信用评分数据分析仪表板（含R分析复刻）", 
                    className="text-center mb-4 mt-4 fw-bold",
                    style={"fontSize": "2.8rem", "letterSpacing": "2px", "color": "#2d3142"}
                ), width=12
            )
        ]),
        dbc.Row([
            dbc.Col([
                # 图1
                dbc.Card([
                    dbc.CardHeader(html.H4("图1. 性别与逾期状态比例分布", className="mb-0 fw-bold")),
                    html.Div("", className="p-3 mb-2 bg-light rounded", style={"minHeight": "60px"}),
                    dbc.CardBody(dcc.Graph(figure=sex_overdue_fig, style={"height": "520px"}, config={'responsive': True}))
                ], className="mb-5 shadow-lg rounded-4 border-0"),
                # 图2
                dbc.Card([
                    dbc.CardHeader(html.H4("图2. 年龄与违约概率的关系", className="mb-0 fw-bold")),
                    html.Div("", className="p-3 mb-2 bg-light rounded", style={"minHeight": "60px"}),
                    dbc.CardBody(dcc.Graph(figure=hex_fig, style={"height": "520px"}, config={'responsive': True}))
                ], className="mb-5 shadow-lg rounded-4 border-0"),
                # 图3
                dbc.Card([
                    dbc.CardHeader(html.H4("图3. 教育水平与信用评分分布分析", className="mb-0 fw-bold")),
                    html.Div("", className="p-3 mb-2 bg-light rounded", style={"minHeight": "60px"}),
                    dbc.CardBody(dcc.Graph(figure=edu_score_fig, style={"height": "520px"}, config={'responsive': True}))
                ], className="mb-5 shadow-lg rounded-4 border-0"),
                # 图4
                dbc.Card([
                    dbc.CardHeader(html.H4("图4. 各地域平均信贷限额对比", className="mb-0 fw-bold")),
                    html.Div("", className="p-3 mb-2 bg-light rounded", style={"minHeight": "60px"}),
                    dbc.CardBody(dcc.Graph(figure=bar_fig, style={"height": "520px"}, config={'responsive': True}))
                ], className="mb-5 shadow-lg rounded-4 border-0"),
            ], width=12)
        ], className="mb-5"),
        # 交互式箱线图
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("图5. 各地域信贷限额分布（可交互）", className="mb-0 fw-bold")),
                    html.Div("", className="p-3 mb-2 bg-light rounded", style={"minHeight": "60px"}),
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
                        dcc.Graph(id='box-plot', style={"height": "520px"}, config={'responsive': True})
                    ])
                ], className="mb-5 shadow-lg rounded-4 border-0")
            ], width=12)
        ], className="mb-5"),
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
        title=f'图5. 信贷限额最高的前 {num_areas} 个地区分布',
        labels={'Living_Area': '居住地区', 'Initial_Limit': '初始信贷限额'},
        template='plotly_white'
    )
    fig.update_layout(
        xaxis_title='居住地区',
        yaxis_title='初始信贷限额(元)',
        height=520,
        showlegend=False,
        xaxis={'categoryorder':'total descending'},
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# 注释掉本地开发时的直接运行方式
# if __name__ == '__main__':
#     app.run(debug=True)

# 如果在 Render 上部署，则不需要此部分，而是通过 wsgi.py 来启动应用