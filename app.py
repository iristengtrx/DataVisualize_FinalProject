import os
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# 初始化Dash应用
app = Dash(__name__)

# 读取数据
df = pd.read_csv("bank_credit_scoring.csv")

# 字段重命名
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

# 获取所有唯一地区并按平均信贷限额排序
area_limit_stats = df.groupby('Living_Area')['Initial_Limit'].mean().sort_values(ascending=False)
all_areas = area_limit_stats.index.tolist()

# 添加模拟年份列用于动画效果
np.random.seed(42)
df['year'] = np.random.choice(range(2018, 2023), size=len(df))

# 新增R分析复刻图表
sex_overdue_fig = px.histogram(
    df.dropna(subset=['Sex', 'Overdue_Group']),
    x="Sex",
    color="Overdue_Group",
    barmode="relative",
    barnorm="percent",
    text_auto=True,
    color_discrete_sequence=px.colors.qualitative.Set2,
    category_orders={"Overdue_Group": ["无逾期", "1-30天", "30天以上"], "Sex": ["男", "女"]}
)
sex_overdue_fig.update_layout(
    title="性别与逾期状态比例分布",
    yaxis_title="比例",
    xaxis_title="性别",
    legend_title="逾期状态",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
sex_overdue_fig.update_yaxes(tickformat=".0%")

age_prob_df = df.dropna(subset=['Age', 'Probability_of_Default'])
hex_fig = px.density_heatmap(
    age_prob_df, x="Age", y="Probability_of_Default", nbinsx=30, nbinsy=30,
    color_continuous_scale=["#fcbba1", "#67090d"]
)
# 平滑线
if not age_prob_df.empty:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smooth = lowess(age_prob_df['Probability_of_Default'], age_prob_df['Age'], frac=0.2)
    hex_fig.add_trace(go.Scatter(x=smooth[:, 0], y=smooth[:, 1], mode='lines', line=dict(color='#440154', width=2), name='平滑线'))
hex_fig.update_layout(
    title="年龄与违约概率的关系",
    xaxis_title="年龄",
    yaxis_title="违约概率"
)

edu_score_df = df.dropna(subset=['Education', 'Scoring_Mark'])
edu_score_fig = px.violin(
    edu_score_df, x="Education", y="Scoring_Mark", box=True, points="all",
    color="Education", color_discrete_sequence=["#4daf4a", "#377eb8", "#984ea3"]
)
edu_score_fig.update_layout(
    title="教育水平与信用评分分布分析",
    xaxis_title="教育水平",
    yaxis_title="信用评分",
    showlegend=False
)

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
    color_discrete_sequence=["#1b9e77", "#d95f02"],
    text="count", barmode="relative"
)
mirror_fig.update_traces(textposition='outside')
mirror_fig.update_layout(
    title="收入分组与逾期天数分布分析",
    xaxis_title="逾期状态",
    yaxis_title="客户数量",
    legend_title="",
    barmode="relative"
)
mirror_fig.update_yaxes(tickvals=[-max(mirror_df['count']), 0, max(mirror_df['count'])], ticktext=[str(max(mirror_df['count'])), "0", str(max(mirror_df['count']))])

# 布局设置
app.layout = html.Div([
    html.H1("银行信用评分数据分析仪表板（含R分析复刻）"),

    # 新增R分析复刻图表
    html.Div([
        html.H3("A. 性别与逾期状态比例分布"),
        dcc.Graph(figure=sex_overdue_fig),
        html.H3("B. 年龄与违约概率的关系"),
        dcc.Graph(figure=hex_fig),
        html.H3("C. 教育水平与信用评分分布分析"),
        dcc.Graph(figure=edu_score_fig),
        html.H3("D. 收入分组与逾期天数分布分析"),
        dcc.Graph(figure=mirror_fig),
    ], style={'width': '100%', 'display': 'inline-block'}),

    # 箱线图部分
    html.Div([
        html.H3("1. 各地域信贷限额分布"),
        html.Label("选择展示的地区数量:"),
        dcc.Slider(
            id='slider-num-areas',
            min=1,
            max=len(all_areas),
            value=10,  # 默认值
            marks={i: f'{i}个' for i in range(1, len(all_areas)+1, 5)},
            step=1
        ),
        dcc.Graph(id='box-plot')
    ], style={'width': '100%', 'display': 'inline-block'}),
    
    # 散点图部分
    html.Div([
        html.H3("2. 收入与信贷限额关系随时间变化"),
        dcc.Graph(figure=px.scatter(
            df, 
            x='Income', 
            y='Initial_Limit', 
            color='Living_Area',
            animation_frame='year',
            size='Probability_of_Default',
            hover_name='Settlement_Name',
            title='各地域收入与信贷限额关系随时间变化',
            labels={'Income': '收入', 'Initial_Limit': '信贷限额', 'Living_Area': '地区', 
                   'Probability_of_Default': '违约概率', 'year': '年份'},
            template='plotly_white'
        ).update_layout(
            height=700,
            width=1200,
            xaxis_title='客户收入(元)',
            yaxis_title='初始信贷限额(元)',
            legend_title='居住地区'
        ))
    ], style={'width': '100%', 'display': 'inline-block'}),
    
    # 条形图部分
    html.Div([
        html.H3("3. 各地域平均信贷限额对比"),
        dcc.Graph(
            figure=px.bar(
                df.groupby('Living_Area')['Initial_Limit'].mean().reset_index(),
                x='Living_Area',
                y='Initial_Limit',
                title='各地域平均信贷限额对比',
                labels={'Living_Area': '地区', 'Initial_Limit': '平均信贷限额'},
                template='plotly_white'
            ).update_layout(
                xaxis_tickangle=-45,
                height=600,
                width=1000,
                xaxis_title='居住地区',
                yaxis_title='平均初始信贷限额(元)',
                showlegend=False
            )
        )
    ], style={'width': '100%', 'display': 'inline-block'}),
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
        title=f'信贷限额最高的前 {num_areas} 个地区分布',
        labels={'Living_Area': '居住地区', 'Initial_Limit': '初始信贷限额'},
        template='plotly_white'
    )

    fig.update_layout(
        xaxis_title='居住地区',
        yaxis_title='初始信贷限额(元)',
        height=600,
        width=1000,
        showlegend=False,
        xaxis={'categoryorder':'total descending'}
    )
    
    return fig

# 注释掉本地开发时的直接运行方式
# if __name__ == '__main__':
#     app.run(debug=True)

# 如果在 Render 上部署，则不需要此部分，而是通过 wsgi.py 来启动应用