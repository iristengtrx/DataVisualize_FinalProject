import os
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np

# 初始化Dash应用
app = Dash(__name__)

# 读取数据
df = pd.read_csv("bank_credit_scoring.csv")

# 获取所有唯一地区并按平均信贷限额排序
area_limit_stats = df.groupby('LV_AREA')['Первоначльный лимит'].mean().sort_values(ascending=False)
all_areas = area_limit_stats.index.tolist()

# 添加模拟年份列用于动画效果
np.random.seed(42)
df['year'] = np.random.choice(range(2018, 2023), size=len(df))

# 布局设置
app.layout = html.Div([
    html.H1("银行信用评分数据分析仪表板"),
    
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
        dcc.Graph(figure=px.scatter(df, 
                                    x='INCOME', 
                                    y='Первоначльный лимит', 
                                    color='LV_AREA',
                                    animation_frame='year',
                                    size='PDN',
                                    hover_name='LV_SETTLEMENTNAME',
                                    title='各地域收入与信贷限额关系随时间变化',
                                    labels={'INCOME': '收入', 'Первоначльный лимит': '信贷限额', 'LV_AREA': '地区', 
                                           'PDN': '违约概率', 'year': '年份'},
                                    template='plotly_white').update_layout(
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
            figure=px.bar(df.groupby('LV_AREA')['Первоначльный лимит'].mean().reset_index(),
                          x='LV_AREA',
                          y='Первоначльный лимит',
                          title='各地域平均信贷限额对比',
                          labels={'LV_AREA': '地区', 'Первоначльный лимит': '平均信贷限额'},
                          template='plotly_white').update_layout(
                              xaxis_tickangle=-45,
                              height=600,
                              width=1000,
                              xaxis_title='居住地区',
                              yaxis_title='平均初始信贷限额(元)',
                              showlegend=False
                          )
        )
    ], style={'width': '100%', 'display': 'inline-block'})
])

# 回调函数：根据滚动条值更新箱线图
@app.callback(
    Output('box-plot', 'figure'),
    [Input('slider-num-areas', 'value')]
)
def update_graph(num_areas):
    top_areas = all_areas[:num_areas]
    filtered_df = df[df['LV_AREA'].isin(top_areas)]
    
    fig = px.box(filtered_df,
                 x='LV_AREA',
                 y='Первоначльный лимит',
                 color='LV_AREA',
                 title=f'信贷限额最高的前 {num_areas} 个地区分布',
                 labels={'LV_AREA': '居住地区', 'Первоначльный лимit': '初始信贷限额'},
                 template='plotly_white')

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