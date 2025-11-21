import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from statsmodels.tsa.seasonal import seasonal_decompose

# Загрузка данных
df = pd.read_csv('Raisin_Dataset.csv')

# Признаки и метки
X = df.drop(columns='Class')
y = df['Class']

# Кодирование меток классов
lb = LabelBinarizer()
y_bin = lb.fit_transform(y)

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Если всего 2 класса, корректируем y_bin для единообразия формы
if y_bin.shape[1] == 1:
    y_bin = np.hstack([1 - y_bin, y_bin])

# Делим на train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_scaled, y_bin, test_size=0.3, random_state=42)

# Обучаем Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Предсказания
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

# Конфузионная матрица
cm = confusion_matrix(y_test, y_pred, labels=lb.classes_)

# ROC и PR-кривые для каждого класса
roc_traces = []
pr_traces = []
for i, class_label in enumerate(lb.classes_):
    y_true = y_test_bin[:, i]
    y_score = y_pred_proba[:, i]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    roc_traces.append(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC {class_label}, AUC={roc_auc:.2f}'))
    pr_traces.append(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR {class_label}, AUC={pr_auc:.2f}'))

# Иерархическая кластеризация linkage (для дендрограммы)
linkage_matrix = linkage(X_scaled, method='ward')

# Бар график распределения классов
class_counts = y.value_counts().reset_index()
class_counts.columns = ['Class', 'Count']

# Опции для выбора признаков (выпадающие списки)
feature_options = [{'label': col, 'value': col} for col in X.columns]

# Создаем Dash приложение
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Классификация"),
    dcc.Graph(
        id='confusion-matrix',
        figure=px.imshow(cm,
                         labels=dict(x="Предсказание", y="Истина", color="Количество"),
                         x=lb.classes_,
                         y=lb.classes_,
                         title='Матрица ошибок')
    ),
    dcc.Graph(
        id='roc-curve',
        figure={
            'data': roc_traces,
            'layout': go.Layout(title='ROC-кривые',
                                xaxis={'title': 'False Positive Rate'},
                                yaxis={'title': 'True Positive Rate'},
                                hovermode='closest')
        }
    ),
    dcc.Graph(
        id='pr-curve',
        figure={
            'data': pr_traces,
            'layout': go.Layout(title='PR-кривые',
                                xaxis={'title': 'Recall'},
                                yaxis={'title': 'Precision'},
                                hovermode='closest')
        }
    ),
    dcc.Graph(
        id='class-distribution',
        figure=px.bar(class_counts, x='Class', y='Count', title='Распределение классов')
    ),

    html.H1("Кластеризация"),
    html.Div([
        dcc.Dropdown(
            id='feature-x',
            options=feature_options,
            value=X.columns[0],
            clearable=False,
            style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%'}
        ),
        dcc.Dropdown(
            id='feature-y',
            options=feature_options,
            value=X.columns[1],
            clearable=False,
            style={'width': '45%', 'display': 'inline-block'}
        )
    ]),
    dcc.Graph(id='scatter-clusters'),

    dcc.Graph(
        id='dendrogram',
        figure=ff.create_dendrogram(X_scaled, linkagefun=lambda x: linkage_matrix)
          .update_layout(title='Дендрограмма иерархической кластеризации')
    ),

    html.H1("Анализ временных рядов"),
    dcc.Interval(id='interval', interval=1, n_intervals=0, max_intervals=1),

    dcc.Graph(id='timeseries-plot'),
    dcc.Graph(id='trend-plot'),
    dcc.Graph(id='seasonal-plot'),
    dcc.Graph(id='residual-plot'),
])

@app.callback(
    Output('scatter-clusters', 'figure'),
    [Input('feature-x', 'value'),
     Input('feature-y', 'value')]
)
def update_scatter_clusters(feature_x, feature_y):
    X_selected = df[[feature_x, feature_y]]
    scaler_local = StandardScaler()
    X_scaled_local = scaler_local.fit_transform(X_selected)

    kmeans_local = KMeans(n_clusters=2, random_state=42)
    clusters_local = kmeans_local.fit_predict(X_scaled_local)

    fig = px.scatter(
        df, x=feature_x, y=feature_y, color=clusters_local.astype(str),
        title=f'Диаграмма рассеяния с KMeans кластерами ({feature_x} vs {feature_y})',
        labels={feature_x: feature_x, feature_y: feature_y}
    )
    return fig

@app.callback(
    [Output('timeseries-plot', 'figure'),
     Output('trend-plot', 'figure'),
     Output('seasonal-plot', 'figure'),
     Output('residual-plot', 'figure')],
    [Input('interval', 'n_intervals')]
)
def update_time_series(n):
    np.random.seed(42)
    time_index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    series_data = pd.Series(df[df.columns[0]].values, index=time_index)
    result = seasonal_decompose(series_data, model='additive', period=7, extrapolate_trend='freq')

    fig_original = go.Figure([go.Scatter(x=series_data.index, y=series_data.values, mode='lines', name='Исходные')])
    fig_original.update_layout(title='Исходный временной ряд')

    fig_trend = go.Figure([go.Scatter(x=result.trend.index, y=result.trend.values, mode='lines', name='Тренд')])
    fig_trend.update_layout(title='Тренд')

    fig_seasonal = go.Figure([go.Scatter(x=result.seasonal.index, y=result.seasonal.values, mode='lines', name='Сезонность')])
    fig_seasonal.update_layout(title='Сезонность')

    fig_resid = go.Figure([go.Scatter(x=result.resid.index, y=result.resid.values, mode='lines', name='Остатки')])
    fig_resid.update_layout(title='Остатки')

    return fig_original, fig_trend, fig_seasonal, fig_resid

if __name__ == '__main__':
    app.run(debug=True, port=8051)
