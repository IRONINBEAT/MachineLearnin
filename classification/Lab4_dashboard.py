import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

import dash
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# ------------------------------

df = pd.read_csv("Raisin_Dataset.csv")
target_col = "Class"
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# стандартизация данных для boxplot
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

# вычисляем корреляционную матрицу
corr_matrix = df[num_cols].corr().round(3)

# ------------------------------
# 2. ОПИСАТЕЛЬНАЯ СТАТИСТИКА
# ------------------------------

# описательная статистика → транспонируем
desc_stats = df.describe().T

# добавляем имя признака
desc_stats = desc_stats.reset_index().rename(columns={"index": "Признак"})

# округлим значения для красоты
for col in desc_stats.columns:
    if col not in ["Признак", "count"]:
        desc_stats[col] = desc_stats[col].round(3)

# ------------------------------
# 3. DASH APP
# ------------------------------

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Дашборд: Дескриптивный анализ", style={"textAlign": "center"}),

    html.H2("1. Выбор признака для гистограммы"),

    dcc.Dropdown(
        id="feature-dropdown",
        options=[{"label": col, "value": col} for col in num_cols],
        value=num_cols[0],
        clearable=False,
        style={"width": "300px"}
    ),

    # контейнер для гистограммы + pie chart + корреляционная матрица
    html.Div(id="single-row-container"),

    html.Br(),

    # кнопка показать/скрыть все
    html.Button(
        "Показать/скрыть все гистограммы",
        id="toggle-all-btn",
        n_clicks=0,
        style={"margin": "10px"}
    ),

    html.Div(id="all-histograms-container"),

    html.H2("2. Boxplot и таблица описательной статистики"),

    # BOX + TABLE в одной строке
    html.Div([

        html.Div(
            dcc.Graph(
                figure=px.box(df_scaled, y=num_cols,
                              title="Boxplot нормализованных признаков")
            ),
            style={"width": "60%"}
        ),

        html.Div(
            dash_table.DataTable(
                data=desc_stats.to_dict("records"),
                columns=[{"name": c, "id": c} for c in desc_stats.columns],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left"},
            ),
            style={"width": "40%"}
        ),

    ], style={
        "display": "flex",
        "flexDirection": "row",
        "justifyContent": "space-between",
        "alignItems": "flex-start"
    }),

    html.H2("3. Попарные диаграммы рассеивания"),

    # Выбор признаков для scatter matrix
    html.Div([
        html.Label("Выберите признаки для попарных диаграмм рассеивания:"),
        dcc.Dropdown(
            id="scatter-matrix-features",
            options=[{"label": col, "value": col} for col in num_cols],
            value=num_cols[:4],  # первые 4 признака по умолчанию
            multi=True,
            clearable=False,
            style={"width": "80%", "margin": "10px 0"}
        )
    ]),

    html.Div(id="scatter-matrix-container"),

    html.H2("4. Дисперсионный анализ (ANOVA)"),

    # Выбор переменных для ANOVA
    html.Div([
        html.Div([
            html.Label("Выберите числовой признак для анализа:"),
            dcc.Dropdown(
                id="anova-feature",
                options=[{"label": col, "value": col} for col in num_cols],
                value=num_cols[0],
                clearable=False,
                style={"width": "90%", "margin": "10px 0"}
            )
        ], style={"width": "48%", "display": "inline-block"}),

        html.Div([
            html.Label("Выберите группирующую переменную:"),
            dcc.Dropdown(
                id="anova-group",
                options=[{"label": col, "value": col} for col in [target_col]],
                value=target_col,
                clearable=False,
                style={"width": "90%", "margin": "10px 0"}
            )
        ], style={"width": "48%", "display": "inline-block", "float": "right"})
    ]),

    # Результаты ANOVA
    html.Div(id="anova-results", style={"margin": "20px 0"}),

    # Графики ANOVA
    html.Div(id="anova-plots-container"),

    html.H2("5. Факторный анализ (PCA)"),

    # Выбор количества факторов
    html.Div([
        html.Div([
            html.Label("Выберите количество факторов:"),
            dcc.Slider(
                id="n-components",
                min=2,
                max=min(8, len(num_cols)),
                value=3,
                marks={i: str(i) for i in range(2, min(8, len(num_cols)) + 1)},
                step=1
            )
        ], style={"width": "48%", "display": "inline-block"}),

        html.Div([
            html.Label("Метод вращения:"),
            dcc.Dropdown(
                id="rotation-method",
                options=[
                    {"label": "Без вращения", "value": "none"},
                    {"label": "Varimax", "value": "varimax"}
                ],
                value="none",
                clearable=False,
                style={"width": "90%", "margin": "10px 0"}
            )
        ], style={"width": "48%", "display": "inline-block", "float": "right"})
    ]),

    # Результаты факторного анализа
    html.Div(id="factor-results", style={"margin": "20px 0"}),

    # Графики факторного анализа
    html.Div(id="factor-plots-container"),

    html.H2("6. Регрессионный анализ"),

    # Выбор целевой переменной и признака для регрессии
    html.Div([
        html.Div([
            html.Label("Выберите целевую переменную (Y):"),
            dcc.Dropdown(
                id="target-regression",
                options=[{"label": col, "value": col} for col in num_cols],
                value=num_cols[0],
                clearable=False,
                style={"width": "90%", "margin": "10px 0"}
            )
        ], style={"width": "48%", "display": "inline-block"}),

        html.Div([
            html.Label("Выберите признак для регрессии (X):"),
            dcc.Dropdown(
                id="feature-regression",
                options=[{"label": col, "value": col} for col in num_cols],
                value=num_cols[1] if len(num_cols) > 1 else num_cols[0],
                clearable=False,
                style={"width": "90%", "margin": "10px 0"}
            )
        ], style={"width": "48%", "display": "inline-block", "float": "right"})
    ]),

    # Метрики модели
    html.Div(id="regression-metrics", style={"margin": "20px 0"}),

    # Графики регрессии
    html.Div(id="regression-plots-container")

])


# ------------------------------
# 4. CALLBACKS
# ------------------------------

# ---- одиночная гистограмма + pie chart + корреляционная матрица ----
@app.callback(
    Output("single-row-container", "children"),
    Input("feature-dropdown", "value")
)
def update_single_histogram(selected_col):
    # Гистограмма
    fig_hist = px.histogram(df, x=selected_col, nbins=30,
                            title=f"Гистограмма: {selected_col}")

    # Круговая диаграмма
    fig_pie = px.pie(df, names=target_col,
                     title="Распределение видов изюма")

    # Тепловая карта корреляционной матрицы
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        text=corr_matrix.values,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        hoverinfo="none",
        showscale=True
    ))
    
    fig_heatmap.update_layout(
        title="Корреляционная матрица",
        xaxis_title="Признаки",
        yaxis_title="Признаки",
        width=500,
        height=500
    )

    return html.Div([
        # Гистограмма (33%)
        html.Div(
            dcc.Graph(figure=fig_hist),
            style={
                "width": "33%",
                "height": "500px",
                "padding": "10px"
            }
        ),
        # Круговая диаграмма (33%)
        html.Div(
            dcc.Graph(figure=fig_pie),
            style={
                "width": "33%",
                "height": "500px",
                "padding": "10px"
            }
        ),
        # Корреляционная матрица (33%)
        html.Div(
            dcc.Graph(figure=fig_heatmap),
            style={
                "width": "33%",
                "height": "500px",
                "padding": "10px"
            }
        )
    ], style={
        "display": "flex",
        "flexDirection": "row",
        "justifyContent": "space-around"
    })


# ---- показать/скрыть все гистограммы ----
@app.callback(
    Output("all-histograms-container", "children"),
    Input("toggle-all-btn", "n_clicks")
)
def toggle_all_histograms(n_clicks):
    if n_clicks % 2 == 1:

        histograms = [
            html.Div(
                dcc.Graph(
                    figure=px.histogram(df, x=col, nbins=30, title=col),
                    style={"height": "300px"}
                )
            )
            for col in num_cols
        ]

        return html.Div(
            children=histograms,
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, 1fr)",
                "gap": "20px",
                "padding": "10px"
            }
        )

    return ""


# ---- попарные диаграммы рассеивания ----
@app.callback(
    Output("scatter-matrix-container", "children"),
    Input("scatter-matrix-features", "value")
)
def update_scatter_matrix(selected_features):
    if not selected_features or len(selected_features) < 2:
        return html.Div("Выберите как минимум 2 признака для построения диаграмм рассеивания")
    
    # Создаем scatter matrix
    fig_scatter = px.scatter_matrix(
        df,
        dimensions=selected_features,
        color=target_col,
        title=f"Попарные диаграммы рассеивания ({len(selected_features)} признаков)",
        height=800
    )
    
    # Улучшаем читаемость подписей
    fig_scatter.update_traces(diagonal_visible=False)
    fig_scatter.update_layout(
        font_size=10,
        title_font_size=16
    )
    
    return dcc.Graph(figure=fig_scatter)


# ---- дисперсионный анализ (ANOVA) ----
@app.callback(
    [Output("anova-results", "children"),
     Output("anova-plots-container", "children")],
    [Input("anova-feature", "value"),
     Input("anova-group", "value")]
)
def update_anova_analysis(feature, group):
    # Проверка данных
    groups = df[group].unique()
    if len(groups) < 2:
        return "Группирующая переменная должна содержать как минимум 2 группы", ""
    
    # Подготовка данных для ANOVA
    group_data = [df[df[group] == g][feature] for g in groups]
    
    # Односторонний ANOVA
    f_stat, p_value = stats.f_oneway(*group_data)
    
    # Дополнительная статистика по группам
    group_stats = df.groupby(group)[feature].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
    group_stats = group_stats.reset_index()
    
    # Создание графиков
    # 1. Boxplot по группам
    fig_boxplot = px.box(
        df, x=group, y=feature, 
        color=group,
        title=f"Boxplot: {feature} по группам {group}",
        points="all"
    )
    fig_boxplot.update_layout(height=400)
    
    # 2. Violin plot
    fig_violin = px.violin(
        df, x=group, y=feature, 
        color=group,
        title=f"Violin plot: {feature} по группам {group}",
        box=True,
        points="all"
    )
    fig_violin.update_layout(height=400)
    
    # 3. Bar plot со средними значениями
    fig_bar = px.bar(
        group_stats, x=group, y='mean',
        color=group,
        title=f"Средние значения {feature} по группам",
        error_y=group_stats['std'],
        text='mean'
    )
    fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_bar.update_layout(height=400)
    
    # Результаты ANOVA
    significance = "статистически значимые" if p_value < 0.05 else "нестатистически значимые"
    
    anova_results = html.Div([
        html.Div([
            html.H4("Результаты дисперсионного анализа (ANOVA)"),
            html.P(f"F-статистика: {f_stat:.4f}"),
            html.P(f"p-значение: {p_value:.4f}"),
            html.P(f"Результат: {significance} различия между группами", 
                  style={'color': 'red' if p_value < 0.05 else 'green', 'fontWeight': 'bold'}),
            html.P(f"Уровень значимости: α = 0.05"),
            html.Br(),
            html.H5("Описательная статистика по группам:"),
            dash_table.DataTable(
                data=group_stats.to_dict("records"),
                columns=[{"name": col, "id": col} for col in group_stats.columns],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", 'padding': '5px'},
                style_header={'backgroundColor': 'lightgray', 'fontWeight': 'bold'}
            )
        ], style={
            'border': '1px solid #ddd',
            'padding': '15px',
            'borderRadius': '5px',
            'backgroundColor': '#f9f9f9'
        })
    ])
    
    # Контейнер с графиками ANOVA
    plots_html = html.Div([
        html.Div([
            html.Div(dcc.Graph(figure=fig_boxplot), style={"width": "33%", "display": "inline-block"}),
            html.Div(dcc.Graph(figure=fig_violin), style={"width": "33%", "display": "inline-block"}),
            html.Div(dcc.Graph(figure=fig_bar), style={"width": "33%", "display": "inline-block"})
        ], style={"display": "flex", "flexDirection": "row", "justifyContent": "space-around"})
    ])
    
    return anova_results, plots_html


# ---- факторный анализ (PCA) ----
@app.callback(
    [Output("factor-results", "children"),
     Output("factor-plots-container", "children")],
    [Input("n-components", "value"),
     Input("rotation-method", "value")]
)
def update_factor_analysis(n_components, rotation):
    # Выполняем PCA
    pca = PCA(n_components=n_components)
    X_scaled = StandardScaler().fit_transform(df[num_cols])
    principal_components = pca.fit_transform(X_scaled)
    
    # Создаем DataFrame с нагрузками
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'Factor {i+1}' for i in range(n_components)],
        index=num_cols
    ).round(3)
    
    # Объясненная дисперсия
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Собственные значения
    eigenvalues = pca.explained_variance_
    
    # Создание графиков
    # 1. Scree plot (график каменистой осыпи)
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Scatter(
        x=list(range(1, len(eigenvalues) + 1)),
        y=eigenvalues,
        mode='lines+markers',
        name='Собственные значения',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    fig_scree.update_layout(
        title='Scree Plot (График каменистой осыпи)',
        xaxis_title='Номер фактора',
        yaxis_title='Собственное значение',
        height=400
    )
    
    # 2. График объясненной дисперсии
    fig_variance = go.Figure()
    fig_variance.add_trace(go.Bar(
        x=[f'Factor {i+1}' for i in range(n_components)],
        y=explained_variance,
        name='Объясненная дисперсия',
        marker_color='lightblue'
    ))
    fig_variance.add_trace(go.Scatter(
        x=[f'Factor {i+1}' for i in range(n_components)],
        y=cumulative_variance,
        mode='lines+markers',
        name='Кумулятивная дисперсия',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))
    fig_variance.update_layout(
        title='Объясненная дисперсия по факторам',
        xaxis_title='Факторы',
        yaxis_title='Доля объясненной дисперсии',
        height=400
    )
    
    # 3. Heatmap факторных нагрузок
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=loadings_df.values,
        x=loadings_df.columns,
        y=loadings_df.index,
        text=loadings_df.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        hoverinfo="none",
        showscale=True
    ))
    fig_heatmap.update_layout(
        title='Тепловая карта факторных нагрузок',
        xaxis_title='Факторы',
        yaxis_title='Признаки',
        height=500
    )
    
    # 4. Biplot (первые два фактора)
    if n_components >= 2:
        # Создаем biplot
        fig_biplot = go.Figure()
        
        # Добавляем точки наблюдений
        fig_biplot.add_trace(go.Scatter(
            x=principal_components[:, 0],
            y=principal_components[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=df[target_col].astype('category').cat.codes,
                colorscale='viridis',
                showscale=True
            ),
            text=df[target_col],
            name='Наблюдения'
        ))
        
        # Добавляем вектора нагрузок
        scale_factor = 3
        for i, feature in enumerate(num_cols):
            fig_biplot.add_trace(go.Scatter(
                x=[0, loadings[i, 0] * scale_factor],
                y=[0, loadings[i, 1] * scale_factor],
                mode='lines',
                line=dict(color='red', width=2),
                name=feature,
                showlegend=False
            ))
            fig_biplot.add_trace(go.Scatter(
                x=[loadings[i, 0] * scale_factor],
                y=[loadings[i, 1] * scale_factor],
                mode='text',
                text=[feature],
                textposition="middle right",
                showlegend=False
            ))
        
        fig_biplot.update_layout(
            title='Biplot (Факторы 1 и 2)',
            xaxis_title=f'Factor 1 ({explained_variance[0]:.1%})',
            yaxis_title=f'Factor 2 ({explained_variance[1]:.1%})',
            height=500
        )
    else:
        fig_biplot = go.Figure()
        fig_biplot.update_layout(
            title='Biplot недоступен (нужно минимум 2 фактора)',
            height=400
        )
    
    # Результаты факторного анализа
    factor_results = html.Div([
        html.Div([
            html.H4("Результаты факторного анализа (PCA)"),
            html.P(f"Количество факторов: {n_components}"),
            html.P(f"Общая объясненная дисперсия: {cumulative_variance[-1]:.1%}"),
            html.Br(),
            html.H5("Факторные нагрузки:"),
            dash_table.DataTable(
                data=loadings_df.reset_index().rename(columns={'index': 'Признак'}).to_dict("records"),
                columns=[{"name": col, "id": col} for col in ['Признак'] + list(loadings_df.columns)],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", 'padding': '5px'},
                style_header={'backgroundColor': 'lightgray', 'fontWeight': 'bold'},
                page_size=10
            )
        ], style={
            'border': '1px solid #ddd',
            'padding': '15px',
            'borderRadius': '5px',
            'backgroundColor': '#f9f9f9'
        })
    ])
    
    # Контейнер с графиками факторного анализа
    plots_html = html.Div([
        html.Div([
            html.Div(dcc.Graph(figure=fig_scree), style={"width": "50%", "display": "inline-block"}),
            html.Div(dcc.Graph(figure=fig_variance), style={"width": "50%", "display": "inline-block"})
        ], style={"display": "flex", "flexDirection": "row", "justifyContent": "space-around"}),
        html.Div([
            html.Div(dcc.Graph(figure=fig_heatmap), style={"width": "50%", "display": "inline-block"}),
            html.Div(dcc.Graph(figure=fig_biplot), style={"width": "50%", "display": "inline-block"})
        ], style={"display": "flex", "flexDirection": "row", "justifyContent": "space-around", "marginTop": "20px"})
    ])
    
    return factor_results, plots_html


# ---- регрессионный анализ ----
@app.callback(
    [Output("regression-metrics", "children"),
     Output("regression-plots-container", "children")],
    [Input("target-regression", "value"),
     Input("feature-regression", "value")]
)
def update_regression_analysis(target, feature):
    if target == feature:
        return "Целевая переменная и признак не должны совпадать", ""
    
    # Подготовка данных
    X = df[[feature]].values
    y = df[target].values
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X)
    y_test_pred = model.predict(X_test)
    
    # Метрики
    r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    
    # Остатки
    residuals = y_test - y_test_pred
    
    # Формирование уравнения регрессии
    coef = model.coef_[0]
    intercept = model.intercept_
    
    # Определяем знак для красивого отображения уравнения
    sign = "+" if intercept >= 0 else "-"
    abs_intercept = abs(intercept)
    
    equation = f"y = {coef:.4f}·x {sign} {abs_intercept:.4f}"
    
    # Создание графиков
    # 1. Scatterplot с линией регрессии
    fig_scatter = px.scatter(
        df, x=feature, y=target, 
        title=f"Линейная регрессия: {feature} vs {target}<br><sup>Уравнение: {equation}</sup>",
        trendline="ols",
        trendline_color_override="red"
    )
    fig_scatter.update_layout(height=400)
    
    # 2. График остатков
    fig_residuals = px.scatter(
        x=y_test_pred, y=residuals,
        title="График остатков",
        labels={"x": "Предсказанные значения", "y": "Остатки"}
    )
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    fig_residuals.update_layout(height=400)
    
    # 3. График предсказанных vs фактических значений
    fig_actual_vs_pred = px.scatter(
        x=y_test, y=y_test_pred,
        title="Предсказанные vs Фактические значения",
        labels={"x": "Фактические значения", "y": "Предсказанные значения"}
    )
    # Добавляем линию идеального предсказания
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    fig_actual_vs_pred.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                  mode='lines', line=dict(dash='dash', color='red'),
                  name='Идеальное предсказание')
    )
    fig_actual_vs_pred.update_layout(height=400)
    
    # Метрики модели и уравнение регрессии в виде карточек
    metrics_html = html.Div([
        html.Div([
            html.H4("Метрики модели"),
            html.P(f"R² Score: {r2:.4f}"),
            html.P(f"MSE: {mse:.4f}"),
            html.P(f"RMSE: {rmse:.4f}"),
            html.P(f"Коэффициент: {coef:.4f}"),
            html.P(f"Свободный член: {intercept:.4f}")
        ], style={
            'border': '1px solid #ddd',
            'padding': '15px',
            'borderRadius': '5px',
            'backgroundColor': '#f9f9f9',
            'width': '48%',
            'display': 'inline-block',
            'verticalAlign': 'top'
        }),
        
        html.Div([
            html.H4("Уравнение регрессии"),
            html.H3(equation, style={'color': '#e74c3c', 'textAlign': 'center'}),
            html.P("где:", style={'marginTop': '10px'}),
            html.P(f"y - целевая переменная ({target})"),
            html.P(f"x - признак ({feature})"),
            html.P(f"Коэффициент наклона: {coef:.4f}"),
            html.P(f"Свободный член: {intercept:.4f}")
        ], style={
            'border': '1px solid #e74c3c',
            'padding': '15px',
            'borderRadius': '5px',
            'backgroundColor': '#fff5f5',
            'width': '48%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'marginLeft': '4%'
        })
    ])
    
    # Контейнер с графиками
    plots_html = html.Div([
        html.Div([
            html.Div(dcc.Graph(figure=fig_scatter), style={"width": "33%", "display": "inline-block"}),
            html.Div(dcc.Graph(figure=fig_residuals), style={"width": "33%", "display": "inline-block"}),
            html.Div(dcc.Graph(figure=fig_actual_vs_pred), style={"width": "33%", "display": "inline-block"})
        ], style={"display": "flex", "flexDirection": "row", "justifyContent": "space-around"})
    ])
    
    return metrics_html, plots_html


# ------------------------------
# RUN APP
# ------------------------------

if __name__ == "__main__":
    app.run(debug=True)