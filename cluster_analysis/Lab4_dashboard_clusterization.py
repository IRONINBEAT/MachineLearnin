import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('fish_dataset.csv', delimiter=';')

# Предобработка данных
df_clean = df.dropna()
species = df_clean['species']
X = df_clean[['length', 'weight']]

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Создание приложения Dash
app = dash.Dash(__name__)
app.title = "Fish Dataset Clustering Dashboard"

app.layout = html.Div([
    html.H1("Fish Dataset Clustering Analysis", 
            style={'textAlign': 'center', 'color': '#2C3E50', 'marginBottom': 30}),
    
    html.Div([
        html.Div([
            html.Label("Number of Clusters (K-means):"),
            dcc.Slider(
                id='kmeans-slider',
                min=2,
                max=8,
                value=3,
                marks={i: str(i) for i in range(2, 9)},
                step=1
            )
        ], className='six columns', style={'padding': 20}),
        
        html.Div([
            html.Label("Number of Clusters (Hierarchical):"),
            dcc.Slider(
                id='hierarchical-slider',
                min=2,
                max=8,
                value=3,
                marks={i: str(i) for i in range(2, 9)},
                step=1
            )
        ], className='six columns', style={'padding': 20})
    ], className='row'),
    
    html.Div([
        dcc.Graph(id='elbow-plot'),
        dcc.Graph(id='silhouette-analysis')
    ], className='row'),
    
    html.Div([
        dcc.Graph(id='scatter-clusters'),
        dcc.Graph(id='dendrogram-plot')
    ], className='row'),
    
    html.Div([
        dcc.Graph(id='heatmap-clusters'),
        dcc.Graph(id='silhouette-plot')
    ], className='row'),
    
    html.Div([
        html.H3("Cluster Statistics", style={'textAlign': 'center', 'marginTop': 40}),
        html.Div(id='cluster-stats')
    ])
], style={'padding': 20})

# Метод локтя
@app.callback(
    Output('elbow-plot', 'figure'),
    Input('kmeans-slider', 'value')
)
def update_elbow_plot(n_clusters):
    wcss = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=wcss,
        mode='lines+markers',
        name='WCSS',
        line=dict(color='#E74C3C', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_vline(x=n_clusters, line_dash="dash", line_color="green", 
                  annotation_text=f"Selected K={n_clusters}")
    
    fig.update_layout(
        title='Elbow Method for Optimal K',
        xaxis_title='Number of Clusters',
        yaxis_title='Within-Cluster Sum of Squares (WCSS)',
        template='plotly_white'
    )
    
    return fig

# Анализ силуэта
@app.callback(
    Output('silhouette-analysis', 'figure'),
    Input('kmeans-slider', 'value')
)
def update_silhouette_analysis(n_clusters):
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=silhouette_scores,
        mode='lines+markers',
        name='Silhouette Score',
        line=dict(color='#3498DB', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_vline(x=n_clusters, line_dash="dash", line_color="green",
                  annotation_text=f"Selected K={n_clusters}")
    
    fig.update_layout(
        title='Silhouette Analysis for Optimal K',
        xaxis_title='Number of Clusters',
        yaxis_title='Silhouette Score',
        template='plotly_white'
    )
    
    return fig

# Scatter plot с кластерами
@app.callback(
    Output('scatter-clusters', 'figure'),
    [Input('kmeans-slider', 'value'),
     Input('hierarchical-slider', 'value')]
)
def update_scatter_plot(kmeans_k, hierarchical_k):
    # K-means кластеризация
    kmeans = KMeans(n_clusters=kmeans_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Иерархическая кластеризация
    hierarchical = AgglomerativeClustering(n_clusters=hierarchical_k)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    # Используем исходные данные для визуализации
    x_axis, y_axis = X_scaled[:, 0], X_scaled[:, 1]
    x_label, y_label = 'Standardized Length', 'Standardized Weight'
    
    fig = go.Figure()
    
    # K-means кластеры
    for cluster in range(kmeans_k):
        mask = kmeans_labels == cluster
        fig.add_trace(go.Scatter(
            x=x_axis[mask],
            y=y_axis[mask],
            mode='markers',
            name=f'K-means Cluster {cluster}',
            marker=dict(size=8, opacity=0.7),
            legendgroup='kmeans'
        ))
    
    # Центроиды K-means
    centroids_vis = kmeans.cluster_centers_
    
    fig.add_trace(go.Scatter(
        x=centroids_vis[:, 0],
        y=centroids_vis[:, 1],
        mode='markers',
        marker=dict(size=15, symbol='x', color='black', line=dict(width=2)),
        name='K-means Centroids',
        legendgroup='kmeans'
    ))
    
    fig.update_layout(
        title=f'K-means Clustering (K={kmeans_k})',
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        height=500
    )
    
    return fig

# Дендрограмма - исправленная версия
@app.callback(
    Output('dendrogram-plot', 'figure'),
    Input('hierarchical-slider', 'value')
)
def update_dendrogram(n_clusters):
    # Используем подвыборку для дендрограммы (для производительности)
    sample_size = min(100, len(X_scaled))
    indices = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[indices]
    species_sample = species.iloc[indices].values
    
    # Вычисляем linkage matrix
    Z = linkage(X_sample, method='ward')
    
    # Создаем дендрограмму с правильными метками
    try:
        fig = ff.create_dendrogram(
            X_sample, 
            orientation='bottom',
            labels=[f'{species_sample[i]}' for i in range(len(species_sample))],
            linkagefun=lambda x: linkage(x, 'ward'),
            color_threshold=Z[-(n_clusters-1), 2] if n_clusters > 1 else 0
        )
    except:
        # Альтернативный метод если возникает ошибка
        fig = ff.create_dendrogram(
            X_sample, 
            orientation='bottom',
            linkagefun=lambda x: linkage(x, 'ward'),
            color_threshold=Z[-(n_clusters-1), 2] if n_clusters > 1 else 0
        )
    
    fig.update_layout(
        title=f'Hierarchical Clustering Dendrogram (Sample of {sample_size} fish)',
        xaxis_title='Samples',
        yaxis_title='Distance',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    # Добавляем горизонтальную линию для отсечения
    if n_clusters > 1:
        cutoff_height = Z[-(n_clusters-1), 2]
        fig.add_hline(y=cutoff_height, line_dash="dash", line_color="red",
                     annotation_text=f"Cutoff: {cutoff_height:.2f}")
    
    return fig

# Альтернативная простая дендрограмма
def create_simple_dendrogram(n_clusters):
    # Используем подвыборку
    sample_size = min(50, len(X_scaled))
    indices = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[indices]
    
    # Вычисляем linkage matrix
    Z = linkage(X_sample, method='ward')
    
    # Создаем фигуру вручную
    fig = go.Figure()
    
    # Функция для отрисовки дендрограммы
    def plot_dendrogram(icoord, dcoord, color):
        for xs, ys, col in zip(icoord, dcoord, color):
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='lines',
                line=dict(color=col, width=2),
                showlegend=False
            ))
    
    # Получаем данные дендрограммы
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    dendro_data = scipy_dendrogram(Z, no_plot=True)
    
    # Отрисовываем дендрограмму
    plot_dendrogram(dendro_data['icoord'], dendro_data['dcoord'], dendro_data['color_list'])
    
    fig.update_layout(
        title=f'Hierarchical Clustering Dendrogram (K={n_clusters})',
        xaxis_title='Samples',
        yaxis_title='Distance',
        template='plotly_white',
        height=500
    )
    
    return fig

# Heatmap
@app.callback(
    Output('heatmap-clusters', 'figure'),
    Input('kmeans-slider', 'value')
)
def update_heatmap(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Используем подвыборку для heatmap
    sample_size = min(100, len(X_scaled))
    indices = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[indices]
    labels_sample = cluster_labels[indices]
    species_sample = species.iloc[indices].values
    
    # Создаем DataFrame для heatmap
    df_heatmap = pd.DataFrame(X_sample, columns=['Length', 'Weight'])
    df_heatmap['Cluster'] = labels_sample
    df_heatmap['Species'] = species_sample
    
    # Сортируем по кластерам
    df_heatmap = df_heatmap.sort_values('Cluster')
    
    # Heatmap данных
    fig = px.imshow(
        df_heatmap[['Length', 'Weight']].T,
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title=f'Heatmap: Features vs Samples (K={n_clusters}, {sample_size} samples)'
    )
    
    # Добавляем разметку кластеров
    cluster_boundaries = []
    current_cluster = df_heatmap['Cluster'].iloc[0]
    start_idx = 0
    
    for i, cluster in enumerate(df_heatmap['Cluster']):
        if cluster != current_cluster:
            cluster_boundaries.append((start_idx, i-1, current_cluster))
            start_idx = i
            current_cluster = cluster
    
    cluster_boundaries.append((start_idx, len(df_heatmap)-1, current_cluster))
    
    for start, end, cluster_num in cluster_boundaries:
        fig.add_vrect(
            x0=start-0.5, x1=end+0.5,
            fillcolor="rgba(0,0,0,0)", line_color="yellow", line_width=2,
            annotation_text=f"Cluster {cluster_num}", annotation_position="top left"
        )
    
    fig.update_layout(
        xaxis_title='Samples (sorted by cluster)',
        yaxis_title='Features',
        template='plotly_white',
        height=500
    )
    
    return fig

# Silhouette plot
@app.callback(
    Output('silhouette-plot', 'figure'),
    Input('kmeans-slider', 'value')
)
def update_silhouette_plot(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Вычисляем силуэтные коэффициенты
    silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
    
    fig = go.Figure()
    
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        ith_cluster_silhouette_vals.sort()
        
        size_cluster_i = ith_cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        fig.add_trace(go.Scatter(
            x=ith_cluster_silhouette_vals,
            y=np.arange(y_lower, y_upper),
            mode='markers',
            name=f'Cluster {i}',
            marker=dict(size=4)
        ))
        
        # Добавляем среднее значение силуэта для кластера
        fig.add_trace(go.Scatter(
            x=[silhouette_vals[cluster_labels == i].mean()],
            y=[(y_lower + y_upper) / 2],
            mode='markers',
            marker=dict(size=10, symbol='x', color='red'),
            name=f'Cluster {i} Avg',
            showlegend=False
        ))
        
        y_lower = y_upper + 10
    
    # Средний силуэтный коэффициент
    avg_silhouette = silhouette_score(X_scaled, cluster_labels)
    fig.add_vline(x=avg_silhouette, line_dash="dash", line_color="red",
                 annotation_text=f"Overall Avg: {avg_silhouette:.3f}")
    
    fig.update_layout(
        title=f'Silhouette Plot (K={n_clusters})',
        xaxis_title='Silhouette Coefficient Values',
        yaxis_title='Cluster Label',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

# Статистика кластеров
@app.callback(
    Output('cluster-stats', 'children'),
    Input('kmeans-slider', 'value')
)
def update_cluster_stats(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Создаем DataFrame с кластерами
    df_clustered = X.copy()
    df_clustered['Cluster'] = cluster_labels
    df_clustered['Species'] = species.values
    
    # Статистика по кластерам
    cluster_stats = []
    for cluster in range(n_clusters):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
        species_counts = cluster_data['Species'].value_counts()
        top_species = species_counts.head(3).to_dict()
        
        stats = html.Div([
            html.H4(f"Cluster {cluster}", style={'color': '#2C3E50'}),
            html.P(f"📊 Number of fish: {len(cluster_data)}"),
            html.P(f"📏 Average length: {cluster_data['length'].mean():.2f}"),
            html.P(f"⚖️ Average weight: {cluster_data['weight'].mean():.2f}"),
            html.P(f"🐟 Top species: {top_species}"),
            html.Hr()
        ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'margin': '10px', 'borderRadius': '5px'})
        cluster_stats.append(stats)
    
    return html.Div(cluster_stats, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})

# CSS стили
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .row {
                display: flex;
                flex-wrap: wrap;
                margin: 0 -10px;
            }
            .six.columns {
                width: 50%;
                padding: 10px;
                box-sizing: border-box;
            }
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=8050)