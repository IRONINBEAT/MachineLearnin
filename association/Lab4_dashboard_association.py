import pandas as pd
import numpy as np
from collections import Counter
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx

import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

df_raw = pd.read_csv("retail_dataset.csv")
np_data = df_raw.to_numpy()
np_data = [[elem for elem in row if isinstance(elem, str)] for row in np_data if any(isinstance(elem, str) for elem in row)]

# Анализ длин транзакций
transaction_lengths = [len(row) for row in np_data if row]
length_hist = px.histogram(
    x=transaction_lengths,
    nbins=max(transaction_lengths),
    title="Распределение длин транзакций",
    labels={"x": "Длина транзакции", "y": "Частота"}
)
length_hist.update_layout(height=370, width=650)

# Топ-10 самых частых товаров
all_items = [item for row in np_data for item in row]
item_counts = Counter(all_items)
top_items = item_counts.most_common(10)
items, counts = zip(*top_items)
top_items_bar = px.bar(
    x=list(counts), y=list(items), orientation='h',
    title="Топ-10 самых частых товаров",
    labels={"x": "Количество покупок", "y": "Товар"},
    text=list(counts)
)
top_items_bar.update_layout(height=370, width=650,
                           yaxis={'categoryorder':'total ascending'})

te = TransactionEncoder()
te_ary = te.fit(np_data).transform(np_data)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Ассоциативные правила (FP-Growth)", style={"textAlign": "center"}),
    html.Div([
        html.H2("1. Анализ структуры транзакций"),
        html.Div([
            html.Div(dcc.Graph(figure=length_hist), style={"width": "49%", "display": "inline-block"}),
            html.Div(dcc.Graph(figure=top_items_bar), style={"width": "49%", "display": "inline-block", "float": "right"})
        ], style={"marginBottom": "30px"}),
        
        html.Label("Минимальная поддержка (min_support)"),
        dcc.Slider(
            id="min-support",
            min=0.01, max=0.5, value=0.05,
            marks={i/100: f'{i}%' for i in range(1, 51, 5)},
            step=0.01
        ),
        html.Label("Минимальная достоверность (min_confidence)"),
        dcc.Slider(
            id="min-confidence",
            min=0.1, max=1.0, value=0.5,
            marks={i/10: f'{i*10:.0f}%' for i in range(1, 11, 1)},
            step=0.05
        ),
        html.Label("Минимальный lift"),
        dcc.Slider(
            id="min-lift",
            min=1.0, max=3.0, value=1.0,
            marks={i: f'{i:.1f}' for i in np.arange(1.0, 3.1, 0.5)},
            step=0.1
        ),
        html.Button("Поиск правил", id="run-btn", n_clicks=0, style={"margin": "15px"})
    ], style={"padding": "15px","border": "1px solid #ccc", "borderRadius": "10px", "marginBottom": "24px"}),
    html.Div(id="results")
])

@app.callback(
    Output("results", "children"),
    [Input("run-btn", "n_clicks")],
    [State("min-support", "value"),
     State("min-confidence", "value"),
     State("min-lift", "value")]
)
def run_fpgrowth(n_clicks, min_support, min_confidence, min_lift):
    if n_clicks == 0:
        return html.Div("Настройте параметры и нажмите кнопку для поиска правил.")

    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    if len(frequent_itemsets) == 0:
        return html.Div("Правила не найдены, уменьшите min_support.")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift]
    if len(rules) == 0:
        return html.Div("Нет правил, уменьшите min_confidence или lift.")

    rules = rules.sort_values('lift', ascending=False)
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    fig_confidence = px.scatter(
        rules,
        x='support',
        y='confidence',
        size='lift',
        color='lift',
        hover_data=['antecedents_str', 'consequents_str'],
        title='График достоверности правил (Support vs Confidence)'
    )
    fig_confidence.update_layout(height=450, width=1000)

    G = nx.DiGraph()
    rules_for_graph = rules.head(min(15, len(rules)))
    for _, row in rules_for_graph.iterrows():
        for ant in row['antecedents']:
            G.add_node(ant, node_type='item')
        for cons in row['consequents']:
            G.add_node(cons, node_type='item')
        if row['antecedents'] and row['consequents']:
            ant = ', '.join(list(row['antecedents'])[:2])
            cons = ', '.join(list(row['consequents']))
            G.add_edge(ant, cons, weight=row['lift'], confidence=row['confidence'])
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=2, color='#888'), showlegend=False)
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(G.degree(node))
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text', marker=dict(size=28, color=node_color, colorscale='YlOrRd', line=dict(width=2, color='white')),
        text=node_text, textposition="top center", showlegend=False)
    fig_network = go.Figure([edge_trace, node_trace])
    fig_network.update_layout(title='Граф ассоциативных правил', height=620, width=1000,
                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    table = dash_table.DataTable(
        data=rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].head(30).round(4).to_dict("records"),
        columns=[
            {"name": "Antecedents (Условия)", "id": "antecedents_str"},
            {"name": "Consequents (Следствия)", "id": "consequents_str"},
            {"name": "Support", "id": "support"},
            {"name": "Confidence", "id": "confidence"},
            {"name": "Lift", "id": "lift"}
        ],
        page_size=15,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "10px"},
        style_header={"backgroundColor": "#9b59b6", "color": "white", "fontWeight": "bold"}
    )

    return html.Div([
        html.H3(f"Найдено правил: {len(rules)}", style={"margin": "20px 0"}),
        dcc.Graph(figure=fig_confidence),
        dcc.Graph(figure=fig_network),
        table
    ])

if __name__ == '__main__':
    app.run(debug=True, port=8051)
