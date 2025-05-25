import plotly.express as px
import plotly.graph_objects as go
import re

def plot_total_matches(total_matches: dict):
    """
    Рисует интерактивный барчарт с общим количеством совпадений по категориям.
    """
    categories = list(total_matches.keys())
    total_counts = [total_matches[cat] for cat in categories]

    # Сортируем по убыванию
    sorted_data = sorted(zip(categories, total_counts), key=lambda x: x[1], reverse=True)
    sorted_categories, sorted_counts = zip(*sorted_data) if sorted_data else ([], [])

    fig = px.bar(
        x=sorted_categories,
        y=sorted_counts,
        labels={"x": "Категория", "y": "Количество совпадений"},
        title="Количество совпадений по категориям",
    )
    fig.update_traces(marker_color="#E4653F")
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()

def plot_matches_by_category(unique_matches: dict, df, token_col: str = "tokens_no_punkt"):
    """
    Рисует интерактивный горизонтальный барчарт с уникальными совпадениями по категориям.
    С выпадающим меню для выбора категории.
    """

    filtered_categories = {cat: words for cat, words in unique_matches.items() if words}
    categories_with_matches = list(filtered_categories.keys())

    data = []
    buttons = []

    # Пустая трасса - "ничего не выбрано"
    data.append(go.Bar(x=[], y=[], orientation='h', marker=dict(color='#E4653F')))
    buttons.append(dict(
        label="Выберите категорию",
        method="update",
        args=[{"visible": [True] + [False]*len(categories_with_matches)},
              {"title": "Уникальные совпадения по категориям"}]
    ))

    for i, category in enumerate(categories_with_matches):
        word_list = list(filtered_categories[category])
        word_counts = [
            df[token_col].str.contains(r'\b' + re.escape(word) + r'\b').sum()
            for word in word_list
        ]
        sorted_data = sorted(zip(word_list, word_counts), key=lambda x: x[1], reverse=True)
        sorted_words, sorted_counts = zip(*sorted_data) if sorted_data else ([], [])

        trace = go.Bar(
            x=sorted_counts,
            y=sorted_words,
            orientation='h',
            name=category,
            visible=False,
            marker=dict(color='#E4653F')
        )
        data.append(trace)

    for i, category in enumerate(categories_with_matches):
        visibility = [False] * (len(categories_with_matches) + 1)
        visibility[i + 1] = True
        buttons.append(dict(
            label=category,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"Уникальные совпадения в категории: {category}"}]
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title="Уникальные совпадения по категориям",
        xaxis_title="Частота",
        yaxis_title="Слово",
        height=600,
        yaxis=dict(autorange="reversed"),
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.9,
            "xanchor": "center",
            "y": 1.15,
            "yanchor": "top"
        }]
    )
    fig.show()
