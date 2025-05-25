import plotly.graph_objects as go

def plot_tfidf_by_year(
    tfidf_df,
    year_column='year',
    word_column='word',
    value_column='TF-IDF',
    top_n=20,
    bar_color='#E4653F',
    title="Топ-слова по годам"
):
    """
    Визуализирует топ-N слов по TF-IDF для каждого года с помощью Plotly.

    Аргументы:
    - tfidf_df: DataFrame с колонками для слов, значений TF-IDF и годов
    - year_column: имя колонки с годами
    - word_column: имя колонки со словами
    - value_column: имя колонки со значениями TF-IDF
    - top_n: количество топ-слов для отображения
    - bar_color: цвет столбцов
    - title: заголовок графика
    """

    fig = go.Figure()
    unique_years = sorted(tfidf_df[year_column].unique())

    for i, year in enumerate(unique_years):
        data_for_year = tfidf_df[tfidf_df[year_column] == year]
        data_for_year_top = data_for_year.nlargest(top_n, value_column)

        fig.add_trace(go.Bar(
            x=data_for_year_top[value_column],
            y=data_for_year_top[word_column],
            name=str(year),
            orientation='h',
            marker_color=bar_color,
            visible=(i == 0)  # видим только первый год
        ))

    # Добавляем dropdown меню
    fig.update_layout(
        title=title,
        xaxis_title=value_column,
        yaxis=dict(autorange='reversed'),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            showactive=True,
            x=0.45,
            xanchor="left",
            y=1.1,
            yanchor="top",
            buttons=[
                dict(
                    label=str(year),
                    method="update",
                    args=[
                        {"visible": [year == yr for yr in unique_years]},
                        {"title": f"{title} — {year}"}
                    ]
                ) for year in unique_years
            ]
        )],
        height=800,
        bargap=0.2
    )

    fig.show()
