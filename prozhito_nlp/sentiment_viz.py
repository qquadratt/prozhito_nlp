import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
import plotly.graph_objects as go

def plot_sentiment_dynamics(
    df: pd.DataFrame,
    date_column: str = 'date',
    score_column: str = 'rusentilex_score',
    window_length: int = 11,
    lowess_frac: float = 0.1,
    title_prefix: str = 'Динамика сентимента'
) -> go.Figure:
    """
    Строит график динамики сентимента по датам с оригинальными значениями и сглаживанием (Savitzky-Golay, LOWESS).
    
    Параметры:
        df (pd.DataFrame): датафрейм с колонками дат и сентимент-оценок
        date_column (str): имя колонки с датами
        score_column (str): имя колонки с сентимент-оценками
        window_length (int): окно сглаживания для фильтра Савицкого-Голея (должно быть нечетным)
        lowess_frac (float): параметр сглаживания для LOWESS
        title_prefix (str): заголовок графика
    
    Возвращает:
        plotly.graph_objects.Figure: интерактивный график
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    dates = df[date_column]
    scores = df[score_column]

    # Savitzky-Golay
    if len(scores) < 5:
        raise ValueError("Недостаточно точек для построения графика со сглаживанием")
    
    if len(scores) < window_length:
        window_length = len(scores) // 2 * 2 + 1
    smoothed_scores_savgol = savgol_filter(scores, window_length=window_length, polyorder=2)

    # LOWESS
    lowess_result = lowess(scores, dates.astype(np.int64), frac=lowess_frac)
    lowess_dates = pd.to_datetime(lowess_result[:, 0])
    lowess_scores = lowess_result[:, 1]

    # Построение графика
    fig = go.Figure()

    # Исходные значения
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name='Исходные значения (RuSentiLex)',
        marker=dict(color='#E4653F', size=4, opacity=0.7),
        line=dict(color='#f08869'),
        visible=True
    ))

    # Savitzky-Golay
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='markers',
        name='Оригинальные значения (RuSentiLex)',
        marker=dict(color='#E4653F', size=4, opacity=0.4),
        visible=False
    ))
    fig.add_trace(go.Scatter(
        x=dates,
        y=smoothed_scores_savgol,
        mode='lines',
        name='Сглаженные значения (Савицкий-Голей)',
        line=dict(color='#E4653F', width=2),
        visible=False
    ))

    # LOWESS
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='markers',
        name='Оригинальные значения (RuSentiLex)',
        marker=dict(color='#E4653F', size=4, opacity=0.4),
        visible=False
    ))
    fig.add_trace(go.Scatter(
        x=lowess_dates,
        y=lowess_scores,
        mode='lines',
        name='Сглаженные значения (LOWESS)',
        line=dict(color='#E4653F', width=2),
        visible=False
    ))

    # Переключатели
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(label='Исходные значения RuSentiLex',
                         method='update',
                         args=[{'visible': [True, False, False, False, False]},
                               {'title': f'{title_prefix} во времени'}]),
                    dict(label='Сглаживание (Савицкий-Голей)',
                         method='update',
                         args=[{'visible': [False, True, True, False, False]},
                               {'title': f'{title_prefix} (Савицкий-Голей)'}]),
                    dict(label='Сглаживание (LOWESS)',
                         method='update',
                         args=[{'visible': [False, False, False, True, True]},
                               {'title': f'{title_prefix} (LOWESS)'}])
                ],
                direction="down",
                showactive=True,
                x=0.6,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ]
    )

    # Общие настройки
    fig.update_layout(
        xaxis_title='Дата',
        yaxis_title='Сентимент',
        hovermode='x unified',
        title=f'{title_prefix} во времени',
        margin=dict(t=80, r=40, b=80),
        height=500,
        width=800,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        )
    )

    return fig

def plot_sentiment_calendar(df, date_col='date', sentiment_col='rusentilex_score'):
    """
    Визуализация сентимента в виде теплового календаря по годам.

    Параметры:
    ----------
    df : pandas.DataFrame
        Таблица с датами и сентиментом.
    date_col : str
        Название колонки с датой.
    sentiment_col : str
        Название колонки с сентимент-оценками.
    """

    df[date_col] = pd.to_datetime(df[date_col])
    sentiment_df = df[[date_col, sentiment_col]].rename(columns={date_col: 'date', sentiment_col: 'sentiment'})

    # --- Усреднение сентимента по дате ---
    daily_sentiment = sentiment_df.groupby('date')['sentiment'].mean().reset_index()
    daily_sentiment['year'] = daily_sentiment['date'].dt.year
    daily_sentiment['week'] = daily_sentiment['date'].dt.strftime('%W').astype(int) + 1
    daily_sentiment['weekday'] = daily_sentiment['date'].dt.weekday

    years = sorted(daily_sentiment['year'].unique())
    total_weeks = 53

    fig = go.Figure()

    # --- Тепловая карта по каждому году ---
    for year in years:
        year_data = daily_sentiment[daily_sentiment['year'] == year]
        z = np.full((7, total_weeks), np.nan)
        customdata = np.full((7, total_weeks, 2), '', dtype=object)

        for _, row in year_data.iterrows():
            week = int(row['week']) - 1
            day = int(row['weekday'])
            if 0 <= week < total_weeks:
                z[day, week] = row['sentiment']
                customdata[day, week, 0] = row['date'].strftime('%Y-%m-%d')
                customdata[day, week, 1] = f"{row['sentiment']:.3f}"

        fig.add_trace(go.Heatmap(
            z=z,
            x=list(range(total_weeks)),
            y=list(range(7)),
            customdata=customdata,
            hovertemplate='Дата: %{customdata[0]}<br>Сентимент: %{customdata[1]}<extra></extra>',
            colorscale='RdYlGn',
            colorbar=dict(title='Сентимент', thickness=40, title_font=dict(size=14)),
            zmin=-1, zmax=1,
            showscale=True,
            name=f'{year}',
            visible=(year == years[0]),
            x0=0.5, dx=1,
            y0=0.5, dy=1,
            xgap=1, ygap=1
        ))

    fig.update_layout(
        title=f"Календарь сентимента: {years[0]}",
        xaxis=dict(
            showgrid=False,
            tickmode='array',
            tickvals=[i * 4.5 for i in range(12)],
            ticktext=['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                      'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'],
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            tickvals=list(range(7)),
            ticktext=['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'],
            autorange='reversed',
            title='',
            zeroline=False
        ),
        updatemenus=[dict(
            type="dropdown",
            x=0.5,
            y=1.1,
            xanchor="center",
            yanchor="bottom",
            buttons=[dict(
                label=f"{year}",
                method="update",
                args=[
                    {"visible": [y == year for y in years]},
                    {"title": f"Календарь сентимента: {year}"}
                ]
            ) for year in years]
        )],
        showlegend=False,
        height=400,
        width=800,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, b=40, t=40)
    )

    fig.show()
