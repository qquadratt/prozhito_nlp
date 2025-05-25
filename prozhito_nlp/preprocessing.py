import pandas as pd
import re

def clean_text_column(df, text_column="text"):
    """
    Очищает текстовую колонку DataFrame от HTML-тегов, markdown-разметки,
    типографических артефактов и раскрытых сокращений.

    Параметры:
    - df: pd.DataFrame
    - text_column: str, название колонки с текстом

    Возвращает:
    - df: pd.DataFrame с очищенной колонкой
    """

    # Общие паттерны для удаления тегов и markdown
    patterns = [
        r'<com id=\d+"/>',
        r'<com id="\d+"/>',
        r'<\w+>',
        r'</\w+>',
        r'<\w+\s*/?>',
        r'\*',
        r'#'
    ]
    for pattern in patterns:
        df[text_column] = df[text_column].str.replace(pattern, '', regex=True)

    # Удаление закодированных HTML-тегов
    df[text_column] = df[text_column].str.replace(r'&lt;.*?&gt;', '', regex=True)
    df[text_column] = df[text_column].str.replace(r'&nbsp;', ' ', regex=True)

    # Замена HTML-сущностей на символы
    html_entities = {
        '&laquo;': '«',
        '&raquo;': '»',
        '&mdash;': '—',
        '&amp;': '&',
        '&copy;': '©',
        '&lt;': '<',
        '&gt;': '>',
    }
    for entity, symbol in html_entities.items():
        df[text_column] = df[text_column].str.replace(entity, symbol, regex=True)

    # Удаление следов плохой типографики (переносов)
    df[text_column] = df[text_column].str.replace(r'(\w+-)\s([го|я|й])', r'\1\2', regex=True)

    # Удаление раскрытий сокращений: М[ария] → Мария
    df[text_column] = df[text_column].str.replace(r'(\w)\[(\w+)\]', r'\1\2', regex=True)

    # Удаление дополнительных HTML-тегов
    df[text_column] = df[text_column].str.replace(r'<br\s*/?>', ' ', regex=True)
    df[text_column] = df[text_column].str.replace(r'<img[^>]*>', '', regex=True)
    df[text_column] = df[text_column].str.replace(r'<a[^>]*>(.*?)</a>', r'\1', regex=True)
    df[text_column] = df[text_column].str.replace(r'<!--.*?-->', '', regex=True)

    return df

def add_year_column(df, date_column="date"):
    """
    Добавляет колонку 'year' на основе даты (в формате YYYY-MM-DD).
    """
    df = df.copy()
    df['year'] = pd.to_datetime(df[date_column], errors='coerce').dt.year
    return df
