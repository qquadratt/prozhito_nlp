import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf_by_year(
    df,
    text_column,
    year_column,
    stop_words_path,
    top_n=20,
    display_year=None
):
    """
    Вычисляет TF-IDF для токенизированных текстов по годам.

    Аргументы:
    - df: DataFrame с текстами
    - text_column: имя колонки с токенами
    - year_column: имя колонки с годами
    - stop_words_path: путь к файлу со стоп-словами
    - top_n: сколько слов сохранять на каждый год
    - display_year: если указан, выводит топ-слова только за этот год
    """

    with open(stop_words_path, 'r', encoding='utf-8') as file:
        stop_words = file.read().split()

    years = df[year_column].unique()
    years.sort()

    result_df = pd.DataFrame(columns=['TF-IDF', 'word', 'year'])

    for year in years:
        year_texts = df[df[year_column] == year][text_column].to_list()
        if not year_texts:
            continue

        vectorizer = TfidfVectorizer(use_idf=True, stop_words=stop_words)
        X = vectorizer.fit_transform(year_texts)

        tfidf_scores = X.sum(axis=0).A1
        words = vectorizer.get_feature_names_out()

        temp_df = pd.DataFrame({
            'TF-IDF': tfidf_scores,
            'word': words
        }).sort_values('TF-IDF', ascending=False)

        temp_df['year'] = year
        result_df = pd.concat([result_df, temp_df.head(top_n)], ignore_index=True)

    # Отображаем результат по указанному году
    if display_year is not None:
        display_df = result_df[result_df['year'] == display_year]
        display_df = display_df.sort_values('TF-IDF', ascending=False).reset_index(drop=True)

        try:
            from IPython.display import display
            display(display_df)
        except ImportError:
            print(display_df)

    return result_df
