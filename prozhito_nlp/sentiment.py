import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Set, List

def load_rusentilex_dict(filepath: Path) -> Dict[str, Dict[str, Set[str]]]:
    """
    Загружает словарь RuSentiLex и организует его по типу лексики и полярности.
    """
    lexicon = defaultdict(lambda: defaultdict(set))  # source -> polarity -> phrases
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            phrase = parts[2].strip()
            polarity = parts[3].strip().lower()
            source = parts[4].strip().lower()
            lexicon[source][polarity].add(phrase)
    return lexicon


def calculate_sentiment_score(pos: int, neu: int, neg: int) -> float:
    """
    Расчитывает сентимент-оценку.
    """
    total = pos + neu + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def analyze_sentiment(
    df: pd.DataFrame,
    text_column: str,
    lexicon: Dict[str, Dict[str, Set[str]]]
) -> Tuple[Dict, int, Dict[str, int], pd.DataFrame]:
    """
    Проводит сентимент-анализ на основе словаря RuSentiLex.
    Возвращает:
    - результаты по категориям,
    - общее количество уникальных слов,
    - общее количество слов в каждой категории словаря,
    - обновлённый DataFrame с колонкой rusentilex_score.
    """
    result = defaultdict(lambda: defaultdict(lambda: {'words': set(), 'count': 0}))
    sentiment_scores = []

    total_unique_words = sum(df[text_column].apply(lambda x: len(set(x.split()))))

    sources = ['opinion', 'feeling', 'fact']
    polarities = ['positive', 'neutral', 'negative']

    for text in df[text_column]:
        normalized_text = ' '.join(text.split())
        pos, neu, neg = 0, 0, 0

        for source in sources:
            for polarity in polarities:
                matches = {phrase for phrase in lexicon[source][polarity] if phrase in normalized_text}
                result[source][polarity]['words'].update(matches)
                result[source][polarity]['count'] += len(matches)

                if polarity == 'positive':
                    pos += len(matches)
                elif polarity == 'neutral':
                    neu += len(matches)
                elif polarity == 'negative':
                    neg += len(matches)

        sentiment_scores.append(calculate_sentiment_score(pos, neu, neg))

    df['rusentilex_score'] = sentiment_scores

    total_category_words = {
        source: sum(len(lexicon[source][pol]) for pol in polarities)
        for source in sources
    }

    return result, total_unique_words, total_category_words, df


def print_sentiment_results(
    result: Dict,
    total_unique_words: int,
    total_category_words: Dict[str, int]
) -> None:
    """
    Печатает краткие статистики сентимент-анализа по категориям.
    """
    categories = {
        'opinion': 'Оценочная лексика',
        'feeling': 'Чувства',
        'fact': 'Факты'
    }
    polarities = ['positive', 'neutral', 'negative']

    for source in categories:
        print(f"\n{categories[source]}:")
        for polarity in polarities:
            words = result[source][polarity]['words']
            count = result[source][polarity]['count']
            percent_unique = (count / total_unique_words * 100) if total_unique_words else 0
            percent_category = (count / total_category_words[source] * 100) if total_category_words[source] else 0
            words_sorted = sorted(words) if words else '—'
            print(f"{polarity}: {percent_unique:.1f}% от уникальных слов, {percent_category:.1f}% от слов в категории ({count}) {words_sorted}")

