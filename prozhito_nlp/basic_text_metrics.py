import pandas as pd
import re
from typing import Dict, Any

def clean_punctuation(text: str) -> str:
    """
    Удаляет все знаки препинания, кроме дефисов.
    """
    return re.sub(r"[!\"#$%&'()*+,./:;<=>?@\[\]^_{|}~«»—]", "", text)

def count_sentences(text: str) -> int:
    """
    Считает количество предложений в тексте на основе разделителей .!?.
    """
    sentence_endings = r'[.!?]'
    sentences = re.split(sentence_endings, text)
    return len([s for s in sentences if s.strip()])

def compute_text_statistics(df: pd.DataFrame, token_column: str = "tokens") -> Dict[str, Any]:
    """
    Вычисляет базовые количественные характеристики текстов:
    - Количество записей
    - Средний объем записей (в токенах)
    - Общее и уникальное число токенов
    - Средняя длина предложения (в токенах)
    
    Параметры:
    - df: pd.DataFrame
    - token_column: str — колонка с лемматизированным текстом

    Возвращает:
    - dict с метриками
    """
    # Подготовка колонки без пунктуации
    df["tokens_no_punkt"] = df[token_column].apply(clean_punctuation)

    num_records = df.shape[0]
    avg_tokens_per_record = df[token_column].str.split().map(len).mean()

    all_tokens = df["tokens_no_punkt"].str.split().sum()
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))

    total_sentences = df[token_column].apply(count_sentences).sum()
    avg_sentence_length = total_tokens / total_sentences if total_sentences > 0 else 0

    return {
        "Количество записей": num_records,
        "Средний объем записей (в токенах)": round(avg_tokens_per_record, 2),
        "Общее количество токенов": total_tokens,
        "Количество уникальных токенов": unique_tokens,
        "Средняя длина предложения (в токенах)": round(avg_sentence_length, 2)
    }
