from natasha import Doc, MorphVocab, NewsMorphTagger, NewsEmbedding, Segmenter
from tqdm import tqdm
import pandas as pd

tqdm.pandas(desc="Лемматизация записей")

class LemmatizerNatasha:
    """
    Класс-обёртка для лемматизации текстов с помощью Natasha.
    """

    def __init__(self):
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.tagger = NewsMorphTagger(self.emb)
        self.morph_vocab = MorphVocab()

    def lemmatize_text(self, text: str) -> str:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        return ' '.join([token.lemma for token in doc.tokens])

def lemmatize_column(df: pd.DataFrame, text_column: str = "text", new_column: str = "tokens") -> pd.DataFrame:
    """
    Лемматизирует тексты из указанной колонки и сохраняет результат в новой колонке.

    Параметры:
    - df: pd.DataFrame — датафрейм с текстами
    - text_column: str — колонка с исходным текстом
    - new_column: str — колонка для записи результата

    Возвращает:
    - df: pd.DataFrame с новой колонкой
    """
    lemmatizer = LemmatizerNatasha()
    df[new_column] = df[text_column].progress_apply(lemmatizer.lemmatize_text)
    return df
