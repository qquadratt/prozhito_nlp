import re
from collections import Counter
from typing import List, Dict, Set, Union
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc


class NatashaAnalyzer:
    """Инициализация и обработка текста с помощью Natasha."""
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)

    def process(self, text: str) -> Doc:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        return doc


def calc_percentage(count: int, total: int) -> float:
    """Вычислить процент с округлением."""
    if total == 0:
        return 0.0
    return round(count * 100 / total, 2)


def analyze_verbs(tokens: List) -> Dict[str, Union[int, Counter, Dict[str, Set[str]]]]:
    tenses = Counter({"прошедшее": 0, "настоящее": 0, "будущее": 0, "инфинитив": 0, "не указано": 0})
    aspects = Counter({"совершенный": 0, "несовершенный": 0})
    verbs_by_aspect = {"совершенный": set(), "несовершенный": set()}
    verbs_by_tense = {key: set() for key in tenses.keys()}

    for token in tokens:
        if token.pos != 'VERB' or not token.feats:
            continue

        feats = token.feats if isinstance(token.feats, dict) else {}

        aspect = feats.get('Aspect')
        if aspect == 'Perf':
            aspects['совершенный'] += 1
            verbs_by_aspect['совершенный'].add(token.text)
        elif aspect == 'Imp':
            aspects['несовершенный'] += 1
            verbs_by_aspect['несовершенный'].add(token.text)
        else:
            continue  # Пропускаем без аспекта

        tense = feats.get('Tense')
        verb_form = feats.get('VerbForm')

        if tense == 'Past':
            tenses['прошедшее'] += 1
            verbs_by_tense['прошедшее'].add(token.text)
        elif tense == 'Pres':
            tenses['настоящее'] += 1
            verbs_by_tense['настоящее'].add(token.text)
        elif tense == 'Fut':
            tenses['будущее'] += 1
            verbs_by_tense['будущее'].add(token.text)
        elif verb_form == 'Inf':
            tenses['инфинитив'] += 1
            verbs_by_tense['инфинитив'].add(token.text)
        else:
            tenses['не указано'] += 1
            verbs_by_tense['не указано'].add(token.text)

    total_verbs = sum(aspects.values())

    return {
        "total_verbs": total_verbs,
        "tenses": tenses,
        "aspects": aspects,
        "verbs_by_aspect": verbs_by_aspect,
        "verbs_by_tense": verbs_by_tense
    }


def analyze_pronouns(tokens: List) -> Dict[str, Union[int, Counter, Dict[str, str], List[str]]]:
    """Анализ местоимений по лицам."""
    pronouns_count = Counter({"1-е лицо": 0, "2-е лицо": 0, "3-е лицо": 0})
    pronouns_list = []

    for token in tokens:
        if token.pos != 'PRON' or not token.feats:
            continue
        feats = token.feats if isinstance(token.feats, dict) else {}
        person = feats.get('Person')
        if person == '1':
            pronouns_count['1-е лицо'] += 1
        elif person == '2':
            pronouns_count['2-е лицо'] += 1
        elif person == '3':
            pronouns_count['3-е лицо'] += 1
        pronouns_list.append(token.text)

    total = sum(pronouns_count.values())
    pronoun_percentages = {key: f"{calc_percentage(val, total)}%" for key, val in pronouns_count.items()}

    return {
        "total_pronouns": total,
        "pronouns_count": pronouns_count,
        "pronoun_percentages": pronoun_percentages,
        "pronouns_list": pronouns_list
    }


def analyze_interjections(tokens: List) -> List[str]:
    """Получить список междометий."""
    return [token.text for token in tokens if token.pos == 'INTJ']


def analyze_sentences(text: str) -> Dict[str, Union[int, List[str]]]:
    """Разбить текст на предложения и классифицировать по типам."""
    # Сохраняем многоточие в виде маркера
    text = re.sub(r'\.{3,}', '<ELLIPSIS>', text)
    # Убираем точки внутри аббревиатур
    text = re.sub(r'(?<=\w)\.(?=\w)', '', text)
    # Делим по знакам конца предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Восстанавливаем многоточия
    sentences = [s.replace('<ELLIPSIS>', '...') for s in sentences]

    exclamatory = [s for s in sentences if s.strip().endswith('!')]
    interrogative = [s for s in sentences if s.strip().endswith('?')]
    declarative = [s for s in sentences if s.strip().endswith('.') and not s.strip().endswith('...')]

    return {
        "total_sentences": len(sentences),
        "exclamatory": exclamatory,
        "interrogative": interrogative,
        "declarative": declarative
    }


class TextAnalyzer:
    def __init__(self, orig_text: str, lemm_text: str):
        self.orig_text = orig_text
        self.lemm_text = lemm_text
        self.analyzer = NatashaAnalyzer()

        # Обработка текстов с Natasha
        self.orig_doc = self.analyzer.process(self.orig_text)
        self.lemm_doc = self.analyzer.process(self.lemm_text)

        # Анализы
        self.verb_data = analyze_verbs(self.orig_doc.tokens)
        self.pronoun_data = analyze_pronouns(self.lemm_doc.tokens)
        self.interjections = analyze_interjections(self.lemm_doc.tokens)
        self.sentence_data = analyze_sentences(self.lemm_text)

    def print_report(self):
        v = self.verb_data
        p = self.pronoun_data
        s = self.sentence_data
        interj = self.interjections

        total_verbs = v["total_verbs"]
        aspects = v["aspects"]
        tenses = v["tenses"]

        print(f"Глаголы (всего с видом): {total_verbs}")
        print(f"Вид глаголов: {calc_percentage(aspects['совершенный'], total_verbs)}% ({aspects['совершенный']}) совершенный, "
              f"{calc_percentage(aspects['несовершенный'], total_verbs)}% ({aspects['несовершенный']}) несовершенный.")
        print(f"Уникальные глаголы совершенного вида: {sorted(v['verbs_by_aspect']['совершенный'])}")
        print(f"Уникальные глаголы несовершенного вида: {sorted(v['verbs_by_aspect']['несовершенный'])}")
        print("__________")

        print(f"Времена глаголов:")
        print(f"  прошедшее: {calc_percentage(tenses['прошедшее'], total_verbs)}% ({tenses['прошедшее']})")
        print(f"  настоящее: {calc_percentage(tenses['настоящее'], total_verbs)}% ({tenses['настоящее']})")
        print(f"  будущее: {calc_percentage(tenses['будущее'], total_verbs)}% ({tenses['будущее']})")
        print(f"  инфинитив: {calc_percentage(tenses['инфинитив'], total_verbs)}% ({tenses['инфинитив']})")
        print(f"  не указано: {calc_percentage(tenses['не указано'], total_verbs)}% ({tenses['не указано']})")
        print(f"Уникальные глаголы по временам:")
        for tense_name in tenses:
            print(f"  {tense_name}: {sorted(v['verbs_by_tense'][tense_name])}")
        print("__________")

        print(f"Типы предложений:")
        print(f"  Вопросительные: {calc_percentage(len(s['interrogative']), s['total_sentences'])}% ({len(s['interrogative'])})")
        print(f"  Восклицательные: {calc_percentage(len(s['exclamatory']), s['total_sentences'])}% ({len(s['exclamatory'])})")
        print(f"  Повествовательные: {calc_percentage(len(s['declarative']), s['total_sentences'])}% ({len(s['declarative'])})")
        print(f"Вопросительные предложения: {s['interrogative']}")
        print(f"Восклицательные предложения: {s['exclamatory']}")
        print(f"Повествовательные предложения: {s['declarative']}")
        print("__________")

        print(f"Междометия: {len(interj)} всего, уникальных: {sorted(set(interj))}")
        print("__________")

        print(f"Местоимения (всего): {p['total_pronouns']}")
        print(f"  1-е лицо: {p['pronoun_percentages']['1-е лицо']} ({p['pronouns_count']['1-е лицо']})")
        print(f"  2-е лицо: {p['pronoun_percentages']['2-е лицо']} ({p['pronouns_count']['2-е лицо']})")
        print(f"  3-е лицо: {p['pronoun_percentages']['3-е лицо']} ({p['pronouns_count']['3-е лицо']})")
        print(f"Уникальные местоимения: {sorted(set(p['pronouns_list']))}")

