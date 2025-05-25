import os
import re
from collections import defaultdict
import pandas as pd

def match_custom_dictionaries(
    df,
    text_column,
    dict_dir,
    dict_names,
    show_details=True
):
    """
    Ищет совпадения с кастомными словарями в лемматизированных текстах.
    Внутри функции встроен список паттернов для составных фразеологизмов.

    Аргументы:
    - df: pandas DataFrame с колонкой лемм
    - text_column: имя колонки с лемматизированным текстом (строка)
    - dict_dir: путь к папке, где лежат словари (файлы вида name_lemm.txt)
    - dict_names: список базовых имен словарей (без _lemm.txt)
    - show_details: выводить ли подробные совпадения (по умолчанию True)

    Возвращает:
    - total_matches: словарь с количеством всех совпадений
    - unique_matches: словарь с уникальными совпадениями
    """

    # Встроенный список паттернов составных фразеологизмов
    compound_patterns = [ 
        r'бросать камни в \S+ огород',
        r'вить из \S+ верёвки',
        r'выворачивать \S+ руки',
        r'доставать \S+',
        r'забить на \S+',
        r'завязать с \S+',
        r'загнать \S+ в угол',
        r'закрывать глаза на \S+',
        r'затмить \S+',
        r'лебезить перед \S+',
        r'мерить всех на \S+ аршин',
        r'не по \S+ части',
        r'отправить \S+ к праотцам',
        r'отправить \S+ на тот свет',
        r'перемывать \S+ косточки',
        r'плакать \S+ в жилетку',
        r'поговорить с \S+ по душам',
        r'подставлять \S+ под удар',
        r'показать \S+ где раки зимуют',
        r'показать \S+ кузькину мать',
        r'попробовать себя в \S+',
        r'принимать \S+ за чистую монету',
        r'пропускать \S+ мимо ушей',
        r'протянуть \S+ руку помощи',
        r'пускать \S+ пыль в глаза',
        r'развязать \S+ руки',
        r'рыться в \S+ грязном белье',
        r'сбить \S+ с панталыку',
        r'связать \S+ по рукам и ногам',
        r'связываться с \S+',
        r'сделать \S+ орудием в своих руках',
        r'скидываться на \S+',
        r'стереть \S+ в порошок',
        r'судьба улыбается \S+',
        r'типун \S+ на язык'
    ]

    # Загрузка словарей из файлов
    phrase_dicts = {
        name: {
            line.strip()
            for line in open(os.path.join(dict_dir, f"{name}_lemm.txt"), encoding='utf-8')
        }
        for name in dict_names
    }

    # Добавляем "словарь" паттернов
    phrase_dicts['phraseologisms_compound'] = set(compound_patterns)

    print("Загружены словари:")
    for name, word_set in phrase_dicts.items():
        print(f"  {name}: {len(word_set)} элементов")

    total_matches = defaultdict(int)
    unique_matches = defaultdict(set)

    for text in df[text_column]:
        words = set(text.split())
        joined_text = ' '.join(words)

        for category, dictionary in phrase_dicts.items():
            if category == 'phraseologisms_compound':
                for pattern in compound_patterns:
                    match = re.search(pattern, joined_text)
                    if match:
                        total_matches[category] += 1
                        unique_matches[category].add(match.group(0))
            else:
                matches = words & dictionary
                total_matches[category] += len(matches)
                unique_matches[category].update(matches)

    print("\nНайденные совпадения:")
    for category in phrase_dicts.keys():
        print(f"\n📚 Словарь: {category}")
        print(f"Всего совпадений: {total_matches[category]}")
        print(f"Уникальных совпадений: {len(unique_matches[category])}")
        if show_details:
            print("Список уникальных совпадений:")
            print(', '.join(sorted(unique_matches[category])) if unique_matches[category] else "Нет совпадений")
        print("-" * 50)

    return total_matches, unique_matches
