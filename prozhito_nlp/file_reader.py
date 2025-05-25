import os
import json
import pandas as pd
import warnings
from typing import Optional

def split_json_to_csv(
    diaries_path: str,
    notes_path: str,
    output_dir: str = "diaries",
    save_csv: bool = True,
    filter_person_id: Optional[int] = None,
    return_dataframe: bool = False
) -> Optional[pd.DataFrame]:
    """
    Загружает данные из JSON-файлов, связывает записи с авторами,
    группирует по авторам и (по желанию) сохраняет в отдельные CSV-файлы.

    Параметры:
    - diaries_path: str — путь к файлу diaries.json
    - notes_path: str — путь к файлу notes.json
    - output_dir: str — директория для сохранения файлов (по умолчанию: "diaries")
    - save_csv: bool — сохранять ли CSV-файлы
    - filter_person_id: Optional[int] — если указан, вернуть только записи этого автора
    - return_dataframe: bool — если True, вернуть DataFrame (только для одного автора)

    Возвращает:
    - pd.DataFrame, если return_dataframe=True и указан filter_person_id. Иначе — None.
    """

    # Чтение данных из JSON-файлов
    with open(diaries_path, "r", encoding="utf-8") as f:
        diaries_data = json.load(f)

    with open(notes_path, "r", encoding="utf-8") as f:
        notes_data = json.load(f)

    # Создание соответствия: diary_id → person_id
    diary_to_person = {entry["id"]: entry["person"] for entry in diaries_data}

    # Добавление person_id к каждой записи
    for note in notes_data:
        note["person"] = diary_to_person.get(note["diary"])

    # Преобразование в DataFrame
    df = pd.DataFrame(notes_data)
    df = df.dropna(subset=["person"])

    # Фильтрация по конкретному автору
    if filter_person_id is not None:
        df = df[df["person"] == filter_person_id]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(by="date")

        if return_dataframe:
            return df.reset_index(drop=True)

        if save_csv:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"author_{int(filter_person_id)}.csv")
            df.to_csv(filename, index=False, encoding="utf-8")

    elif save_csv:
        os.makedirs(output_dir, exist_ok=True)
        grouped = df.groupby("person")
        for person_id, group in grouped:
            filename = os.path.join(output_dir, f"author_{int(person_id)}.csv")
            group.to_csv(filename, index=False, encoding="utf-8")

    return None

def load_diary_from_csv(file_path: str) -> pd.DataFrame:
    """
    Загружает дневник из CSV-файла и возвращает его в виде DataFrame.

    Параметры:
    ----------
    file_path : str
        Путь к CSV-файлу (например, 'diaries/author_12.csv').

    Возвращает:
    -----------
    pd.DataFrame
        DataFrame с содержимым дневника.
    """
    # Подавляем предупреждения
    warnings.simplefilter("ignore")
    
    # Настройка отображения pandas
    pd.set_option("display.max_colwidth", None)

    # Чтение файла
    df = pd.read_csv(file_path, sep=",")

    return df