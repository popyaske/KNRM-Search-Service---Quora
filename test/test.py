"""
Тестовый запуск сервиса поиска похожих вопросов KNRM
"""

import pandas as pd
import requests
import json
from typing import List, Dict
import time
import sys

def load_qqp_data(dev_path: str):
    """
    Загружает dev.tsv
    
    Args:
        dev_path: путь к dev.tsv
    
    Returns:
        DataFrame с данными QQP
    """
    print(f"\n{'='*60}")
    print(f"📂 ЗАГРУЗКА ДАННЫХ QQP")
    print(f"{'='*60}")
    
    # Загружаем dev
    print(f"\n[1/5] Загрузка dev данных из {dev_path}")
    try:
        dev_df = pd.read_csv(dev_path, sep='\t')
        print(f"  ✅ Dev: {len(dev_df)} строк")
    except Exception as e:
        print(f"  ❌ Ошибка загрузки: {e}")
        sys.exit(1)
    
    print(f"\n✅ Объединенный датасет:")
    print(f"  Всего строк: {len(dev_df)}")
    print(f"  Уникальных пар вопросов: {dev_df[['qid1', 'qid2']].nunique().to_dict()}")
    
    return dev_df

def prepare_documents_for_index(df: pd.DataFrame) -> Dict[str, str]:
    """
    Подготавливает документы для /update_index из данных QQP
    
    Args:
        df: DataFrame с колонками qid1, qid2, question1, question2
    
    Returns:
        Словарь {id: текст} для уникальных вопросов
    """
    print(f"\n[2/5] Подготовка документов для индексации...")
    
    documents = {}
    
    # Собираем уникальные вопросы из qid1
    for _, row in df.iterrows():
        qid = str(row['qid1'])
        question = row['question1']
        if qid not in documents:
            documents[qid] = question
    
    # Собираем уникальные вопросы из qid2
    for _, row in df.iterrows():
        qid = str(row['qid2'])
        question = row['question2']
        if qid not in documents:
            documents[qid] = question
    
    print(f"  ✅ Подготовлено {len(documents)} уникальных документов")
    return documents

def update_index(base_url: str, documents: Dict[str, str], timeout: int = 200):
    """
    Отправляет запрос на /update_index
    
    Args:
        base_url: базовый URL сервиса
        documents: словарь документов {id: текст}
        timeout: таймаут в секундах
    """
    url = f"{base_url}/update_index"
    
    payload = {
        "documents": documents
    }
    
    print(f"\n[3/5] Обновление FAISS индекса...")
    print(f"  URL: {url}")
    print(f"  Документов: {len(documents)}")
    print(f"  Ожидание ответа до {timeout} сек...")
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Индекс успешно обновлен")
            print(f"  📊 Размер индекса: {result.get('index_size')} документов")
            return result
        else:
            print(f"  ❌ Ошибка: {response.status_code}")
            print(f"  {response.text}")
            return None
    except requests.exceptions.Timeout:
        print(f"  ❌ Таймаут! Сервер не ответил за {timeout} сек")
        return None
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return None

def prepare_queries(df: pd.DataFrame, num_queries: int = 10) -> List[str]:
    """
    Подготавливает список запросов из данных QQP
    
    Args:
        df: DataFrame с вопросами
        num_queries: количество запросов
    
    Returns:
        Список вопросов для поиска
    """
    print(f"\n[4/5] Подготовка тестовых запросов...")
    
    # Берем случайные num_queries вопросов
    queries = df['question1'].sample(n=min(num_queries, len(df))).tolist()
    
    print(f"  ✅ Подготовлено {len(queries)} запросов:")
    for i, q in enumerate(queries[:5], 1):
        preview = q[:80] + "..." if len(q) > 80 else q
        print(f"    {i}. {preview}")
    if len(queries) > 5:
        print(f"    ... и еще {len(queries)-5} запросов")
    
    return queries

def search_queries(base_url: str, queries: List[str], timeout: int = 30):
    """
    Отправляет запросы на /query
    
    Args:
        base_url: базовый URL сервиса
        queries: список запросов
        timeout: таймаут в секундах
    """
    url = f"{base_url}/query"
    
    payload = {
        "queries": queries
    }
    
    print(f"\n[5/5] Поиск похожих вопросов...")
    print(f"  URL: {url}")
    print(f"  Запросов: {len(queries)}")
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Статус: {data.get('status')}")
            
            results = data.get('results', [])
            print(f"  📊 Получено результатов: {len(results)}")
            return data
        else:
            print(f"  ❌ Ошибка: {response.status_code}")
            print(f"  {response.text}")
            return None
    except requests.exceptions.Timeout:
        print(f"  ❌ Таймаут! Сервер не ответил за {timeout} сек")
        return None
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return None

def calculate_success_rate(search_results: dict) -> float:
    """
    Считает долю запросов с дубликатом в топ-10
    
    Args:
        search_results: результаты поиска
    
    Returns:
        Доля успешных запросов (0-1)
    """
    if not search_results:
        return 0.0
    
    results = search_results.get('results', [])
    english_results = [r for r in results if r.get('lang_check', False)]
    
    if not english_results:
        return 0.0
    
    successful = 0
    for r in english_results:
        suggestions = [suggestion for _, suggestion in r.get('suggestions', [])]
        if r.get('query') in suggestions:
            successful += 1
    
    return successful / len(english_results)

def print_results_summary(search_results: dict, rate: float):
    """
    Выводит сводку результатов
    
    Args:
        search_results: результаты поиска
        rate: доля успешных запросов
    """
    print(f"\n{'='*60}")
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print(f"{'='*60}")
    
    if not search_results:
        print("❌ Нет результатов для отображения")
        return
    
    results = search_results.get('results', [])
    
    print(f"\n📈 Общая статистика:")
    print(f"  Всего запросов: {len(results)}")
    
    english_count = sum(1 for r in results if r.get('lang_check', False))
    non_english_count = len(results) - english_count
    print(f"  ✅ Английских: {english_count}")
    print(f"  ❌ Не английских: {non_english_count}")
    
    total_suggestions = sum(len(r.get('suggestions', [])) for r in results)
    print(f"  💡 Всего предложений: {total_suggestions}")
    print(f"  🎯 Доля успешных: {rate:.4f} ({rate*100:.2f}%)")
    
    print(f"\n📝 Детальные результаты по запросам:")
    for i, result in enumerate(results, 1):
        query = result.get('query', '')
        lang_check = result.get('lang_check', False)
        suggestions = result.get('suggestions', [])
        
        status = "✅" if lang_check else "❌"
        lang_status = "EN" if lang_check else "Non-EN"
        
        print(f"\n  {i}. {status} {lang_status}: {query[:70]}...")
        print(f"     Предложений: {len(suggestions)}")
        
        if suggestions:
            print(f"     Топ-3 предложения:")
            for j, (doc_id, text) in enumerate(suggestions[:3], 1):
                is_duplicate = "✓ ДУБЛИКАТ" if text == query else ""
                print(f"       {j}. [{doc_id}] {text[:60]}... {is_duplicate}")
    
    print(f"\n{'='*60}")
    if rate >= 0.5:
        print(f"✅ ПОРОГ ДОСТИГНУТ: {rate:.4f} >= 0.5")
    else:
        print(f"⚠️  ПОРОГ НЕ ДОСТИГНУТ: {rate:.4f} < 0.5")
    print(f"{'='*60}")

def wait_for_service(base_url: str, max_attempts: int = 30, delay: int = 5):
    """
    Ожидает готовности сервиса
    
    Args:
        base_url: базовый URL сервиса
        max_attempts: максимальное количество попыток
        delay: задержка между попытками в секундах
    """
    print(f"\n{'='*60}")
    print(f"🔍 ПРОВЕРКА ГОТОВНОСТИ СЕРВИСА")
    print(f"{'='*60}")
    
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(f"{base_url}/ping", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok':
                    print(f"✅ Сервис готов! (попытка {attempt})")
                    return True
                else:
                    print(f"⏳ Сервис инициализируется... (попытка {attempt}/{max_attempts})")
            else:
                print(f"⏳ Сервис не отвечает... (попытка {attempt}/{max_attempts})")
        except:
            print(f"⏳ Ожидание запуска сервиса... (попытка {attempt}/{max_attempts})")
        
        time.sleep(delay)
    
    print(f"❌ Сервис не запустился за {max_attempts * delay} секунд")
    return False

def run_test():
    """
    Основная функция тестирования
    """
    print(f"\n{'='*60}")
    print(f"🧪 ТЕСТОВЫЙ ЗАПУСК KNRM SEARCH SERVICE")
    print(f"{'='*60}")
    
    # Конфигурация
    BASE_URL = "http://localhost:11000"
    QQP_DEV_PATH = "QQP/dev.tsv"
    NUM_QUERIES = 10
    
    # 1. Ожидание сервиса
    if not wait_for_service(BASE_URL):
        print("\n❌ Не удалось подключиться к сервису. Убедитесь, что сервер запущен.")
        return
    
    # 2. Загрузка данных
    try:
        df = load_qqp_data(QQP_DEV_PATH)
    except Exception as e:
        print(f"\n❌ Ошибка загрузки данных: {e}")
        return
    
    # 3. Подготовка документов
    documents = prepare_documents_for_index(df)
    
    # 4. Обновление индекса
    result = update_index(BASE_URL, documents)
    if not result:
        print("\n❌ Не удалось обновить индекс")
        return
    
    # 5. Подготовка запросов
    queries = prepare_queries(df, NUM_QUERIES)
    
    # 6. Поиск
    search_results = search_queries(BASE_URL, queries)
    
    # 7. Расчет метрики
    if search_results:
        rate = calculate_success_rate(search_results)
        
        # 8. Вывод результатов
        print_results_summary(search_results, rate)
        
        return rate
    else:
        print("\n❌ Не удалось выполнить поиск")
        return None

if __name__ == "__main__":
    # Запуск теста
    success_rate = run_test()
    
    if success_rate is not None:
        print(f"\n🏆 ИТОГОВАЯ ДОЛЯ УСПЕШНЫХ ЗАПРОСОВ: {success_rate:.4f} ({success_rate*100:.2f}%)")