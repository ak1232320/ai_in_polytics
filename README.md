# AI in Politics - RAG System

Система для генерации политических диалогов и комментариев с использованием RAG (Retrieval-Augmented Generation).

## Установка

```bash
uv sync
```

## Настройка

1. Скопируйте `.env.example` в `.env`:
```bash
cp .env.example .env
```

2. Добавьте OpenAI API ключ в `.env`:
```
OPENAI_API_KEY=sk-your-api-key-here
```

Получить ключ: https://platform.openai.com/api-keys

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `OPENAI_API_KEY` | API ключ OpenAI | - |
| `LLM_MODEL` | Модель OpenAI | `gpt-4o-mini` |
| `USE_RAG` | Использовать RAG | `true` |
| `QDRANT_PATH` | Путь к Qdrant | `./qdrant_store` |
| `EMB_MODEL` | Модель эмбеддингов | `BAAI/bge-m3` |
| `TOP_K` | Количество документов RAG | `6` |

**Совет**: Установите `USE_RAG=false` для быстрого запуска без загрузки модели эмбеддингов (~2GB).

## Запуск

```bash
# Development (с автоперезагрузкой)
uv run uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

# Production
uv run python src/app.py
```

API: http://localhost:8000
- Документация: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Режимы генерации

| Mode | Паттерн | Описание |
|------|---------|----------|
| `dialogue` | A-B-A-B-A-B | Стандартный диалог с чередованием |
| `dialogue_double` | A-B-B | Один начинает, другой добивает двумя репликами |
| `single` | A | Одиночный комментарий |
| `multi_independent` | A, B, C | Независимые комментарии от 2-3 аккаунтов |

## API

### POST /generate_dialogue

Генерация контента с выбором режима.

**Параметры:**

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|----------|--------------|
| `message` | string | Тема/описание поста | *обязательный* |
| `mode` | string | Режим генерации | `dialogue` |
| `turns` | int | Число ходов (для dialogue) | `6` |
| `chars_per_turn` | int | Лимит символов на реплику | `400` |
| `speakers` | int | Число участников (для multi_independent) | `2` |
| `top_k` | int | Количество RAG документов | `6` |
| `stance_A` | string | Позиция участника A | *опционально* |
| `stance_B` | string | Позиция участника B | *опционально* |

**Примеры запросов:**

```bash
# Диалог A-B-A-B (по умолчанию)
curl -X POST http://localhost:8000/generate_dialogue \
  -H "Content-Type: application/json" \
  -d '{"message": "Повышение минимальной зарплаты", "mode": "dialogue", "turns": 6}'

# Диалог A-B-B
curl -X POST http://localhost:8000/generate_dialogue \
  -H "Content-Type: application/json" \
  -d '{"message": "Повышение минимальной зарплаты", "mode": "dialogue_double"}'

# Одиночный комментарий
curl -X POST http://localhost:8000/generate_dialogue \
  -H "Content-Type: application/json" \
  -d '{"message": "Повышение минимальной зарплаты", "mode": "single"}'

# Независимые комментарии от 3 аккаунтов
curl -X POST http://localhost:8000/generate_dialogue \
  -H "Content-Type: application/json" \
  -d '{"message": "Повышение минимальной зарплаты", "mode": "multi_independent", "speakers": 3}'
```

**Пример ответа (dialogue):**
```json
{
  "query": "Повышение минимальной зарплаты",
  "mode": "dialogue",
  "rag_used": [],
  "result": {
    "meta": {"topic": "минималка", "turns": 6},
    "dialogue": [
      {"role": "A", "turn": 1, "text": "..."},
      {"role": "B", "turn": 2, "text": "..."}
    ],
    "summary": {"verdict": "A звучит убедительнее"}
  }
}
```

### POST /retrieve

Поиск релевантных документов в Qdrant.

### GET /health

Проверка статуса сервера.

## Структура проекта

```
ai_in_polytics/
├── src/
│   ├── app.py              # FastAPI endpoints
│   ├── generate_dialogue.py # RAG логика, prompt builders
│   └── index_corpus.py      # Индексация в Qdrant
├── prompts/                 # Системные промпты
│   ├── dialogue.txt         # A-B-A-B диалог
│   ├── dialogue_double.txt  # A-B-B паттерн
│   ├── single.txt           # Одиночный комментарий
│   └── multi_independent.txt # Независимые комментарии
├── qdrant_store/            # Локальная БД Qdrant
├── .env                     # Конфигурация
└── pyproject.toml           # Зависимости
```

## Docker

```bash
# Сборка
docker build -t ai-in-politics .

# Запуск
docker run -d -p 8000:8000 -e OPENAI_API_KEY=sk-... ai-in-politics

# Логи
docker logs -f <container_id>
```

## Другие скрипты

```bash
# Индексация корпуса в Qdrant
uv run python src/index_corpus.py

# Линтинг
uv run ruff check .
uv run ruff format .
```
