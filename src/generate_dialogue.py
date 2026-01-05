import os
import logging
from pathlib import Path
from typing import Optional, Dict, List

# Настройка логирования
logger = logging.getLogger(__name__)

# Путь к директории с промптами
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ===================== Загрузка промптов =====================
def load_system_prompt(mode: str) -> str:
    """Загружает системный промпт для указанного режима."""
    prompt_file = PROMPTS_DIR / f"{mode}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    logger.debug(f"📄 Loading prompt from {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")

# ===================== Билдеры промптов для разных режимов =====================

def build_dialogue_prompt(
    message: str,
    rag_ctx: str,
    turns: int,
    chars_per_turn: int,
    stance_A: Optional[str],
    stance_B: Optional[str]
) -> Dict[str, str]:
    """Диалог A-B-A-B (чередование)."""
    system_msg = load_system_prompt("dialogue")
    posA = stance_A or "человек, которому близки идеи партии и который их защищает"
    posB = stance_B or "критик этих идей"

    user_msg = f"""
=== ТЕМА / СООБЩЕНИЕ ===
{message}

=== ФОРМАТ ДИАЛОГА ===
- Роль A — уверенная, спокойная ({posA})
- Роль B — спорит, но рыхлее ({posB})
- Короткие, живые реплики без официоза.
- Общение исключительно на "ты".
- Примерный лимит: {chars_per_turn} символов на реплику.
- Всего ходов: {turns}.

=== RAG-КОНТЕКСТ ===
{rag_ctx}
"""
    return {"system": system_msg, "user": user_msg}


def build_double_prompt(
    message: str,
    rag_ctx: str,
    chars_per_turn: int,
    stance_A: Optional[str],
    stance_B: Optional[str]
) -> Dict[str, str]:
    """Диалог A-B-B (один начинает, другой добивает двумя)."""
    system_msg = load_system_prompt("dialogue_double")
    posA = stance_A or "критик или нейтральный человек"
    posB = stance_B or "сторонник партии"

    user_msg = f"""
=== ТЕМА / СООБЩЕНИЕ ===
{message}

=== ФОРМАТ: A-B-B ===
- A — задаёт вопрос или критикует ({posA})
- B — парирует и добавляет второе сообщение ({posB})
- Всего 3 реплики: A, B, B
- Примерный лимит: {chars_per_turn} символов на реплику.

=== RAG-КОНТЕКСТ ===
{rag_ctx}
"""
    return {"system": system_msg, "user": user_msg}


def build_single_prompt(
    message: str,
    rag_ctx: str,
    chars_per_turn: int
) -> Dict[str, str]:
    """Один самостоятельный комментарий."""
    system_msg = load_system_prompt("single")

    user_msg = f"""
=== ТЕМА / ПОСТ ===
{message}

=== ЗАДАЧА ===
Напиши один комментарий к посту.
Примерный лимит: {chars_per_turn} символов.

=== RAG-КОНТЕКСТ ===
{rag_ctx}
"""
    return {"system": system_msg, "user": user_msg}


def build_multi_prompt(
    message: str,
    rag_ctx: str,
    speakers: int,
    chars_per_turn: int
) -> Dict[str, str]:
    """Независимые комментарии от разных людей."""
    system_msg = load_system_prompt("multi_independent")
    roles = ["A", "B", "C"][:speakers]
    roles_str = ", ".join(roles)

    user_msg = f"""
=== ТЕМА / ПОСТ ===
{message}

=== ЗАДАЧА ===
Напиши {speakers} независимых комментария от разных людей ({roles_str}).
Комментарии не связаны друг с другом.
Примерный лимит: {chars_per_turn} символов на комментарий.

=== RAG-КОНТЕКСТ ===
{rag_ctx}
"""
    return {"system": system_msg, "user": user_msg}


# ===================== Универсальный роутер =====================

def build_prompt(
    mode: str,
    message: str,
    rag_ctx: str,
    turns: int = 6,
    chars_per_turn: int = 400,
    speakers: int = 2,
    stance_A: Optional[str] = None,
    stance_B: Optional[str] = None
) -> Dict[str, str]:
    """Универсальная функция построения промпта с роутингом по режиму."""
    logger.debug(f"🔀 Building prompt for mode: {mode}")

    if mode == "dialogue":
        return build_dialogue_prompt(message, rag_ctx, turns, chars_per_turn, stance_A, stance_B)
    elif mode == "dialogue_double":
        return build_double_prompt(message, rag_ctx, chars_per_turn, stance_A, stance_B)
    elif mode == "single":
        return build_single_prompt(message, rag_ctx, chars_per_turn)
    elif mode == "multi_independent":
        return build_multi_prompt(message, rag_ctx, speakers, chars_per_turn)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ===================== RAG Утилиты =====================
def qdrant_query(qdrant, vector, collection: str, k: int):
    """
    Совместимость с разными версиями qdrant-client:
    - новые: .query_points
    - старые: .search
    """
    try:
        # новые клиенты
        return qdrant.query_points(
            collection_name=collection,
            query=vector,
            with_payload=True,
            with_vectors=False,
            limit=k,
        )
    except AttributeError:
        # старые клиенты
        return qdrant.search(
            collection_name=collection,
            query_vector=vector,
            with_payload=True,
            limit=k,
        )


def retrieve(qdrant, emb_model, query: str, collection: str, k: int = 6) -> List[Dict]:
    """Поиск релевантных документов в Qdrant."""
    logger.debug(f"🔍 Encoding query: '{query[:50]}...'")
    qv = emb_model.encode([query], normalize_embeddings=True).tolist()[0]
    logger.debug(f"✅ Query encoded to vector of dimension {len(qv)}")
    
    logger.debug(f"📊 Querying Qdrant collection '{collection}' for top {k} results...")
    res = qdrant_query(qdrant, qv, collection, k)

    points = getattr(res, "points", None) or res  # search() возвращает список
    hits = []
    for i, p in enumerate(points, 1):
        payload = getattr(p, "payload", None) or {}
        score = getattr(p, "score", None) if hasattr(p, "score") else None
        hits.append({
            "rank": i,
            "score": score,
            "title": payload.get("title", ""),
            "source": payload.get("source", ""),
            "chunk": payload.get("chunk", ""),
        })
    logger.debug(f"✅ Retrieved {len(hits)} hits from Qdrant")
    return hits


def build_rag_context(hits: List[Dict]) -> str:
    """Форматирует результаты RAG в текстовый контекст."""
    logger.debug(f"📝 Building RAG context from {len(hits)} chunks...")
    context = "\n\n".join([f"[Источник {i+1}] {h['chunk']}" for i, h in enumerate(hits)])
    logger.debug(f"✅ RAG context built: {len(context)} characters")
    return context


# ===================== OpenAI Генератор =====================
def call_openai(oa_client, system_msg: str, user_msg: str, model: str, temperature: float, expect_json: bool = True) -> str:
    """Вызов OpenAI API для генерации ответа."""
    logger.debug(f"🤖 Preparing OpenAI request: model={model}, temperature={temperature}, expect_json={expect_json}")
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=temperature,
    )
    if expect_json:
        kwargs["response_format"] = {"type": "json_object"}

    logger.debug("📡 Sending request to OpenAI API...")
    resp = oa_client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    logger.debug(f"✅ OpenAI response received: {len(content)} characters")
    return content
