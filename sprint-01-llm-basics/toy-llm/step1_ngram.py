"""Шаг 1. Самая простая «языковая модель»: подсчёт, что за чем идёт.

Только встроенные типы: dict, list, re, random.
Идея та же, что у LLM: по предыдущим токенам предсказать следующий.
Отличие: «знания» здесь — таблица частот, а не выученные веса,
и модель видит только N последних токенов, а не весь контекст.
"""
import math
import random
import re
from collections import Counter, defaultdict

N = 2  # сколько предыдущих токенов модель «видит» (размер её контекстного окна)
BOS, EOS = "<s>", "</s>"  # специальные токены: начало и конец текста


# --- 1. Токенизатор -------------------------------------------------------
# Упрощение: токен = слово или знак препинания. Настоящий токенизатор режет
# на куски слов (см. Tokenizer в toy_llm.py), но для идеи это не важно.
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


# --- 2. «Обучение»: считаем, какой токен шёл после каждых N токенов -------
def train(lines: list[str]) -> dict[tuple[str, ...], Counter]:
    model: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    for line in lines:
        tokens = [BOS] * N + tokenize(line) + [EOS]
        for i in range(N, len(tokens)):
            context = tuple(tokens[i - N : i])
            model[context][tokens[i]] += 1
    return model


# --- 3. Предсказание: распределение вероятностей следующего токена --------
def next_token_probs(model, context: tuple[str, ...], temperature: float) -> dict[str, float]:
    counts = model.get(context)
    if not counts:
        return {EOS: 1.0}  # такого контекста не видели — модель «не знает», что дальше
    # Логарифм частоты играет роль логита. Делим на temperature и делаем softmax,
    # ровно как в настоящей LLM.
    logits = {tok: math.log(c) / temperature for tok, c in counts.items()}
    top = max(logits.values())
    exps = {tok: math.exp(l - top) for tok, l in logits.items()}
    total = sum(exps.values())
    return {tok: e / total for tok, e in exps.items()}


# --- 4. Генерация: выбираем токен, дописываем, повторяем ------------------
def generate(model, prompt: str, temperature: float = 1.0, max_tokens: int = 20) -> str:
    tokens = [BOS] * N + tokenize(prompt)
    for _ in range(max_tokens):
        probs = next_token_probs(model, tuple(tokens[-N:]), temperature)
        tok = random.choices(list(probs), weights=list(probs.values()))[0]
        if tok == EOS:
            break
        tokens.append(tok)
    return " ".join(t for t in tokens if t != BOS)


if __name__ == "__main__":
    random.seed(0)
    with open("corpus.txt", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    model = train(lines)

    print(f"Контекстов в таблице: {len(model)}")
    ctx = ("пользователь", "оформил")
    print(f"\nЧто идёт после {ctx}:")
    for t in (0.1, 1.0, 3.0):
        probs = next_token_probs(model, ctx, t)
        shown = ", ".join(f"{tok} {p:.0%}" for tok, p in sorted(probs.items(), key=lambda x: -x[1]))
        print(f"  temperature={t}: {shown}")

    print("\nГенерация:")
    for prompt in ["пользователь", "курьер", "менеджер подтвердил"]:
        for t in (0.1, 1.0):
            print(f"  [{prompt!r}, T={t}] {generate(model, prompt, temperature=t)}")

    print("\nНезнакомое начало:")
    print("  ", generate(model, "кладовщик собрал"))
