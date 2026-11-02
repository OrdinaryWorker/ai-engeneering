"""Шаг 2. Крошечный трансформер на чистом Python: list, dict, re, math, random.

Та же архитектура, что у настоящих LLM (см. 01b-monday-tokenizer-transformer.md):
  токенизатор → эмбеддинги → слой (внимание + полносвязная сеть) → выход → выбор токена.
Отличие только в размерах: здесь ~4 тысячи параметров, у gpt-oss-20b ~21 миллиард.

Главная разница с шагом 1: модель не хранит таблицу «что за чем идёт».
Она подбирает числа в матрицах (веса) так, чтобы предсказывать следующий токен.
"""
import math
import random
import re

random.seed(42)


# =========================================================================
# 0. Автоматическое дифференцирование.
# Чтобы обучить веса, нужно знать, в какую сторону подвинуть каждое число,
# чтобы ошибка уменьшилась (градиент). Класс Value запоминает, из чего получено
# каждое число, и умеет пройти назад по этой цепочке. Это то, что делает
# PyTorch, только для одного числа за раз и в тысячи раз медленнее.
# =========================================================================
class Value:
    __slots__ = ("data", "grad", "_parents", "_local_grads")

    def __init__(self, data, parents=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self._parents = parents
        self._local_grads = local_grads  # производная результата по каждому родителю

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, k):
        return Value(self.data**k, (self,), (k * self.data ** (k - 1),))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        return Value(max(0.0, self.data), (self,), (float(self.data > 0),))

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def backward(self):
        order, seen = [], set()

        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    visit(p)
                order.append(v)

        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            for parent, local in zip(v._parents, v._local_grads):
                parent.grad += local * v.grad


# =========================================================================
# 1. Токенизатор: слово или знак → номер. Словарь строим по корпусу.
# =========================================================================
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


with open("corpus.txt", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

EOS = "<end>"  # специальный токен «конец текста»
vocab = sorted({tok for line in lines for tok in tokenize(line)}) + [EOS]
tok_to_id = {tok: i for i, tok in enumerate(vocab)}
id_to_tok = {i: tok for tok, i in tok_to_id.items()}


def encode(text):
    return [tok_to_id[t] for t in tokenize(text)]


# =========================================================================
# Гиперпараметры: размеры модели (у настоящих — тысячи и десятки слоёв).
# =========================================================================
V = len(vocab)  # размер словаря
D = 16          # длина вектора одного токена
CONTEXT = 12    # максимум токенов на входе — контекстное окно
FFN_HIDDEN = 4 * D


def matrix(rows, cols, scale=0.3):
    return [[Value(random.gauss(0, scale)) for _ in range(cols)] for _ in range(rows)]


# Все веса модели. Именно эти числа «знает» модель после обучения.
weights = {
    "token_emb": matrix(V, D),         # шаг 2: таблица эмбеддингов, строка на токен
    "pos_emb": matrix(CONTEXT, D),     # вектор «я на позиции i» (порядок слов)
    "Wq": matrix(D, D), "Wk": matrix(D, D), "Wv": matrix(D, D), "Wo": matrix(D, D),  # внимание
    "W1": matrix(D, FFN_HIDDEN), "W2": matrix(FFN_HIDDEN, D),                       # полносвязная сеть
    "W_out": matrix(D, V),             # шаг 4: выходная матрица, вектор → оценка каждого токена
}
params = [v for m in weights.values() for row in m for v in row]


# =========================================================================
# Вспомогательные операции над векторами (list[Value]).
# =========================================================================
def linear(x, W):  # вектор × матрица
    return [sum((x[i] * W[i][j] for i in range(len(x))), Value(0.0)) for j in range(len(W[0]))]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def softmax(xs):
    top = max(x.data for x in xs)
    exps = [(x - top).exp() for x in xs]
    total = sum(exps, Value(0.0))
    return [e / total for e in exps]


def norm(x):  # нормализация: держим числа в разумном масштабе (RMSNorm)
    scale = (sum((v * v for v in x), Value(0.0)) / len(x) + 1e-5) ** -0.5
    return [v * scale for v in x]


# =========================================================================
# 2–4. Прямой проход: номера токенов → оценки (логиты) следующего токена
# для каждой позиции.
# =========================================================================
def forward(ids):
    # Шаг 2. Эмбеддинги: берём строку таблицы по номеру + вектор позиции.
    xs = [add(weights["token_emb"][t], weights["pos_emb"][i]) for i, t in enumerate(ids)]

    # Шаг 3, блок 1. Внимание: каждый токен собирает информацию с предыдущих.
    hs = [norm(x) for x in xs]
    qs = [linear(h, weights["Wq"]) for h in hs]  # запрос: «что я ищу»
    ks = [linear(h, weights["Wk"]) for h in hs]  # ключ: «по чему меня искать»
    vs = [linear(h, weights["Wv"]) for h in hs]  # значение: «что я отдам»
    for i in range(len(xs)):
        # Сравниваем запрос токена i с ключами токенов 0..i (будущее не видно).
        scores = [sum((a * b for a, b in zip(qs[i], ks[j])), Value(0.0)) / math.sqrt(D) for j in range(i + 1)]
        attn = softmax(scores)  # веса от 0 до 1, в сумме 1
        mixed = [sum((attn[j] * vs[j][d] for j in range(i + 1)), Value(0.0)) for d in range(D)]
        xs[i] = add(xs[i], linear(mixed, weights["Wo"]))  # прибавляем поправку к вектору

    # Шаг 3, блок 2. Полносвязная сеть: каждый токен отдельно.
    for i in range(len(xs)):
        hidden = [h.relu() for h in linear(norm(xs[i]), weights["W1"])]
        xs[i] = add(xs[i], linear(hidden, weights["W2"]))

    # Шаг 4. Выход: вектор каждой позиции → оценка каждого токена словаря.
    return [linear(norm(x), weights["W_out"]) for x in xs]


# =========================================================================
# Обучение: показываем фразы и подкручиваем веса, чтобы вероятность
# правильного следующего токена росла.
# =========================================================================
def loss_on(line):
    ids = (encode(line) + [tok_to_id[EOS]])[: CONTEXT + 1]
    inputs, targets = ids[:-1], ids[1:]
    logits = forward(inputs)
    # Ошибка = −log(вероятность правильного токена), среднее по позициям.
    losses = [-softmax(l)[t].log() for l, t in zip(logits, targets)]
    return sum(losses, Value(0.0)) / len(losses)


def train(steps=600, lr=0.01):
    # Adam — стандартный способ двигать веса по градиенту.
    m = [0.0] * len(params)
    v = [0.0] * len(params)
    for step in range(1, steps + 1):
        loss = loss_on(random.choice(lines))
        for p in params:
            p.grad = 0.0
        loss.backward()
        for i, p in enumerate(params):
            m[i] = 0.9 * m[i] + 0.1 * p.grad
            v[i] = 0.99 * v[i] + 0.01 * p.grad**2
            p.data -= lr * (m[i] / (1 - 0.9**step)) / ((v[i] / (1 - 0.99**step)) ** 0.5 + 1e-8)
        if step % 100 == 0:
            print(f"  шаг {step}: ошибка {loss.data:.2f}")


# =========================================================================
# 5. Генерация: прогнать, взять оценки последней позиции, temperature,
# softmax, выбрать токен, дописать, повторить.
# =========================================================================
def generate(prompt, temperature=1.0, max_tokens=10):
    ids = encode(prompt)
    for _ in range(max_tokens):
        logits = forward(ids[-CONTEXT:])[-1]
        probs = softmax([l / temperature for l in logits])
        next_id = random.choices(range(V), weights=[p.data for p in probs])[0]
        if id_to_tok[next_id] == EOS:
            break
        ids.append(next_id)
    return " ".join(id_to_tok[i] for i in ids)


def top_next(prompt, k=3):
    probs = softmax(forward(encode(prompt))[-1])
    best = sorted(range(V), key=lambda i: -probs[i].data)[:k]
    return ", ".join(f"{id_to_tok[i]} {probs[i].data:.0%}" for i in best)


if __name__ == "__main__":
    print(f"Словарь: {V} токенов, параметров: {len(params)}")
    print(f"До обучения, после 'курьер доставил': {top_next('курьер доставил')}")
    print("Обучение:")
    train()
    print(f"После обучения, после 'курьер доставил': {top_next('курьер доставил')}")
    # После «и» правильное слово зависит от того, КТО в начале фразы. Это видно только через внимание.
    for prompt in ["пользователь оформил заказ и", "менеджер подтвердил заказ и", "курьер забрал посылку и"]:
        print(f"После обучения, после {prompt!r}: {top_next(prompt)}")
    print("\nГенерация:")
    for prompt in ["пользователь", "курьер", "менеджер"]:
        for t in (0.1, 1.0):
            print(f"  [{prompt!r}, T={t}] {generate(prompt, temperature=t)}")
