"""ToyLLM — учебный прототип LLM на чистом Python (только стандартная библиотека).

Что внутри, в терминах урока 01:
  • токенизатор: BPE + специальные токены + шаблон чата;
  • веса: таблица эмбеддингов, N слоёв трансформера (внимание с RoPE + полносвязная сеть),
    выходная матрица;
  • выбор токена: softmax + temperature;
  • KV-кэш, prefill и decode — как в уроке 02.

Что НЕ так, как в настоящих моделях (ради простоты и скорости на чистом Python):
  • токенизатор строится из символов, а не из байтов (иначе словарь начинается с 256 элементов);
  • размеры крошечные: вектор из 16 чисел, 2 слоя, 2 головы — несколько тысяч параметров;
  • prefill — это цикл по токенам, а не одна параллельная матричная операция на GPU;
    математически результат тот же;
  • нет MoE, квантования и батчей.

Запуск (из папки toy-llm):
  python3 toy_llm.py train                      # обучение, ~несколько минут; сохраняет weights.json
  python3 toy_llm.py chat "кто доставил заказ?" # инференс одного запроса с подробным логом
  python3 toy_llm.py demo                       # сценарий: запрос, второй ход диалога, temperature
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

HERE = Path(__file__).parent
WEIGHTS_FILE = HERE / "weights.json"

# Специальные токены (урок 01, 4.1): размечают структуру диалога, не являются обычным текстом.
USER, ASSISTANT, END = "<|user|>", "<|assistant|>", "<|end|>"
SPECIAL_TOKENS = [USER, ASSISTANT, END]

# Обучающие данные: диалоги «вопрос — ответ». Настоящие модели проходят предобучение на
# триллионах токенов, а потом дообучение на диалогах (урок 01, 5.2). У нас сразу второе,
# и всего на дюжине примеров, поэтому модель просто выучит их почти наизусть.
DIALOGS = [
    ("кто оформил заказ?", "пользователь оформил заказ."),
    ("кто оплатил заказ?", "пользователь оплатил заказ."),
    ("кто подтвердил заказ?", "менеджер подтвердил заказ."),
    ("кто отправил заказ?", "менеджер отправил заказ."),
    ("кто доставил заказ?", "курьер доставил заказ."),
    ("кто привез посылку?", "курьер привез посылку."),
    ("что сделал курьер?", "курьер доставил заказ."),
    ("что сделал менеджер?", "менеджер отправил заказ."),
    ("что сделал пользователь?", "пользователь оплатил заказ."),
    ("заказ не оплачен. что делать?", "напомнить об оплате."),
    ("заказ оплачен. что делать?", "передать заказ менеджеру."),
    ("привет", "здравствуйте! чем помочь?"),
]


def log(text=""):
    print(text)


def stage(number, title):
    log(f"\n{'═' * 72}\nЭтап {number}. {title}\n{'═' * 72}")


# ═════════════════════════════════════════════════════════════════════════════
# Автоматическое дифференцирование.
#
# Нужно только для ОБУЧЕНИЯ (урок 01, раздел 2): чтобы подобрать веса, надо знать,
# в какую сторону сдвинуть каждое число, чтобы ошибка уменьшилась (градиент).
# Value запоминает, из каких чисел и как получено каждое число, и умеет пройти
# по этой цепочке назад. Это то, что делает PyTorch, только по одному числу.
# При ИНФЕРЕНСЕ градиенты не нужны, мы просто читаем .data.
# ═════════════════════════════════════════════════════════════════════════════
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
        if isinstance(other, Value):
            return Value(self.data * other.data, (self, other), (other.data, self.data))
        return Value(self.data * other, (self,), (other,))

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
        return self * -1.0

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


def vsum(xs):
    """Сумма списка Value одним узлом (быстрее, чем цепочка сложений)."""
    return Value(sum(x.data for x in xs), tuple(xs), (1.0,) * len(xs))


def dot(xs, ws):
    """Скалярное произведение двух векторов одним узлом: Σ x_i · w_i."""
    data = sum(x.data * w.data for x, w in zip(xs, ws))
    return Value(data, tuple(xs) + tuple(ws), tuple(w.data for w in ws) + tuple(x.data for x in xs))


# ═════════════════════════════════════════════════════════════════════════════
# Операции над векторами (вектор = list[Value]) и матрицами (list[list[Value]]).
# ═════════════════════════════════════════════════════════════════════════════
def linear(x, W):
    """Вектор × матрица-преобразование (урок 01, раздел 2).

    W хранится «по столбцам»: W[j] — столбец, описывающий, как из всех входных чисел
    посчитать j-е выходное. Матрица 16 × 64 принимает вектор из 16 чисел, выдаёт из 64.
    """
    return [dot(x, column) for column in W]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def rmsnorm(x):
    """Нормализация (урок 01, 4.4): приводим числа вектора к стандартному масштабу."""
    mean_square = vsum([v * v for v in x]) * (1.0 / len(x))
    scale = (mean_square + 1e-5) ** -0.5
    return [v * scale for v in x]


def softmax(logits):
    """Логиты → вероятности (урок 01, 4.7): всё от 0 до 1, сумма равна 1."""
    top = max(v.data for v in logits)  # вычитаем максимум, чтобы exp не переполнился
    exps = [(v - top).exp() for v in logits]
    total = vsum(exps)
    return [e / total for e in exps]


def softmax_floats(xs):
    top = max(xs)
    exps = [math.exp(x - top) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def rope(vec, pos):
    """RoPE (урок 01, 4.3): поворачиваем пары чисел вектора на угол «позиция × частота».

    Пары (0,1), (2,3), … — точки на плоскости. У первых пар частота высокая (крутятся
    быстро), у последних низкая. Применяется к запросу и ключу перед сравнением,
    и тогда их скалярное произведение зависит от расстояния между токенами.
    """
    out = []
    dim = len(vec)
    for i in range(0, dim, 2):
        freq = 10000 ** (-i / dim)
        angle = pos * freq
        c, s = math.cos(angle), math.sin(angle)
        a, b = vec[i], vec[i + 1]
        out += [a * c - b * s, a * s + b * c]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Токенизатор (урок 01, 4.1)
# ═════════════════════════════════════════════════════════════════════════════
class Tokenizer:
    """Словарь «кусок → номер» + правила склейки BPE + специальные токены."""

    # Претокенизация: режем строку на слова (вместе с пробелом перед ними) и знаки.
    # Склеивать куски BPE можно только внутри одного такого фрагмента.
    PRETOKEN_RE = re.compile(r" ?\w+| ?[^\w\s]+|\s+")
    SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")

    def __init__(self, vocab, merges):
        self.vocab = vocab  # список кусков; номер куска = его индекс
        self.merges = [tuple(m) for m in merges]  # упорядоченные правила склейки
        self.piece_to_id = {p: i for i, p in enumerate(vocab)}

    @classmethod
    def train(cls, texts, num_merges, verbose=True):
        """BPE: начинаем с отдельных символов и раз за разом склеиваем самую частую пару."""
        chunks = [list(c) for t in texts for c in cls.PRETOKEN_RE.findall(t)]
        vocab = SPECIAL_TOKENS + sorted({ch for c in chunks for ch in c})
        if verbose:
            log(f"Стартовый словарь: {len(SPECIAL_TOKENS)} специальных токена + "
                f"{len(vocab) - len(SPECIAL_TOKENS)} символов")
        merges = []
        for step in range(num_merges):
            pairs = {}
            for c in chunks:
                for a, b in zip(c, c[1:]):
                    pairs[(a, b)] = pairs.get((a, b), 0) + 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            merges.append(best)
            vocab.append(best[0] + best[1])
            chunks = [cls._apply_merge(c, best) for c in chunks]
            if verbose and step < 10:
                log(f"  склейка {step + 1:2d}: {best[0]!r} + {best[1]!r} → {best[0] + best[1]!r}"
                    f"  (встретилась {pairs[best]} раз)")
        if verbose:
            log(f"  … всего склеек: {len(merges)}, итоговый словарь: {len(vocab)} токенов")
        return cls(vocab, merges)

    @staticmethod
    def _apply_merge(pieces, pair):
        out, i = [], 0
        while i < len(pieces):
            if i < len(pieces) - 1 and (pieces[i], pieces[i + 1]) == pair:
                out.append(pieces[i] + pieces[i + 1])
                i += 2
            else:
                out.append(pieces[i])
                i += 1
        return out

    def encode_pieces(self, text):
        """Текст → список кусков. Специальные токены не режутся."""
        pieces = []
        for part in self.SPECIAL_RE.split(text):
            if not part:
                continue
            if part in SPECIAL_TOKENS:
                pieces.append(part)
                continue
            for chunk in self.PRETOKEN_RE.findall(part):
                chunk_pieces = list(chunk)
                for pair in self.merges:  # склейки применяем в порядке обучения
                    chunk_pieces = self._apply_merge(chunk_pieces, pair)
                pieces += [p for p in chunk_pieces if p in self.piece_to_id]  # незнакомые символы теряются
        return pieces

    def encode(self, text):
        return [self.piece_to_id[p] for p in self.encode_pieces(text)]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    @staticmethod
    def apply_chat_template(messages, add_generation_prompt=True):
        """Шаблон чата (урок 01, 4.1; урок 02, этап 2): сообщения с ролями → одна строка.

        add_generation_prompt=True оставляет в конце ОТКРЫТОЕ сообщение ассистента:
        модели остаётся только продолжить текст, и продолжение станет ответом.
        """
        tag = {"user": USER, "assistant": ASSISTANT}
        text = "".join(f"{tag[m['role']]}{m['content']}{END}" for m in messages)
        return text + (ASSISTANT if add_generation_prompt else "")


# ═════════════════════════════════════════════════════════════════════════════
# Модель
# ═════════════════════════════════════════════════════════════════════════════
class ToyLLM:
    def __init__(self, tokenizer, d_model=16, n_layers=2, n_heads=2, d_ff=64, seed=42):
        self.tok = tokenizer
        self.config = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
                           vocab_size=len(tokenizer.vocab))
        self.d, self.n_layers, self.n_heads = d_model, n_layers, n_heads
        self.head_dim = d_model // n_heads
        rnd = random.Random(seed)

        def matrix(n_out, n_in, scale=0.3):
            # n_out столбцов по n_in чисел: преобразует вектор из n_in чисел в вектор из n_out
            return [[Value(rnd.gauss(0, scale)) for _ in range(n_in)] for _ in range(n_out)]

        V = len(tokenizer.vocab)
        # Все веса модели (урок 01, раздел 2). Именно эти числа подбираются при обучении.
        self.w = {
            # Таблица эмбеддингов: строка на каждый токен словаря, в строке d_model чисел.
            "embedding": [[Value(rnd.gauss(0, 0.3)) for _ in range(d_model)] for _ in range(V)],
            # Выходная матрица: из итогового вектора — по логиту на каждый токен словаря.
            "output": matrix(V, d_model),
        }
        for l in range(n_layers):
            self.w[f"L{l}.Wq"] = matrix(d_model, d_model)   # запрос: «что я ищу»
            self.w[f"L{l}.Wk"] = matrix(d_model, d_model)   # ключ: «по чему меня искать»
            self.w[f"L{l}.Wv"] = matrix(d_model, d_model)   # значение: «что я отдам»
            self.w[f"L{l}.Wo"] = matrix(d_model, d_model)   # смешивание выходов голов
            self.w[f"L{l}.W1"] = matrix(d_ff, d_model)      # полносвязная сеть, вход
            self.w[f"L{l}.W2"] = matrix(d_model, d_ff)      # полносвязная сеть, выход
        self.params = [v for m in self.w.values() for row in m for v in row]

        # KV-кэш между вызовами (урок 02, этап 9): помним, для каких токенов он посчитан.
        self._cache = None
        self._cache_tokens = []

    # ─────────────────────────────────────────────────────────────────────────
    # Прямой проход. ОДНА функция для обучения, prefill и decode.
    # ─────────────────────────────────────────────────────────────────────────
    def new_cache(self):
        # Для каждого слоя: список ключей и значений всех обработанных токенов.
        # Ключ и значение токена — по вектору на каждую голову.
        return [{"k": [], "v": []} for _ in range(self.n_layers)]

    def forward(self, token_ids, cache, trace=None):
        """Прогоняет НОВЫЕ токены через модель, дописывая их ключи и значения в KV-кэш.

        Позиция каждого нового токена = сколько токенов уже в кэше. Возвращает логиты
        для каждого нового токена (предсказание следующего за ним).

        trace — словарь, куда складываем веса внимания последнего токена для лога.
        """
        start_pos = len(cache[0]["k"])

        # Шаг 2. Эмбеддинги: берём строку таблицы по номеру токена, как SELECT по ключу.
        # Позиционных векторов не прибавляем: позицию вносит RoPE внутри внимания.
        xs = [list(self.w["embedding"][t]) for t in token_ids]

        # Шаг 3. Слои трансформера.
        for l in range(self.n_layers):
            W = lambda name: self.w[f"L{l}.{name}"]
            for i in range(len(xs)):
                pos = start_pos + i

                # ── Блок 1: внимание (урок 01, 4.5) ──
                h = rmsnorm(xs[i])
                q, k, v = linear(h, W("Wq")), linear(h, W("Wk")), linear(h, W("Wv"))
                hd = self.head_dim
                q_heads = [rope(q[j:j + hd], pos) for j in range(0, self.d, hd)]
                k_heads = [rope(k[j:j + hd], pos) for j in range(0, self.d, hd)]
                v_heads = [v[j:j + hd] for j in range(0, self.d, hd)]

                # Кладём ключ и значение этого токена в KV-кэш. Предыдущие токены уже там:
                # их ключи и значения не пересчитываются — в этом весь смысл кэша.
                cache[l]["k"].append(k_heads)
                cache[l]["v"].append(v_heads)

                head_outputs = []
                for hidx in range(self.n_heads):
                    # Сравниваем запрос токена с ключами всех токенов в кэше: это токены
                    # 0..pos. Будущих там ещё нет — так работает каузальная маска.
                    scores = [dot(q_heads[hidx], past_k[hidx]) * (1 / math.sqrt(hd))
                              for past_k in cache[l]["k"]]
                    weights = softmax(scores)  # веса от 0 до 1, сумма 1
                    if trace is not None and i == len(xs) - 1:
                        trace[(l, hidx)] = [w.data for w in weights]
                    # Взвешенная смесь значений: каждый токен отдаёт свою долю информации.
                    mixed = [vsum([weights[j] * cache[l]["v"][j][hidx][d] for j in range(len(weights))])
                             for d in range(hd)]
                    head_outputs += mixed
                # Остаточная связь: прибавляем поправку к вектору, а не заменяем его.
                xs[i] = add(xs[i], linear(head_outputs, W("Wo")))

                # ── Блок 2: полносвязная сеть (урок 01, 4.6), каждый токен отдельно ──
                hidden = [u.relu() for u in linear(rmsnorm(xs[i]), W("W1"))]
                xs[i] = add(xs[i], linear(hidden, W("W2")))

        # Шаг 4. Выходная матрица: вектор → логит для каждого токена словаря.
        return [linear(rmsnorm(x), self.w["output"]) for x in xs]

    # ─────────────────────────────────────────────────────────────────────────
    # Обучение (урок 01, раздел 2 и 5.2)
    # ─────────────────────────────────────────────────────────────────────────
    def train(self, dialogs, steps=1500, lr=0.02):
        texts = [Tokenizer.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
            add_generation_prompt=False) for q, a in dialogs]
        sequences = [self.tok.encode(t) for t in texts]
        log(f"Пример обучающей строки: {texts[0]!r}")
        log(f"  → {len(sequences[0])} токенов: {self.tok.encode_pieces(texts[0])}")
        log(f"Параметров модели: {len(self.params)}")

        m = [0.0] * len(self.params)
        v = [0.0] * len(self.params)
        rnd = random.Random(0)
        started = time.perf_counter()
        for step in range(1, steps + 1):
            seq = rnd.choice(sequences)
            inputs, targets = seq[:-1], seq[1:]
            logits = self.forward(inputs, self.new_cache())
            # Ошибка = −log(вероятность правильного следующего токена), среднее по позициям.
            loss = vsum([-softmax(lg)[t].log() for lg, t in zip(logits, targets)]) * (1 / len(targets))

            for p in self.params:
                p.grad = 0.0
            loss.backward()
            # Adam: сдвигаем каждый вес против градиента. Скорость обучения плавно падает.
            step_lr = lr * (1 - step / (steps + 1))
            for i, p in enumerate(self.params):
                m[i] = 0.9 * m[i] + 0.1 * p.grad
                v[i] = 0.99 * v[i] + 0.01 * p.grad**2
                p.data -= step_lr * (m[i] / (1 - 0.9**step)) / ((v[i] / (1 - 0.99**step)) ** 0.5 + 1e-8)
            if step % 100 == 0:
                log(f"  шаг {step:4d}/{steps}: ошибка {loss.data:.3f}  "
                    f"({time.perf_counter() - started:.0f} с)")

    def save(self, path):
        data = {
            "config": self.config,
            "tokenizer": {"vocab": self.tok.vocab, "merges": self.tok.merges},
            "weights": {name: [[round(v.data, 6) for v in row] for row in m] for name, m in self.w.items()},
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = Tokenizer(data["tokenizer"]["vocab"], data["tokenizer"]["merges"])
        cfg = data["config"]
        model = cls(tok, cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_ff"])
        for name, m in data["weights"].items():
            for row, values in zip(model.w[name], m):
                for p, value in zip(row, values):
                    p.data = value
        return model

    # ─────────────────────────────────────────────────────────────────────────
    # Инференс: путь одного запроса, этапы урока 02
    # ─────────────────────────────────────────────────────────────────────────
    def chat(self, messages, temperature=0.2, max_tokens=30, num_ctx=96, seed=None, verbose=True):
        say = log if verbose else (lambda *a: None)
        rnd = random.Random(seed)

        if verbose:
            stage(1, "Запрос пришёл в движок")
            say(json.dumps({"messages": messages, "temperature": temperature,
                            "max_tokens": max_tokens}, ensure_ascii=False, indent=2))

        # ── Этап 2. Шаблон чата ──
        prompt = Tokenizer.apply_chat_template(messages)
        if verbose:
            stage(2, "Сообщения склеиваются по шаблону чата")
            say(f"Строка, которую увидит модель:\n  {prompt!r}")
            say("В конце — открытое сообщение ассистента: модели остаётся продолжить текст.")

        # ── Этап 3. Токенизация ──
        pieces = self.tok.encode_pieces(prompt)
        ids = [self.tok.piece_to_id[p] for p in pieces]
        if verbose:
            stage(3, "Токенизация: строка → куски → номера")
            for p, i in zip(pieces, ids):
                say(f"  {p!r:>16} → {i}")
            say(f"Итого токенов входа: {len(ids)} (из них специальных: "
                f"{sum(p in SPECIAL_TOKENS for p in pieces)})")

        # ── Этап 4. Проверка контекстного окна ──
        if verbose:
            stage(4, "Проверка контекстного окна")
            say(f"Окно num_ctx = {num_ctx} токенов; вход {len(ids)} + max_tokens {max_tokens} "
                f"= {len(ids) + max_tokens}")
        if len(ids) + max_tokens > num_ctx:
            keep = max(num_ctx - max_tokens, 1)
            say(f"  ⚠ Не помещается! Отрезаем начало: остаются последние {keep} токенов. "
                f"Ошибки не будет, модель просто не увидит часть текста.")
            ids = ids[-keep:]
        elif verbose:
            say("  Помещается.")

        # ── Этап 5. Prefill ──
        if verbose:
            stage(5, "Prefill: модель читает весь вход и заполняет KV-кэш")
        reused = self._reuse_cache(ids)
        cache = self._cache
        new_ids = ids[reused:]
        if verbose:
            if reused:
                say(f"Начало входа совпало с прошлым запросом: из KV-кэша взято готовыми {reused} "
                    f"ток., считаем только {len(new_ids)} новых (урок 02, этап 9).")
            else:
                say(f"KV-кэш пуст, считаем все {len(new_ids)} токенов.")
        trace = {}
        t0 = time.perf_counter()
        logits = self.forward(new_ids, cache, trace)[-1]
        prefill_time = time.perf_counter() - t0
        if verbose:
            self._log_attention(ids, trace)
            say(f"KV-кэш теперь хранит ключи и значения {len(cache[0]['k'])} токенов "
                f"в каждом из {self.n_layers} слоёв.")

        # ── Этап 6. Decode ──
        if verbose:
            stage(6, "Decode: ответ рождается по одному токену")
            say(f"{'шаг':>3} | {'топ-3 кандидата (вероятность после temperature)':<52} | выбран")
            say("-" * 72)
        answer_ids, done_reason = [], "length"
        t1 = time.perf_counter()
        first_token_at = None
        for step in range(1, max_tokens + 1):
            probs = softmax_floats([lg.data / temperature for lg in logits])
            next_id = rnd.choices(range(len(probs)), weights=probs)[0]
            if first_token_at is None:
                first_token_at = time.perf_counter()
            if verbose:
                top = sorted(range(len(probs)), key=lambda i: -probs[i])[:3]
                cand = ", ".join(f"{self.tok.vocab[i]!r} {probs[i]:.0%}" for i in top)
                say(f"{step:>3} | {cand:<52} | {self.tok.vocab[next_id]!r}")

            # ── Этап 7. Остановка: модель выбрала токен конца сообщения ──
            if self.tok.vocab[next_id] == END:
                done_reason = "stop"
                break
            answer_ids.append(next_id)
            ids.append(next_id)
            # Через модель проходит ТОЛЬКО новый токен; прошлые берутся из KV-кэша.
            logits = self.forward([next_id], cache)[-1]
        decode_time = time.perf_counter() - t1
        self._cache_tokens = list(ids)

        if verbose:
            stage(7, "Остановка")
            if done_reason == "stop":
                say(f"Модель выбрала {END!r} — нормальное завершение, done_reason = 'stop'.")
            else:
                say(f"Достигнут max_tokens = {max_tokens} — ответ обрезан, done_reason = 'length'.")

        # ── Этап 8. Ответ: детокенизация, usage, тайминги ──
        text = self.tok.decode(answer_ids)
        ttft = (first_token_at or t1) - t0
        response = {
            "message": {"role": "assistant", "content": text},
            "done_reason": done_reason,
            "prompt_eval_count": len(new_ids),
            "prompt_eval_duration_s": round(prefill_time, 4),
            "eval_count": len(answer_ids),
            "eval_duration_s": round(decode_time, 4),
            "time_to_first_token_s": round(ttft, 4),
        }
        if verbose:
            stage(8, "Ответ возвращается")
            say(json.dumps(response, ensure_ascii=False, indent=2))
            if answer_ids:
                say(f"Скорость prefill: {len(new_ids) / prefill_time:.0f} ток/с; "
                    f"decode: {len(answer_ids) / decode_time:.0f} ток/с.")
                say("У нас обе фазы — циклы на Python, поэтому скорости близки. На GPU prefill "
                    "считается одной матричной операцией для всех токенов сразу и на токен бывает "
                    "в десятки раз быстрее decode (урок 02, «Где тратится время»).")
        return response

    def _reuse_cache(self, ids):
        """Сколько первых токенов входа совпадает с тем, что уже лежит в KV-кэше."""
        if self._cache is None:
            self._cache, self._cache_tokens = self.new_cache(), []
        common = 0
        for a, b in zip(ids, self._cache_tokens):
            if a != b:
                break
            common += 1
        common = min(common, len(ids) - 1)  # последний токен входа всегда считаем заново
        for layer in self._cache:  # отрезаем из кэша всё, что после совпавшего начала
            del layer["k"][common:]
            del layer["v"][common:]
        return common

    def _log_attention(self, ids, trace):
        """Показываем, на какие токены «смотрел» последний токен входа в последнем слое."""
        l = self.n_layers - 1
        log(f"Внимание последнего токена {self.tok.vocab[ids[-1]]!r} в слое {l + 1} "
            f"(на кого он смотрит сильнее всего):")
        for h in range(self.n_heads):
            weights = trace[(l, h)][-len(ids):]
            top = sorted(range(len(weights)), key=lambda i: -weights[i])[:3]
            parts = ", ".join(f"{self.tok.vocab[ids[i]]!r} {weights[i]:.0%}" for i in top)
            log(f"  голова {h + 1}: {parts}")

    def describe(self):
        """Состав модели и бюджет памяти (урок 01, раздел 2; урок 02, «Где тратится память»)."""
        c = self.config
        log("Состав модели:")
        log(f"  токенизатор: словарь {c['vocab_size']} токенов, {len(self.tok.merges)} склеек BPE")
        log(f"  таблица эмбеддингов: {c['vocab_size']} × {c['d_model']}")
        log(f"  слоёв: {c['n_layers']}, в каждом внимание ({c['n_heads']} головы по {self.head_dim}) "
            f"и полносвязная сеть {c['d_model']} → {c['d_ff']} → {c['d_model']}")
        log(f"  выходная матрица: {c['d_model']} × {c['vocab_size']}")
        n = len(self.params)
        log(f"  параметров: {n} → {n * 4 / 1024:.0f} КБ в float32 (у gpt-oss-20b ~21 млрд, ~14 ГБ в 4 битах)")
        kv = 2 * c["n_layers"] * c["n_heads"] * self.head_dim * 4
        log(f"  KV-кэш на 1 токен: 2 × {c['n_layers']} слоя × {c['n_heads']} головы × {self.head_dim} "
            f"× 4 байта = {kv} байт")


# ═════════════════════════════════════════════════════════════════════════════
# Командная строка
# ═════════════════════════════════════════════════════════════════════════════
def cmd_train(args):
    stage("Т1", "Обучаем токенизатор (BPE) на текстах диалогов")
    texts = [t for pair in DIALOGS for t in pair]
    tok = Tokenizer.train(texts, num_merges=args.merges)
    stage("Т2", "Обучаем веса модели")
    model = ToyLLM(tok)
    model.describe()
    model.train(DIALOGS, steps=args.steps)
    model.save(WEIGHTS_FILE)
    log(f"\nВеса сохранены в {WEIGHTS_FILE.name}. Это и есть «модель на диске»: токенизатор + числа.")


def load_model():
    if not WEIGHTS_FILE.exists():
        raise SystemExit("Нет weights.json — сначала обучи модель: python3 toy_llm.py train")
    return ToyLLM.load(WEIGHTS_FILE)


def cmd_chat(args):
    model = load_model()
    model.describe()
    model.chat([{"role": "user", "content": args.prompt}], temperature=args.temperature,
               max_tokens=args.max_tokens, num_ctx=args.num_ctx, seed=args.seed)


def cmd_demo(args):
    model = load_model()
    model.describe()

    log("\n\n######## Сценарий 1. Первый запрос ########")
    history = [{"role": "user", "content": "кто доставил заказ?"}]
    r = model.chat(history, temperature=0.2, seed=1)

    log("\n\n######## Сценарий 2. Второй ход диалога: история отправляется целиком ########")
    history += [r["message"], {"role": "user", "content": "что сделал менеджер?"}]
    model.chat(history, temperature=0.2, seed=1)

    log("\n\n######## Сценарий 3. Temperature: один вопрос, по 5 ответов ########")
    for t in (0.1, 1.0, 3.0):
        answers = [model.chat([{"role": "user", "content": "что сделал курьер?"}], temperature=t,
                              seed=s, verbose=False)["message"]["content"] for s in range(5)]
        log(f"temperature={t}:")
        for a in answers:
            log(f"   {a!r}")

    log("\n\n######## Сценарий 4. max_tokens слишком мал ########")
    model.chat([{"role": "user", "content": "заказ оплачен. что делать?"}], max_tokens=3, seed=1)


def main():
    parser = argparse.ArgumentParser(description="Учебный прототип LLM на чистом Python")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train", help="обучить токенизатор и веса, сохранить weights.json")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--merges", type=int, default=60)
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("chat", help="один запрос с подробным логом всех этапов")
    p.add_argument("prompt")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=30)
    p.add_argument("--num-ctx", type=int, default=96)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_chat)
    p = sub.add_parser("demo", help="сценарии: запрос, второй ход, temperature, max_tokens")
    p.set_defaults(func=cmd_demo)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
