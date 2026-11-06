"""Модель (урок 01, раздел 2): веса и формула прямого прохода через трансформер.

Модель — это данные (веса) плюс фиксированный алгоритм их применения (forward).
Она ничего не знает про шаблон чата, temperature и остановку — это работа движка
(engine.py), как в жизни: файл весов отдельно, Ollama отдельно.
"""
import json
import math
import random
from pathlib import Path

from .autograd import Value, dot, vsum
from .logs import log
from .ops import add, linear, rmsnorm, rope, softmax
from .tokenizer import Tokenizer


class ToyLLM:
    """Decoder-only трансформер (урок 01, 3.2) с крошечными размерами.

    d_model  — размерность вектора токена;
    n_layers — число слоёв трансформера;
    n_heads  — число голов внимания в слое (d_model делится между ними поровну);
    d_ff     — размерность внутреннего слоя полносвязной сети.
    """

    def __init__(self, tokenizer, d_model=16, n_layers=2, n_heads=2, d_ff=64, seed=42):
        """Создаёт модель со случайными весами. Чтобы она что-то умела, её нужно обучить."""
        self.tok = tokenizer
        self.config = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
                           vocab_size=len(tokenizer.vocab))
        self.d, self.n_layers, self.n_heads = d_model, n_layers, n_heads
        self.head_dim = d_model // n_heads
        rnd = random.Random(seed)

        def matrix(n_out, n_in, scale=0.3):
            """Матрица-преобразование: n_out столбцов по n_in случайных чисел."""
            return [[Value(rnd.gauss(0, scale)) for _ in range(n_in)] for _ in range(n_out)]

        V = len(tokenizer.vocab)
        # Все веса модели. Именно эти числа подбираются при обучении.
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

    def new_cache(self):
        """Создаёт пустой KV-кэш (урок 02, этап 6).

        Для каждого слоя — списки ключей и значений всех уже обработанных токенов.
        Ключ и значение одного токена хранятся как список векторов, по вектору на голову.
        """
        return [{"k": [], "v": []} for _ in range(self.n_layers)]

    def forward(self, token_ids, cache, trace=None):
        """Прямой проход: прогоняет НОВЫЕ токены через модель и дописывает их в KV-кэш.

        Одна и та же функция обслуживает обучение, prefill и decode:
          • обучение — все токены примера, пустой кэш;
          • prefill  — все токены входа (или те, что не нашлись в кэше);
          • decode   — один только что выбранный токен, кэш уже заполнен.

        Позиция каждого нового токена равна числу токенов, уже лежащих в кэше.
        Возвращает логиты для каждого нового токена: оценку того, какой токен идёт за ним.

        trace — необязательный словарь, куда складываются веса внимания последнего
        токена в каждом слое и голове (движок печатает их в лог).
        """
        start_pos = len(cache[0]["k"])

        # Шаг 2. Эмбеддинги: берём строку таблицы по номеру токена, как SELECT по ключу.
        # Позиционных векторов не прибавляем: позицию вносит RoPE внутри внимания.
        xs = [list(self.w["embedding"][t]) for t in token_ids]

        # Шаг 3. Слои трансформера.
        for l in range(self.n_layers):
            for i in range(len(xs)):
                pos = start_pos + i
                # Блок 1: внимание (урок 01, 4.5). Остаточная связь: прибавляем, не заменяем.
                attn_out = self._attention(l, rmsnorm(xs[i]), pos, cache[l],
                                           trace if i == len(xs) - 1 else None)
                xs[i] = add(xs[i], attn_out)
                # Блок 2: полносвязная сеть (урок 01, 4.6), каждый токен отдельно.
                xs[i] = add(xs[i], self._feed_forward(l, rmsnorm(xs[i])))

        # Шаг 4. Выходная матрица: вектор → логит для каждого токена словаря.
        return [linear(rmsnorm(x), self.w["output"]) for x in xs]

    def _attention(self, l, h, pos, layer_cache, trace=None):
        """Блок внимания слоя l для одного токена на позиции pos.

        1. Из нормализованного вектора h получаем запрос, ключ и значение (матрицы Wq, Wk, Wv)
           и делим каждый на головы.
        2. Поворачиваем запрос и ключ каждой головы через RoPE.
        3. Кладём ключ и значение токена в KV-кэш слоя. Ключи и значения предыдущих токенов
           уже там: их не пересчитываем — в этом весь смысл кэша.
        4. Для каждой головы сравниваем запрос со всеми ключами в кэше (это токены 0..pos,
           будущих там ещё нет — так работает каузальная маска), получаем веса через softmax
           и складываем значения с этими весами.
        5. Склеиваем выходы голов и смешиваем их матрицей Wo.
        """
        hd = self.head_dim
        q = linear(h, self.w[f"L{l}.Wq"])
        k = linear(h, self.w[f"L{l}.Wk"])
        v = linear(h, self.w[f"L{l}.Wv"])
        q_heads = [rope(q[j:j + hd], pos) for j in range(0, self.d, hd)]
        k_heads = [rope(k[j:j + hd], pos) for j in range(0, self.d, hd)]
        v_heads = [v[j:j + hd] for j in range(0, self.d, hd)]

        layer_cache["k"].append(k_heads)
        layer_cache["v"].append(v_heads)

        head_outputs = []
        for hidx in range(self.n_heads):
            scores = [dot(q_heads[hidx], past_k[hidx]) * (1 / math.sqrt(hd))
                      for past_k in layer_cache["k"]]
            weights = softmax(scores)
            if trace is not None:
                trace[(l, hidx)] = [w.data for w in weights]
            mixed = [vsum([weights[j] * layer_cache["v"][j][hidx][d] for j in range(len(weights))])
                     for d in range(hd)]
            head_outputs += mixed
        return linear(head_outputs, self.w[f"L{l}.Wo"])

    def _feed_forward(self, l, h):
        """Полносвязная сеть слоя l: расширяем вектор (W1), ReLU, сжимаем обратно (W2)."""
        hidden = [u.relu() for u in linear(h, self.w[f"L{l}.W1"])]
        return linear(hidden, self.w[f"L{l}.W2"])

    def save(self, path):
        """Сохраняет модель в JSON: конфигурацию, токенизатор и все веса.

        Это и есть «модель на диске» в миниатюре — как файл весов, который скачивает Ollama.
        """
        data = {
            "config": self.config,
            "tokenizer": {"vocab": self.tok.vocab, "merges": self.tok.merges},
            "weights": {name: [[round(v.data, 6) for v in row] for row in m] for name, m in self.w.items()},
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        """Загружает модель из JSON, сохранённого методом save."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = Tokenizer(data["tokenizer"]["vocab"], data["tokenizer"]["merges"])
        cfg = data["config"]
        model = cls(tok, cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_ff"])
        for name, m in data["weights"].items():
            for row, values in zip(model.w[name], m):
                for p, value in zip(row, values):
                    p.data = value
        return model

    def describe(self):
        """Печатает состав модели и бюджет памяти (урок 01, раздел 2; урок 02, «Где тратится память»)."""
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
