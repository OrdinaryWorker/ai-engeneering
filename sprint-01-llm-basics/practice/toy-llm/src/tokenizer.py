"""Токенизатор (урок 01, 4.1): текст ↔ номера токенов, плюс шаблон чата.

Токенизатор — не нейросеть, а алгоритм со словарём. Его словарь строится до обучения
модели простым подсчётом частот (BPE) и потом не меняется.
"""
import re

from .logs import log

# Специальные токены: размечают структуру диалога и не являются обычным текстом.
USER, ASSISTANT, END = "<|user|>", "<|assistant|>", "<|end|>"
SPECIAL_TOKENS = [USER, ASSISTANT, END]


class Tokenizer:
    """Словарь «кусок → номер» + упорядоченные правила склейки BPE + специальные токены.

    vocab  — список кусков; номер куска равен его индексу в списке;
    merges — правила склейки (a, b) в том порядке, в котором они были выучены.
    """

    # Претокенизация: режем строку на слова (вместе с пробелом перед ними) и знаки.
    # Склеивать куски BPE можно только внутри одного такого фрагмента.
    PRETOKEN_RE = re.compile(r" ?\w+| ?[^\w\s]+|\s+")
    SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")

    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = [tuple(m) for m in merges]
        self.piece_to_id = {p: i for i, p in enumerate(vocab)}

    @classmethod
    def train(cls, texts, num_merges, verbose=True):
        """Строит словарь алгоритмом BPE и возвращает готовый токенизатор.

        1. Режем тексты на фрагменты (слова и знаки) и каждый фрагмент — на символы.
        2. Стартовый словарь: специальные токены + все встретившиеся символы.
        3. num_merges раз: находим самую частую пару соседних кусков, добавляем
           их склейку в словарь как новый токен и склеиваем эту пару во всех фрагментах.
        """
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
        """Склеивает в списке кусков каждое вхождение пары pair = (a, b) в один кусок a+b."""
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
        """Режет текст на куски-токены (строки), не переводя их в номера.

        Специальные токены вырезаются целиком и не режутся. Остальной текст режется
        на фрагменты, каждый фрагмент — на символы, и к ним применяются склейки BPE
        в том порядке, в котором они были выучены. Символы, которых нет в словаре,
        теряются (настоящие токенизаторы этого избегают, начиная словарь с байтов).
        """
        pieces = []
        for part in self.SPECIAL_RE.split(text):
            if not part:
                continue
            if part in SPECIAL_TOKENS:
                pieces.append(part)
                continue
            for chunk in self.PRETOKEN_RE.findall(part):
                chunk_pieces = list(chunk)
                for pair in self.merges:
                    chunk_pieces = self._apply_merge(chunk_pieces, pair)
                pieces += [p for p in chunk_pieces if p in self.piece_to_id]
        return pieces

    def encode(self, text):
        """Текст → список номеров токенов."""
        return [self.piece_to_id[p] for p in self.encode_pieces(text)]

    def decode(self, ids):
        """Список номеров токенов → текст (просто склеиваем куски словаря)."""
        return "".join(self.vocab[i] for i in ids)

    @staticmethod
    def apply_chat_template(messages, add_generation_prompt=True):
        """Шаблон чата (урок 01, 4.1; урок 02, этап 2): сообщения с ролями → одна строка.

        Каждое сообщение оборачивается в «<|роль|>текст<|end|>». Если
        add_generation_prompt=True, в конце добавляется ОТКРЫТОЕ сообщение
        ассистента: модели остаётся только продолжить текст, и продолжение станет
        ответом. Для обучающих примеров открытое сообщение не нужно.
        """
        tag = {"user": USER, "assistant": ASSISTANT}
        text = "".join(f"{tag[m['role']]}{m['content']}{END}" for m in messages)
        return text + (ASSISTANT if add_generation_prompt else "")
