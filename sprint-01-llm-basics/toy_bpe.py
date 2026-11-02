"""Игрушечный byte-level BPE: обучаем словарь на маленьком корпусе и токенизируем."""
from collections import Counter

corpus = (
    "the user placed an order. the order was paid. the user got the order. "
    "the payment failed, the user placed the order again. "
) * 20 + "пользователь оформил заказ. заказ оплачен. "  # английского в 20 раз больше


def train(text: str, num_merges: int) -> dict[tuple[int, int], int]:
    ids = list(text.encode("utf-8"))  # старт: каждый байт отдельный токен (0..255)
    merges = {}
    for step in range(num_merges):
        pairs = Counter(zip(ids, ids[1:]))
        best = max(pairs, key=pairs.get)  # самая частая пара соседних токенов
        new_id = 256 + step
        merges[best] = new_id
        ids = merge(ids, best, new_id)
    return merges


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def encode(text: str, merges: dict) -> list[int]:
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges.items():  # применяем слияния в порядке обучения
        ids = merge(ids, pair, new_id)
    return ids


def build_vocab(merges: dict) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab


merges = train(corpus, num_merges=40)
vocab = build_vocab(merges)

print("Первые 10 слияний:")
for (a, b), new_id in list(merges.items())[:10]:
    print(f"  {new_id}: {vocab[a]!r} + {vocab[b]!r} -> {vocab[new_id]!r}")

for text in ["the user placed the order", "пользователь оформил заказ"]:
    ids = encode(text, merges)
    pieces = [vocab[i].decode("utf-8", errors="replace") for i in ids]
    print(f"\n{text!r}: {len(text)} символов, {len(text.encode())} байт, {len(ids)} токенов")
    print("  ids:   ", ids)
    print("  куски: ", pieces)
