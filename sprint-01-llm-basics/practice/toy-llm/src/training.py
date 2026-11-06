"""Обучение (урок 01, раздел 2 и 5.2): подбираем веса так, чтобы модель предсказывала следующий токен."""
import random
import time

from .autograd import vsum
from .logs import log
from .ops import softmax
from .tokenizer import Tokenizer


def train(model, dialogs, steps=1500, lr=0.02):
    """Обучает веса модели на диалогах «вопрос — ответ».

    На каждом шаге:
      1. Берём случайный диалог, оформляем его по шаблону чата и токенизируем.
      2. Прогоняем через модель все токены, кроме последнего, и для каждой позиции
         получаем предсказание следующего токена.
      3. Ошибка (loss) = среднее по позициям от −log(вероятность правильного токена).
         Если модель уверена в правильном токене, вероятность близка к 1, и ошибка ≈ 0.
      4. backward() считает градиент ошибки по каждому весу.
      5. Оптимизатор Adam сдвигает каждый вес против градиента. Скорость обучения lr
         плавно уменьшается к концу, чтобы веса «успокоились».
    """
    texts = [Tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
        add_generation_prompt=False) for q, a in dialogs]
    sequences = [model.tok.encode(t) for t in texts]
    log(f"Пример обучающей строки: {texts[0]!r}")
    log(f"  → {len(sequences[0])} токенов: {model.tok.encode_pieces(texts[0])}")
    log(f"Параметров модели: {len(model.params)}")

    params = model.params
    m = [0.0] * len(params)  # Adam: скользящее среднее градиента
    v = [0.0] * len(params)  # Adam: скользящее среднее квадрата градиента
    rnd = random.Random(0)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        seq = rnd.choice(sequences)
        inputs, targets = seq[:-1], seq[1:]
        logits = model.forward(inputs, model.new_cache())
        loss = vsum([-softmax(lg)[t].log() for lg, t in zip(logits, targets)]) * (1 / len(targets))

        for p in params:
            p.grad = 0.0
        loss.backward()
        step_lr = lr * (1 - step / (steps + 1))
        for i, p in enumerate(params):
            m[i] = 0.9 * m[i] + 0.1 * p.grad
            v[i] = 0.99 * v[i] + 0.01 * p.grad**2
            p.data -= step_lr * (m[i] / (1 - 0.9**step)) / ((v[i] / (1 - 0.99**step)) ** 0.5 + 1e-8)
        if step % 100 == 0:
            log(f"  шаг {step:4d}/{steps}: ошибка {loss.data:.3f}  "
                f"({time.perf_counter() - started:.0f} с)")
