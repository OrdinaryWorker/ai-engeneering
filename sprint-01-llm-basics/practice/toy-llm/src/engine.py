"""Движок инференса (урок 01, раздел 2): исполняет модель и проводит запрос по этапам урока 02.

Движок отвечает за всё, чего нет в весах: шаблон чата, токенизацию, проверку
контекстного окна, KV-кэш между запросами, выбор токена с temperature, остановку
и сбор ответа. Это наш маленький аналог Ollama.
"""
import json
import random
import time

from .logs import log, stage
from .ops import softmax_floats
from .tokenizer import END, SPECIAL_TOKENS, Tokenizer


class InferenceEngine:
    """Обёртка вокруг модели, которая принимает сообщения и возвращает ответ.

    Хранит KV-кэш между вызовами и помнит, для каких токенов он посчитан,
    чтобы следующий ход диалога не пересчитывал совпадающее начало (урок 02, этап 9).
    """

    def __init__(self, model):
        self.model = model
        self.tok = model.tok
        self._cache = None
        self._cache_tokens = []

    def chat(self, messages, temperature=0.2, max_tokens=30, num_ctx=96, seed=None, verbose=True):
        """Отвечает на список сообщений и, если verbose, печатает каждый этап урока 02.

        messages    — список {"role": "user" | "assistant", "content": "..."};
        temperature — острота распределения при выборе токена (урок 01, 4.8);
        max_tokens  — максимум токенов ответа;
        num_ctx     — размер контекстного окна: вход + ответ должны в него поместиться;
        seed        — зерно случайности, чтобы выбор токенов был повторяемым.

        Возвращает словарь в духе ответа Ollama: текст, причина остановки, счётчики
        токенов и тайминги prefill и decode.
        """
        say = log if verbose else (lambda *a: None)
        rnd = random.Random(seed)

        if verbose:
            stage(1, "Запрос пришёл в движок")
            say(json.dumps({"messages": messages, "temperature": temperature,
                            "max_tokens": max_tokens}, ensure_ascii=False, indent=2))

        prompt = self._build_prompt(messages, verbose)
        ids = self._tokenize(prompt, verbose)
        ids = self._fit_context(ids, max_tokens, num_ctx, verbose)

        new_ids, logits, prefill_time = self._prefill(ids, verbose)
        answer_ids, done_reason, decode_time, ttft_from_decode = self._decode(
            ids, logits, temperature, max_tokens, rnd, verbose)
        self._cache_tokens = list(ids)

        if verbose:
            stage(7, "Остановка")
            if done_reason == "stop":
                say(f"Модель выбрала {END!r} — нормальное завершение, done_reason = 'stop'.")
            else:
                say(f"Достигнут max_tokens = {max_tokens} — ответ обрезан, done_reason = 'length'.")

        # Этап 8. Ответ: детокенизация, счётчики, тайминги.
        response = {
            "message": {"role": "assistant", "content": self.tok.decode(answer_ids)},
            "done_reason": done_reason,
            "prompt_eval_count": len(new_ids),
            "prompt_eval_duration_s": round(prefill_time, 4),
            "eval_count": len(answer_ids),
            "eval_duration_s": round(decode_time, 4),
            "time_to_first_token_s": round(prefill_time + ttft_from_decode, 4),
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

    # ─── Этапы 2–4: подготовка входа ──────────────────────────────────────────
    def _build_prompt(self, messages, verbose):
        """Этап 2: склеивает сообщения по шаблону чата в одну строку."""
        prompt = Tokenizer.apply_chat_template(messages)
        if verbose:
            stage(2, "Сообщения склеиваются по шаблону чата")
            log(f"Строка, которую увидит модель:\n  {prompt!r}")
            log("В конце — открытое сообщение ассистента: модели остаётся продолжить текст.")
        return prompt

    def _tokenize(self, prompt, verbose):
        """Этап 3: режет строку на токены и переводит их в номера словаря."""
        pieces = self.tok.encode_pieces(prompt)
        ids = [self.tok.piece_to_id[p] for p in pieces]
        if verbose:
            stage(3, "Токенизация: строка → куски → номера")
            for p, i in zip(pieces, ids):
                log(f"  {p!r:>16} → {i}")
            log(f"Итого токенов входа: {len(ids)} (из них специальных: "
                f"{sum(p in SPECIAL_TOKENS for p in pieces)})")
        return ids

    def _fit_context(self, ids, max_tokens, num_ctx, verbose):
        """Этап 4: проверяет, что вход + max_tokens помещаются в окно, иначе молча отрезает начало."""
        if verbose:
            stage(4, "Проверка контекстного окна")
            log(f"Окно num_ctx = {num_ctx} токенов; вход {len(ids)} + max_tokens {max_tokens} "
                f"= {len(ids) + max_tokens}")
        if len(ids) + max_tokens > num_ctx:
            keep = max(num_ctx - max_tokens, 1)
            if verbose:
                log(f"  ⚠ Не помещается! Отрезаем начало: остаются последние {keep} ток. "
                    f"Ошибки не будет, модель просто не увидит часть текста.")
            return ids[-keep:]
        if verbose:
            log("  Помещается.")
        return ids

    # ─── Этап 5: prefill ──────────────────────────────────────────────────────
    def _prefill(self, ids, verbose):
        """Этап 5: прогоняет вход через модель и заполняет KV-кэш.

        Сначала ищет, сколько первых токенов уже посчитано в кэше с прошлого запроса,
        и прогоняет только остальные. Возвращает (новые токены, логиты последнего
        токена, время prefill в секундах).
        """
        if verbose:
            stage(5, "Prefill: модель читает весь вход и заполняет KV-кэш")
        reused = self._reuse_cache(ids)
        new_ids = ids[reused:]
        if verbose:
            if reused:
                log(f"Начало входа совпало с прошлым запросом: из KV-кэша взято готовыми {reused} "
                    f"ток., считаем только {len(new_ids)} новых (урок 02, этап 9).")
            else:
                log(f"KV-кэш пуст, считаем все {len(new_ids)} токенов.")
        trace = {}
        t0 = time.perf_counter()
        logits = self.model.forward(new_ids, self._cache, trace)[-1]
        prefill_time = time.perf_counter() - t0
        if verbose:
            self._log_attention(ids, trace)
            log(f"KV-кэш теперь хранит ключи и значения {len(self._cache[0]['k'])} токенов "
                f"в каждом из {self.model.n_layers} слоёв.")
        return new_ids, logits, prefill_time

    def _reuse_cache(self, ids):
        """Возвращает, сколько первых токенов входа совпадает с уже посчитанными в KV-кэше.

        Всё, что в кэше после совпавшего начала, выбрасывается. Последний токен входа
        всегда считаем заново: нам нужны его логиты, чтобы выбрать первый токен ответа.
        """
        if self._cache is None:
            self._cache, self._cache_tokens = self.model.new_cache(), []
        common = 0
        for a, b in zip(ids, self._cache_tokens):
            if a != b:
                break
            common += 1
        common = min(common, len(ids) - 1)
        for layer in self._cache:
            del layer["k"][common:]
            del layer["v"][common:]
        return common

    def _log_attention(self, ids, trace):
        """Печатает, на какие токены сильнее всего «смотрел» последний токен входа в последнем слое."""
        l = self.model.n_layers - 1
        log(f"Внимание последнего токена {self.tok.vocab[ids[-1]]!r} в слое {l + 1} "
            f"(на кого он смотрит сильнее всего):")
        for h in range(self.model.n_heads):
            weights = trace[(l, h)][-len(ids):]
            top = sorted(range(len(weights)), key=lambda i: -weights[i])[:3]
            parts = ", ".join(f"{self.tok.vocab[ids[i]]!r} {weights[i]:.0%}" for i in top)
            log(f"  голова {h + 1}: {parts}")

    # ─── Этапы 6–7: decode и остановка ────────────────────────────────────────
    def _decode(self, ids, logits, temperature, max_tokens, rnd, verbose):
        """Этап 6: генерирует ответ по одному токену, пока не выпадет <|end|> или не кончится max_tokens.

        На каждом шаге: логиты делятся на temperature, softmax даёт вероятности,
        токен выбирается случайно с этими вероятностями. Если это не <|end|>, он
        дописывается к входу и прогоняется через модель — один, остальные берутся
        из KV-кэша. ids дополняется на месте.

        Возвращает (номера токенов ответа, причина остановки, время decode,
        время от начала decode до выбора первого токена).
        """
        if verbose:
            stage(6, "Decode: ответ рождается по одному токену")
            log(f"{'шаг':>3} | {'топ-3 кандидата (вероятность после temperature)':<52} | выбран")
            log("-" * 72)
        answer_ids, done_reason = [], "length"
        t1 = time.perf_counter()
        first_token_at = None
        for step in range(1, max_tokens + 1):
            next_id, probs = self._sample(logits, temperature, rnd)
            if first_token_at is None:
                first_token_at = time.perf_counter()
            if verbose:
                top = sorted(range(len(probs)), key=lambda i: -probs[i])[:3]
                cand = ", ".join(f"{self.tok.vocab[i]!r} {probs[i]:.0%}" for i in top)
                log(f"{step:>3} | {cand:<52} | {self.tok.vocab[next_id]!r}")
            if self.tok.vocab[next_id] == END:  # этап 7: модель сама закончила сообщение
                done_reason = "stop"
                break
            answer_ids.append(next_id)
            ids.append(next_id)
            logits = self.model.forward([next_id], self._cache)[-1]
        decode_time = time.perf_counter() - t1
        return answer_ids, done_reason, decode_time, (first_token_at or t1) - t1

    @staticmethod
    def _sample(logits, temperature, rnd):
        """Выбор токена (урок 01, 4.8): делим логиты на temperature, softmax, случайный выбор.

        Возвращает (номер выбранного токена, список вероятностей всех токенов).
        """
        probs = softmax_floats([lg.data / temperature for lg in logits])
        return rnd.choices(range(len(probs)), weights=probs)[0], probs
