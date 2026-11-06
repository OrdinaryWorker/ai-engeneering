"""Командная строка ToyLLM.

Запуск (из папки toy-llm):
  python3 main.py train                      # обучение, ~4 минуты; сохраняет weights.json
  python3 main.py chat "кто доставил заказ?" # инференс одного запроса с подробным логом
  python3 main.py demo                       # сценарии: запрос, второй ход диалога, temperature, max_tokens
"""
import argparse
from pathlib import Path

from src import InferenceEngine, Tokenizer, ToyLLM
from src.data import DIALOGS
from src.logs import log, stage
from src.training import train

WEIGHTS_FILE = Path(__file__).parent / "weights.json"


def cmd_train(args):
    """Обучает токенизатор и веса с нуля и сохраняет результат в weights.json."""
    stage("Т1", "Обучаем токенизатор (BPE) на текстах диалогов")
    texts = [t for pair in DIALOGS for t in pair]
    tok = Tokenizer.train(texts, num_merges=args.merges)
    stage("Т2", "Обучаем веса модели")
    model = ToyLLM(tok)
    model.describe()
    train(model, DIALOGS, steps=args.steps)
    model.save(WEIGHTS_FILE)
    log(f"\nВеса сохранены в {WEIGHTS_FILE.name}. Это и есть «модель на диске»: токенизатор + числа.")


def load_engine():
    """Загружает модель из weights.json, печатает её состав и оборачивает в движок."""
    if not WEIGHTS_FILE.exists():
        raise SystemExit("Нет weights.json — сначала обучи модель: python3 main.py train")
    model = ToyLLM.load(WEIGHTS_FILE)
    model.describe()
    return InferenceEngine(model)


def cmd_chat(args):
    """Отвечает на один вопрос и печатает все этапы пути запроса."""
    engine = load_engine()
    engine.chat([{"role": "user", "content": args.prompt}], temperature=args.temperature,
                max_tokens=args.max_tokens, num_ctx=args.num_ctx, seed=args.seed)


def cmd_demo(args):
    """Прогоняет четыре сценария, которые иллюстрируют урок 02."""
    engine = load_engine()

    log("\n\n######## Сценарий 1. Первый запрос ########")
    history = [{"role": "user", "content": "кто доставил заказ?"}]
    r = engine.chat(history, temperature=0.2, seed=1)

    log("\n\n######## Сценарий 2. Второй ход диалога: история отправляется целиком ########")
    history += [r["message"], {"role": "user", "content": "что сделал менеджер?"}]
    engine.chat(history, temperature=0.2, seed=1)

    log("\n\n######## Сценарий 3. Temperature: один вопрос, по 5 ответов ########")
    for t in (0.1, 1.0, 3.0):
        answers = [engine.chat([{"role": "user", "content": "что сделал курьер?"}], temperature=t,
                               seed=s, verbose=False)["message"]["content"] for s in range(5)]
        log(f"temperature={t}:")
        for a in answers:
            log(f"   {a!r}")

    log("\n\n######## Сценарий 4. max_tokens слишком мал ########")
    engine.chat([{"role": "user", "content": "заказ оплачен. что делать?"}], max_tokens=3, seed=1)


def main():
    """Разбирает аргументы командной строки и запускает нужную команду."""
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
