# ai-engeneering

Учебный репозиторий курса AI-инженерии: от основ LLM до RAG, агентов и эксплуатации. Курс идёт спринтами без привязки ко времени: спринт закрывается, когда его темы пройдены. Модели запускаются локально (Ollama на рабочей станции), все вызовы модели идут через `llm/client.py` (появится в конце спринта 1).

## Структура

| Папка | Что там |
|---|---|
| `sprint-01-llm-basics/` | Спринт 1: как устроены LLM, первый вызов локальной модели |
| `sprint-01-llm-basics/practice/primitive-llm/` | Примитивная языковая модель на подсчёте частот |
| `sprint-01-llm-basics/practice/toy-llm/` | Учебный прототип ToyLLM на чистом Python: путь запроса в консоли |
| `docs/` | Справочные материалы: разборы статей и т.п. |

## Спринт 1: основы LLM

- [00-ai-agents-ai-engineering.md](sprint-01-llm-basics/00-ai-agents-ai-engineering.md): AI, LLM, агенты, AI-инженерия, квантование
- [01-llm-basics.md](sprint-01-llm-basics/01-llm-basics.md): урок 1 — что такое LLM, из чего состоит, трансформер по частям, контекст, temperature, галлюцинации
- [02-how-llm-works.md](sprint-01-llm-basics/02-how-llm-works.md): урок 2 — путь одного запроса от твоего кода до ответа: prefill, decode, KV-кэш, память и время
- [03-ollama-setup.md](sprint-01-llm-basics/03-ollama-setup.md): драйвер, Ollama, модель, SSH-туннель, первый вызов
- [primitive-llm/](sprint-01-llm-basics/practice/primitive-llm/README.md): примитивная языковая модель на `dict` — следующее слово по частотам, temperature
- [toy-llm/](sprint-01-llm-basics/practice/toy-llm/README.md): учебный прототип ToyLLM — токенизатор, трансформер, KV-кэш, prefill и decode с подробным логом

## Документы

- [Attention Is All You Need (2017): подробный разбор на русском](docs/attention-is-all-you-need-ru.md)
