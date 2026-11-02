# ai-engeneering

Учебный репозиторий 14-недельного курса AI-инженерии: от основ LLM до RAG, агентов и эксплуатации. Модели запускаются локально (Ollama на рабочей станции), все вызовы модели идут через `llm/client.py` (появится в конце недели 1).

## Структура

| Папка | Что там |
|---|---|
| `week-01/` | Неделя 1: как устроены LLM, первый вызов локальной модели |
| `week-01/toy-llm/` | Игрушечная LLM на чистом Python: n-граммы и мини-трансформер |
| `docs/` | Справочные материалы: разборы статей и т.п. |

## Неделя 1

- [00-ai-agents-ai-engineering.md](week-01/00-ai-agents-ai-engineering.md): AI, LLM, агенты, AI-инженерия, квантование
- [01-monday-llm-basics.md](week-01/01-monday-llm-basics.md): токены, контекстное окно, temperature, галлюцинации
- [01b-monday-tokenizer-transformer.md](week-01/01b-monday-tokenizer-transformer.md): как устроена LLM, путь одного запроса
- [02-tuesday-ollama-setup.md](week-01/02-tuesday-ollama-setup.md): драйвер, Ollama, модель, SSH-туннель, первый вызов
- [toy-llm/](week-01/toy-llm/README.md): игрушечная LLM, только стандартная библиотека
- [toy_bpe.py](week-01/toy_bpe.py): игрушечный BPE-токенизатор

## Документы

- [Attention Is All You Need (2017): подробный разбор на русском](docs/attention-is-all-you-need-ru.md)
