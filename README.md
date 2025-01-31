# LLM Intro Telegram RAG Bot

A Telegram bot that uses LLM and RAG (Retrieval Augmented Generation) to answer questions.

## Requirements

- Python 3.12.3
- Poetry for dependency management

## Installation

1. Clone the repository:
```bash
git clone git@github.com:Tialo/llm-intro-tg-rag.git
cd llm-intro-tg-rag
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:
```bash
poetry install
```

4. Set up pre-commit hooks:
```bash
poetry run pre-commit install
```

## Project Structure

```
llm-intro-tg-rag/
├── Dockerfile         # Docker configuration
├── README.md          # Project documentation
├── poetry.lock        # Dependencies lock file
├── pyproject.toml     # Poetry and tools configuration
└── tgrag/             # Main package directory
    ├── data.py        # Data handling module
    ├── example_data/  # Example data directory
    │   ├── 1.json
    │   ├── 2.json
    │   ├── 3.json
    │   └── 4.json
    ├── tg_bot.py      # Telegram bot implementation
    └── tg_parser.py   # Telegram message parser
```

## Development

The project uses the following tools to ensure code quality:

- **ruff**: Linter and code formatter
- **pre-commit**: Automatic checks before commit
  - Check for trailing whitespace
  - Check file endings
  - Check YAML files
  - Check large files
  - Verify poetry.lock synchronization
  - Format and lint with ruff

### Code Style

The project follows these code style rules:
- Line length: 88 characters
- Python version: 3.12
- Double quotes for strings
- Spaces for indentation

## Usage

- Complete installation
- Create .env file with following variables:
  1. OPENAI_API_KEY
  2. OPENAI_API_BASE (for specifying base url to openai-like api)
  3. USE_LOCAL_MODELS=1 (to use local models instead of api)
  4. MONITORED_CHANNELS (list of tg channels separated with commas)
  5. BOT_TOKEN ([telegram bot token](https://core.telegram.org/bots/faq#how-do-i-create-a-bot) from Botfather)
  6. TG_API_ID (to fetch new messages [guide](https://core.telegram.org/api/obtaining_api_id))
  7. TG_API_HASH (to fetch new messages [guide](https://core.telegram.org/api/obtaining_api_id))
- Run `python tgrag/tb_bot.py`

## Metrics

RAG was evaluated by Model Assited Expert Judgement.
Result metric is 8/10 
