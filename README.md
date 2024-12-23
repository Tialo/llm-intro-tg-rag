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

TODO: Add bot usage instructions

## License

TODO: Add license information
