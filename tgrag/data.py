import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

proj_path = Path(__file__).parent


def get_documents(documents : list[str] = None):
    if documents is None:
        documents = [
            proj_path / "./example_data/1.json",
            proj_path / "./example_data/2.json",
            proj_path / "./example_data/3.json",
            proj_path / "./example_data/4.json",
        ]

    raw_data = []
    for document in documents:
        with open(document, 'r') as f:
            data = json.load(f)

        raw_data.append("\n".join(msg["message_text"] for msg in data if msg["message_text"]))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.create_documents(raw_data)
