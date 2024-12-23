import json
from pathlib import Path

from langchain.docstore.document import Document

proj_path = Path(__file__).parent


def get_documents(file_paths: list[str] = None):
    if file_paths is None:
        file_paths = [
            proj_path / "./example_data/1.json",
            proj_path / "./example_data/2.json",
            proj_path / "./example_data/3.json",
            proj_path / "./example_data/4.json",
        ]

    documents = []
    for file_path in file_paths:
        with open(file_path, encoding="utf-8") as f:
            messages = json.load(f)

        documents.extend(
            Document(
                page_content=msg["message_text"],
                metadata={"message_url": msg["message_url"]},
            )
            for msg in messages
        )

    return documents
