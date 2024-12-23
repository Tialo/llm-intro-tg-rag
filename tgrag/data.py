import json
import os
from pathlib import Path
from typing import Dict, Optional

from langchain.docstore.document import Document
from tg_parser import get_channel_messages

proj_path = Path(__file__).parent


class MessageTracker:
    def __init__(self, storage_file: str = "message_tracker.json"):
        self.storage_file = proj_path / storage_file
        self.last_message_ids: Dict[str, int] = {}
        self.load_state()

    def load_state(self):
        if self.storage_file.exists():
            with open(self.storage_file, "r", encoding="utf-8") as f:
                self.last_message_ids = json.load(f)
        else:
            self.last_message_ids = {}

    def save_state(self):
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(self.last_message_ids, f)

    def update_last_message_id(self, channel: str, message_id: int):
        self.last_message_ids[channel] = message_id
        self.save_state()

    def get_last_message_id(self, channel: str) -> Optional[int]:
        return self.last_message_ids.get(channel)


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


async def fetch_new_messages(channels: list[str], message_tracker: MessageTracker):
    api_id = os.getenv("TG_API_ID")
    api_hash = os.getenv("TG_API_HASH")

    new_documents = []

    for channel in channels:
        try:
            messages = await get_channel_messages(api_id, api_hash, channel)
            last_message_id = message_tracker.get_last_message_id(channel)

            if messages:
                message_tracker.update_last_message_id(
                    channel, messages[0]["message_id"]
                )
                new_messages = messages
                if last_message_id is not None:
                    new_messages = [
                        msg for msg in messages if msg["message_id"] > last_message_id
                    ]
                new_documents.extend(
                    [
                        Document(
                            page_content=msg["message_text"],
                            metadata={"message_url": msg["message_url"]},
                        )
                        for msg in new_messages
                        if msg["message_text"].strip()
                    ]
                )

        except Exception as e:
            print(f"Error fetching messages from {channel}: {e}")

    return new_documents
