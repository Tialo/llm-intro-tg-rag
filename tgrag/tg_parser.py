import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import nest_asyncio
from telethon.sync import TelegramClient

nest_asyncio.apply()

MAX_RETRIES = 3
RETRY_DELAY = 2


@asynccontextmanager
async def get_telegram_client(api_id: str, api_hash: str):
    """Context manager for handling Telegram client connection."""
    client = TelegramClient("session_name", api_id, api_hash)
    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()


async def get_messages_with_retry(
    client: TelegramClient, channel_username: str
) -> List[Dict[Any, Any]]:
    """Get messages with retry logic for handling database locks."""
    for attempt in range(MAX_RETRIES):
        try:
            channel = await client.get_entity(channel_username)
            messages = await client.get_messages(channel, limit=100)

            if channel_username.startswith("@"):
                channel_username = channel_username[1:]

            return [
                {
                    "message_id": message.id,
                    "date": str(message.date),
                    "sender_id": message.sender_id,
                    "message_text": message.text or "",
                    "message_url": f"https://t.me/{channel_username}/{message.id}",
                }
                for message in messages
            ]
        except Exception as e:
            if "database is locked" in str(e) and attempt < MAX_RETRIES - 1:
                print(
                    f"Database is locked, retrying in {RETRY_DELAY} seconds... (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(RETRY_DELAY)
                continue
            raise


async def get_channel_messages(api_id: str, api_hash: str, channel_username: str):
    """Main function to get channel messages."""
    async with get_telegram_client(api_id, api_hash) as client:
        try:
            return await get_messages_with_retry(client, channel_username)
        except Exception as e:
            print(f"Error fetching messages from {channel_username}: {e}")
            return []
