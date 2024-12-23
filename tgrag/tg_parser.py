import nest_asyncio
from telethon.sync import TelegramClient

nest_asyncio.apply()


async def get_channel_messages(api_id: str, api_hash: str, channel_username: str):
    client = TelegramClient("session_name", api_id, api_hash)
    await client.start()
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
