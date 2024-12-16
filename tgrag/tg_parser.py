from telethon.sync import TelegramClient
import nest_asyncio

nest_asyncio.apply()


async def get_channel_messages(api_id: str, api_hash: str, channel_username: str):
    client = TelegramClient('session_name', api_id, api_hash)
    await client.start()
    channel = await client.get_entity(channel_username)

    messages = await client.get_messages(channel, limit=100)

    return [
        {
            "message_id": message.id,
            "date": str(message.date),
            "sender_id": message.sender_id,
            "message_text": message.text
        }
        for message in messages
    ]
