import asyncio
import os
from collections import defaultdict
from datetime import datetime

from data import MessageTracker, fetch_new_messages, get_documents
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

USE_LOCAL_MODELS = int(os.getenv("USE_LOCAL_MODELS", 0))
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 1200))
MONITORED_CHANNELS = os.getenv("MONITORED_CHANNELS", "").split(",")

DEBUG = False
HISTORY_LIMIT = 10


class RAGTelegramBot:
    def __init__(self):
        self.qa_chain = None
        self.user_histories = defaultdict(list)
        self.history_limit = HISTORY_LIMIT
        self.vector_store = None
        self.message_tracker = MessageTracker()
        self.embeddings = None
        self.setup_rag()

    def setup_rag(self):
        if USE_LOCAL_MODELS:
            llm = OllamaLLM(model="llama3.2:3b")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                model_kwargs={"device": "cuda"},
            )
        else:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
            )
            self.embeddings = OpenAIEmbeddings()
        if os.path.isdir("vector_store"):
            self.vector_store = Chroma(
                embedding_function=self.embeddings, persist_directory="vector_store"
            )
        else:
            self.vector_store = Chroma.from_documents(
                get_documents(), self.embeddings, persist_directory="vector_store"
            )
        system_prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. "
            "Provide url of the message you use to give an answer, do NOT format it with braces [url](source), if there are few sources, provide all of them. "
            "Use the same language as the question."
            "\nContext: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )

        self.qa_chain = (
            {
                "context": self.vector_store.as_retriever(),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Split chains for debugging
        self.first_chain = {
            "context": self.vector_store.as_retriever(),
            "question": RunnablePassthrough(),
        } | prompt
        self.last_chain = llm | StrOutputParser()

    async def update_index(self):
        while True:
            try:
                print(f"[{datetime.now()}] Checking for new messages...")
                new_documents = await fetch_new_messages(
                    MONITORED_CHANNELS, self.message_tracker
                )

                if new_documents:
                    print(
                        f"[{datetime.now()}] Adding {len(new_documents)} new documents to the vector store"
                    )
                    self.vector_store.add_documents(new_documents)
                    print(f"[{datetime.now()}] Successfully added new documents")
                else:
                    print(f"[{datetime.now()}] No new messages found")

            except Exception as e:
                print(f"[{datetime.now()}] Error updating index: {e}")

            await asyncio.sleep(UPDATE_INTERVAL)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /start is issued."""
        await update.message.reply_text(
            "Hi! I am your RAG-powered Telegram bot. Send me a question, and I will try to help!"
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /help is issued."""
        await update.message.reply_text(
            "Send me any question, and I will search through my knowledge base to help you!"
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages and respond using RAG."""
        try:
            user_message = update.message.text
            user_id = update.message.from_user.id
            prompt_response = f"Based on this context:{self.user_histories[user_id]} \nRespond to this message: {user_message}"
            self.user_histories[user_id].append(f"User: {user_message}")
            if len(self.user_histories[user_id]) > self.history_limit:
                self.user_histories[user_id] = self.user_histories[user_id][
                    -self.history_limit :
                ]

            history_context = "\n".join(self.user_histories[user_id])  # noqa

            if DEBUG:
                first = self.first_chain.invoke(user_message)
                print(first)
                response = self.last_chain.invoke(first)
            else:
                response = self.qa_chain.invoke(prompt_response)

            self.user_histories[user_id].append(f"Bot: {response}")
            if len(self.user_histories[user_id]) > self.history_limit:
                self.user_histories[user_id] = self.user_histories[user_id][
                    -self.history_limit :
                ]

            await update.message.reply_text(response)
        except Exception as e:
            if DEBUG:
                raise
            await update.message.reply_text(f"Sorry, I encountered an error: {str(e)}")

    def run(self):
        """Run the bot."""
        application = Application.builder().token(os.getenv("BOT_TOKEN")).build()

        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        asyncio.get_event_loop().create_task(self.update_index())

        print("Bot is running...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    bot = RAGTelegramBot()
    bot.run()
