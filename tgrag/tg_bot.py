import os

from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from langchain import hub
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from data import get_documents

load_dotenv()


class RAGTelegramBot:
    def __init__(self):
        self.qa_chain = None
        self.setup_rag()

    def setup_rag(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cuda'}
        )
        llm = OllamaLLM(model="llama3.2:3b")
        vector_store = Chroma.from_documents(get_documents(), embeddings)
        prompt = hub.pull("rlm/rag-prompt")

        self.qa_chain = (
            {
                "context": vector_store.as_retriever(),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /start is issued."""
        await update.message.reply_text(
            'Hi! I am your RAG-powered Telegram bot. Send me a question, and I will try to help!'
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /help is issued."""
        await update.message.reply_text(
            'Send me any question, and I will search through my knowledge base to help you!'
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages and respond using RAG."""
        try:
            user_message = update.message.text

            response = self.qa_chain.invoke(user_message)

            await update.message.reply_text(response)
        except Exception as e:
            await update.message.reply_text(
                f"Sorry, I encountered an error: {str(e)}"
            )

    def run(self):
        """Run the bot."""
        application = Application.builder().token(os.getenv('BOT_TOKEN')).build()

        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        print("Bot is running...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    bot = RAGTelegramBot()
    bot.run()
