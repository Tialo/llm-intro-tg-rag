from tg_bot import RAGTelegramBot
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv

load_dotenv()

class Evaluator:
    def __init__(self, model_name = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
                model= model_name,
                temperature=0.3,
            )
    def evaluate(self, question, context, generated_answer):
        prompt_template = PromptTemplate(
            input_variables=["question", "context", "generated_answer"],
            template = (
                "Question: {question}\n"
                "Answer: {generated_answer}\n\n"
                "Evaluate the quality of the answer based on the following criteria:\n"
                "- Accuracy (from 1 to 10): How accurate is the answer.\n"
                "- Completeness (from 1 to 10): How well does the answer cover all aspects of the question.\n"
                "Return the result in JSON format, for example:\n"
                "{\"accuracy\": 9, \"completeness\": 8}"
            )
        )
        chain = prompt_template | self.llm
        evaluation = chain.invoke({
            "question": question,
            "generated_answer": generated_answer
        })

        return evaluation.content


if __name__ == "__main__":
    bot = RAGTelegramBot()
    question = "Расскажи что нибудь о LLM"
    rag_response = bot.qa_chain.invoke(question)
    eval = Evaluator()
    print(eval.evaluate(question=question, generated_answer=rag_response, context=""))

