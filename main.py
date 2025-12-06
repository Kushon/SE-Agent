import math
import os

from dotenv import find_dotenv, load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

load_dotenv(find_dotenv())

# Langfuse configuration
Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    base_url=os.getenv("LANGFUSE_BASE_URL"),
)
langfuse = get_client()
langfuse_handler = CallbackHandler()

llm = ChatOllama(model="qwen3:8b", temperature=0.7, base_url="http://localhost:11434")


@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression. Example: '2 + 2 * 3'"""
    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {"sin": math.sin, "cos": math.cos, "sqrt": math.sqrt, "pi": math.pi},
        )
        return float(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_word_length(word: str) -> int:
    """Get the length of a word"""
    return len(word)


@tool
def uppercase(text: str) -> str:
    """Convert text to uppercase"""
    return text.upper()


@tool
def reverse_text(text: str) -> str:
    """Reverse the text"""
    return text[::-1]


tools = [calculate, get_word_length, uppercase, reverse_text]

agent = create_agent(model=llm, tools=tools, debug=True)


def run_agent(query: str):
    """Run the agent with a user query"""
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"{'=' * 60}\n")

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"callbacks": [langfuse_handler]},
    )
    langfuse.flush()

    final_message = result["messages"][-1]
    print(f"\nFinal Response:\n{final_message.content}\n")
    return final_message.content


if __name__ == "__main__":
    queries = [
        "What is 25 * 4 plus 100?",
        "How many letters are in the word 'programming'?",
        "Convert 'hello world' to uppercase and tell me how many characters it has",
        "Reverse the word 'agent' and calculate 2**3",
    ]
    for query in queries:
        run_agent(query)
