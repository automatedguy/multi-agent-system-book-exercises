import asyncio
from picoagents import Agent
from picoagents.llm import OpenAIChatCompletionClient
import os

# Setup the language model client
client = OpenAIChatCompletionClient(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

