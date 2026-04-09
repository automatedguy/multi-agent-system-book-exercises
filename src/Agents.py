import asyncio
from picoagents import Agent
from picoagents.llm import OpenAIChatCompletionClient
from picoagents.orchestration import RoundRobinOrchestrator
from picoagents.termination import MaxMessageTermination, TextMentionTermination

# Setup the language model client
client = OpenAIChatCompletionClient(
    model="qwen3:8b",
    base_url="http://192.168.1.9:11434/v1",
    api_key="ollama"
)

poet = Agent(
    name="Poet",
    description="Haiku poet",
    instructions="You are a haiku poet.",
    model_client=client,
)

critic = Agent(
    name="Critic",
    description="Haiku critic you provide constructive feedback on haikus.",
    instructions="You are a haiku critic. When you critique a haiku, "
                 "you should provide feedback on the structure, "
                 "imagery, and emotional impact of the poem.",
    model_client=client,
)

termination = (MaxMessageTermination(max_messages=4), TextMentionTermination(text="APPROVED"))

orchestrator = RoundRobinOrchestrator(
    agents=[poet, critic],
    termination=termination,
    max_iterations=4
)


async def test_poet():
    response = await poet.run("Write a haiku about the ocean.")
    print(f"Poet says: {response}")


async def test_critique():
    response = await critic.run("The ocean is vast and deep, waves crash upon the shore, endless blue expanse.")
    print(f"Critic says: {response}")


async def run_orchestration():
    task = "Write a haiku about the ocean."
    stream = orchestrator.run(task)
    async for message in stream:
        print(f"{message.agent_name} says: {message.text}")

asyncio.run(run_orchestration())
