import os

import chainlit as cl
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import OpenWeatherMapAPIWrapper

load_dotenv()

# OpenAI API key
OPENAI_API_KEY = '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'
# OpenWeather API key
os.environ["OPENWEATHERMAP_API_KEY"] = '0e4b0b27974960b0dd09faeb7dfde1fb'

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

search = DuckDuckGoSearchRun()
weather = OpenWeatherMapAPIWrapper(openweathermap_api_key='0e4b0b27974960b0dd09faeb7dfde1fb')

# Web Search Tool
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="A useful tool for searching the Internet to find information on world events, issues, etc. Worth "
                "using for general topics. Use precise questions.",
)

# Open Weather Tool
open_weather_tool = Tool(
    name="Open Weather",
    func=weather.run,
    description="Open Weather Tool",
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = PromptTemplate(
    template="""Plan: {input}

History: {chat_history}

Let's think about answer step by step.
If it's information retrieval task, solve it like a professor in particular field.""",
    input_variables=["input", "chat_history"],
)

plan_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""Prepare plan for task execution. (e.g. retrieve current date to find weather forecast)

    Tools to use: web search, Open Weather

    Question: {input}

    History: {chat_history}

    Output look like this:
    '''
        Question: {input}

        Execution plan: [execution_plan]

        Rest of needed information: [rest_of_needed_information]
    '''

    '''
        input: {input}
    '''
    """,
)

plan_chain = ConversationChain(
    llm=llm,
    memory=memory,
    input_key="input",
    prompt=plan_prompt,
    output_key="output",
)

# Initialize Agent
agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=[search_tool, open_weather_tool],
    llm=llm,
    verbose=True,  # verbose option is for printing logs (only for development)
    max_iterations=3,
    prompt=prompt,
    memory=memory,
)


@cl.langchain_factory(use_async=False)
def main():
    chain = plan_chain
    return chain


@cl.langchain_run
async def run(agent, input_str):
    res = await cl.make_async(agent)(input_str, callbacks=[cl.LangchainCallbackHandler()])
    await cl.Message(content=res["output"]).send()
