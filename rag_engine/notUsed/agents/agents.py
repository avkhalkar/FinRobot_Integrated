import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.agents import create_react_agent, AgentExecutor

# ✅ IMPORT TOOLS DIRECTLY (NO Tool() WRAPPING)
from finance_tools import (
    fetch_stock_price_tool,
    fetch_historical_prices_tool,
    fetch_news_tool
)

# -----------------------------
# TOOL LIST (AS-IS)
# -----------------------------
tools_list = [
    fetch_stock_price_tool,
    fetch_historical_prices_tool,
    fetch_news_tool
]

# -----------------------------
# STRICT ReAct PROMPT
# -----------------------------
FINANCE_AGENT_PROMPT = """
You are a financial analysis assistant.

You have access to the following tools:
{tools}

Tool names:
{tool_names}

You MUST follow this exact format:

Thought: explain your reasoning
Action: one of [{tool_names}]
Action Input: JSON input
Observation: tool result
... (repeat if needed)
Final Answer: final response to user

Question:
{input}

Thought:
{agent_scratchpad}
"""

prompt_template = PromptTemplate(
    input_variables=[
        "input",
        "tools",
        "tool_names",
        "agent_scratchpad"
    ],
    template=FINANCE_AGENT_PROMPT
)

# -----------------------------
# LLM (HUGGING FACE)
# -----------------------------
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.0,
    huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY")
)

model = ChatHuggingFace(llm=llm)

# -----------------------------
# AGENT
# -----------------------------
agent = create_react_agent(
    llm=model,
    tools=tools_list,
    prompt=prompt_template
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools_list,
    verbose=True,
    handle_parsing_errors=True
)

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    query1 = "What is the latest price of Apple and why is it moving today?"
    # query2 = "MSFT news"
    # query3 = "last 1 month prices of AAPL"


    response = agent_executor.invoke({
        "input": query1
    })

    print("\nFINAL OUTPUT:\n")
    print(response["output"])