"""
LangGraph Streaming Demo
========================
Demonstrates 3 ways to consume agent output as it runs:
  1. astream(stream_mode=["values"])  -> full state snapshots after each step
  2. astream(stream_mode=["updates"]) -> only the diff each node contributed
  3. astream_events(version="v1")     -> fine-grained events from every component

Source: github.com/benman1/generative_ai_with_langchain (chapter 6, second_edition)
"""

import asyncio
from datasets import load_dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits.load_tools import load_tools
#from langchain.agents import create_agent, create_react_agent
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState


import os
from dotenv import load_dotenv

load_dotenv()
# ---------------------------------------------------------------------------
# 1. Setup: model, tools, prompt, agent
# ---------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0)

# Tools the ReAct loop can call. Some tools (like ddg-search) need an LLM
# under the hood for summarization, hence the llm= argument.
research_tools = load_tools(
    tool_names=["ddg-search", "arxiv", "wikipedia"],
    llm=llm,
)

system_prompt = (
    "You're a hard-working, curious and creative student. "
    "You're working on exam question. Think step by step. "
    "Always provide an argumentation for your answer. "
    "Do not assume anything, use available tools to search "
    "for evidence and supporting statements."
)

raw_prompt_template = (
    "Answer the following multiple-choice question."
    "\nQUESTION:\n{question}\n\nANSWER OPTIONS:\n{options}\n"
)

# ChatPromptTemplate with three slots:
#   - "system": persona / instructions
#   - "user": the question + options (filled from state)
#   - "placeholder": where the ReAct agent injects its tool-call scratchpad
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", raw_prompt_template),
    ("placeholder", "{messages}"),
])


# State schema: extends AgentState (which already has `messages`) with the
# question + options fields the prompt template references.
class ResearchState(AgentState):
    question: str
    options: str


research_agent = create_react_agent(
    model=llm,
    tools=research_tools,
    state_schema=ResearchState,
    prompt=prompt,
)


# ---------------------------------------------------------------------------
# 2. Pull a question from MMLU to feed the agent
# ---------------------------------------------------------------------------

ds = load_dataset("cais/mmlu", "high_school_geography")
ds_dict = ds["test"].take(100).to_dict()

i = 6
question = ds_dict["question"][i]
options = "\n".join([f"{i}. {a}" for i, a in enumerate(ds_dict["choices"][i])])

print(f"QUESTION: {question}")
print(f"OPTIONS:\n{options}\n")
print("=" * 70)


# ---------------------------------------------------------------------------
# 3. Streaming Mode 1: "values" - full state after each step
# ---------------------------------------------------------------------------
# Use this when you want to re-render the entire conversation each turn
# (e.g. a chat UI that redraws on every update).

async def demo_values_mode():
    print("\n[ values mode ] -- full state after each node\n")
    async for _, event in research_agent.astream(
        {"question": question, "options": options},
        stream_mode=["values"],
    ):
        # event["messages"] is the FULL conversation history at this point
        print(f"  total messages in state: {len(event['messages'])}")
    # Expected progression: 0 -> 1 -> 5 -> 6
    #   0: initial state
    #   1: agent emitted AIMessage with 4 parallel tool calls
    #   5: tools node added 4 ToolMessages (1 + 4 = 5)
    #   6: agent emitted final AIMessage synthesizing results (5 + 1 = 6)


# ---------------------------------------------------------------------------
# 4. Streaming Mode 2: "updates" - per-node diffs
# ---------------------------------------------------------------------------
# Use this for progress indicators / per-node logging.
# Event shape: {node_name: {field: new_value}}

async def demo_updates_mode_compact():
    print("\n[ updates mode (compact) ] -- which node ran, how many msgs added\n")
    async for _, event in research_agent.astream(
        {"question": question, "options": options},
        stream_mode=["updates"],
    ):
        node = list(event.keys())[0]
        msg_count = len(event[node].get("messages", []))
        print(f"  node={node!r}  messages_added={msg_count}")
    # Expected: agent 1 -> tools 1 -> agent 1
    # NOTE: the "tools 1" event has a single update payload that contains
    # 4 ToolMessages inside (parallel tool calls collapse into one update).


async def demo_updates_mode_full():
    print("\n[ updates mode (full payload) ] -- complete dict each step\n")
    async for _, event in research_agent.astream(
        {"question": question, "options": options},
        stream_mode=["updates"],
    ):
        print(event)
        print("-" * 70)


# ---------------------------------------------------------------------------
# 5. Streaming Mode 3: astream_events - fine-grained events
# ---------------------------------------------------------------------------
# Use this for token-by-token streaming, tracing (Langfuse/LangSmith),
# or any custom hook below the node level.

async def demo_astream_events():
    print("\n[ astream_events v1 ] -- what event types fire?\n")
    seen_events = set()
    async for event in research_agent.astream_events(
        {"question": question, "options": options},
        version="v1",
    ):
        seen_events.add(event["event"])
    print(f"  unique event types: {seen_events}")
    # Expected events include:
    #   on_prompt_start / on_prompt_end           -- prompt template
    #   on_chat_model_start / on_chat_model_stream / on_chat_model_end
    #                                             -- LLM calls (stream = per-token chunks)
    #   on_tool_start / on_tool_end               -- each tool invocation
    #   on_chain_start / on_chain_stream / on_chain_end
    #                                             -- the wrapping chains


async def demo_astream_events_token_streaming():
    """
    Practical example: stream LLM tokens as they're generated.
    This is what powers a "typing" effect in chat UIs.
    """
    print("\n[ astream_events -- token streaming ]\n")
    async for event in research_agent.astream_events(
        {"question": question, "options": options},
        version="v1",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # chunk is an AIMessageChunk; .content is the text piece
            if chunk.content:
                print(chunk.content, end="", flush=True)
        elif event["event"] == "on_tool_start":
            print(f"\n\n[calling tool: {event['name']}]\n")
        elif event["event"] == "on_tool_end":
            print(f"\n[tool finished: {event['name']}]\n")
    print()


# ---------------------------------------------------------------------------
# 6. Run all demos
# ---------------------------------------------------------------------------

async def main():
    await demo_values_mode()
    await demo_updates_mode_compact()
    await demo_updates_mode_full()
    await demo_astream_events()
    await demo_astream_events_token_streaming()


if __name__ == "__main__":
    asyncio.run(main())