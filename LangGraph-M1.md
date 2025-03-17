# Introduction to LangGraph (Module 1)

This document provides a comprehensive guide to the fundamentals of LangGraph, a specialized framework for building agent and multi-agent applications with LLMs. It covers key concepts from basic graph construction to advanced agent architectures with memory and deployment options.

## Table of Contents

- [Introduction](#introduction)
- [Setup and Prerequisites](#setup-and-prerequisites)
- [Understanding Chat Models](#understanding-chat-models)
- [Building Your First Graph](#building-your-first-graph)
- [Creating a Chain](#creating-a-chain)
- [Building a Router](#building-a-router)
- [Agent Architecture](#agent-architecture)
- [Adding Memory to Agents](#adding-memory-to-agents)
- [Deployment Options](#deployment-options)

## Introduction

### What is LangGraph?

LangGraph is a framework built within the LangChain ecosystem designed for creating stateful, multi-actor applications with Large Language Models (LLMs). It addresses the challenges of building reliable agent systems that can automate complex tasks.

The core design philosophy of LangGraph is to provide developers with greater precision and control in agent workflows. This makes it particularly suitable for complex real-world systems where you might need an agent to follow specific execution patterns, such as always calling a particular tool first or using different prompts based on its state.

### Key Features

LangGraph offers three core benefits compared to other LLM frameworks:

1. **Cycles** - LangGraph allows you to define flows that involve cycles, which are essential for most agent architectures. This differentiates it from DAG-based (Directed Acyclic Graph) solutions.

2. **Controllability** - It provides fine-grained control over the execution flow, allowing for precise orchestration of multi-step reasoning and tool usage.

3. **Persistence** - LangGraph enables state persistence across interactions, essential for maintaining context in long-running conversations.

## Setup and Prerequisites

Before diving into LangGraph, let's set up our environment with the necessary dependencies.

### Installation

```python
%pip install --quiet -U langchain_openai langchain_core langchain_community langgraph tavily-python
```

### API Keys Setup

LangGraph will typically require API keys for various services. Here's how to set them up:

```python
import os, getpass

def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:  # If the key does not exist
        print("OPENAI_API_KEY is not set.")
        api_key = input("Please enter your OpenAI API Key: ").strip()
        
        if api_key:  # Ensure the user enters a non-empty key
            os.environ["OPENAI_API_KEY"] = api_key
            print("OPENAI_API_KEY has been set successfully.")
        else:
            print("Error: API Key cannot be empty.")
    
    return os.getenv("OPENAI_API_KEY")

# A simpler helper function for setting environment variables
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Example usage
_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")  # If using Tavily search
```

## Understanding Chat Models

Chat models in LangChain take a sequence of messages as inputs and return chat messages as outputs. LangChain doesn't host models but integrates with third-party providers like OpenAI.

### Initializing Chat Models

```python
from langchain_openai import ChatOpenAI

# Initialize models with different parameters
gpt4o_chat = ChatOpenAI(model="gpt-4o", temperature=0)
gpt35_chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
```

The most common parameters for chat models include:

- `model`: The specific model to use
- `temperature`: Controls randomness (0 = deterministic, 1 = creative)
- `timeout`: Request timeout
- `max_tokens`: Maximum tokens to generate
- `stop`: Default stop sequences
- `max_retries`: Maximum number of retries for requests

### Working with Messages

Chat models use messages with different roles to structure conversations:

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Create a message
msg = HumanMessage(content="Hello world", name="Lance")

# Message list
messages = [msg]

# Invoke the model with a list of messages 
response = gpt4o_chat.invoke(messages)
```

You can also invoke a chat model directly with a string, which will be converted to a `HumanMessage`:

```python
response = gpt4o_chat.invoke("hello world")
```

### Using Search Tools

Some modules use external tools like Tavily, a search engine optimized for LLMs and RAG:

```python
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_search = TavilySearchResults(max_results=3)
search_docs = tavily_search.invoke("What is LangGraph?")
```

## Building Your First Graph

Let's build a simple graph with 3 nodes and one conditional edge to understand the basics of LangGraph.

![Simple Graph](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dba5f465f6e9a2482ad935_simple-graph1.png)

### State in LangGraph

First, we need to define the **State** of our graph, which serves as the input schema for all Nodes and Edges:

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

The State schema defines what data structure will be passed between nodes in the graph. Here, we're using a simple string value `graph_state`.

### Nodes

Nodes in LangGraph are just Python functions. The first positional argument of each node function is the state, as defined above:

```python
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}
```

Each node returns a new value for the `graph_state` key, which by default will override the prior state value.

### Edges

Edges connect the nodes in a graph. There are two types:

1. **Normal Edges** - Always go from one node to another
2. **Conditional Edges** - Optionally route between nodes based on some logic

Here's an example of a conditional edge:

```python
import random
from typing import Literal

def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:
        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"
```

This function uses the current state to determine which node to visit next based on a random condition.

### Graph Construction

Now let's build the graph from our components:

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Compile
graph = builder.compile()

# Visualize
display(Image(graph.get_graph().draw_mermaid_png()))
```

Special nodes:
- `START`: A special node that sends user input to the graph
- `END`: A special node that represents a terminal node

### Graph Invocation

The compiled graph implements the "runnable" protocol, providing a standard way to execute LangChain components:

```python
# Invoke the graph with an initial state
result = graph.invoke({"graph_state" : "Hi, this is Lance."})
```

When `invoke` is called:
1. The graph starts execution from the `START` node
2. It progresses through nodes based on the defined edges
3. The conditional edge will route from node 1 to node 2 or 3 based on the decision function
4. Execution continues until it reaches the `END` node
5. It returns the final state after all nodes have executed

Output example:
```
---Node 1---
---Node 3---
{'graph_state': 'Hi, this is Lance. I am sad!'}
```

## Creating a Chain

Now we'll build a more sophisticated chain that combines several key concepts:

1. Using chat messages as graph state
2. Using chat models in graph nodes
3. Binding tools to our chat model
4. Executing tool calls in graph nodes

![Chain Diagram](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbab08dd607b08df5e1101_chain1.png)

### Messages in LangChain

Chat models can use different types of messages that represent various roles in a conversation:

```python
from langchain_core.messages import AIMessage, HumanMessage

# Create a conversation
messages = [AIMessage(content="So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content="Yes, that's right.", name="Lance"))
messages.append(AIMessage(content="Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content="I want to learn about the best place to see Orcas in the US.", name="Lance"))

# Print the conversation
for m in messages:
    m.pretty_print()
```

### Working with Tools

Tools are useful for allowing a model to interact with external systems. When we bind an API as a tool, we give the model awareness of the required input schema.

```python
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# Bind the tool to our model
llm_with_tools = llm.bind_tools([multiply])

# Example use - the model will call the tool when appropriate
tool_call = llm_with_tools.invoke([HumanMessage(content="What is 2 multiplied by 3", name="Lance")])
print(tool_call.tool_calls)
```

The model will choose to call a tool based on the natural language input and return output that adheres to the tool's schema.

### Using Messages as State

To use messages in our graph state, we'll define a custom state type:

```python
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

The `add_messages` reducer ensures that new messages are appended to the existing list rather than replacing them.

For convenience, LangGraph includes a pre-built `MessagesState`:

```python
from langgraph.graph import MessagesState

class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    pass
```

### Building the Chain Graph

Now let's create a simple chain graph using our message state:

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
    
# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

# Visualize
display(Image(graph.get_graph().draw_mermaid_png()))
```

When we invoke this graph, the LLM will either respond directly or make a tool call:

```python
# Regular response
messages = graph.invoke({"messages": HumanMessage(content="Hello!")})

# Tool call response
messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3")})
```

## Building a Router

Now we'll extend our graph to work with either type of output from the LLM - either a direct response or a tool call.

![Router Diagram](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbac6543c3d4df239a4ed1_router1.png)

We'll use two key components:

1. Add a node that will execute our tool
2. Add a conditional edge that routes to our tool-calling node or ends based on the LLM's output

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# Node for LLM with bound tools
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # Routes based on whether the LLM output is a tool call
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

# Visualize
display(Image(graph.get_graph().draw_mermaid_png()))
```

In this graph:
- The `tool_calling_llm` node invokes the LLM with tools bound to it
- The `tools_condition` function checks if the LLM output contains a tool call:
  - If yes, it routes to the `tools` node
  - If no, it goes directly to `END`
- The `tools` node executes the tool and adds the result as a ToolMessage

## Agent Architecture

We can extend the router into a complete agent architecture by creating a loop that allows the agent to make multiple tool calls as needed.

![Agent Diagram](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbac0ba0bd34b541c448cc_agent1.png)

This implements the [ReAct pattern](https://react-lm.github.io/), which consists of:

- **Reason**: The model thinks about what to do based on current context
- **Act**: The model calls specific tools
- **Observe**: Tool outputs are passed back to the model

```python
# Define our tools
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# System message to guide the agent
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Assistant node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph with loop
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
# Critical line: loop back to the assistant after executing tools
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Visualize
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
```

Example usage for a multi-step calculation:

```python
messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
result = react_graph.invoke({"messages": messages})
```

In this example, the agent will:
1. First add 3 and 4 (getting 7)
2. Then multiply 7 by 2 (getting 14)
3. Finally divide 14 by 5 (getting 2.8)
4. Provide a final answer: "The final result after performing the operations \( (3 + 4) \times 2 \div 5 \) is 2.8."

Each step involves the model making a tool call, receiving the result, and deciding what to do next.

## Adding Memory to Agents

One limitation of the basic agent is that state is transient during a single graph execution. This limits our ability to have multi-turn conversations with interruptions.

![Agent Memory Diagram](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbab7453080e6802cd1703_agent-memory1.png)

To address this, we can use LangGraph's persistence capabilities, which allow the graph to store state and resume from where it left off.

### Using a Checkpointer

One of the easiest persistence mechanisms is the `MemorySaver`, an in-memory key-value store for graph state:

```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
```

### Threading for State Management

When using memory, we need to specify a `thread_id` to store our collection of graph states:

```python
# Specify a thread
config = {"configurable": {"thread_id": "1"}}

# First interaction
messages = [HumanMessage(content="Add 3 and 4.")]
result = react_graph_memory.invoke({"messages": messages}, config)

# Second interaction (continuing the conversation)
messages = [HumanMessage(content="Multiply that by 2.")]
result = react_graph_memory.invoke({"messages": messages}, config)
```

The checkpointer:
1. Saves the state at every step of the graph
2. Stores these checkpoints in a thread
3. Allows accessing the thread in the future using the `thread_id`

![State Persistence Diagram](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e0e9f526b41a4ed9e2d28b_agent-memory2.png)

This enables the agent to understand references to previous conversation elements, like "that" referring to the result of the previous calculation.

## Deployment Options

There are several ways to deploy and interact with LangGraph applications:

### Key Concepts

- **LangGraph**: Python and JavaScript library for creating agent workflows
- **LangGraph API**: Bundles the graph code with a task queue and persistence
- **LangGraph Cloud**: Hosted service for LangGraph API deployments
- **LangGraph Studio**: IDE for developing and testing LangGraph applications
- **LangGraph SDK**: Client library for interacting with LangGraph applications

### Testing Locally

You can run LangGraph Studio locally with:

```bash
langgraph dev
```

This starts a local server with:
- API endpoint: http://127.0.0.1:2024
- Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

### Using the SDK

You can interact with locally hosted graphs using the SDK:

```python
from langgraph_sdk import get_client

# Connect to local server
URL = "http://127.0.0.1:2024"
client = get_client(url=URL)

# List available graphs
assistants = await client.assistants.search()

# Create a thread for tracking state
thread = await client.threads.create()

# Run the graph
input = {"messages": [HumanMessage(content="Multiply 3 by 2.")]}
async for chunk in client.runs.stream(
        thread['thread_id'],
        "agent",
        input=input,
        stream_mode="values",
    ):
    if chunk.data and chunk.event != "metadata":
        print(chunk.data['messages'][-1])
```

### Deploying to the Cloud

You can deploy LangGraph applications to LangGraph Cloud through a GitHub repository:

1. **Create a repository** for your LangGraph application
2. **Connect LangSmith** to your repository via the Deployments tab
3. **Configure your deployment** with the appropriate paths and API keys
4. **Interact with your deployed graph** using the SDK or LangGraph Studio

Once deployed, you can use the SDK to interact with your cloud deployment:

```python
# Connect to cloud deployment
URL = "https://your-deployment-url.langgraph.app"
client = get_client(url=URL)

# Create a thread
thread = await client.threads.create()

# Run the agent
input = {"messages": [HumanMessage(content="Multiply 3 by 2.")]}
async for chunk in client.runs.stream(
        thread['thread_id'],
        "agent",
        input=input,
        stream_mode="values",
    ):
    if chunk.data and chunk.event != "metadata":
        print(chunk.data['messages'][-1])
```

## Conclusion

In this module, we've covered the fundamentals of LangGraph, progressing from basic graph concepts to sophisticated agent architectures with memory and deployment options. Here's a summary of what we've learned:

1. **Foundational Concepts**: State, nodes, edges, and graph construction
2. **Working with LLMs**: Message handling, tool binding, and execution
3. **Agent Patterns**: Building routers and ReAct agents
4. **Advanced Features**: Adding memory and persistence
5. **Deployment**: Local testing and cloud deployment options

LangGraph provides a powerful framework for building complex, stateful agent applications with LLMs. By focusing on precision and control, it enables developers to create more reliable systems for real-world applications.
