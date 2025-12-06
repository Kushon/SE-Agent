# Agent

This project demonstrates creating an intelligent agent that uses the local language model qwen3:8b through Ollama and performs various tasks using special tools.

## Agent Capabilities

The agent can:

- Calculate mathematical expressions (with support for sin, cos, sqrt, pi)

- Count word lengths

- Convert text to uppercase
-  Reverse text

## Requirements

- Python 3.12
- Ollama with model qwen3:8b
- uv



## Installation

Clone the repo:
```bash
git clone https://github.com/Kushon/SE-Agent.git
```

### Local 
Download the model:
```bash
ollama pull qwen3:8b
```

Run the project:
```bash
uv run main.py
```
