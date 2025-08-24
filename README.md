# RAG Learning Project

This repository contains a Streamlit based demo for a Retrieval-Augmented Generation (RAG) system. It implements
basic scaffolding for login, file uploads and a placeholder chat interface.

## Features
- **Login system**: ten built-in demo accounts defined in `config/accounts.json`.
- **Knowledge base page**: upload files to per-user folders under `data/<user>/uploads`.
- **Chat page**: placeholder interface for asking questions; currently returns a static message.

The project is structured so that further development can add document parsing, chunking, embedding with Qwen
`text-embedding-v4`, storage in Chroma and retrieval augmented generation following the detailed specification.

## Running
Install dependencies (including `openai>=1.0`) and start the Streamlit app. Before
launching the app, set `DASHSCOPE_API_KEY` in the environment so the client can
authenticate against DashScope. Optionally set `DASHSCOPE_BASE_URL` to override
the default endpoint.

```bash
pip install -r requirements.txt

export DASHSCOPE_API_KEY="<your_api_key>"
# export DASHSCOPE_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

streamlit run app/main.py
```

## Repository layout
```
app/            # Streamlit application entry point
config/         # demo account configuration
requirements.txt
README.md
```
