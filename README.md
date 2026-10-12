# Sharlok

![Sharlok logo](static/logo.png)

**AI threat modeling agent for system architectures.** Uses the STRIDE methodology and runs on local models through Ollama.

Sharlok is a comprehensive threat modeling agent that uses the **STRIDE methodology** to analyze system architectures and identify security vulnerabilities. It's designed for offline or corporate environments and leverages local models via Ollama.

**⚠️ Important Info:** 
1. Sharlok uses `nomic-embed-text` for embeddings and `gemma4:latest` for analysis and image extraction
2. You do not need to start Ollama or pull models by hand. `make setup` installs them, and `make run` starts Ollama and pulls any missing model before the server starts
3. Output quality depends on architecture diagram clarity and knowledge base content

## Features

- **STRIDE Threat Modeling**: Comprehensive analysis covering all threat categories
  - Spoofing (Identity)
  - Tampering (Data/System)
  - Repudiation (Action Denial)
  - Information Disclosure
  - Denial of Service
  - Elevation of Privilege

- **Web Interface**: Modern, responsive UI for uploading diagrams and viewing threat models
- **Local Models**: Analysis runs on Ollama, with no cloud LLM APIs
- **Threat Model Generation**: Automatic analysis and documentation
- **Security Recommendations**: Prioritized, actionable recommendations
- **Interactive Chat**: Ask follow-up questions about threats and mitigations
- **Knowledge Base Integration**: Leverage your own security documentation and best practices

## High-Level Behavior

1. **Upload Architecture Diagram**: Drag & drop or click to upload
2. **Analyze with STRIDE**: Auto-generates threat model
3. **View Threat Model**: See categorized threats with mitigations
4. **Generate Recommendations**: Get prioritized security recommendations
5. **Chat for Details**: Ask questions about specific threats or implementation

## Quick Start

### Prerequisites

- macOS or Linux. Windows works through WSL or Git Bash.
- `make` and `curl`
- Internet access for the first setup only (Ollama and model downloads)

### 1. Setup (First Time)

```bash
make setup
```

`make setup` runs these steps in order:

| Step | Target | What it does |
|---|---|---|
| 1 | `install-uv` | Installs [uv](https://docs.astral.sh/uv/) if missing (Homebrew on macOS, official installer otherwise) |
| 2 | `install-ollama` | Installs the Ollama CLI for your OS: Homebrew on macOS, official install script on Linux, `winget` on Windows |
| 3 | `deps` | Creates `.venv` and installs Python dependencies with `uv sync` |
| 4 | `start-ollama` | Starts `ollama serve` in the background if it is not already running |
| 5 | `models` | Pulls `gemma4:latest` and `nomic-embed-text` |
| 6 | `rerank` | Downloads the reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` from Hugging Face |

The Makefile reads the same environment variables as the app (see [Local Models](#local-models)).

### 2. Run

```bash
make run
```

Open http://localhost:8000 in your browser.

The server checks Ollama on every start through `ready.py`. It starts `ollama serve` if the server is down and pulls any required model that is missing. If the check fails, the server does not start. The check runs in the app's startup hook, so `make run`, `uv run main.py`, `uvicorn app:app`, and `python app.py` all enforce it. Ollama keeps running after Sharlok exits.

### 3. Usage

1. Click the upload area or drag a diagram (image, PDF, PPTX, or Mermaid/text file)
2. Click the "Analyze with STRIDE" button
3. Wait for threat model generation (1-2 minutes depending on model and diagram)
4. View the threat model, recommendations, and chat about findings
5. Download or save the threat model markdown

## Knowledge Base

The knowledge base gives the analysis security context from your own documents. Sources are embedded into a local FAISS store at `Knowledgebase/.faiss_store/` (generated automatically, do not edit).

```
Knowledgebase/
  PDF/            # Security whitepapers and guides (.pdf)
  Repos/          # Security-relevant source code and architecture docs
  Tools/          # Security tool and framework documentation
  .faiss_store/   # Generated index
webpages.yml      # Topic-keyed URL lists, fetched and indexed at build time
```

### Web Pages (`webpages.yml`)

URLs are grouped by topic. Each topic is a list of URLs:

```yaml
aws:
  - https://docs.aws.amazon.com/security/
  - https://aws.amazon.com/architecture/security-identity-compliance/
azure:
  - https://docs.microsoft.com/en-us/security/
other:
  - https://owasp.org/
  - https://www.nist.gov/
```

Topic keys are `aws`, `azure`, `gcp`, `ai`, and `other`.

### Building and Rebuilding the Index

- On startup, Sharlok builds the index in the background if the store is missing. Until the build finishes, analysis runs without knowledge base grounding.
- `make build-index` adds only new or changed files. Deduplication uses content hashes.
- To rebuild from scratch: `rm -rf Knowledgebase/.faiss_store && make build-index`

### Make Targets

| Command | Purpose |
|---|---|
| `make setup` | Full first-time setup (see Quick Start) |
| `make run` | Check Ollama and models, then start the server on port 8000 |
| `make build-index` | Update the knowledge base index and exit |
| `make models` | Pull the Ollama models only |
| `make rerank` | Download the reranker model only |
| `make help` | List all targets |

`uv run main.py` is the same as `make run`, and `uv run main.py --build-index` is the same as `make build-index`.

## Customizing System Prompts

All threat modeling and chat prompts are in `prompts.yaml`:

```yaml
threat_model_system_prompt      # STRIDE analysis instructions
threat_model_user_prompt        # Threat model output format
chat_system_prompt              # Chat assistant behavior
recommendation_prompt           # Security recommendations
```

Edit these files to customize behavior without changing code. Changes apply on next request (no restart needed).

To reload prompts in a running Python session:

```python
from threat_modeling import reload_prompts
reload_prompts()
```

## API Endpoints

### Sessions
- `POST /api/session` - Create new threat modeling session
- `GET /api/session/{session_id}` - Get session data
- `DELETE /api/session/{session_id}` - Clean up session

### Analysis
- `POST /api/upload` - Upload architecture diagram
- `POST /api/analyze` - Generate STRIDE threat model
- `POST /api/recommendations` - Generate security recommendations
- `GET /api/threat_models` - List all saved threat models
- `GET /threat_models/{filename}` - Download threat model

### Chat
- `POST /api/chat` - Send message about threat model
- `GET /uploads/{filename}` - Serve uploaded image

### Health
- `GET /health` - Health check

## Local Models

Defaults (set in `Makefile` and `ready.py`):

| Variable | Default | Purpose |
|---|---|---|
| `Sharlok_MODEL` | `gemma4:latest` | Threat model generation and recommendations |
| `Sharlok_VISION_MODEL` | `gemma4:latest` | Diagram and image understanding |
| `Sharlok_EMBED_MODEL` | `nomic-embed-text` | Knowledge base embeddings |
| `Sharlok_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker for retrieved context |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

Override any variable on the command line. `make run` pulls the new model before starting:

```bash
Sharlok_MODEL=llama3.1:8b make run
```

Sharlok auto-starts Ollama only when `OLLAMA_HOST` points to localhost. For a remote host, start the server on that machine.

Changing `Sharlok_EMBED_MODEL` requires rebuilding the index, because the vector size changes:

```bash
rm -rf Knowledgebase/.faiss_store && make build-index
```

## Supported Input Formats

- **Images:** PNG (`.png`), JPEG (`.jpg`, `.jpeg`), WebP (`.webp`), GIF (`.gif`)
- **Documents:** PDF (`.pdf`, text-based; scanned PDFs with little extractable text are rejected), PowerPoint (`.pptx`)
- **Text diagrams:** Mermaid and plain text (`.mmd`, `.md`, `.txt`)

Maximum recommended: 4MB per file (larger files process slower)

## Threat Model Output

Each threat model includes:

```markdown
# Threat Model: [System Name]

## Executive Summary
[Overview of system and primary threats]

## Architecture Overview
[Component description]

## Threat Analysis by Category
### Spoofing (Identity Spoofing)
- Threat: [Description]
  Mitigation: [Specific actions]

### Tampering
### Repudiation
### Information Disclosure
### Denial of Service
### Elevation of Privilege

## Risk Summary Table
[Threat, Severity, Component, Mitigation]

## Security Recommendations
[Prioritized actionable steps]
```

## Saving & Exporting

- Threat models are automatically saved to `threat_models/` as markdown
- Download button on the UI exports current threat model
- Chat transcripts can be saved from the chat interface
- All files are stored locally — no cloud sync

## Troubleshooting

### Ollama Not Reachable

`make run` starts Ollama automatically on localhost. If startup still fails:

```bash
# Check the server
curl http://localhost:11434/api/tags

# Start it manually and watch the output
ollama serve

# Confirm installed models
ollama list
```

- `Ollama CLI not found`: run `make install-ollama` or `make setup`
- `Ollama did not respond within 30s`: run `ollama serve` in another terminal to see the error
- Remote `OLLAMA_HOST`: Sharlok does not start remote servers, so start Ollama on that host

### Model Pull Fails

- Check network access, then run `make models`
- Confirm the tag exists in the Ollama library, for example `ollama pull <model>:<tag>`

### Analysis Takes Too Long

- Large models (26B, 34B) can take 2-5 minutes
- Use a smaller model with `Sharlok_MODEL=<model>:<tag> make run`
- Reduce image complexity or split the diagram into smaller parts

### Empty or Incomplete Threat Model

- Make sure the image is clear and readable
- Try a different format (JPG vs PNG)
- For PDFs, make sure the text is extractable and not only scanned images
- Check the Ollama server log (macOS: `~/.ollama/logs/server.log`)

### Knowledge Base Not Used

- Check that `Knowledgebase/.faiss_store/` exists
- Run `make build-index` and check the logs for `No FAISS index`

## Performance Tips

1. Use medium-sized models (7B-13B) for fast analysis
2. Start with clear, simple architecture diagrams
3. Build the knowledge base for better recommendations
4. Run Ollama on a machine with adequate RAM (16GB+ recommended)

## Security & Privacy

✅ **What's Private:**
- All analysis happens locally
- No data sent to cloud services
- No external embedding APIs
- Internet is used only during setup (Ollama, models, reranker), when building the knowledge base from `webpages.yml`, and for optional live web search (off by default)
- Conversations stored only in session memory

❌ **What's Public:**
- Downloaded threat model files (you manage storage)
- Knowledge base content (you control)

## Requirements

- Python 3.10 or newer (`uv` manages the virtual environment)
- `make`, `curl`, and `uv` (installed by `make setup` if missing)
- Ollama (installed by `make setup`)
- macOS or Linux. Windows via WSL or Git Bash.
- 16GB+ RAM recommended
- 10GB+ disk space for models and the reranker
- Modern browser (Chrome, Firefox, Safari, Edge)

## Notes

- Threat models are incremental — you can refine by uploading new diagrams
- Chat history is session-specific and lost on page refresh
- Knowledge base indexing runs automatically at startup if the store is missing. The web UI works without it, but answers are not grounded in your documents until the build finishes.
- All threat model files use STRIDE framework (security industry standard)

## Next Steps

1. Add security documentation to `Knowledgebase/`
2. Create organization-specific threat templates in `prompts.yaml`
3. Integrate with your architecture review process
4. Use threat models for security testing roadmaps

## Support & Feedback

For issues, improvements, or questions:
- Check Ollama status: `ollama list`
- Review browser console for client errors
- Check server logs for API errors

## Related Projects

- **brodit**: Technical blog writer using the same Ollama stack
- **Ollama**: Local model serving (https://ollama.com)
- **STRIDE**: Microsoft threat modeling methodology
- **OWASP**: Security resources and guidance
