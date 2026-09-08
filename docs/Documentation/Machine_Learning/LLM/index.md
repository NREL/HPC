---
layout: default
title: LLM
parent: Machine Learning
---

# Large Language Model (LLM) Assistants

Kestrel hosts two locally-served LLM-based assistants for coding and HPC-support workflows. Both run open-weight models on Kestrel's own GPU nodes, so no code or data leaves NLR: **[OnField Assistant (`ofa`)](#onfield-assistant-ofa)**, a retrieval-augmented assistant built and maintained in-house and tuned for OpenFOAM/AMReX/MARBLES/VASP/HPC-support workflows, and **[OpenCode](#OpenCode)**, a general-purpose terminal coding agent that can also reach NLR's centrally-hosted ServeAI Gateway models.

## OnField Assistant (`ofa`)

[OnField Assistant](https://github.com/nileshsawant/onfield-assistant) (`ofa`) is a locally-hosted, retrieval-augmented-generation (RAG) LLM assistant for HPC and scientific-computing workflows. It runs Gemma 4 (31B, Google, Apache 2.0) via Ollama on a Kestrel GPU node, and layers RAG over Kestrel's documentation plus OpenFOAM, AMReX, MARBLES, VASP, ReFrame, and quantum-computing source/paper corpora. It currently ships eight specialized modes &mdash; `--code` (default), `--openfoam`, `--hpc`, `--amrex`, `--marbles`, `--quantum-computing`, `--vasp`, and `--rhel9_reframe` &mdash; each swapping in a different system prompt and RAG index.

```
module load assistant
```

puts `ofa` on `$PATH` (and the `ofa_client` Python module on `$PYTHONPATH`). A GPU is allocated automatically via SLURM the first time you run `ofa` (override the account/partition/walltime with the `OFA_ACCOUNT` / `OFA_PARTITION` / `OFA_WALLTIME` environment variables).

!!! note
    Full installation notes, the RAG-maintenance playbook, and an in-depth technical writeup live in the project's own docs: [nileshsawant.github.io/onfield-assistant](https://nileshsawant.github.io/onfield-assistant/).

There are five ways to use `ofa`:

### 1. Command line

Run `ofa` with no arguments for an interactive chat, or pass a single quoted prompt for a one-shot answer. A mode flag selects the system prompt + RAG index:

```
$ ofa                            # interactive chat, default --code mode
$ ofa "explain this SLURM error" # one-shot: answer and exit
$ ofa --hpc "how do I request an H100 node?"
$ ofa --openfoam --save ./mycase # OpenFOAM case generator, writes files to ./mycase
$ ofa --amrex                    # AMReX C++ framework assistant
$ ofa --marbles                  # MARBLES lattice-Boltzmann solver
$ ofa --vasp                     # VASP assistant
$ ofa --quantum-computing        # quantum-computing assistant
$ ofa --rhel9_reframe            # ReFrame (RHEL9) testing assistant
```

Useful flags (see `ofa --help` for the complete list):

| Flag | Effect |
| --- | --- |
| `--resume` | Reload and continue your previous session. |
| `--model <id>` | Use a specific model for this run (see `--list-models`). |
| `--save <dir>` | Write any files the assistant produces into `<dir>`. |
| `--fast` | OpenFOAM: skip the multi-file planning stage (single-shot). |
| `--no-rag` | Skip retrieval; ask the base model directly. |
| `--list-models` | Print the model registry and exit (runs on the login node, no GPU). |

Inside an interactive session, type `quit` (or `exit`/`q`) to leave, and use these slash commands:

| Command | Effect |
| --- | --- |
| `/help` | List all commands. |
| `/clear` | Reset the conversation (keeps the system prompt). |
| `/compact` | Compress history now to reclaim context space. |
| `/history` | Show the current session size. |
| `/memory`, `/remember <text>`, `/forget` | Inspect / add / clear long-term memory. |
| `/skills`, `/skill <name>` | List and load skill files. |
| `/models` | List pulled models and how to switch. |
| `@<path>` | Inline a file's contents into your prompt. |
| `$ <command>` | Run a shell command without leaving the chat. |
| `save <dir>` | Save the last response into `<dir>`. |

`ofa` remembers durable preferences you state in conversation (e.g. "always use 4-space indentation") across sessions, and auto-compresses long sessions so extended debugging runs don't lose earlier context.

### 2. VS Code Chat

The OnField Assistant VS Code extension puts every `ofa` mode directly in VS Code Copilot Chat's model picker. It handles the SLURM allocation, the compute-node connection, and the API key for you &mdash; there is nothing to configure on your laptop.

Setup, once per machine:

1. **Connect VS Code to Kestrel** using Remote-SSH (the standard "Connect to Host" flow &mdash; open a VS Code window attached to a Kestrel login node).
2. **Install the extension on the Kestrel side.** Open the Extensions view (`Ctrl+Shift+X` / `Cmd+Shift+X`), click the `⋯` menu at the top &rarr; **Install from VSIX…**, and select:

    ```
    /nopt/nrel/apps/cpu_stack/software/openfoam/assistant/vscode-ext/ofa-vscode.vsix
    ```

    Reload the window when prompted. The extension appears under **SSH: KESTREL** as `ofa-vscode`.

Each session:

3. **Connect.** Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and run **OFA: Connect**. This allocates a GPU node and starts the server; watch progress with **OFA: Show Logs**.
4. **Chat.** Open Copilot Chat, open the model picker, and choose any `ofa · …` entry (e.g. `ofa · code`, `ofa · openfoam`). Ask mode answers questions; Agent mode can additionally apply edits and run commands.

Run **OFA: Disconnect** to release the GPU allocation when you're done. If a previous allocation is still alive, the extension reconnects to it automatically on the next launch.

### 3. Python (`ofa_client`)

`ofa_client` is a stdlib-only Python module (no extra packages to install) that talks to a running `ofa --serve` over HTTP. Start the server once in your allocation, then call it from any Python process on the same node:

```bash
module load assistant
ofa --serve            # leave running; auto-allocates a GPU node if needed
```

```python
from ofa_client import ask

# One-shot question
text = ask("what is a good turbulence model for cavity flow at Re=1e4?")

# Pick a mode with model=, e.g. the OpenFOAM or HPC assistant
text = ask("draft a controlDict for a 2s transient run", model="ofa-openfoam")
```

The client auto-detects the server URL and API token from `$OFA_SCRATCH`, so no configuration is needed when it runs alongside the server. `ask()` also accepts optional context:

```python
# Attach a text file (last 32 KB by default; full_file=True for all of it)
ask("why did this solver diverge?", file="log.simpleFoam")

# Attach an image — great for having a running job summarise a plot mid-run
ask("summarise the pressure field", image="output/step_0100.png")

# Inline arbitrary text context
ask("is this residual trend converging?", context=my_residual_table)
```

For multi-turn conversations, use a `Session`, which keeps history client-side so follow-up questions see earlier turns:

```python
from ofa_client import Session

sess = Session(model="ofa-code")
sess.ask("what turbulence model for cavity flow?")
sess.ask("now show me a controlDict for that")   # remembers the first answer
```

A common pattern is a long-running simulation that periodically asks `ofa` to interpret its own output &mdash; e.g. summarise a freshly written plot or diagnose a crash &mdash; and logs the reply, all without leaving the node.

### 4. As the LLM backend for other agent frameworks

Because `ofa --serve` speaks the standard OpenAI `/v1/chat/completions` API, any existing agent framework that supports a custom `base_url` + `api_key` can use `ofa` as its backend LLM. For example, [AMReX Agent](https://github.com/AMReX-Codes/amrex-agent) has a `litellm` provider for exactly this; pointed at `ofa`:

```bash
module load assistant
ofa --serve --serve-enable-tools
export LITELLM_BASE_URL="http://localhost:$(cat $OFA_SCRATCH/.ofa_serve_port)/v1"
export LITELLM_API_KEY="$(cat $OFA_SCRATCH/.ofa_api_key)"
export LITELLM_MODEL="ofa-code"
```

See ofa's [Bring your own agent](https://github.com/nileshsawant/onfield-assistant#bring-your-own-agent) section for the full recipe and other integrations.

### 5. Build your own agent (index your own data)

You can point `ofa` at your **own** files &mdash; notes, papers, a codebase, documentation &mdash; and it will retrieve from them automatically in every mode, alongside the shared corpora. This effectively turns `ofa` into an assistant grounded in your material, and requires no write access to the shared install:

```bash
ofa --add-private ~/my-notes                 # index a directory
ofa --add-private ~/papers --private-name lit-review   # give it a name
ofa --list-private                           # see what you've indexed
ofa --forget-private lit-review              # remove one (or 'all')
```

Then just ask `ofa` a question &mdash; from the command line, from VS Code, or through any of the other access methods above &mdash; and answers will draw on your indexed material where relevant.

Supported inputs are text and code files, `.pdf`, and Office `.docx`/`.xlsx`. For scanned or equation-dense PDFs, `ofa` automatically re-reads hard-to-parse pages with the local vision model (rendering each page to an image and transcribing it, including equations) &mdash; on by default, using a vision-capable model such as the default Gemma 4. Your indexed data is stored per-user under `$OFA_SCRATCH` with owner-only permissions and never written to the shared install. Everything stays on Kestrel; as with any `ofa` use, keep in mind that answers (which may quote your indexed content) are returned to whichever client you're using.

## OpenCode

[OpenCode](https://opencode.ai) is a terminal-native, open-source AI coding agent (also available as a desktop app / IDE extension). NLR's build is enabled to communicate with two kinds of models:

* **ServeAI Gateway**: NLR-managed [HALO models](https://pages.github.nrel.gov/HALO/halo-docs/) hosted centrally on OpenStack, with no personal GPU allocation required. OpenCode currently supports access to Devstral 2 123B, GPT-OSS 120B, Gemma 4 31B, Nemotron 3 Super 120B, and Nemotron 3 Nano 30B, all with tool-calling enabled.
* **Local Node Model**: models you run yourself via Ollama on a GPU allocation (`gpt-oss:120b`, `gemma4:31b`, `muse-glimmer:30b`), reached at `$OLLAMA_HOST` (which dynamically sets the access port by default). These models are each able to easily fit within the memory constraints of a single 80GB H100 GPU for local inference.

!!! note
    OpenCode's Kestrel rules instruct it to warn you if it is about to run a local-Ollama model from a login node &mdash; that workload needs a GPU job (`salloc`/`sbatch`), not the shared login node itself. Only ServeAI Gateway models can be used through OpenCode on login nodes.

### Running OpenCode on Kestrel

After loading the module, run `opencode models <provider>` on a login node to see the LLMs available to use in OpenCode, where `<provider>` can be one of `local-ollama`  (which can be directly run on a Kestrel GPU compute node by any user) or `serveai` (one of the NLR-internal [HALO models](https://pages.github.nrel.gov/HALO/halo-docs/)).

Each time you load the module, it generates a per-user config at `~/.cache/opencode/opencode.json` with both providers above pre-wired and a conservative [default permission policy](#safety-and-permissions-considerations). This configuration file is generated upon each launch so that the user is always provided the most up-to-date list of supported LLMs to access. Additionally, when running [node-local models](#compute-node-access-node-local-model-serving-and-halo-models), when OpenCode is launched, the configuration file is automatically updated with an available port through which Ollama may use for serving. 

Ongoing work aims to integrate the OpenCode module with `ofa` as a provider, which would give you the entire `ofa` mode family alongside OpenCode's built-in providers in the same model picker.

#### Login node access: HALO models only

`module load opencode` is only available on Kestrel's GPU login node that runs [RHEL9](../../../RHEL9_upgrade/index.md) (i.e., `kl5`). OpenCode is not available on CPU nodes or any RHEL8 nodes:

```
ssh kl5.hpc.nlr.gov
module load opencode
opencode
```

After launching `opencode`, users may type `/models` in the interactive prompt to see the available models. Although `Ollama` (i.e., node-local) models may appear in the "Recent" list, attempting to connect to them on a login node will fail with an error. Only `ServeAI Gateway` provider models are usable when launching OpenCode from a login node.

!!! note
    Users are encouraged to use OpenCode on compute nodes instead of login nodes whenever possible.

#### Compute node access: Node-local model serving and HALO models

This example reflects requesting an interactive debug job for one hour to launch the `muse-glimmer:30b` model via OpenCode on a single H100 device:

```
ssh kl5.hpc.nlr.gov
salloc -A <project-handle> -p debug -t 01:00:00 -c 32 -n 1 --mem=85G --gres=gpu:1
module load opencode
opencode --model muse-glimmer:30b
```

### Safety and permissions considerations

The OpenCode module on Kestrel is designed to prevent the agent from mistakenly moving, deleting, or otherwise changing files on the user's behalf without the user's knowledge. To that end, the OpenCode module is deployed as a container and is given **read-only access** to Kestrel's filesystem **except for the directory from which the agent is launched**, which also has write access. Users are encouraged to keep the agent focused on the current working directory (as opposed to multiple /projects folders, for example), as it will only be able to write or modify files there. Additionally, `git push`, `chmod`/`chown`, and `rm` require confirmation from the user in each session or are denied outright.
