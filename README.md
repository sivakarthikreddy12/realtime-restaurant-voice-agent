# Realtime Restaurant Voice Agent (Local Mac Prototype)

A real-time English restaurant ordering demo built on Hugging Face
[`speech-to-speech`](https://github.com/huggingface/speech-to-speech). Microphone
audio is streamed into VAD → STT → LLM → TTS, the assistant speaks back in real
time, and it can call local Python tools for the menu, cart, and a **mock** order.

**This is a demo.** It never calls Toast, never uses Toast credentials, and never
places a real order. The final tool returns a `DEMO-xxxxxx` receipt and clears the
in-memory cart.

## Verified environment

- macOS on Apple Silicon
- CPython **3.14.5** (Python 3.11–3.14 is supported by `speech-to-speech` 1.1.0)
- `speech-to-speech==1.1.0`
- Microphone + speakers (headphones recommended to reduce echo)
- VS Code terminal
- First launch needs internet and roughly 8 GB of model downloads with the
  default models. Later launches can run from the local cache.

## 1. Set up the environment

From the project folder in the VS Code terminal:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.14 is not installed, Python 3.11, 3.12, or 3.13 also work. Use the
matching `python3.x -m venv .venv` command.

`requirements.txt` pins `speech-to-speech==1.1.0` and includes
`audioop-lts`, which restores the `audioop` module removed in Python 3.13+.
`certifi` supplies a trusted CA bundle if the interpreter's platform CA store is
empty.

## 2. Verify the installation

```bash
python main.py --check
```

This checks Python, the `speech-to-speech` version, `audioop`, NLTK data, TLS
certificate verification, the five tool schemas, and the default microphone and
speaker. It also lists audio device indices for later use.

Optional hardware checks:

```bash
python main.py --mic-test       # records 1 second and prints the signal level
python main.py --speaker-test   # plays a short 440 Hz tone
```

The first `--check` run downloads the small NLTK `punkt_tab` and
`averaged_perceptron_tagger_eng` resources into `.nltk_data/` in this project.

## 3. Launch the voice assistant

```bash
source .venv/bin/activate
python main.py
```

`main.py` is the recommended launcher. It fixes the console-script import path,
configures TLS trust safely, downloads/verifies NLTK data, validates the tool
module, and then starts `speech-to-speech local` with:

- the `order_tools` module
- an English restaurant system prompt
- `--qwen3_tts_language en`
- Apple Silicon `--mac-optimal-settings` automatically

Useful launcher options:

```bash
python main.py --check
python main.py --verbose
python main.py --input-device 0 --output-device 1
python main.py --block-mic-during-playback
python main.py --port 8766
python main.py --prompt "You are a terse English ordering assistant."
```

Use the indices printed by `--check` for `--input-device` and `--output-device`.
Extra `speech-to-speech local` flags go after `--`:

```bash
python main.py -- --stt mlx-audio-whisper --log_level debug
```

The first launch downloads and loads models. Wait for the terminal to say the
pipeline is ready, then speak normally. Press `Ctrl+C` to stop.

### Smaller/faster model set (optional)

The defaults are Parakeet TDT 0.6B, Qwen3.5-4B 4-bit, and Qwen3-TTS 1.7B. On a
machine with less RAM or for a quicker smoke test, override them:

```bash
python main.py -- \
  --stt mlx-audio-whisper \
  --mlx_audio_whisper_model_name mlx-community/whisper-base-mlx \
  --model_name mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --qwen3_tts_model_name mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit
```

Tool-calling quality is lower with the 1.5B model than with the default 4B model.

## 4. Optional web UI

A small local browser UI is included. It starts the Realtime voice backend
automatically, shows the conversation, supports a microphone button, plays the
spoken replies, and displays the live cart and tool results.

```bash
source .venv/bin/activate
python ui_server.py
```

Open the URL printed in the terminal, normally:

```text
http://127.0.0.1:8000
```

While testing voice turns, follow `latency.log` from any terminal with
`tail -f latency.log`. The server also prints these `[LATENCY]` lines to its
terminal. They show speech-end detection, transcription completion, response
start, first generated text/audio, response completion, and tool execution time
without logging the recognized words.

Controls:

- Tap **Talk** once to start browser microphone streaming and allow microphone
  access when the browser asks. Speak naturally; server-side voice activity
  detection responds after a 700 ms pause, so you do not need to tap Stop after
  each turn. Speak over the assistant to interrupt it and take the turn.
  **Stop listening** disables the microphone when you are finished.
- The text box works without a microphone, which is useful for testing.
- The right panel shows the cart, subtotal, and the last tool result.
- **Create mock order** appears after items are in the cart. It asks for
  confirmation, calls `get_cart` and `create_mock_order` through the same guarded
  tool path, and shows the `DEMO-xxxxxx` receipt. Use this when a very small LLM
  skips the order tool; it is not a real order.
- **New session** reloads the page and resets the in-memory cart.

If port 8765 is already used by `python main.py`, choose another internal port:

```bash
python ui_server.py --s2s-port 8766 --ui-port 8001
```

For a faster demo with the smaller cached model set:

```bash
python ui_server.py --small-models
```

To connect the UI to a Realtime server you already started separately:

```bash
python ui_server.py --no-start --s2s-port 8765
```

`Ctrl+C` stops the UI and, unless `--no-start` was used, the backend it started.
The UI binds to `127.0.0.1` by default and has no authentication; keep it local.
Only one browser session is active at a time because the backend pool size is 1.

## 5. Test conversation

Speak naturally in English:

- "Hi, what do you have on the menu?"
- "I'd like two cheeseburgers and one Coke."
- "Add fries as well."
- "Actually, remove one cheeseburger."
- "Can you read back my order?"
- After the assistant reads the items and subtotal: "Yes, place the order."

Expected tool sequence:

1. `show_menu`
2. `add_to_cart` (twice, or once per item)
3. `remove_from_cart`
4. `get_cart`
5. `create_mock_order` only after explicit confirmation

The mock receipt contains `status: MOCK_ORDER_CREATED`,
`real_order_placed: false`, an `order_id` such as `DEMO-A1B2C3`, the item lines,
and the subtotal. The cart is then cleared. Nothing is sent to Toast or any real
restaurant system.

`order_tools.py` also enforces the sequence: `create_mock_order` refuses to run
until `get_cart` has been read back, and it rejects negative/ambiguous replies
such as "no", "not yet", or "do not place the order".

## 6. Using the bare `speech-to-speech` command

The installed console script starts with `.venv/bin` on `sys.path`, not the
project folder. That is why `--tool-module order_tools` fails with
`ModuleNotFoundError` when the CLI is run by itself. Either use `main.py` (which
adds the project root and also bootstraps TLS/NLTK) or set the environment
explicitly:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"
export NLTK_DATA="$PWD/.nltk_data"
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
speech-to-speech local \
  --tool-module order_tools \
  --qwen3_tts_language en \
  --mac-optimal-settings \
  --init_chat_prompt "You are a friendly English-speaking restaurant order-taking assistant for Demo Burger House. Use the tools for the menu and cart. Call get_cart and read back the full cart before asking for confirmation. Only call create_mock_order after explicit confirmation. Never claim a real Toast order was placed."
```

Run `python main.py --check` first so the NLTK data has been downloaded. If you
set `SSL_CERT_FILE` yourself as shown above, certificate verification remains
enabled.

## 7. Run the tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

The suite covers menu/cart operations, aliases and plurals, quantity validation,
confirmation parsing, mock order safety, TLS fallback, launcher arguments, the
real `speech-to-speech` tool loader, session tool wiring, the client's tool
execution coordinator, and the web UI launcher/static page.

## Files

- `main.py` — robust launcher: `sys.path`, TLS/certifi, NLTK bootstrap, checks,
  microphone/speaker tests, English local pipeline.
- `ui_server.py` — local FastAPI web UI, Realtime WebSocket bridge, tool execution,
  and voice-backend lifecycle.
- `ui/index.html` — single-page chat UI with microphone capture, audio playback,
  cart, and tool status.
- `order_tools.py` — menu, cart, validation, explicit-confirmation logic, and the
  mock order tool. Returns plain JSON-serializable dictionaries; `CREATE_RESPONSE = True`.
- `requirements.txt` — pinned runtime dependencies, including Python 3.13+
  `audioop` compatibility.
- `tests/` — unit, Realtime-contract, and web UI tests.
- `README.md` — this guide.

## Troubleshooting

### `ModuleNotFoundError: No module named 'order_tools'`

Use `python main.py`, or set `PYTHONPATH="$PWD"` when invoking the console script
directly. The project root is not on `sys.path` for an installed console script.

### `CERTIFICATE_VERIFY_FAILED` / NLTK SSL errors

`main.py` checks whether Python's default TLS context has any CA certificates.
If it does not, it points `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, and
`CURL_CA_BUNDLE` at `certifi`'s bundle **while keeping certificate and hostname
verification enabled**. Then re-run:

```bash
python -m pip install --upgrade certifi
python main.py --check
```

If the platform store is present but stale, point Python at certifi explicitly
(this still verifies certificates and hostnames):

```bash
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
python main.py --check
```

If you are behind a corporate proxy, configure the proxy and its trusted CA
bundle in the normal way. Do **not** disable TLS verification, set
`verify=False`, or use `ssl.CERT_NONE`.

### `ModuleNotFoundError: No module named 'audioop'`

Python 3.13+ removed `audioop`. Reinstall the pinned requirements:

```bash
python -m pip install -r requirements.txt
```

`audioop-lts` should appear in `python main.py --check`.

### `speech-to-speech: command not found`

```bash
source .venv/bin/activate
pip show speech-to-speech
```

If it is missing, run `python -m pip install -r requirements.txt`.

### Microphone is silent or not detected

1. Run `python main.py --mic-test`.
2. Allow microphone access for Terminal/VS Code in **System Settings → Privacy &
   Security → Microphone**.
3. Run `python main.py --check` and choose the microphone index with
   `--input-device`.

### Assistant hears itself / echo

Use headphones. Alternatively start with `--block-mic-during-playback`, which
mutes microphone capture while assistant audio is playing.

### Speaker output is wrong

Run `python main.py --check` to list device indices, then pass
`--output-device N`. Test with `python main.py --speaker-test --output-device N`
(the launcher accepts both; the tone test uses the same device option).

### Port already in use

```bash
python main.py --port 8766
```

### Models are slow / memory pressure is high

Close memory-heavy apps. The first run downloads and loads models; later runs use
the Hugging Face cache. On 16 GB machines, the default 4B LLM is the documented
starting point. The optional smaller model set above uses substantially less RAM.

### Tool calls do not happen

Start with `python main.py --verbose` and confirm the terminal shows `TOOL: ...`
lines. Check that `python main.py --check` reports all five tools. Small LLMs may
ignore the tool prompt; use the default 4B model for better tool-calling.

### Assistant says the order is placed but no receipt appears

An order exists only if the terminal shows a `create_mock_order` tool call and
the returned receipt has `status: MOCK_ORDER_CREATED` plus a `DEMO-xxxxxx` ID.
The launcher prompt now explicitly requires `create_mock_order` when the customer
confirms and forbids claiming success without it. If a smaller model still skips
the call, switch to the default 4B model. If the call is skipped, the cart is not
cleared and no mock (or real) order was created.

## Scope and safety limits

- English-only test flow; sample USD menu prices.
- Cart state is in-memory and shared by the single default pipeline. Do not treat
  this as multi-tenant or production ordering software.
- No taxes, customizations, inventory, payments, or real Toast integration.
- Tool calls are model-directed. The code adds deterministic guards for quantity,
  cart state, read-back, and explicit confirmation, but this remains a demo.
- Never connect this project directly to a live ordering system without
  authorization, audit logging, idempotency, and integration tests.
