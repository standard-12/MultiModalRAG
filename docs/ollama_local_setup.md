# Ollama Local Setup Guide

How to run Ollama natively on macOS (especially Apple Silicon M1/M2/M3) for use with this project.

> **Why native instead of Docker?**
> Docker on Mac runs inside a Linux VM with no access to the Metal GPU.
> Every model runs on CPU — responses take 10–30 minutes.
> Running natively gives full GPU acceleration: responses in 1–3 seconds.

---

## Requirements

- macOS 12+ (Monterey or later)
- Apple Silicon (M1/M2/M3) or Intel Mac with 8 GB+ RAM
- [Homebrew](https://brew.sh) installed

---

## Step 1 — Install Ollama

```bash
brew install ollama
```

Verify the install:

```bash
ollama --version
```

---

## Step 2 — Pull the model

This project uses `qwen2:0.5b` — 352 MB, fast on CPU and GPU:

```bash
ollama pull qwen2:0.5b
```

Other model options (ordered by size):

| Model | Size | Notes |
|-------|------|-------|
| `qwen2:0.5b` | 352 MB | Default for this project — fast, low RAM |
| `tinyllama` | 637 MB | Slightly larger, good for Q&A |
| `phi3:mini` | 2.2 GB | Smarter responses, needs more RAM |
| `llama3.2:3b` | 2.0 GB | Strong general-purpose |
| `llava` | 4.7 GB | Vision model — used for image captioning |

To switch models, update `MODEL` in `app/app.py`:

```python
MODEL = "qwen2:0.5b"   # change this
```

---

## Step 3 — Starting Ollama manually

Use this if you want full control over when Ollama runs — no background service, no auto-login.

**Local only** (only your machine can connect):

```bash
ollama serve
```

**Network-accessible** (others on your network can connect too):

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

Ollama runs on port `11434`. Keep this terminal open while the app is running.
To stop it, press `Ctrl+C`.

---

## Step 4 — Auto-start at login

Use this if you want Ollama to start automatically every time you log in — no need to run anything manually.

**4a. Copy the plist to LaunchAgents:**

```bash
cp /opt/homebrew/opt/ollama/homebrew.mxcl.ollama.plist \
   ~/Library/LaunchAgents/homebrew.mxcl.ollama.plist
```

**4b. Edit `~/Library/LaunchAgents/homebrew.mxcl.ollama.plist`** and add `OLLAMA_HOST` to the `EnvironmentVariables` block so it's network-accessible:

```xml
<key>EnvironmentVariables</key>
<dict>
    <key>OLLAMA_HOST</key>
    <string>0.0.0.0</string>
    <key>OLLAMA_FLASH_ATTENTION</key>
    <string>1</string>
    <key>OLLAMA_KV_CACHE_TYPE</key>
    <string>q8_0</string>
</dict>
```

**4c. Load the service:**

```bash
launchctl load ~/Library/LaunchAgents/homebrew.mxcl.ollama.plist
```

Ollama will now start automatically on every login.

**To stop and disable auto-start:**

```bash
launchctl unload ~/Library/LaunchAgents/homebrew.mxcl.ollama.plist
```

---

## Step 5 — Verify

```bash
# Check server is up
curl http://localhost:11434/api/tags

# Check model is loaded and GPU is active
ollama ps
# Should show: PROCESSOR = 100% GPU
```

---

## Sharing with others on the same network

Find your Mac's IP:

```bash
ipconfig getifaddr en0
```

Others connect to `http://<your-ip>:11434`.

Update `OLLAMA_HOST` in `app/app.py` to your IP if the app runs on a different machine:

```python
OLLAMA_HOST = "http://192.168.x.x:11434"
```

If the app runs on the **same Mac**, keep it as:

```python
OLLAMA_HOST = "http://localhost:11434"
```

---

## Useful commands

```bash
ollama list              # list downloaded models
ollama ps                # show running models + GPU usage
ollama pull <model>      # download a model
ollama rm <model>        # delete a model
ollama run <model>       # interactive chat in terminal
tail -f /opt/homebrew/var/log/ollama.log   # live logs
```

---

## Troubleshooting

**Model takes 20+ seconds on first request**
Normal — this is the cold start while the model loads into GPU memory.
Subsequent requests will be fast (1–3 s for qwen2:0.5b).

**`ollama ps` shows CPU instead of GPU**
Ollama is not using Metal. Make sure you installed natively via Homebrew, not via Docker.

**Port 11434 already in use**
Another Ollama instance is running. Kill it:

```bash
pkill -f "ollama serve"
```

**Connection refused on `localhost:11434`**
Ollama isn't running. Start it:

```bash
ollama serve
# or if using launchctl:
launchctl load ~/Library/LaunchAgents/homebrew.mxcl.ollama.plist
```

**Others on the network can't connect**
Make sure you started with `OLLAMA_HOST=0.0.0.0` and check your Mac's firewall settings under System Settings → Network → Firewall.
