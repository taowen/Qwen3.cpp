# vendor-runtime

Vendored runtime sources for custom C++ apps (chatbot, service wrappers, etc.).

## Included

- `src/chatbot_main.cpp` (editable REPL chatbot entry)
- `src/runner.cpp`, `include/runner.h` (vendored from ExecuTorch)
- `src/llama_tiktoken.cpp`, `include/llama_tiktoken.h` (vendored from ExecuTorch)
- `upstream/*` (snapshot copy for diff/reference)

## Build

```powershell
cd C:\Apps\qwen3.cpp
.\scripts\build_chatbot.ps1
```

## Run

```powershell
cd C:\Apps\qwen3.cpp
.\scripts\run_chatbot.ps1
```

## Notes

- Export still uses `executorch.examples.models.llama.export_llama`.
- Runtime app source is now local and editable.
- If upstream runner changes, run `scripts/vendor_sync_runtime.ps1` to resync.
