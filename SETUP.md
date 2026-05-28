# SETUP

Step-by-step guide to take a fresh Windows machine to a working POC. Read top-to-bottom the first time; the **Quick start** at §1 is for repeat setups after the first install.

For *what* this POC is and *why* it exists, see [README.md](README.md). For the full architecture, schemas, and design decisions, see [implementation_plan.md](implementation_plan.md). For the on-screen demo script, see [poc/README.md](poc/README.md).

---

## 1. Quick start (returning user, everything already installed)

```powershell
cd c:\Users\Dell\Desktop\Project\Floor_Plan\poc
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Browser opens at `http://localhost:8501`. The app loads the cached run automatically and is demoable in one click.

If you're setting up on a new machine or after a `git clone`, continue with §2.

---

## 2. Prerequisites

| Requirement | Version | Notes |
| --- | --- | --- |
| Operating system | Windows 10 / 11 | Linux/macOS work too; replace PowerShell commands with shell equivalents. The Tesseract path detection in [poc/pipeline/ocr.py](poc/pipeline/ocr.py) is Windows-specific and falls back to PATH lookup on other OSes. |
| Python | 3.10 – 3.13 | Tested against 3.13.9 |
| Disk space | ~2 GB | Mostly `torch` CPU build (~600 MB) + the HuggingFace model cache (~80 MB) + a Python venv |
| Anthropic API key | required | Get from <https://console.anthropic.com/> — the BOM-assembly stage and the demo-override prompt both need it. |
| Tesseract OCR binary | v5.x | Installed separately as a Windows binary, not via pip |

You do **not** need a GPU. Everything runs CPU-only.

---

## 3. Get the code

If you already have the project folder at `c:\Users\Dell\Desktop\Project\Floor_Plan\`, skip this step.

```powershell
cd c:\Users\Dell\Desktop\Project
git clone <repo-url> Floor_Plan
cd Floor_Plan
```

Verify the tree:

```powershell
ls
# expect: README.md, SETUP.md, implementation_plan.md, .gitignore, samples\, poc\
```

---

## 4. Install Tesseract OCR (Windows binary)

`pytesseract` is just a Python wrapper. The real Tesseract engine must be installed as a system binary.

1. Download `tesseract-ocr-w64-setup-v5.x.x.exe` from <https://github.com/UB-Mannheim/tesseract/wiki>.
2. Run the installer. Default install path is `C:\Program Files\Tesseract-OCR\`.
3. (Optional) During install, tick *"Add to system PATH"*. The POC's [poc/pipeline/ocr.py](poc/pipeline/ocr.py) auto-locates the binary at the default path even if it's not on PATH, but adding it makes other tools find it too.

**Verify:**

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
# Should print: tesseract v5.5.0...  leptonica-...  libgif... etc.
```

If the binary is at a non-standard location, set the `TESSERACT_CMD` environment variable to the full path before launching anything — `pipeline/ocr.py` respects it.

---

## 5. Create the Python virtualenv

```powershell
cd c:\Users\Dell\Desktop\Project\Floor_Plan\poc
python -m venv venv
.\venv\Scripts\Activate.ps1
python --version
# Should print: Python 3.x.x  (≥ 3.10)
```

If `Activate.ps1` is blocked by execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# then retry .\venv\Scripts\Activate.ps1
```

---

## 6. Install Python dependencies

The full set (~700 MB download, mostly `torch`):

```powershell
pip install -r requirements.txt
```

This installs:

| Package | Used by | Approx size |
| --- | --- | --- |
| `streamlit` | UI (Phase 5) | small |
| `anthropic` | Claude SDK (Phase 4) | small |
| `python-dotenv` | `.env` loader | tiny |
| `pydantic` | Schemas across all phases | small |
| `pypdfium2` | PDF rendering (Phases 1, 2, 5) | ~20 MB |
| `pillow` | Image handling | ~10 MB |
| `numpy` | CV array ops | ~20 MB |
| `opencv-python` | Template matching (Phase 2) | ~50 MB |
| `pytesseract` | Tesseract wrapper (Phase 3) | tiny — needs the binary from §4 |
| `sentence-transformers` | Embeddings (Phase 3) — pulls `torch` | ~600 MB |
| `faiss-cpu` | Vector index (Phase 3) | ~30 MB |

**Verify:**

```powershell
python -c "import streamlit, anthropic, pypdfium2, cv2, pytesseract, sentence_transformers, faiss; print('all deps OK')"
# Should print: all deps OK
```

---

## 7. Set the Anthropic API key

```powershell
copy .env.example .env
notepad .env
```

Edit `.env` to look like:

```ini
ANTHROPIC_API_KEY=sk-ant-api03-...your-real-key-here...
CLAUDE_MODEL=claude-sonnet-4-6
```

Save and close. The app and all smoke tests read this file via `python-dotenv`.

**Verify:**

```powershell
python -c "from dotenv import load_dotenv; load_dotenv('.env'); import os; print('key present:', bool(os.environ.get('ANTHROPIC_API_KEY','').strip()))"
# Should print: key present: True
```

If you ever see `Could not resolve authentication method`, the key is missing or has a trailing space.

---

## 8. Generate the synthetic test fixture

This only needs to happen once per machine (or after editing `library.json`).

```powershell
python symbol_library\generate_symbols.py
# Writes 15 symbol PNGs to symbol_library\
```

```powershell
$env:PYTHONPATH = (Get-Location).Path
python synth\generate_sample.py
# Writes samples\synthetic_layout.pdf and samples\synthetic_layout_ground_truth.json
```

**Verify:**

```powershell
ls ..\samples\synthetic_layout*.*
# expect: synthetic_layout.pdf  synthetic_layout_ground_truth.json
ls symbol_library\*.png | Measure-Object | Select-Object Count
# expect: Count = 15
```

---

## 9. Capture the cached run (Streamlit demo fallback)

The Streamlit app boots in *cached* mode by default — instant load, no Claude call needed. Capture one clean run so that cache exists:

```powershell
python scripts\capture_cached_run.py
```

First run takes ~150 seconds because the `sentence-transformers/all-MiniLM-L6-v2` embedding model downloads (~80 MB) and FAISS embeds the library. Subsequent runs are much faster because of the on-disk cache at `poc\.rag_cache\`.

**Verify:**

```powershell
ls cached_runs\synthetic\
# expect: page.png, run.json, source.pdf
```

The cached run feeds the Streamlit app on first launch. Re-capture it any time you change the pipeline (`scripts\capture_cached_run.py` again).

---

## 10. Optional: run each pipeline stage as a CLI smoke test

These are useful for debugging and for sanity-checking before a demo.

```powershell
# CV symbol detection + recall vs ground truth
python scripts\cv_smoke_test.py

# OCR span dump + bbox overlay
python scripts\ocr_smoke_test.py

# RAG queries (exact / alias / free-text) with top-3 scores
python scripts\rag_smoke_test.py

# Full pipeline: PDF -> CV -> OCR -> RAG -> Claude -> BOM
python scripts\bom_smoke_test.py

# Same, but with the deliberate-deviation prompt so the override mechanism triggers
python scripts\bom_smoke_test.py --force-deviation
```

Each test prints results to the console and saves an artefact to `..\samples\synthetic_layout_*.png` or `..\samples\synthetic_layout_bom.json`.

Expected from `cv_smoke_test.py`: ~90% recall, 13/15 classes at 100% counts.

Expected from `bom_smoke_test.py --force-deviation`:
```
Validation findings (deterministic source-of-truth enforced):
  qty overrides:    [{'library_key': 'ceiling_fan', 'claude': 5, 'truth': 3}]
  cost overrides:   [{'library_key': 'ceiling_light', 'claude': 52.0, 'truth': 42.0}]
  spurious dropped: ['cable_run']
  missing added:    ['wall_light', 'wp_gpo']
```
…with the final subtotal *unchanged* from the clean run.

---

## 11. Optional: ingest a custom symbol library (Phase 6)

The POC supports loading user-defined symbols from a legend PDF page. There are two paths:

**CLI:**
```powershell
python scripts\ingest_legend.py --pdf ..\samples\synthetic_procalc_style_electrical_set.pdf --page 4 --dry-run
# Preview only — prints the 15 extracted rows but writes nothing.

python scripts\ingest_legend.py --pdf ..\samples\synthetic_procalc_style_electrical_set.pdf --page 4
# Real ingest — writes 15 PNGs to symbol_library\user\ and merges into user_additions.json.

python scripts\ingest_legend.py --reset
# Removes all user-ingested symbols.
```

**Streamlit:** open the app, go to **tab 6 "Symbol Library"**, upload the PDF, pick the legend page, click *Extract legend (preview)*, optionally fix OCR fuzz in the per-row editor, click *Ingest all rows*. The sidebar pack auto-flips to `user`.

User additions are stored in:
- `poc/symbol_library/user/*.png` — one PNG per ingested symbol
- `poc/symbol_library/user_additions.json` — metadata merged at library-load time

Both paths are in [.gitignore](.gitignore) — they may contain client-sensitive material.

---

## 12. Launch the Streamlit app

```powershell
streamlit run app.py
```

Browser opens at `http://localhost:8501` automatically. First load shows the cached run on the BOM tab; the five pipeline panels are populated instantly.

Stop the app with `Ctrl+C` in the terminal.

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Could not resolve authentication method. Expected one of api_key, auth_token, or credentials...` | `ANTHROPIC_API_KEY` empty or missing | Re-do §7. Make sure there's no whitespace after the `=`. Restart Streamlit. |
| `TesseractNotInstalledError` | Tesseract binary not installed or not findable | Re-do §4. If you installed to a non-standard path, set `$env:TESSERACT_CMD = "C:\path\to\tesseract.exe"` before launching. |
| `ModuleNotFoundError: No module named 'pipeline'` when running a script | Working directory wrong, or PYTHONPATH not set | `cd` into `poc\` first. If running `synth\generate_sample.py` directly, prepend `$env:PYTHONPATH = (Get-Location).Path; `. Scripts in `scripts\` add `poc/` to `sys.path` automatically. |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell execution policy too strict | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` then retry. |
| First `streamlit run` hangs on "Loading sentence-transformers embedding model" | ~80 MB model downloading from HuggingFace | One-time; let it finish. Future runs read from `~/.cache/huggingface/`. If the network is dead, the RAG layer falls back to deterministic string-similarity automatically. |
| `pip install` of `torch` is slow | CPU build is ~600 MB | Expected. Be patient on first install. |
| Streamlit warning about `huggingface_hub` symlinks on Windows | Cosmetic | Either enable Windows Developer Mode, run Python as administrator, or ignore (the warning is harmless — caching still works). |
| `streamlit run app.py` says ANTHROPIC_API_KEY not set, but you set it | App was launched before the `.env` was created | Restart Streamlit. `load_dotenv()` only runs once at startup. |
| CV detection takes ~20 s per page on the synthetic fixture | 300 template-match operations per page (15 classes × 5 scales × 4 rotations) | Expected. The Streamlit progress bar shows each stage. |
| Detection on a user-uploaded PDF finds almost nothing | Either no user pack ingested yet (so built-in 15 symbols are mismatched with the user's drawing style) OR the page is rendered too small (default 2000 px); the runner auto-bumps to 4500 px when user pack is active | Ingest the legend page first (§11), or pre-render at higher DPI. |
| Streamlit app loads but the BOM tab is empty | No cached run captured yet | Run §9 (`python scripts\capture_cached_run.py`). |

---

## 14. What got installed where

For reference / cleanup:

```
c:\Users\Dell\Desktop\Project\Floor_Plan\
├── poc\
│   ├── venv\                              # Python virtualenv (~1.5 GB on disk)
│   ├── .env                               # YOUR API KEY (gitignored, never commit)
│   ├── .rag_cache\                        # FAISS embeddings cache (gitignored)
│   ├── audit_store\                       # per-Claude-call audit trail (gitignored)
│   ├── cached_runs\synthetic\             # Streamlit demo fallback
│   └── symbol_library\
│       ├── *.png                          # 15 built-in symbol templates
│       ├── library.json                   # built-in metadata
│       ├── user\                          # ingested user symbols (gitignored)
│       └── user_additions.json            # ingested metadata (gitignored)
└── samples\
    ├── synthetic_layout.pdf               # generated test fixture
    ├── synthetic_layout_ground_truth.json # placement truth
    ├── synthetic_layout_*.png             # smoke-test outputs
    └── synthetic_procalc_style_electrical_set*.pdf   # Phase-6 demo input (user-supplied)

C:\Program Files\Tesseract-OCR\            # system-wide Tesseract binary (Windows)
~\.cache\huggingface\                      # sentence-transformers model cache
```

To fully tear down: delete the project folder, delete `~/.cache/huggingface/`, and uninstall Tesseract from Add/Remove Programs.

---

## 15. Next steps after setup

- [poc/README.md §4](poc/README.md) — the ~5-minute interview demo script with what to click in each tab and what to say.
- [implementation_plan.md](implementation_plan.md) — architecture, schemas, design decisions, and the post-audit + Phase-6 backlogs.
- `poc\app.py` — the Streamlit entry point if you want to read the wiring.
