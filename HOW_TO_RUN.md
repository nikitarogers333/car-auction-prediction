# How to Use This Project

## 1. What you need

- **Python 3.11 or newer** (check: `python3 --version`)
- **Terminal** (macOS Terminal, Windows Command Prompt or PowerShell, or VS Code terminal)

Optional (for real OpenAI predictions):

- `pip install openai`
- An OpenAI API key in the environment: `export OPENAI_API_KEY=your-key`

---

## 2. Open the project

In the terminal, go to the project folder:

```bash
cd "/Users/nikitarogers/Anton Project"
```

(Or wherever you saved the project; use your actual path.)

---

## 3. Run without an API key (mock mode)

Everything works **without** an OpenAI key. The pipeline uses a deterministic mock predictor.

**Single prediction (one car):**

```bash
python3 main.py --mock single
```

**Same run but also write an audit log:**

```bash
python3 main.py --mock --log single
```

**Variance check (run 5 times, see mean/std/CV):**

```bash
python3 main.py --mock --repeats 5 consistency
```

**Compare all four conditions (P1–P4) on a few vehicles:**

```bash
python3 main.py --mock --repeats 2 --limit 4 experiments
```

**Run KNN and linear regression baselines:**

```bash
python3 main.py baselines
```

Important: put **global options** (`--mock`, `--repeats`, `--log`, `--index`, etc.) **before** the command word (`single`, `consistency`, `experiments`, `baselines`).

---

## 4. Run with the real OpenAI API

1. Install the client: `pip install openai`
2. Set your key (replace with your key, and prefer a new key if you ever shared one):

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Run without `--mock`:

   ```bash
   python3 main.py single
   python3 main.py --repeats 3 consistency
   ```

---

## 5. Your data

- **Input:** `data/vehicles.json` — edit this file to add your own cars (keep `make`, `model`, `year`, `mileage`; optional `price` for evaluation).
- **Results:**  
  - `eval/` — experiment summaries and `baselines_summary.json`  
  - `logs/` — audit logs (when you use `--log`)

---

## 6. Quick reference

| Goal                    | Command                                              |
|-------------------------|------------------------------------------------------|
| One prediction (mock)   | `python3 main.py --mock single`                      |
| One prediction (real)   | `python3 main.py single` (with `OPENAI_API_KEY` set) |
| Variance over 5 runs    | `python3 main.py --mock --repeats 5 consistency`     |
| P1–P4 comparison        | `python3 main.py --mock --repeats 2 --limit 4 experiments` |
| Baselines               | `python3 main.py baselines`                          |
| Use 2nd vehicle in file | `python3 main.py --mock --index 1 single`            |

For more detail, see **README.md**.
