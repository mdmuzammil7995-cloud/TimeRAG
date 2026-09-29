# TimeRAG

LLM-enhanced time-series forecasting with Retrieval-Augmented Generation (RAG).
Upload any dataset, get a forecast, and see a plain-English explanation of
*why* the forecast looks the way it does - grounded in similar patterns
from the series' own history.

## How it works

1. **Train** (`train.py`) fits an XGBoost forecasting model on your dataset
   and builds a RAG knowledge base (a FAISS index over descriptions of past
   30-day windows).
2. **Run the app** (`app.py`, a multi-page Streamlit dashboard) loads the
   trained model, lets you upload a dataset (any date/numeric column names -
   they're detected automatically, or picked manually), forecasts forward,
   retrieves the most similar historical windows, and asks an LLM to
   explain the forecast using that retrieved context.

## Project structure

```
TimeRAG/
├── app.py                 # Multi-page Streamlit dashboard
├── train.py                # Training script (CLI, also runnable from inside the app)
├── pipeline.py              # Shared cleaning / feature engineering / forecasting logic
├── rag_utils.py              # RAG corpus building + FAISS retrieval
├── llm_utils.py               # LLM explanation calls (Gemini / OpenAI)
├── requirements.txt
├── .env.example              # Copy to .env and fill in your own key(s)
├── .gitignore
├── data/raw/                 # Put dataset CSVs here
├── models_saved/              # train.py writes model_xgb_v1.pkl + metrics.json here
└── rag/vector_store/           # train.py writes the FAISS index here
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Edit `.env` - Gemini is the default provider and has a free tier (get a key
at https://aistudio.google.com/apikey, no card required):
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

## 2. Train

```bash
python train.py --data data/raw/your_dataset.csv
```

Any CSV with a date-like column and a numeric column works - they don't
need to be named `date`/`value` specifically when using the app's upload
flow (see below), but `train.py`'s CLI does expect literal `date`/`value`
column names, since the app normalizes any uploaded file to that shape
before pointing `train.py` at it.

This will:
- Clean the data, engineer calendar + lag + rolling features
- Fit a linear trend and train XGBoost on the detrended residual (so the
  model doesn't have to extrapolate past values it's never seen on a
  trending series)
- Evaluate on a holdout split and save `models_saved/model_xgb_v1.pkl` +
  `models_saved/metrics.json`
- Build the RAG index at `rag/vector_store/`

## 3. Run the app

```bash
streamlit run app.py
```

Pages: **Overview**, **Dataset**, **Forecast**, **Historical Patterns**,
**AI Explanation**, **Model Performance**, **Reports**.

On the **Dataset** page, upload any CSV. A date/time column and a numeric
target column are guessed automatically, but you can always override either
pick - including choosing "use row order" when there's no real date column
at all, so even a dataset with no time dimension can still be forecast
against its row sequence.

If no LLM API key is configured, or the LLM call fails for any reason, the
app still shows the full forecast - it just skips the explanation and shows
a note instead. A failed explanation never blocks the core forecast.

## Security

- API keys are only ever read from environment variables via
  `python-dotenv` - never hardcoded, printed, or logged.
- `.env` is git-ignored; only `.env.example` (placeholders) is tracked.
- When deploying, set keys in your hosting platform's secret manager, not
  in a committed file.

## Notes on the RAG implementation

Historical windows are embedded with scikit-learn's TF-IDF vectorizer (on
short auto-generated text summaries of each window) and indexed with FAISS
for similarity search - a deliberately simple, dependency-light stand-in
for a semantic embedding model.

## Notes on the LLM implementation

Gemini is called via Google's OpenAI-compatible endpoint (using the
standard `openai` client pointed at Google's base URL), rather than the
native `google-genai` SDK - this sidesteps a known issue where some newly
issued Gemini API keys are rejected by the native SDK/REST endpoint.

## Troubleshooting

- **"Model not trained"** in the app → use the "Train now" widget on the
  Forecast page, or run `train.py` from a terminal first.
- **`ModuleNotFoundError`** → make sure your virtual environment is active
  and `pip install -r requirements.txt` completed without errors.
- **`AttributeError: 'dict' object has no attribute 'predict'`** → this
  means `pipeline.py` and `app.py` are out of sync (an old `pipeline.py`
  expecting a bare model, paired with a newer `app.py` passing a model
  bundle dict). Replace both files together from the same source.
- **Explanation always unavailable** → check `.env` exists, `LLM_PROVIDER`
  matches the key you filled in, and the key is valid.

## Possible extensions

- Add a second model (LSTM/Transformer) alongside XGBoost.
- Split into a separate FastAPI backend + Streamlit frontend.
- Swap ChromaDB in for FAISS if you need persistent metadata filtering.
