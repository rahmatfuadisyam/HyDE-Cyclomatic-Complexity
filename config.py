import os

# Muat .env jika ada (tanpa dependensi eksternal)
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env):
    for _l in open(_env):
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# === REPOSITORI TARGET (Python, kecil, populer) ===
REPOS = [
    "https://github.com/psf/requests.git",
    "https://github.com/pallets/flask.git",
    "https://github.com/tiangolo/fastapi.git",
]

# === PATH ===
DATA_DIR = "data"
REPO_DIR = os.path.join(DATA_DIR, "repos")
METRICS_FILE = os.path.join(DATA_DIR, "metrics.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

# === EMBEDDING MODEL (ringan, lokal) ===
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# === LLM UNTUK HyDE (dibaca dari environment / .env; tidak ada default rahasia) ===
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# === HYDE PROMPT (FEW-SHOT VERSION - FINAL) ===
HYDE_PROMPT_TEMPLATE = """You are a senior developer writing INTENTIONALLY COMPLEX business logic code for testing purposes.

EXAMPLE OF COMPLEX BUSINESS CODE (follow this style, but create different logic):
if customer.is_premium:
    if order.total_amount > 1000:
        if order.items:
            for item in order.items:
                if item.category == 'electronics':
                    if item.price > 500:
                        discount = item.price * 0.2
                        total += item.price - discount
                        applied_discounts.append(('electronics_premium', discount))
                    else:
                        total += item.price
                elif item.category == 'clothing':
                    if item.quantity > 5:
                        discount = item.price * 0.15
                        total += (item.price - discount) * item.quantity
                        applied_discounts.append(('bulk_clothing', discount * item.quantity))
                    else:
                        total += item.price * item.quantity
                else:
                    total += item.price * item.quantity
        else:
            return {{'error': 'No items in order'}}
    else:
        for item in order.items:
            total += item.price * item.quantity
else:
    if order.items:
        for item in order.items:
            if item.is_on_sale:
                total += item.sale_price * item.quantity
            else:
                total += item.price * item.quantity
    else:
        return {{'error': 'Empty order'}}

if total > 0:
    try:
        payment_result = payment_gateway.charge(customer.payment_method, total)
        if payment_result.success:
            if customer.email:
                notification_service.send_email(customer.email, 'Order confirmed')
            if customer.phone:
                notification_service.send_sms(customer.phone, 'Order confirmed')
            return {{'status': 'success', 'total': total, 'discounts': applied_discounts}}
        else:
            return {{'status': 'failed', 'error': payment_result.error}}
    except Exception as e:
        return {{'status': 'error', 'message': str(e)}}
else:
    return {{'error': 'Invalid total'}}


NOW GENERATE YOUR OWN COMPLEX BUSINESS FUNCTION based on this query:
{query}

RULES:
- Create a REALISTIC business function (e-commerce, banking, healthcare, logistics, etc.)
- Include 4+ levels of nested if/else
- Include multiple loops with conditions inside
- Include try/except blocks
- Make it 30-50 lines long
- DO NOT write meta-code (no AST parsing, no reflection, no code that analyzes other code)
- Output ONLY the Python code. No markdown, no explanations.

GENERATED CODE:"""

# === EVALUATION ===
TOP_K = 3

# =====================================================================
# === KONFIGURASI EKSPERIMEN v2 =======================================
# =====================================================================
V2_DIR = os.environ.get("MAHYDE_V2_DIR", os.path.join(DATA_DIR, "v2"))
V2_CORPUS_FILE = os.path.join(V2_DIR, "corpus.json")          # fungsi + metrik
V2_EMB_FILE = os.path.join(V2_DIR, "corpus_emb.npy")          # embedding korpus
V2_HYPO_CACHE = os.path.join(V2_DIR, "hypothetical_cache.json")  # output LLM (cache)
V2_RESULTS_FILE = os.path.join(V2_DIR, "eval_results_v2.json")
V2_STATS_FILE = os.path.join(V2_DIR, "stats_v2.json")
V2_FIG_DIR = os.environ.get("MAHYDE_FIG_DIR", os.path.join("figures", "v2"))

# Hanya kode produksi: buang test, dokumentasi/tutorial, contoh, dan skrip CI
EXCLUDE_PATH_PARTS = ("tests", "test", "testing_utils", "docs", "docs_src",
                      "examples", "scripts", "venv", ".git", "vendor", "node_modules")

K = 10                       # cutoff evaluasi (P@10, nDCG@10, MeanCC@10)
RELEVANT_CC = 11             # relevan bila CC >= 11 (rank radon C ke atas)
FILTER_CC = 5                # ambang metadata filter mode Hybrid (sama dengan v1)
HYDE_RUNS = 3                # jumlah generasi hipotetis independen per query
HYDE_TEMPERATURE = 0.7
RANDOM_SEED = 42
N_BOOTSTRAP = 10000

# Instruksi query resmi BGE untuk retrieval (query pendek -> passage)
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# 20 query niat-struktural (bahasa alami). Q1-Q5 identik dengan v1.
QUERIES_V2 = [
    ("Q01", "Find Python functions with very high cyclomatic complexity"),
    ("Q02", "Show me deeply nested code with many if/else branches"),
    ("Q03", "Find functions with many parameters and complex logic"),
    ("Q04", "Show me code with multiple nested loops"),
    ("Q05", "Find long functions with many conditional branches"),
    ("Q06", "Which functions have the most decision points and execution paths?"),
    ("Q07", "Find hard-to-test functions with a large number of independent paths"),
    ("Q08", "Show functions that are good candidates for refactoring because of complex control flow"),
    ("Q09", "Find code with long chains of elif statements"),
    ("Q10", "Show me functions mixing loops, try/except blocks and conditionals"),
    ("Q11", "Find functions with complicated error handling and many exception branches"),
    ("Q12", "Find the most complex functions in this codebase"),
    ("Q13", "Show me spaghetti code with tangled control flow"),
    ("Q14", "Find functions with high McCabe complexity score"),
    ("Q15", "Show methods with many boolean conditions combined with and/or"),
    ("Q16", "Find functions containing loops with nested conditionals inside"),
    ("Q17", "Show code that handles many different cases in a single function"),
    ("Q18", "Find large functions that do too many things"),
    ("Q19", "Show functions with deep indentation levels"),
    ("Q20", "Find technical debt hotspots with high branching complexity"),
]

# Prompt HyDE zero-shot (setia pada Gao et al., 2023): tanpa domain, tanpa contoh
HYDE_ZS_PROMPT_TEMPLATE = """Write a Python function that would be a relevant search result for the following code-search query.
Query: {query}
Output ONLY the Python code. No markdown, no explanations."""
