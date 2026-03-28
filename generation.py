"""
Prompt construction and SQL generation with the quantized LLM.
"""

import re
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from config import PipelineConfig

logger = logging.getLogger(__name__)

# SQL keywords that indicate a multi-table / complex query
# NOTE: we match "SELECT\b" via regex below rather than the string "SELECT "
# (trailing-space version) — the space was load-bearing but invisible, causing
# every SELECT query to match and skip the single-table simplification branch.
_COMPLEX_KEYWORDS = frozenset(["JOIN", "INTERSECT", "EXCEPT", "UNION"])
_HAS_SUBQUERY = re.compile(r"\bSELECT\b.*\bSELECT\b", re.IGNORECASE | re.DOTALL)

# Regex patterns compiled once
_NULLS_PATTERN = re.compile(r"\s+NULLS\s+(LAST|FIRST)", re.IGNORECASE)
_ILIKE_PATTERN = re.compile(r"\bILIKE\b", re.IGNORECASE)
_CAST_PATTERN = re.compile(r"::text\b", re.IGNORECASE)
_ALIAS_INLINE = re.compile(r"(?i)\s+AS\s+[a-zA-Z0-9_]+(?=[\s,]|FROM|$)")
_TABLE_PREFIX = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\.(?=[a-zA-Z_])")
_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    cfg: PipelineConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the quantized causal LM and its tokenizer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_use_double_quant=cfg.bnb_double_quant,
        bnb_4bit_quant_type=cfg.bnb_quant_type,
        bnb_4bit_compute_dtype=cfg.bnb_compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.empty_cache()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(
    question: str,
    schema_context: str,
    extracted_values: dict,
    few_shot_block: str = "",
) -> str:
    """
    Construct the prompt that instructs the LLM to generate SQL.

    Args:
        question         : natural language question
        schema_context   : CREATE TABLE DDL string from GraphRAG
        extracted_values : dict with keys "strings" and "numbers"
        few_shot_block   : pre-formatted few-shot examples string (optional).
                           Pass the output of few_shot.format_few_shot_block().
                           If empty, the prompt is zero-shot.
    """
    value_hints = _format_value_hints(extracted_values)

    return (
        "### Task\n"
        "Generate a valid SQLite query to answer the following question.\n"
        "### Database Schema\n"
        f"{schema_context}\n"
        f"{value_hints}"
        f"{few_shot_block}"
        "### Question\n"
        f"{question}\n"
        "### Instructions\n"
        "1. Only use the tables and columns provided in the schema above. DO NOT make up names.\n"
        "2. JOIN ALIASING (CRITICAL): If you use JOINs, you MUST explicitly declare table aliases "
        "in the FROM clause using `AS T1`, `AS T2`, etc.\n"
        "3. SINGLE TABLE RULES: If querying a single table (NO JOIN), DO NOT use any table aliases.\n"
        "4. ALWAYS use `COUNT(*)` instead of `COUNT(column)`.\n"
        "5. SET OPERATIONS: Use `INTERSECT` to find items matching two separate criteria. "
        "Use `EXCEPT` for exclusionary conditions.\n"
        "6. Output ONLY the raw SQL query. Do not add markdown or explanations.\n"
        "### Answer\n"
        "```sql\n"
    )


def _format_value_hints(extracted_values: dict) -> str:
    strings = extracted_values.get("strings", [])
    numbers = extracted_values.get("numbers", [])
    if not strings and not numbers:
        return ""

    lines = ["-- Important values\n"]
    lines += [f"-- Use string value '{s}'\n" for s in strings]
    lines += [f"-- Use numeric value {n}\n" for n in numbers]
    return "".join(lines)


# ---------------------------------------------------------------------------
# SQL extraction & cleaning
# ---------------------------------------------------------------------------

def _clean_sql(sql: str) -> str:
    """
    Normalise the raw LLM output into a clean SQL string.

    Steps:
    1. Strip markdown fences and backticks.
    2. Extract the SELECT statement.
    3. Remove dialect-specific syntax (NULLS LAST, ILIKE, ::text casts).
    4. For simple single-table queries, strip table aliases and prefixes.
    5. Collapse whitespace.
    """
    sql = sql.replace("```sql", "").replace("```", "").replace("`", "").strip()

    # Extract the first SELECT … statement
    match = re.search(r"(SELECT\s+.*?)(?:;|$)", sql, re.DOTALL | re.IGNORECASE)
    sql = match.group(1).strip() if match else sql.strip()

    if not sql:
        return "SELECT 1"

    # Remove non-SQLite constructs
    sql = _NULLS_PATTERN.sub("", sql)
    sql = _ILIKE_PATTERN.sub("LIKE", sql)
    sql = _CAST_PATTERN.sub("", sql)

    # For single-table queries, strip aliases and table-name prefixes
    # A query is "simple" (single-table, no set ops, no subquery) when it has
    # none of the complex keywords AND does not contain a second SELECT.
    is_simple = (
        not any(kw in sql.upper() for kw in _COMPLEX_KEYWORDS)
        and not _HAS_SUBQUERY.search(sql)
    )
    if is_simple:
        sql = _ALIAS_INLINE.sub("", sql)
        parts = re.split(r"\bFROM\b", sql, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            select_part, from_part = parts[0], parts[1].strip()
            select_part = _TABLE_PREFIX.sub("", select_part)

            table_match = re.match(r"^([a-zA-Z0-9_]+)", from_part)
            if table_match:
                table_name = table_match.group(1)
                kw_match = re.search(
                    r"(?i)\b(WHERE|GROUP|ORDER|HAVING|LIMIT)\b.*", from_part, re.DOTALL
                )
                from_part = f"{table_name} {kw_match.group(0)}" if kw_match else table_name

            sql = f"{select_part} FROM {from_part}"

    return _WHITESPACE.sub(" ", sql).strip()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_sql(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cfg: PipelineConfig,
) -> str:
    """Run a single greedy-decode pass and return the cleaned SQL."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return _clean_sql(generated_text)
