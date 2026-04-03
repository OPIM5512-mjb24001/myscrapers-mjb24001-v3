# main.py
# Purpose: PoC LLM extractor that reads your existing per-listing JSONL records,
# fetches the original TXT, asks an LLM (Vertex AI) to extract fields, and writes
# a sibling "<post_id>_llm.jsonl" to the NEW 'jsonl_llm/' sub-directory.
#
# FINAL FIXES INCLUDED:
# 1. Schema updated to use "type": "string" + "nullable": True.
# 2. system_instruction removed from GenerationConfig and merged into prompt.
# 3. LLM_MODEL set to 'gemini-2.5-flash' (Fixes 404/NotFound error).
# 4. "additionalProperties": False removed from schema (Fixes internal ParseError).
# 5. Non-breaking spaces (U+00A0) replaced with standard spaces (U+0020). <--- FIX FOR THIS ERROR

import os
import re
import json
import logging
import traceback
import time
from datetime import datetime, timezone

from flask import Request, jsonify
from google.api_core import retry as gax_retry
from google.cloud import storage

# ---- REQUIRED VERTEX AI IMPORTS ----
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.api_core.exceptions import ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded

# -------------------- ENV --------------------
PROJECT_ID           = os.getenv("PROJECT_ID", "")
REGION               = os.getenv("REGION", "us-central1")
BUCKET_NAME          = os.getenv("GCS_BUCKET", "")
STRUCTURED_PREFIX    = os.getenv("STRUCTURED_PREFIX", "structured")
LLM_PROVIDER         = os.getenv("LLM_PROVIDER", "vertex").lower()
LLM_MODEL            = os.getenv("LLM_MODEL", "gemini-2.5-flash")
OVERWRITE_DEFAULT    = os.getenv("OVERWRITE", "false").lower() == "true"
MAX_FILES_DEFAULT    = int(os.getenv("MAX_FILES", "0") or 0)

# GCS READ RETRY - Use default transient error logic
READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=120.0
)

# LLM API RETRY PREDICATE
def _if_llm_retryable(exception):
    """Checks if the exception is transient and should trigger a retry."""
    return isinstance(exception, (ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded))

# LLM CALL RETRY
LLM_RETRY = gax_retry.Retry(
    predicate=_if_llm_retryable,
    initial=5.0, maximum=30.0, multiplier=2.0, deadline=180.0,
)

storage_client = storage.Client()
_CACHED_MODEL_OBJ = None

# Accept BOTH run id styles:
RUN_ID_ISO_RE    = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")

# Keys from regex JSONL passed to the LLM as hints (regex-first, LLM-second merge).
REGEX_HINT_KEYS = frozenset({
    "price", "year", "make", "model", "mileage",
    "transmission", "color", "city", "state", "zip_code",
    "drive", "fuel", "condition", "title_status", "type",
    "cylinders", "seller_type",
})

OUTPUT_FIELD_KEYS = [
    "price", "year", "make", "model", "mileage",
    "transmission", "color", "city", "state", "zip_code",
    "drive", "fuel", "condition", "title_status", "type",
    "cylinders", "seller_type",
]


# -------------------- HELPERS --------------------
def _get_vertex_model() -> GenerativeModel:
    """Initializes and returns the cached Vertex AI model object (thread-safe for CF)."""
    global _CACHED_MODEL_OBJ
    if _CACHED_MODEL_OBJ is None:
        if not PROJECT_ID:
            raise RuntimeError("PROJECT_ID environment variable is missing.")
        
        # Initialize client once per container lifecycle
        vertexai.init(project=PROJECT_ID, location=REGION)
        _CACHED_MODEL_OBJ = GenerativeModel(LLM_MODEL)
        logging.info(f"Initialized Vertex AI model: {LLM_MODEL} in {REGION}")
    return _CACHED_MODEL_OBJ


def _list_structured_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    """
    List 'structured/run_id=*/' directories and return normalized run_ids.
    """
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:
        pass

    runs = []
    for pref in getattr(it, "prefixes", []):
        tail = pref.rstrip("/").split("/")[-1]
        if tail.startswith("run_id="):
            cand = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(cand) or RUN_ID_PLAIN_RE.match(cand):
                runs.append(cand)
    return sorted(runs)


def _normalize_run_id_iso(run_id: str) -> str:
    """
    Normalize run_id to ISO8601 Z string for provenance.
    """
    try:
        if RUN_ID_ISO_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        elif RUN_ID_PLAIN_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        else:
            raise ValueError("unsupported run_id")
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _list_per_listing_jsonl_for_run(bucket: str, run_id: str) -> list[str]:
    """
    Return *input* per-listing JSONL object names for a given run_id
    (assumes inputs are in 'jsonl/').
    """
    prefix = f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl/"
    bucket_obj = storage_client.bucket(bucket)
    names = []
    for b in bucket_obj.list_blobs(prefix=prefix):
        if not b.name.endswith(".jsonl"):
            continue
        names.append(b.name)
    return names


def _download_text(blob_name: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.download_as_text(retry=READ_RETRY, timeout=120)


def _upload_jsonl_line(blob_name: str, record: dict):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    blob.upload_from_string(line, content_type="application/x-ndjson")


def _blob_exists(blob_name: str) -> bool:
    bucket = storage_client.bucket(BUCKET_NAME)
    return bucket.blob(blob_name).exists()


def _safe_int(x):
    try:
        if x is None or x == "":
            return None
        return int(str(x).replace(",", "").strip())
    except Exception:
        return None


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _optional_str(x) -> str | None:
    """Empty string -> None; strip whitespace."""
    if x is None:
        return None
    s = _collapse_ws(str(x).replace("\u00a0", " "))
    return s if s else None


def _norm_make(s: str | None) -> str | None:
    if not s:
        return None
    return s.title()


def _norm_model(s: str | None) -> str | None:
    """Strip and collapse whitespace; do not over-normalize spelling."""
    return _optional_str(s)


def _normalize_transmission(raw: str | None) -> str | None:
    s = _optional_str(raw)
    if not s:
        return None
    low = s.lower()
    if "cvt" in low:
        return "cvt"
    if "automatic transmission" in low or low in ("auto", "automatic", "at") or low.startswith("auto "):
        return "automatic"
    if "manual" in low or "stick" in low or re.search(r"\b[45678]-speed\b", low):
        return "manual"
    allowed = {"automatic", "manual", "cvt"}
    if low in allowed:
        return low
    return None


def _normalize_fuel(raw: str | None) -> str | None:
    s = _optional_str(raw)
    if not s:
        return None
    low = s.lower().replace("-", " ")
    low = re.sub(r"\s+", " ", low)
    if ("plug" in low and "hybrid" in low) or "phev" in low or "plug in hybrid" in low:
        return "plug-in hybrid"
    if "hybrid electric" in low or (low == "hybrid"):
        return "hybrid"
    if low in ("ev", "electric", "bev") or "electric vehicle" in low:
        return "electric"
    if low in ("gasoline", "gas", "petrol"):
        return "gas"
    if "diesel" in low:
        return "diesel"
    if ("flex" in low and "fuel" in low) or "e85" in low:
        return "flex fuel"
    allowed = {"gas", "diesel", "hybrid", "electric", "plug-in hybrid", "flex fuel"}
    if low in allowed:
        return low
    return None


def _normalize_type(raw: str | None) -> str | None:
    """Vehicle body / listing type (rubric: type)."""
    s = _optional_str(raw)
    if not s:
        return None
    low = s.lower()
    if "sport utility" in low or "crossover" in low or low in ("suv", "cuv"):
        return "suv"
    if "pickup" in low or low in ("pick up", "pick-up"):
        return "truck"
    allowed = {
        "sedan", "suv", "truck", "coupe", "hatchback", "wagon", "van",
        "minivan", "convertible",
    }
    if low in allowed:
        return low
    return None


def _normalize_drive(raw: str | None) -> str | None:
    s = _optional_str(raw)
    if not s:
        return None
    low = re.sub(r"[\s\-]+", "", s.lower())
    if "4wd" in low or "4x4" in low:
        return "4wd"
    if "awd" in low:
        return "awd"
    if "fwd" in low or "frontwheel" in low:
        return "fwd"
    if "rwd" in low or "rearwheel" in low:
        return "rwd"
    allowed = {"4wd", "awd", "fwd", "rwd"}
    stripped = s.lower().strip()
    if stripped in allowed:
        return stripped
    return None


def _normalize_state(raw: str | None) -> str | None:
    s = _optional_str(raw)
    if not s:
        return None
    s2 = re.sub(r"[^A-Za-z]", "", s).upper()
    if len(s2) == 2:
        return s2
    return None


def _normalize_cylinders(raw) -> int | None:
    n = _safe_int(raw)
    if n is None:
        return None
    if 0 < n <= 16:
        return n
    return None


def _normalize_seller_type(raw: str | None) -> str | None:
    s = _optional_str(raw)
    if not s:
        return None
    low = s.lower()
    if "dealer" in low:
        return "dealer"
    if "private" in low or "owner" in low:
        return "private"
    allowed = {"dealer", "private"}
    if low in allowed:
        return low
    return None


def _validate_zip_in_text(zip_candidate: str | None, raw_text: str) -> str | None:
    """Reject ZIP unless the exact 5-digit token appears in listing text (anti-hallucination)."""
    if not zip_candidate:
        return None
    z = re.sub(r"\D", "", str(zip_candidate).strip())
    if len(z) != 5:
        return None
    if re.search(rf"\b{z}\b", raw_text or ""):
        return z
    return None


def _finalize_zip_for_submission(zip_code: str | None, state: str | None) -> str | None:
    """
    ZIP as a 5-digit string only. If state is CT, expect 06xxx (USPS); else drop inconsistent ZIP.
    """
    if not zip_code:
        return None
    z = re.sub(r"\D", "", str(zip_code).strip())
    if len(z) != 5 or not z.isdigit():
        return None
    if state == "CT" and not z.startswith("06"):
        return None
    return z


def _normalize_title_status(raw: str | None) -> str | None:
    s = _optional_str(raw)
    if not s:
        return None
    low = s.lower()
    if "clean title" in low or low == "clean":
        return "clean"
    if "rebuilt title" in low or low == "rebuilt":
        return "rebuilt"
    if "salvage title" in low or low == "salvage":
        return "salvage"
    if "lien" in low:
        return "lien"
    if "parts only" in low or "parts-only" in low:
        return "parts only"
    if "missing" in low:
        return "missing"
    allowed = {"clean", "rebuilt", "salvage", "lien", "parts only", "missing"}
    if low in allowed:
        return low
    return None


def _normalize_condition(raw: str | None) -> str | None:
    s = _optional_str(raw)
    if not s:
        return None
    low = s.lower()
    if "very clean" in low or "excellent condition" in low:
        return "excellent"
    if "runs good" in low or low == "runs great":
        return "good"
    if "mechanic special" in low or "project car" in low or low == "project":
        return "project"
    if "like new" in low or low.replace("-", " ") == "like new":
        return "like new"
    allowed = {"like new", "excellent", "good", "fair", "poor", "project"}
    if low in allowed:
        return low
    return None


def _normalize_color(raw: str | None) -> str | None:
    """Plain English color if stated; do not invent."""
    s = _optional_str(raw)
    if not s:
        return None
    return s.title()


def _regex_hints_from_record(base_rec: dict) -> dict:
    out = {}
    for k in REGEX_HINT_KEYS:
        if k not in base_rec:
            continue
        v = base_rec[k]
        if v is None or v == "":
            continue
        out[k] = v
    return out


def _postprocess_llm_dict(parsed: dict) -> dict:
    """Normalize LLM output fields."""
    parsed["price"] = _safe_int(parsed.get("price"))
    parsed["year"] = _safe_int(parsed.get("year"))
    parsed["mileage"] = _safe_int(parsed.get("mileage"))
    parsed["cylinders"] = _normalize_cylinders(parsed.get("cylinders"))

    parsed["make"] = _norm_make(_optional_str(parsed.get("make")))
    parsed["model"] = _norm_model(parsed.get("model"))
    parsed["transmission"] = _normalize_transmission(parsed.get("transmission"))
    parsed["fuel"] = _normalize_fuel(parsed.get("fuel"))
    parsed["type"] = _normalize_type(parsed.get("type"))
    parsed["color"] = _normalize_color(parsed.get("color"))
    parsed["title_status"] = _normalize_title_status(parsed.get("title_status"))
    parsed["condition"] = _normalize_condition(parsed.get("condition"))
    parsed["drive"] = _normalize_drive(parsed.get("drive"))
    parsed["city"] = _optional_str(parsed.get("city"))
    if parsed["city"]:
        parsed["city"] = parsed["city"].title()
    parsed["state"] = _normalize_state(parsed.get("state"))
    zraw = parsed.get("zip_code")
    if zraw is None or str(zraw).strip() == "":
        parsed["zip_code"] = None
    else:
        digits = re.sub(r"\D", "", str(zraw))
        if len(digits) >= 9:
            digits = digits[:5]
        parsed["zip_code"] = digits if len(digits) == 5 and digits.isdigit() else None
    parsed["seller_type"] = _normalize_seller_type(parsed.get("seller_type"))
    return parsed


def _merge_llm_and_regex(llm_norm: dict, regex_hints: dict, raw_text: str) -> dict:
    """
    Prefer LLM when it returns a value; fall back to regex hints.
    ZIP is only kept if the digit sequence appears verbatim in the listing text.
    """
    out = {}
    for k in OUTPUT_FIELD_KEYS:
        lv = llm_norm.get(k)
        rv = regex_hints.get(k)

        if k == "zip_code":
            z_llm = _validate_zip_in_text(_optional_str(lv), raw_text)
            z_re = _validate_zip_in_text(_optional_str(rv), raw_text)
            out[k] = z_llm or z_re or None
            continue

        if k in ("price", "year", "mileage", "cylinders"):
            if lv is not None:
                out[k] = lv
            else:
                out[k] = rv if rv is not None else None
            continue

        # Strings / categoricals
        if lv is not None and lv != "":
            out[k] = lv
        else:
            out[k] = rv if rv not in (None, "") else None

    out["zip_code"] = _finalize_zip_for_submission(out.get("zip_code"), out.get("state"))
    return out


# -------------------- VERTEX AI CALL --------------------
def _vertex_extract_fields(raw_text: str, regex_hints: dict) -> dict:
    """
    Ask Gemini for strict JSON; regex_hints are advisory (regex-first pipeline).
    """
    model = _get_vertex_model()

    raw_text = (raw_text or "").replace("\u00a0", " ")
    hints_json = json.dumps(regex_hints, ensure_ascii=False, default=str)

    schema = {
        "type": "object",
        "properties": {
            "price": {"type": "integer", "nullable": True},
            "year": {"type": "integer", "nullable": True},
            "make": {"type": "string", "nullable": True},
            "model": {"type": "string", "nullable": True},
            "mileage": {"type": "integer", "nullable": True},
            "transmission": {"type": "string", "nullable": True},
            "color": {"type": "string", "nullable": True},
            "city": {"type": "string", "nullable": True},
            "state": {"type": "string", "nullable": True},
            "zip_code": {"type": "string", "nullable": True},
            "drive": {"type": "string", "nullable": True},
            "fuel": {"type": "string", "nullable": True},
            "condition": {"type": "string", "nullable": True},
            "title_status": {"type": "string", "nullable": True},
            "type": {"type": "string", "nullable": True},
            "cylinders": {"type": "integer", "nullable": True},
            "seller_type": {"type": "string", "nullable": True},
        },
        "required": [
            "price", "year", "make", "model", "mileage",
            "transmission", "color", "city", "state", "zip_code",
            "drive", "fuel", "condition", "title_status", "type",
            "cylinders", "seller_type",
        ],
    }

    sys_instr = (
        "You extract structured vehicle listing fields from the TEXT below.\n"
        "Return STRICT JSON ONLY — one JSON object, no markdown, no commentary.\n"
        "Use EXACTLY these keys: price, year, make, model, mileage, transmission, color, "
        "city, state, zip_code, drive, fuel, condition, title_status, type, cylinders, seller_type.\n"
        "Use null if a value is unclear, ambiguous, or not explicitly stated in the TEXT.\n"
        "Do NOT invent facts. Do NOT guess zip_code — use null unless a 5-digit US ZIP appears "
        "explicitly in the TEXT.\n"
        "Infer city and state ONLY if they clearly appear as location in the listing (e.g. "
        "'Hartford, CT' or 'CT' near the title); otherwise null.\n"
        "REGEX_HINTS_JSON (from a separate regex pass; may be incomplete or wrong): "
        "prefer TEXT over hints when they conflict; use hints only to disambiguate when helpful.\n"
        "Normalize categoricals only when the TEXT clearly supports a label; otherwise null "
        "(do not use a catch-all like 'other').\n"
        "- transmission: automatic | manual | cvt | null\n"
        "- fuel: gas | diesel | hybrid | electric | plug-in hybrid | flex fuel | null\n"
        "- type (body): sedan | suv | truck | coupe | hatchback | wagon | van | minivan | "
        "convertible | null\n"
        "- drive: fwd | rwd | awd | 4wd | null\n"
        "- title_status: clean | rebuilt | salvage | lien | parts only | missing | null\n"
        "- condition: like new | excellent | good | fair | poor | project | null\n"
        "- color: plain English if clearly stated; else null\n"
        "- seller_type: dealer | private | null\n"
        "- state: 2-letter USPS code if explicit; else null\n"
        "Integers: price (USD), year (4-digit), mileage (miles), cylinders (count). null if unknown."
    )

    prompt = f"{sys_instr}\n\nREGEX_HINTS_JSON:\n{hints_json}\n\nTEXT:\n{raw_text}"

    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=40,
        candidate_count=1,
        response_mime_type="application/json",
        response_schema=schema,
    )

    max_attempts = 3
    resp = None
    for attempt in range(max_attempts):
        try:
            resp = model.generate_content(prompt, generation_config=gen_cfg)
            break
        except Exception as e:
            if not _if_llm_retryable(e) or attempt == max_attempts - 1:
                logging.error(f"Fatal/non-retryable LLM error or max retries reached: {e}")
                raise

            sleep_time = LLM_RETRY._calculate_sleep(attempt)
            logging.warning(
                f"Transient LLM error on attempt {attempt+1}/{max_attempts}. Retrying in {sleep_time:.2f}s..."
            )
            time.sleep(sleep_time)

    if resp is None:
        raise RuntimeError("LLM call failed after all retries.")

    parsed = json.loads(resp.text)
    parsed = _postprocess_llm_dict(parsed)
    return _merge_llm_and_regex(parsed, regex_hints, raw_text)


# -------------------- HTTP ENTRY --------------------
def llm_extract_http(request: Request):
    """
    Reads latest (or requested) run's per-listing JSONL inputs and writes LLM outputs.
    """
    logging.getLogger().setLevel(logging.INFO)

    if not BUCKET_NAME:
        return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500
    if not PROJECT_ID:
        return jsonify({"ok": False, "error": "missing PROJECT_ID env"}), 500
    if LLM_PROVIDER != "vertex":
        return jsonify({"ok": False, "error": "PoC supports LLM_PROVIDER='vertex' only"}), 400

    # Body overrides
    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    run_id = body.get("run_id")
    max_files = int(body.get("max_files") or MAX_FILES_DEFAULT or 0)
    overwrite = bool(body.get("overwrite")) if "overwrite" in body else OVERWRITE_DEFAULT

    # Pick newest run if not provided
    if not run_id:
        runs = _list_structured_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        if not runs:
            return jsonify({"ok": False, "error": f"no run_ids found under {STRUCTURED_PREFIX}/"}), 200
        run_id = runs[-1]

    structured_iso = _normalize_run_id_iso(run_id)

    inputs = _list_per_listing_jsonl_for_run(BUCKET_NAME, run_id)
    if not inputs:
        return jsonify({"ok": True, "run_id": run_id, "processed": 0, "written": 0, "skipped": 0, "errors": 0}), 200
    if max_files > 0:
        inputs = inputs[:max_files]

    logging.info(f"Starting LLM extraction for run_id={run_id} ({len(inputs)} files to process)")

    processed = written = skipped = errors = 0

    for in_key in inputs:
        processed += 1
        try:
            # Read the tiny JSON line (single record)
            raw_line = _download_text(in_key).strip()
            if not raw_line:
                raise ValueError("empty input jsonl")
            base_rec = json.loads(raw_line)

            post_id = base_rec.get("post_id")
            if not post_id:
                raise ValueError("missing post_id in input record")

            source_txt_key = base_rec.get("source_txt")
            if not source_txt_key:
                raise ValueError("missing source_txt in input record")

            # Output path: uses 'jsonl_llm/' folder
            out_prefix = in_key.rsplit("/", 2)[0] + "/jsonl_llm"
            out_key = out_prefix + f"/{post_id}_llm.jsonl"

            if not overwrite and _blob_exists(out_key):
                skipped += 1
                continue

            # Fetch the raw listing TXT; send to LLM (regex hints from jsonl row)
            raw_listing = _download_text(source_txt_key)
            hints = _regex_hints_from_record(base_rec)
            merged = _vertex_extract_fields(raw_listing, hints)

            out_record = {
                "post_id": post_id,
                "run_id": base_rec.get("run_id", run_id),
                "scraped_at": base_rec.get("scraped_at", structured_iso),
                "source_txt": source_txt_key,
                **{k: merged.get(k) for k in OUTPUT_FIELD_KEYS},
                "llm_provider": "vertex",
                "llm_model": LLM_MODEL,
                "llm_ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

            _upload_jsonl_line(out_key, out_record)
            written += 1

        except Exception as e:
            errors += 1
            logging.error(f"LLM extraction failed for {in_key}: {e}\n{traceback.format_exc()}")

    result = {
        "ok": True,
        "version": "extractor-llm-v2-hybrid-schema",
        "run_id": run_id,
        "processed": processed,
        "written": written,
        "skipped": skipped,
        "errors": errors,
    }
    logging.info(json.dumps(result))
    return jsonify(result), 200
