# main.py
# Purpose: Convert raw TXT -> one-line JSON records (.jsonl) in GCS.
# Compatible input layouts:
#   gs://<bucket>/<SCRAPES_PREFIX>/<RUN>/*.txt
#   gs://<bucket>/<SCRAPES_PREFIX>/<RUN>/txt/*.txt
# where <RUN> is either 20251026T170002Z or 20251026170002.
# Output:
#   gs://<bucket>/<STRUCTURED_PREFIX>/run_id=<RUN>/jsonl/<post_id>.jsonl

import os
import re
import json
import logging
import traceback
from datetime import datetime, timezone

from flask import Request, jsonify
from google.api_core import retry as gax_retry
from google.cloud import storage

# -------------------- ENV --------------------
PROJECT_ID         = os.getenv("PROJECT_ID")
BUCKET_NAME        = os.getenv("GCS_BUCKET")                        # REQUIRED
SCRAPES_PREFIX     = os.getenv("SCRAPES_PREFIX", "scrapes")         # input
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured")   # output

# Accept BOTH run id styles:
RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")  # 20251026T170002Z
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")        # 20251026170002

READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=120.0
)

storage_client = storage.Client()

# -------------------- SIMPLE REGEX EXTRACTORS --------------------
PRICE_RE      = re.compile(r"\$\s?([0-9,]+)")
YEAR_RE       = re.compile(r"\b(19|20)\d{2}\b")
MAKE_MODEL_RE = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][A-Za-z0-9]+)")

# -------------------- HELPERS --------------------
def _list_run_ids(bucket: str, scrapes_prefix: str) -> list[str]:
    """
    List run folders under gs://bucket/<scrapes_prefix>/ and return normalized run_ids.
    Accept:
      - <scrapes_prefix>/run_id=20251026T170002Z/
      - <scrapes_prefix>/20251026170002/
    """
    it = storage_client.list_blobs(bucket, prefix=f"{scrapes_prefix}/", delimiter="/")
    for _ in it:
        pass  # populate it.prefixes

    run_ids: list[str] = []
    for pref in getattr(it, "prefixes", []):
        # e.g., 'scrapes/run_id=20251026T170002Z/' OR 'scrapes/20251026170002/'
        tail = pref.rstrip("/").split("/")[-1]
        cand = tail.split("run_id=", 1)[1] if tail.startswith("run_id=") else tail
        if RUN_ID_ISO_RE.match(cand) or RUN_ID_PLAIN_RE.match(cand):
            run_ids.append(cand)
    return sorted(run_ids)

def _txt_objects_for_run(run_id: str) -> list[str]:
    """
    Return .txt object names for a given run_id.
    Tries (in order) and returns the first non-empty list:
      scrapes/run_id=<run_id>/txt/
      scrapes/run_id=<run_id>/
      scrapes/<run_id>/txt/
      scrapes/<run_id>/
    """
    bucket = storage_client.bucket(BUCKET_NAME)
    candidates = [
        f"{SCRAPES_PREFIX}/run_id={run_id}/txt/",
        f"{SCRAPES_PREFIX}/run_id={run_id}/",
        f"{SCRAPES_PREFIX}/{run_id}/txt/",
        f"{SCRAPES_PREFIX}/{run_id}/",
    ]
    for pref in candidates:
        names = [b.name for b in bucket.list_blobs(prefix=pref) if b.name.endswith(".txt")]
        if names:
            return names
    return []

def _download_text(blob_name: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.download_as_text(retry=READ_RETRY, timeout=120)

def _upload_jsonl_line(blob_name: str, record: dict):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    blob.upload_from_string(line, content_type="application/x-ndjson")

def _parse_run_id_as_iso(run_id: str) -> str:
    """Normalize either run_id style to ISO8601 Z (fallback = now UTC)."""
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

def _norm_attr_value(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if s else ""


def _parse_attr_line(text: str, label: str) -> str | None:
    """Craigslist-style 'label: value' on same line (case-insensitive)."""
    pat = re.compile(rf"(?im)^\s*{re.escape(label)}\s*[:\-]\s*([^\n]+?)\s*$")
    m = pat.search(text)
    if not m:
        return None
    v = _norm_attr_value(m.group(1))
    return v if v else None


def _normalize_transmission_regex(raw: str | None) -> str | None:
    if not raw:
        return None
    low = raw.lower()
    if "cvt" in low:
        return "cvt"
    if "automatic" in low or low in ("auto", "at"):
        return "automatic"
    if "manual" in low or "stick" in low:
        return "manual"
    return None


def _normalize_drive_regex(raw: str | None) -> str | None:
    if not raw:
        return None
    low = re.sub(r"[\s\-]+", "", raw.lower())
    if "4wd" in low or "4x4" in low or "fourwheel" in low.replace(" ", ""):
        return "4wd"
    if "awd" in low:
        return "awd"
    if "fwd" in low or "front" in low:
        return "fwd"
    if "rwd" in low or "rear" in low:
        return "rwd"
    return None


def _normalize_fuel_regex(raw: str | None) -> str | None:
    if not raw:
        return None
    low = raw.lower().replace("-", " ")
    if "diesel" in low:
        return "diesel"
    if "electric" in low or low.strip() == "ev":
        return "electric"
    if "hybrid" in low:
        return "hybrid"
    if "flex" in low or "e85" in low:
        return "flex fuel"
    if "gas" in low or "petrol" in low:
        return "gas"
    return None


def _normalize_title_regex(raw: str | None) -> str | None:
    if not raw:
        return None
    low = raw.lower()
    for token in ("clean", "rebuilt", "salvage", "lien", "missing"):
        if token in low:
            return token if token != "missing" else "missing"
    if "parts" in low:
        return "parts only"
    return None


def _normalize_condition_regex(raw: str | None) -> str | None:
    if not raw:
        return None
    low = raw.lower()
    for lab in ("like new", "excellent", "good", "fair", "poor", "project"):
        if lab in low or low == lab.replace(" ", ""):
            return lab
    return None


def _normalize_type_regex(raw: str | None) -> str | None:
    if not raw:
        return None
    low = raw.lower()
    mapping = {
        "sedan": "sedan", "suv": "suv", "truck": "truck", "pickup": "truck",
        "coupe": "coupe", "hatchback": "hatchback", "wagon": "wagon",
        "van": "van", "minivan": "minivan", "convertible": "convertible",
    }
    for k, v in mapping.items():
        if k in low:
            return v
    return None


def _normalize_seller_regex(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\bby\s+dealer\b", t) or re.search(r"\bdealer\b", t) and "private" not in t:
        return "dealer"
    if re.search(r"\bby\s+owner\b", t) or re.search(r"\bprivate\b", t):
        return "private"
    return None


def _extract_zip_regex(text: str) -> str | None:
    """Only explicit 5-digit US ZIP (do not guess from partial digits)."""
    m = re.search(r"\b(\d{5})(?:-\d{4})?\b", text)
    return m.group(1) if m else None


def _finalize_zip_for_submission(zip_code: str | None, state: str | None) -> str | None:
    """
    Keep zip_code as a 5-character digit string only when valid.
    Conservative: if state is CT, USPS ZIPs are 06xxx — null mismatches (bad geography).
    """
    if not zip_code:
        return None
    z = re.sub(r"\D", "", str(zip_code).strip())
    if len(z) != 5 or not z.isdigit():
        return None
    if state == "CT" and not z.startswith("06"):
        return None
    return z


def _extract_state_city_regex(text: str) -> tuple[str | None, str | None]:
    """
    Heuristic: 'City, ST' or 'ST' two-letter state near location lines.
    Conservative: only when pattern is clear.
    """
    city, state = None, None
    m = re.search(
        r"\b([A-Za-z][A-Za-z\s\.\-]{1,40}?)\s*,\s*([A-Z]{2})\b",
        text,
    )
    if m:
        city = _norm_attr_value(m.group(1))
        state = m.group(2).upper()
        if len(city) < 2:
            city = None
    if state is None:
        m2 = re.search(r"\b([A-Z]{2})\s+\d{5}\b", text)
        if m2:
            state = m2.group(1)
    return city, state


# -------------------- PARSE A LISTING --------------------
def parse_listing(text: str) -> dict:
    d = {}

    m = PRICE_RE.search(text)
    if m:
        try:
            d["price"] = int(m.group(1).replace(",", ""))
        except ValueError:
            pass

    y = YEAR_RE.search(text)
    if y:
        try:
            d["year"] = int(y.group(0))
        except ValueError:
            pass

    mm = MAKE_MODEL_RE.search(text)
    if mm:
        d["make"] = mm.group(1)
        d["model"] = mm.group(2)

    # mileage variants
    mi = None
    m1 = re.search(r"(?:mileage|odometer)\s*[:\-]?\s*([\d,]+)", text, re.I)
    if m1:
        try:
            mi = int(m1.group(1).replace(",", ""))
        except ValueError:
            mi = None
    if mi is None:
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*k\s*(?:mi|mile|miles)\b", text, re.I)
        if m2:
            try:
                mi = int(float(m2.group(1)) * 1000)
            except ValueError:
                mi = None
    if mi is None:
        m3 = re.search(r"(\d{1,3}(?:[,\d]{3})*)\s*(?:mi|mile|miles)\b", text, re.I)
        if m3:
            try:
                mi = int(re.sub(r"[^\d]", "", m3.group(1)))
            except ValueError:
                mi = None
    if mi is not None:
        d["mileage"] = mi

    # --- Structured attribute lines (Craigslist-style) ---
    trans_raw = _parse_attr_line(text, "transmission")
    if trans_raw:
        nt = _normalize_transmission_regex(trans_raw)
        if nt:
            d["transmission"] = nt

    color_raw = _parse_attr_line(text, "paint color") or _parse_attr_line(text, "color")
    if color_raw:
        d["color"] = color_raw.title()

    drive_raw = _parse_attr_line(text, "drive")
    if drive_raw:
        nd = _normalize_drive_regex(drive_raw)
        if nd:
            d["drive"] = nd

    fuel_raw = _parse_attr_line(text, "fuel")
    if fuel_raw:
        nf = _normalize_fuel_regex(fuel_raw)
        if nf:
            d["fuel"] = nf

    cond_raw = _parse_attr_line(text, "condition")
    if cond_raw:
        nc = _normalize_condition_regex(cond_raw)
        if nc:
            d["condition"] = nc

    title_raw = _parse_attr_line(text, "title status") or _parse_attr_line(text, "title")
    if title_raw:
        nt = _normalize_title_regex(title_raw)
        if nt:
            d["title_status"] = nt

    type_raw = _parse_attr_line(text, "type")
    if type_raw:
        nty = _normalize_type_regex(type_raw)
        if nty:
            d["type"] = nty

    cyl_raw = _parse_attr_line(text, "cylinders")
    if cyl_raw:
        cm = re.search(r"(\d+)", cyl_raw)
        if cm:
            try:
                n = int(cm.group(1))
                if 0 < n <= 16:
                    d["cylinders"] = n
            except ValueError:
                pass

    seller = _normalize_seller_regex(text)
    if seller:
        d["seller_type"] = seller

    z = _extract_zip_regex(text)
    cty, st = _extract_state_city_regex(text)
    if cty:
        d["city"] = cty
    if st:
        d["state"] = st
    zf = _finalize_zip_for_submission(z, st)
    if zf:
        d["zip_code"] = zf

    return d

# -------------------- HTTP ENTRY --------------------
def extract_http(request: Request):
    """
    Reads latest (or requested) run's TXT listings and writes ONE-LINE JSON records to:
      gs://<bucket>/<STRUCTURED_PREFIX>/run_id=<run_id>/jsonl/<post_id>.jsonl
    Request JSON (optional):
      { "run_id": "<...>", "max_files": 0, "overwrite": false }
    """
    logging.getLogger().setLevel(logging.INFO)

    if not BUCKET_NAME:
        return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    run_id    = body.get("run_id")
    max_files = int(body.get("max_files") or 0)        # 0 = unlimited
    overwrite = bool(body.get("overwrite") or False)

    # Pick newest run if not provided
    if not run_id:
        runs = _list_run_ids(BUCKET_NAME, SCRAPES_PREFIX)
        if not runs:
            return jsonify({"ok": False, "error": f"no run_ids found under {SCRAPES_PREFIX}/"}), 200
        run_id = runs[-1]

    scraped_at_iso = _parse_run_id_as_iso(run_id)

    txt_blobs = _txt_objects_for_run(run_id)
    if not txt_blobs:
        return jsonify({"ok": False, "run_id": run_id, "error": "no .txt files found for run"}), 200
    if max_files > 0:
        txt_blobs = txt_blobs[:max_files]

    processed = written = skipped = errors = 0
    bucket = storage_client.bucket(BUCKET_NAME)

    for name in txt_blobs:
        try:
            text = _download_text(name)
            fields = parse_listing(text)

            post_id = os.path.splitext(os.path.basename(name))[0]
            record = {
                "post_id": post_id,
                "run_id": run_id,
                "scraped_at": scraped_at_iso,
                "source_txt": name,
                **fields,
            }

            out_key = f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl/{post_id}.jsonl"

            if not overwrite and bucket.blob(out_key).exists():
                skipped += 1
            else:
                _upload_jsonl_line(out_key, record)
                written += 1

        except Exception as e:
            errors += 1
            logging.error(f"Failed {name}: {e}\n{traceback.format_exc()}")

        processed += 1

    result = {
        "ok": True,
        "version": "extractor-v4-regex-enriched",
        "run_id": run_id,
        "processed_txt": processed,
        "written_jsonl": written,
        "skipped_existing": skipped,
        "errors": errors
    }
    logging.info(json.dumps(result))
    return jsonify(result), 200
