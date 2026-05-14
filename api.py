"""
DawaScan — Drug Lookup API
===========================
Endpoints:
  GET  /health
  GET  /drugs/barcode/{barcode}   — lookup by scanned barcode
  GET  /drugs/search?q=augmentin  — fuzzy name search
  POST /drugs/verify-image        — AI packaging check (free Gemini)

Run:
  uvicorn api:app --reload --port 8000

.env file:
  NEXT_PUBLIC_SUPABASE_URL=https://xxxx.supabase.co
  NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=sb_publishable_...
  GEMINI_API_KEY=AIza...   (optional — free at aistudio.google.com/app/apikey)

Install:
  pip install fastapi uvicorn supabase google-genai python-dotenv
"""

import os
import json
from contextlib import asynccontextmanager

from supabase import create_client, Client
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import logging

load_dotenv()


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    url = os.environ["NEXT_PUBLIC_SUPABASE_URL"]
    key = os.environ["NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY"]
    app.state.db: Client = create_client(url, key)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        from google import genai

        app.state.gemini = genai.Client(api_key=gemini_key)
    else:
        app.state.gemini = None

    yield


app = FastAPI(title="DawaScan API", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────────


class DrugResult(BaseModel):
    found: bool
    id: str | None = None
    moph_code: str | None = None
    trade_name: str | None = None
    scientific_name: str | None = None
    dosage_form: str | None = None
    strength: str | None = None
    manufacturer: str | None = None
    country_origin: str | None = None
    registration_status: str | None = None
    price_usd: float | None = None
    verdict: str | None = None
    verdict_detail: str | None = None


class ImageVerifyResult(BaseModel):
    verdict: str
    confidence: float
    flags: list[str]
    explanation: str


# ── Helpers ────────────────────────────────────────────────────────────────────

DRUG_FIELDS = [
    "id",
    "moph_code",
    "trade_name",
    "scientific_name",
    "dosage_form",
    "strength",
    "manufacturer",
    "country_origin",
    "registration_status",
    "price_usd",
]


def row_to_drug_result(row: dict, verdict: str, detail: str) -> DrugResult:
    return DrugResult(
        found=True,
        verdict=verdict,
        verdict_detail=detail,
        **{k: row.get(k) for k in DRUG_FIELDS},
    )


def status_to_verdict(drug: dict) -> tuple[str, str]:
    name = drug.get("trade_name", "")
    code = drug.get("moph_code", "")
    status = drug.get("registration_status") or "unknown"

    if status == "cancelled":
        return "cancelled", (
            f"{name} ({code}) has been CANCELLED by MoPH. "
            "This drug should not be dispensed."
        )
    if status == "suspended":
        return "suspended", (
            f"{name} registration is currently SUSPENDED. "
            "Do not use without consulting a pharmacist."
        )
    if status == "active":
        return "registered", f"Registered with MoPH. Code: {code}"
    return "unknown", "Registration status could not be determined."


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.3.0",
        "image_verify": "available" if app.state.gemini else "no GEMINI_API_KEY set",
    }


@app.get("/drugs/barcode/{barcode}", response_model=DrugResult)
async def lookup_barcode(barcode: str):
    res = (
        app.state.db.table("drug_barcodes")
        .select("*, drugs(*)")
        .eq("barcode", barcode.strip())
        .limit(1)
        .execute()
    )

    if not res.data:
        return DrugResult(
            found=False,
            verdict="not_found",
            verdict_detail=(
                "This barcode is not in our database yet. "
                "This does NOT mean the drug is counterfeit — "
                "it may simply not be registered with us yet. "
                "Use the image scan for a visual check, or ask your pharmacist."
            ),
        )

    drug = res.data[0]["drugs"]
    verdict, detail = status_to_verdict(drug)
    return row_to_drug_result(drug, verdict, detail)


@app.get("/drugs/search", response_model=list[DrugResult])
async def search_drugs(
    q: str = Query(..., min_length=2, description="Drug name to search"),
    limit: int = Query(10, ge=1, le=50),
):
    res = (
        app.state.db.table("drugs")
        .select("*")
        .ilike("trade_name", f"%{q}%")
        .limit(limit)
        .execute()
    )
    rows = res.data or []

    res2 = (
        app.state.db.table("drugs")
        .select("*")
        .ilike("scientific_name", f"%{q}%")
        .limit(limit)
        .execute()
    )
    seen = {r["moph_code"] for r in rows}
    for r in res2.data or []:
        if r["moph_code"] not in seen:
            rows.append(r)
            seen.add(r["moph_code"])

    if not rows:
        return []

    results = []
    for r in rows[:limit]:
        verdict, detail = status_to_verdict(r)
        results.append(row_to_drug_result(r, verdict, detail))
    return results


@app.post("/drugs/verify-image", response_model=ImageVerifyResult)
async def verify_image(
    drug_name: str = Query(..., description="Name of the drug shown in the photo"),
    image: UploadFile = File(...),
):
    if not app.state.gemini:
        raise HTTPException(
            503,
            detail=(
                "Image verification unavailable — GEMINI_API_KEY not set. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            ),
        )

    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(400, "Image file is empty.")
    if len(img_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image too large. Max 10MB.")

    res = (
        app.state.db.table("drugs")
        .select("*")
        .ilike("trade_name", drug_name)
        .limit(1)
        .execute()
    )
    drug = res.data[0] if res.data else None

    drug_context = ""
    if drug:
        drug_context = f"""
Known MoPH registration:
- Code: {drug['moph_code']}
- Manufacturer: {drug['manufacturer']} ({drug['country_origin']})
- Form: {drug['dosage_form']}, Strength: {drug['strength']}
- Status: {drug['registration_status']}
"""

    prompt = f"""You are a pharmaceutical packaging expert helping detect suspicious medications in Lebanon.

The user photographed a drug package they believe is: {drug_name}
{drug_context}
Examine the image for these issues:
1. Font problems — wrong size, weight, typos, mixed fonts
2. Logo quality — blurry, pixelated, stretched, misaligned
3. Color accuracy — off-brand colors
4. Security features — missing or suspicious hologram/seal
5. Batch/expiry format — wrong format for this manufacturer
6. Print quality — bleeding ink, misalignment, fading
7. Text errors — spelling or grammar mistakes in any language
8. Barcode — damaged, photocopied, or low quality

Respond ONLY in this exact JSON, no other text:
{{
  "verdict": "likely_authentic",
  "confidence": 0.85,
  "flags": [],
  "explanation": "The packaging appears consistent with genuine product."
}}

verdict must be exactly: "likely_authentic", "suspicious", or "unknown"
confidence is 0.0 to 1.0
flags is a list of strings (empty list if no issues)
explanation is one plain sentence a non-expert can understand

Rules:
- Never use "fake" or "counterfeit" — only "suspicious"
- Image too blurry/dark/partial to assess → "unknown"
- If unsure, lower confidence rather than flagging suspicious
- For suspicious results, mention consulting a pharmacist"""

    try:
        from google.genai import types

        response = app.state.gemini.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=image.content_type or "image/jpeg",
                ),
            ],
        )

        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        logging.info(f"Gemini raw: {raw[:300]}")
        result = json.loads(raw)

        verdict = result.get("verdict", "unknown")
        if verdict not in ("likely_authentic", "suspicious", "unknown"):
            verdict = "unknown"

        return ImageVerifyResult(
            verdict=verdict,
            confidence=max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            flags=result.get("flags", []),
            explanation=result.get(
                "explanation", "Unable to assess. Please consult a pharmacist."
            ),
        )

    except json.JSONDecodeError:
        text = response.text.lower() if hasattr(response, "text") else ""
        verdict = (
            "suspicious"
            if any(w in text for w in ("suspicious", "fake", "counterfeit", "problem"))
            else "unknown"
        )
        return ImageVerifyResult(
            verdict=verdict,
            confidence=0.3,
            flags=["Analysis returned unexpected format"],
            explanation="Please consult a pharmacist to verify this medication.",
        )

    except Exception as e:
        logging.error(f"Gemini error: {str(e)}")
        return ImageVerifyResult(
            verdict="unknown",
            confidence=0.0,
            flags=[str(e)],
            explanation="Image analysis temporarily unavailable. Please consult a pharmacist.",
        )


# ── Crowdsource: link a barcode to a drug ─────────────────────────────────────


class LinkBarcodeRequest(BaseModel):
    barcode: str
    drug_id: str
    barcode_type: str = "EAN13"


class LinkBarcodeResult(BaseModel):
    success: bool
    message: str


@app.post("/drugs/link-barcode", response_model=LinkBarcodeResult)
async def link_barcode(body: LinkBarcodeRequest):
    """
    Crowdsource endpoint — saves a user-submitted barcode → drug link.
    Marked as unverified until an admin confirms it.
    Called when a user scans an unknown barcode and identifies the drug.
    """
    # Check drug exists
    drug_res = (
        app.state.db.table("drugs")
        .select("id, trade_name")
        .eq("id", body.drug_id)
        .limit(1)
        .execute()
    )
    if not drug_res.data:
        raise HTTPException(404, "Drug not found.")

    # Check barcode not already linked
    existing = (
        app.state.db.table("drug_barcodes")
        .select("id")
        .eq("barcode", body.barcode.strip())
        .limit(1)
        .execute()
    )
    if existing.data:
        return LinkBarcodeResult(
            success=False,
            message="This barcode is already linked to a drug in our database.",
        )

    # Insert unverified link
    app.state.db.table("drug_barcodes").insert(
        {
            "drug_id": body.drug_id,
            "barcode": body.barcode.strip(),
            "barcode_type": body.barcode_type,
            "verified": False,  # admin must verify before it's trusted
        }
    ).execute()

    drug_name = drug_res.data[0]["trade_name"]
    return LinkBarcodeResult(
        success=True,
        message=f"Thank you! Barcode linked to {drug_name}. Our team will verify it shortly.",
    )
