"""
VoxSales — AI Voice Sales Bot
AI-Powered Voice Sales Agent
Production-minded Flask backend with Google Gemini SDK, SQLite persistence, and MediaRecorder handling.
"""

import os
import json
import sqlite3
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
import io
import unicodedata
import time
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Google Gemini SDK
try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# Optional gTTS library
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False


class GeminiAnalysisError(Exception):
    """Raised when Gemini AI analysis fails or cannot be generated."""
    pass


class GeminiQuotaExhaustedError(GeminiAnalysisError):
    """Raised specifically when Gemini API returns 429 / RESOURCE_EXHAUSTED."""
    pass


def is_quota_exhausted_error(err):
    """
    Detects if an exception or error message indicates quota exhaustion,
    rate limit, or resource exhaustion (429 / RESOURCE_EXHAUSTED / quota exceeded).
    """
    if err is None:
        return False
    msg = str(err).lower()
    code = str(getattr(err, "code", "") or "").lower()
    status = str(getattr(err, "status", "") or "").lower()
    combined = f"{msg} {code} {status}".lower()
    quota_keywords = [
        "resource_exhausted",
        "429",
        "quota",
        "rate limit",
        "rate_limit_exceeded",
        "insufficient_quota",
        "credit_balance_exhausted",
        "exceeded your current quota"
    ]
    return any(k in combined for k in quota_keywords)


def is_transient_gemini_error(err):
    """
    Determines if an error is transient (e.g. 500, 502, 503, 504, timeout, network error),
    which is eligible for bounded exponential backoff retry.
    Permanent errors (400 Invalid Argument, 401/403 Invalid API key, 429 Quota exhausted)
    are NOT retried.
    """
    if err is None:
        return False
    if is_quota_exhausted_error(err):
        return False
    code = getattr(err, "code", None)
    msg = str(err).lower()
    if code in (400, 401, 403) or any(k in msg for k in [
        "api key not valid",
        "permission_denied",
        "unauthenticated",
        "invalid_argument",
        "forbidden"
    ]):
        return False
    if code in (500, 502, 503, 504) or any(k in msg for k in [
        "timeout",
        "timed out",
        "unavailable",
        "deadline_exceeded",
        "connection",
        "temporary"
    ]):
        return True
    return False


def execute_with_transient_retry(operation, max_retries=2, initial_delay=0.4):
    """
    Executes a callable with bounded exponential backoff only for transient failures.
    Does not retry permanent authentication errors or exhausted quotas.
    """
    delay = initial_delay
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as e:
            last_err = e
            if attempt < max_retries and is_transient_gemini_error(e):
                app.logger.warning(
                    f"Transient Gemini failure (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("Retry operation exhausted without an exception captured.")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "salesbot.db"
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 35 * 1024 * 1024  # 35MB limit for call recordings

STATUSES = [
    "Converted",
    "Interested",
    "Follow-up Required",
    "Not Interested",
    "Demo Requested",
    "No Response",
]

# Configurable SaaS Product Catalog
PRODUCT_PROFILES = {
    "FlowPilot": {
        "name": "FlowPilot",
        "tagline": "AI-Powered Workflow Automation & Orchestration",
        "description": "FlowPilot is a modern cloud orchestration platform that connects SaaS business applications, automates multi-step processes, and deploys intelligent AI agent actions with zero code.",
        "features": [
            "Visual drag-and-drop workflow builder",
            "150+ pre-built enterprise connectors (Slack, Salesforce, HubSpot, Stripe)",
            "Automated retry logic and SLA breach notifications",
            "Native LLM prompt and action nodes",
            "Complete audit logs and SOC2 compliant permissions"
        ],
        "benefits": "Reduces operational overhead by 70%, accelerates ticket triage, eliminates manual data re-entry across tools.",
        "pricing": "Starter plan at $49/month (up to 10k actions). Pro plan at $199/month (100k actions, premium connectors). Enterprise tier available with custom SLA and dedicated support.",
        "target_audience": "Operations leaders, engineering leads, customer success directors, and growing tech teams.",
        "limitations": "Self-hosted on-premises deployment requires an Enterprise contract. Not built for legacy mainframe systems.",
        "demo_link": "https://flowpilot.example.com/demo"
    },
    "CloudShield": {
        "name": "CloudShield",
        "tagline": "Autonomous Cloud Security & Compliance Automation",
        "description": "CloudShield provides continuous cloud security posture management (CSPM) and automated compliance evidence collection for AWS, GCP, and Microsoft Azure environments.",
        "features": [
            "Real-time multi-cloud asset discovery and inventory mapping",
            "Continuous misconfiguration scanning and instant remediation",
            "Automated audit-ready reports for SOC2 Type II, ISO 27001, HIPAA, and PCI-DSS",
            "IAM drift detection and least-privilege policy generator",
            "CI/CD infrastructure-as-code security gating"
        ],
        "benefits": "Pinpoints high-risk security flaws before attackers do, cuts compliance audit preparation time from months to hours.",
        "pricing": "Growth plan at $299/month (up to 3 cloud accounts). Enterprise plan at $899/month (unlimited accounts, custom policies, 24/7 security SLA).",
        "target_audience": "DevOps engineers, CISOs, cloud architects, and security compliance leads.",
        "limitations": "Supports public clouds (AWS, GCP, Azure). Private OpenStack or bare-metal data centers are not supported.",
        "demo_link": "https://cloudshield.example.com/demo"
    },
    "DataPulse": {
        "name": "DataPulse",
        "tagline": "Real-Time Sales Intelligence & Pipeline Forecasting",
        "description": "DataPulse analyzes CRM pipeline velocity, sales rep activities, and call transcripts to provide accurate AI revenue forecasts and deal risk warnings.",
        "features": [
            "Bi-directional synchronization with Salesforce and HubSpot",
            "AI deal risk scoring and churn alert triggers",
            "Automated activity logging and next-action recommendations",
            "Executive revenue forecast simulations with confidence bounds"
        ],
        "benefits": "Increases team win rates by 22%, eliminates manual spreadsheet forecasts, identifies at-risk deals before quarter close.",
        "pricing": "$35 per seat/month billed annually. Free 14-day fully-featured trial for teams with 5 or more reps.",
        "target_audience": "VP of Sales, CROs, Sales Operations, and Growth Account Executives.",
        "limitations": "Requires an existing CRM deployment (Salesforce or HubSpot).",
        "demo_link": "https://datapulse.example.com/demo"
    }
}

SUPPORTED_LANGUAGES = [
    {"code": "en-US", "name": "English (US)", "speechLang": "en-US"},
    {"code": "en-IN", "name": "English (India)", "speechLang": "en-IN"},
    {"code": "hi-IN", "name": "Hindi (हिंदी)", "speechLang": "hi-IN"},
    {"code": "te-IN", "name": "Telugu (తెలుగు)", "speechLang": "te-IN"}
]

# Verified server-side TTS language specifications supporting Telugu, Hindi, and English
TTS_LANGUAGE_SPECS = {
    "te-IN": {
        "lang_code": "te",
        "voice_id": "google-te-standard",
        "provider": "google",
        "name": "Telugu (India)",
        "tld": "com"
    },
    "hi-IN": {
        "lang_code": "hi",
        "voice_id": "google-hi-standard",
        "provider": "google",
        "name": "Hindi (India)",
        "tld": "com"
    },
    "en-IN": {
        "lang_code": "en",
        "voice_id": "google-en-in-standard",
        "provider": "google",
        "name": "English (India)",
        "tld": "co.in"
    },
    "en-US": {
        "lang_code": "en",
        "voice_id": "google-en-us-standard",
        "provider": "google",
        "name": "English (US)",
        "tld": "com"
    }
}


def generate_tts_audio(text, language_code="te-IN"):
    """
    Generates genuine speech audio bytes for the exact text in the selected language.
    Strictly preserves the input text in its original language without translating it.
    Explicitly supports Telugu (te-IN), Hindi (hi-IN), and English (en-US, en-IN).
    """
    clean_text = (text or "").strip()
    clean_text = re.sub(r"[\*#_`~]", "", clean_text).strip()
    if not clean_text:
        raise ValueError("Text is required for TTS generation")

    spec = TTS_LANGUAGE_SPECS.get(language_code)
    if not spec:
        primary = language_code.split("-")[0]
        matched_key = next((k for k in TTS_LANGUAGE_SPECS if k.startswith(primary)), None)
        if matched_key:
            spec = TTS_LANGUAGE_SPECS[matched_key]
        else:
            raise ValueError(f"Language '{language_code}' is not supported by TTS provider. Supported: {list(TTS_LANGUAGE_SPECS.keys())}")

    # High-availability primary provider: gTTS
    lang_code = spec["lang_code"]
    tld = spec.get("tld", "com")

    if HAS_GTTS:
        try:
            fp = io.BytesIO()
            tts = gTTS(text=clean_text, lang=lang_code, tld=tld)
            tts.write_to_fp(fp)
            fp.seek(0)
            data = fp.read()
            if data and len(data) > 100:
                return data, "audio/mpeg"
        except Exception as e:
            app.logger.warning(f"gTTS library call failed, trying direct endpoint: {e}")

    # Direct Google Translate TTS endpoint fallback
    try:
        tts_url = "https://translate.google.com/translate_tts"
        sentences = [s.strip() for s in re.split(r"(?<=[।!?.\n])\s*", clean_text) if s.strip()]
        if not sentences:
            sentences = [clean_text]

        chunks = []
        curr = ""
        for s in sentences:
            if len(curr) + len(s) + 1 <= 180:
                curr = (curr + " " + s).strip()
            else:
                if curr:
                    chunks.append(curr)
                curr = s[:180]
        if curr:
            chunks.append(curr)
        if not chunks:
            chunks = [clean_text[:180]]

        combined_audio = io.BytesIO()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        for chunk in chunks:
            params = {
                "ie": "UTF-8",
                "q": chunk,
                "tl": lang_code,
                "client": "tw-ob"
            }
            resp = requests.get(tts_url, params=params, headers=headers, timeout=12)
            if resp.status_code == 200 and resp.content:
                combined_audio.write(resp.content)
            else:
                app.logger.warning(f"Direct Google TTS chunk returned status {resp.status_code}")

        combined_audio.seek(0)
        audio_content = combined_audio.read()
        if audio_content and len(audio_content) > 100:
            return audio_content, "audio/mpeg"
    except Exception as e:
        app.logger.exception(f"Direct Google TTS fallback failed: {e}")

    raise RuntimeError(f"Failed to generate TTS audio for language '{language_code}'")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY,
                customer TEXT,
                company TEXT DEFAULT '',
                product TEXT,
                language TEXT,
                notes TEXT DEFAULT '',
                started_at TEXT,
                ended_at TEXT,
                duration_seconds INTEGER DEFAULT 0,
                transcript TEXT DEFAULT '[]',
                summary TEXT DEFAULT '',
                customer_requirements TEXT DEFAULT '',
                objections TEXT DEFAULT '',
                buying_signals TEXT DEFAULT '',
                next_steps TEXT DEFAULT '',
                sales_status TEXT DEFAULT 'No Response',
                recording TEXT DEFAULT '',
                consent INTEGER DEFAULT 0,
                processing_status TEXT DEFAULT 'completed'
            )
        """)

        # Backward compatibility migration for existing columns
        existing_cols = {col[1] for col in conn.execute("PRAGMA table_info(calls)").fetchall()}
        columns_to_add = [
            ("company", "TEXT DEFAULT ''"),
            ("notes", "TEXT DEFAULT ''"),
            ("customer_requirements", "TEXT DEFAULT ''"),
            ("objections", "TEXT DEFAULT ''"),
            ("buying_signals", "TEXT DEFAULT ''"),
            ("next_steps", "TEXT DEFAULT ''"),
            ("processing_status", "TEXT DEFAULT 'completed'"),
            ("sales_status", "TEXT DEFAULT 'No Response'"),
            ("status", "TEXT DEFAULT 'No Response'"),
            ("analysis_mode", "TEXT DEFAULT 'ai'")
        ]
        for col_name, col_def in columns_to_add:
            if col_name not in existing_cols:
                try:
                    conn.execute(f"ALTER TABLE calls ADD COLUMN {col_name} {col_def}")
                except Exception as e:
                    app.logger.warning(f"Could not add column {col_name}: {e}")

        # Keep status and sales_status synced for both legacy and new schema callers
        try:
            conn.execute("UPDATE calls SET sales_status = status WHERE (sales_status IS NULL OR sales_status = '' OR sales_status = 'No Response') AND (status IS NOT NULL AND status != '')")
            conn.execute("UPDATE calls SET status = sales_status WHERE (status IS NULL OR status = '' OR status = 'No Response') AND (sales_status IS NOT NULL AND sales_status != '')")
        except Exception:
            pass


init_db()


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    if not HAS_GEMINI_SDK:
        app.logger.warning("google-genai SDK is not installed or available.")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        app.logger.error(f"Failed to initialize Gemini client: {e}")
        return None


def get_gemini_model():
    model = (os.getenv("GEMINI_MODEL") or "").strip()
    if not model or model.lower() in ("undefined", "none"):
        return "gemini-3.6-flash"
    return model


def get_gemini_fallback_models():
    """
    Returns the configured secondary fallback model names to attempt sequentially when earlier models encounter 429 quota exhaustion.
    Defaults to ['gemini-3.8-flash', 'gemini-3-flash-preview', 'gemini-3.1-flash-lite'].
    Returns [] if explicitly set to 'none', 'disabled', or empty string.
    """
    val = os.getenv("GEMINI_FALLBACK_MODEL")
    if val is None:
        return ["gemini-3.8-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite"]
    val = val.strip()
    if not val or val.lower() in ("none", "disabled", "false", "off", "undefined"):
        return []
    models = [m.strip() for m in val.split(",") if m.strip()]
    return models


def get_gemini_fallback_model():
    """
    Returns the primary configured fallback model name or None if disabled.
    """
    models = get_gemini_fallback_models()
    return models[0] if models else None


def build_generate_content_config(model_name, **kwargs):
    """
    Builds a GenerateContentConfig for the specified model.
    Adds thinking_config=types.ThinkingConfig(thinking_budget=0) for Gemini 3 models to disable latency-inducing thoughts.
    """
    config_kwargs = dict(kwargs)
    if model_name and "gemini-3" in str(model_name).lower():
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
    return types.GenerateContentConfig(**config_kwargs)


def generate_content_with_quota_fallback(
    gemini_client,
    contents,
    base_config_kwargs,
    primary_model=None,
    fallback_model=None
):
    """
    Executes content generation using the primary model.
    If a 429 / RESOURCE_EXHAUSTED / quota error occurs, attempts using the configured
    fallback model(s) in sequence.
    Does NOT retry quota errors indefinitely on the same model. Non-quota errors (e.g. 400, 401) immediately raise.
    Returns: (response, used_model_name, used_fallback_bool)
    """
    if primary_model is None:
        primary_model = get_gemini_model()

    if fallback_model is not None:
        if isinstance(fallback_model, list):
            fallback_models = fallback_model
        elif isinstance(fallback_model, str):
            fallback_models = [m.strip() for m in fallback_model.split(",") if m.strip() and m.strip().lower() not in ("none", "disabled")]
        else:
            fallback_models = []
    else:
        fallback_models = get_gemini_fallback_models()

    # Filter out primary model from fallback models
    fallback_models = [m for m in fallback_models if m.lower() != primary_model.lower()]

    # 1. Attempt primary model
    try:
        config_obj = build_generate_content_config(primary_model, **base_config_kwargs)
        response = execute_with_transient_retry(
            lambda: gemini_client.models.generate_content(
                model=primary_model,
                contents=contents,
                config=config_obj
            )
        )
        return response, primary_model, False
    except Exception as e:
        if not is_quota_exhausted_error(e) or not fallback_models:
            raise e

        last_fb_err = e
        for fb_model in fallback_models:
            app.logger.warning(
                f"Model '{primary_model if fb_model == fallback_models[0] else 'prior fallback'}' quota exhausted. "
                f"Attempting fallback model '{fb_model}'..."
            )
            try:
                fb_config_obj = build_generate_content_config(fb_model, **base_config_kwargs)
                response = execute_with_transient_retry(
                    lambda: gemini_client.models.generate_content(
                        model=fb_model,
                        contents=contents,
                        config=fb_config_obj
                    )
                )
                app.logger.info(f"Fallback model '{fb_model}' successfully generated content.")
                return response, fb_model, True
            except Exception as fb_err:
                last_fb_err = fb_err
                if is_quota_exhausted_error(fb_err):
                    app.logger.warning(f"Fallback model '{fb_model}' also quota exhausted.")
                    continue
                else:
                    app.logger.warning(f"Fallback model '{fb_model}' failed with non-quota error: {fb_err}")
                    raise fb_err

        raise last_fb_err


def get_product_context(product_name, custom_details=None):
    if custom_details and isinstance(custom_details, dict) and custom_details.get("description"):
        return custom_details

    profile = PRODUCT_PROFILES.get(product_name)
    if profile:
        return profile

    # Default fallback profile if user typed a custom product name
    return {
        "name": product_name,
        "tagline": f"{product_name} Cloud SaaS",
        "description": f"{product_name} is a dedicated SaaS solution designed to streamline enterprise workflows.",
        "features": ["Cloud-based accessibility", "Collaborative team workspace", "Security and analytics"],
        "benefits": "Improves efficiency, simplifies complex operations, and saves team time.",
        "pricing": "Flexible subscription tiers available. Custom quote provided after discovering requirements.",
        "target_audience": "Modern enterprises and growing business teams.",
        "limitations": "Specific feature compatibility depends on your team's existing technology stack.",
        "demo_link": "Available upon request"
    }


def get_language_info(language_code):
    return next((l for l in SUPPORTED_LANGUAGES if l["code"] == language_code), {
        "code": language_code, "name": language_code, "speechLang": language_code
    })


def get_initial_greeting(language_code, prospect_name="Guest", product_name="FlowPilot"):
    p_name = f" {prospect_name.strip()}" if prospect_name and prospect_name.strip() and prospect_name.strip().lower() != "guest" else ""
    if language_code == "hi-IN":
        return f"नमस्ते{p_name}! कॉल पर बात करने के लिए धन्यवाद। मैं {product_name} के लिए AI सेल्स विशेषज्ञ हूँ। वर्तमान में आपके वर्कफ़्लो में सबसे बड़ी चुनौती क्या आ रही है?"
    elif language_code == "te-IN":
        return f"నమస్కారం{p_name}! కాల్ మాట్లాడినందుకు ధన్యవాదాలు. నేను {product_name} సేల్స్ AI నిపుణుడిని. మీ ప్రస్తుత వర్క్‌ఫ్లోలో మీరు ఎదుర్కొంటున్న అతిపెద్ద సవాలు ఏమిటి?"
    elif language_code == "en-IN":
        return f"Hello{p_name}, thank you for taking my call. I am the AI sales specialist for {product_name}. What is the biggest bottleneck your team is facing in your workflow today?"
    else:
        return f"Hello{p_name}! Thanks for taking my call. I am the AI sales specialist for {product_name}. What is your biggest challenge with your current workflow right now?"


def build_system_prompt(product_name, language_code, prospect_name="Guest", company="", notes="", custom_details=None, past_assistant_texts=None):
    product = get_product_context(product_name, custom_details)
    lang_info = get_language_info(language_code)
    lang_name = lang_info["name"]

    features_str = ", ".join(product["features"]) if isinstance(product.get("features"), list) else str(product.get("features", ""))
    prospect_desc = f"{prospect_name}" + (f" from {company}" if company else "")

    lang_rules = {
        "hi-IN": """MANDATORY LANGUAGE DIRECTIVE: HINDI (हिंदी)
- You MUST converse ONLY in Hindi (हिंदी).
- Every single customer-facing reply, question, explanation, objection handling, and closing MUST be in pure, fluent Hindi using Devanagari script (देवनागरी लिपि).
- Understand the customer's speech in Hindi.
- Answer product questions and objections naturally and idiomatically in Hindi.
- Maintain a professional, polite, and consultative sales tone.
- NEVER switch to English or reply in English or Latin script unless English is explicitly selected or the customer explicitly asks you to switch languages.
- You may pronounce SaaS product names (e.g., FlowPilot, CloudShield, DataPulse) as their proper brand names, but all sentences, verbs, and questions MUST be in Hindi.""",

        "te-IN": """MANDATORY LANGUAGE DIRECTIVE: TELUGU (తెలుగు)
- You MUST converse ONLY in Telugu (తెలుగు).
- Every single customer-facing reply, question, explanation, objection handling, and closing MUST be in pure, fluent Telugu using Telugu script (తెలుగు లిపి).
- Understand the customer's speech in Telugu.
- Answer product questions and objections naturally and idiomatically in Telugu.
- Maintain a professional, polite, and consultative sales tone.
- NEVER switch to English or reply in English or Latin script unless English is explicitly selected or the customer explicitly asks you to switch languages.
- You may pronounce SaaS product names (e.g., FlowPilot, CloudShield, DataPulse) as their proper brand names, but all sentences, verbs, and questions MUST be in Telugu.""",

        "en-IN": """MANDATORY LANGUAGE DIRECTIVE: ENGLISH (INDIA)
- Speak and generate all responses in fluent, professional Indian English.
- Maintain a courteous, professional, and consultative Indian corporate sales tone.
- Address questions, features, pricing, and objections clearly in English (India).""",

        "en-US": """MANDATORY LANGUAGE DIRECTIVE: ENGLISH (US)
- Speak and generate all responses in natural, fluent American English.
- Maintain an articulate, confident, and consultative US B2B sales tone.
- Address questions, features, pricing, and objections clearly in English (US)."""
    }

    selected_directive = lang_rules.get(language_code, f"""MANDATORY LANGUAGE DIRECTIVE: {lang_name} ({language_code})
- Speak exclusively in {lang_name}.
- Generate every customer-facing response in {lang_name}.
- Maintain a professional, friendly sales tone.
- Never switch to English unless English is the selected language or the customer explicitly asks to change languages.""")

    prompt = f"""You are VoxSales, an articulate, natural, and ethical AI voice sales assistant representing the SaaS product '{product['name']}'.
You are having a real-time telephone/voice conversation with {prospect_desc}.

CONVERSATION LANGUAGE MANDATE:
{selected_directive}

VOICE CONVERSATION GUIDELINES:
1. Spoken Delivery: This is spoken aloud via Text-to-Speech. Keep responses short and conversational (1 to 3 sentences maximum per turn).
2. Formatting: Never use bullet points, markdown formatting, numbered lists, emojis, asterisks, or URLs.
3. CONVERSATION PROGRESSION & DIRECT ANSWERS:
   - Carefully review the entire conversation history.
   - Respond directly and specifically to what the customer just said in their latest message.
   - NEVER repeat your previous question or re-ask the opening greeting question.
   - If the customer answers your question or describes their pain point, acknowledge their specific problem with empathy, explain how {product['name']} solves it, and offer the next step (features, pricing, or demo).
   - If the customer asks about features, explain 1-2 core features relevant to their need.
   - If the customer asks about pricing, answer with the verified pricing and ask if that fits their budget.
   - If the customer raises an objection or concern, address it constructively with value, ROI, and flexibility.
   - If the customer asks for a demo or expresses interest, confirm enthusiasm and propose a concrete scheduling time (e.g., tomorrow afternoon for a quick 20-minute guided session).
4. Accuracy & Honesty: Only discuss verified features and pricing from the product profile below. NEVER invent unverified guarantees, custom integrations, or false customer testimonials. If an answer is unknown, politely offer to check with the technical sales team.
5. Objection Handling: Acknowledge objections empathetically and address them constructively without being pushy or aggressive.
6. Next Steps: Ask permission before offering a live product demo or scheduling a technical follow-up.
7. Conversions: Never claim a transaction, sale, or contract has closed unless the customer explicitly agrees to purchase.

PRODUCT PROFILE:
- Product Name: {product['name']}
- Tagline: {product.get('tagline', '')}
- Description: {product.get('description', '')}
- Key Features: {features_str}
- Proven Benefits: {product.get('benefits', '')}
- Pricing: {product.get('pricing', 'Pricing details available on consultation')}
- Target Audience: {product.get('target_audience', '')}
- Known Limitations: {product.get('limitations', 'Standard cloud limits apply')}
- Demo Info: {product.get('demo_link', 'Live guided demo available')}
"""
    if notes:
        prompt += f"\nPRE-CALL SALES NOTES ON PROSPECT:\n{notes}\n"

    if past_assistant_texts:
        formatted_past = "\n".join(f"- \"{t}\"" for t in past_assistant_texts[-5:])
        prompt += f"""
CRITICAL ANTI-REPETITION MANDATE:
You have ALREADY spoken the following responses earlier in this conversation:
{formatted_past}

YOU MUST NEVER:
1. Repeat or rephrase any response or question listed above.
2. Re-ask what their biggest workflow problem, challenge, or bottleneck is.
3. Repeat pricing or features without advancing to the next step.
Directly address the prospect's latest statement and advance the sales progression.
"""

    return prompt


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    try:
        with get_db() as conn:
            conn.execute("SELECT 1").fetchone()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    return jsonify({
        "status": "healthy",
        "service": "VoxSales AI Voice Sales Bot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "aiConfigured": bool(api_key),
        "aiProvider": "gemini"
    })


@app.get("/api/config")
def config():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    tts_provider = os.getenv("TTS_PROVIDER", "google").lower().strip()
    return jsonify({
        "aiConfigured": bool(api_key),
        "aiProvider": "gemini",
        "model": get_gemini_model(),
        "fallbackModel": get_gemini_fallback_model(),
        "speechRecognitionSupported": True,
        "languages": SUPPORTED_LANGUAGES,
        "products": PRODUCT_PROFILES,
        "statuses": STATUSES,
        "tts": {
            "supported": True,
            "provider": tts_provider,
            "supportedLanguages": list(TTS_LANGUAGE_SPECS.keys())
        }
    })


@app.post("/api/tts")
def tts_generate():
    """
    Server-side Text-To-Speech endpoint.
    Generates genuine MP3 audio from the exact text in the requested language.
    Never translates the text.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Text is required for TTS generation"}), 400

    language = (data.get("language") or "en-US").strip()

    try:
        audio_bytes, mimetype = generate_tts_audio(text, language)
        return send_file(
            io.BytesIO(audio_bytes),
            mimetype=mimetype,
            as_attachment=False,
            download_name=f"tts_{language}_{uuid.uuid4().hex[:8]}.mp3"
        )
    except ValueError as ve:
        return jsonify({
            "error": str(ve),
            "supportedLanguages": list(TTS_LANGUAGE_SPECS.keys())
        }), 400
    except Exception as e:
        app.logger.exception(f"TTS generation error: {e}")
        return jsonify({"error": f"Failed to generate speech audio: {str(e)}"}), 500


@app.get("/api/tts/voices")
def tts_voices():
    """
    Returns verified server-side TTS voice options and providers.
    """
    provider = os.getenv("TTS_PROVIDER", "google").lower().strip()
    voices_list = []
    for code, spec in TTS_LANGUAGE_SPECS.items():
        voices_list.append({
            "code": code,
            "name": spec["name"],
            "voiceId": spec["voice_id"],
            "provider": spec["provider"],
            "supported": True
        })
    return jsonify({
        "provider": provider,
        "voices": voices_list
    })


@app.post("/api/calls/start")
@app.post("/api/calls")
def create_call():
    data = request.get_json(silent=True) or {}
    if not data.get("consent"):
        return jsonify({"error": "Affirmative recording consent is required before starting a call."}), 400

    customer = (data.get("customer") or data.get("prospect_name") or "").strip()[:120]
    if not customer:
        return jsonify({"error": "Prospect name is required to start a call."}), 400

    product = (data.get("product") or data.get("product_name") or "").strip()[:120]
    if not product:
        return jsonify({"error": "SaaS product selection is required to start a call."}), 400

    language = (data.get("language") or "en-US").strip()
    call_id = str(uuid.uuid4())
    company = (data.get("company") or "").strip()[:120]
    notes = (data.get("notes") or "").strip()[:1000]
    started = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        conn.execute("""
            INSERT INTO calls (id, customer, company, product, language, notes, started_at, consent, status, sales_status, processing_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, 'In Progress', 'In Progress', 'active')
        """, (call_id, customer, company, product, language, notes, started))

    greeting = get_initial_greeting(language, customer, product)

    return jsonify({
        "id": call_id,
        "callId": call_id,
        "startedAt": started,
        "customer": customer,
        "product": product,
        "language": language,
        "greeting": greeting
    })


def normalize_speech_text(text):
    """
    Normalizes spoken or transcript text by stripping punctuation, extra spaces,
    and lowercasing for accurate duplicate detection across languages.
    Preserves all Unicode letters, numbers, and Indic combining marks.
    """
    if not text:
        return ""
    cleaned = "".join(c for c in str(text) if not unicodedata.category(c).startswith(("P", "S")))
    return " ".join(cleaned.lower().split())


def is_greeting_reask(text, language_code):
    """
    Checks if a candidate assistant response repeats or re-asks the opening greeting question.
    """
    low = (text or "").lower()
    is_hi = language_code.startswith("hi")
    is_te = language_code.startswith("te")

    if is_te:
        return any(q in low for q in [
            "అతిపెద్ద సవాలు ఏమిటి", "సవాలు ఏమిటి", "వర్క్‌ఫ్లోలో మీరు ఎదుర్కొంటున్న",
            "ప్రస్తుతం మీ వ్యాపారానికి ఏది అత్యంత అవసరం"
        ])
    elif is_hi:
        return any(q in low for q in [
            "सबसे बड़ी चुनौती क्या आ रही है", "बड़ी चुनौती क्या आ रही है",
            "वर्कफ़्लो में सबसे बड़ी चुनौती", "वर्तमान में आपके वर्कफ़्लो में"
        ])
    else:
        return any(q in low for q in [
            "biggest challenge with your current workflow",
            "biggest bottleneck your team is facing",
            "biggest challenge right now",
            "challenge with your current workflow right now"
        ])


def is_repetitive_response(candidate_reply, past_assistant_texts, language_code):
    """
    Detects if a candidate assistant reply is an exact, normalized, or substantially
    similar duplicate of any past assistant turn in this call, or re-asks the greeting question.
    """
    if not candidate_reply or not candidate_reply.strip():
        return True

    clean_candidate = candidate_reply.strip()
    norm_candidate = normalize_speech_text(clean_candidate)
    if not norm_candidate:
        return True

    if is_greeting_reask(clean_candidate, language_code):
        return True

    cand_words = set(norm_candidate.split())

    for past in past_assistant_texts:
        if not past or not past.strip():
            continue
        clean_past = past.strip()
        norm_past = normalize_speech_text(clean_past)
        if not norm_past:
            continue

        if clean_candidate == clean_past or norm_candidate == norm_past:
            return True

        if len(norm_past) > 20 and len(norm_candidate) > 20:
            if norm_candidate in norm_past or norm_past in norm_candidate:
                return True

        past_words = set(norm_past.split())
        if cand_words and past_words:
            overlap = len(cand_words & past_words)
            sim = overlap / max(len(cand_words), len(past_words))
            if sim >= 0.70:
                return True

    return False


def generate_conversational_reply(user_text, history, product_name, language, prospect_name="Guest", company="", custom_product=None):
    """
    Stateful, context-aware conversational engine for VoxSales.
    Tracks prior turns, understands customer intent across Telugu, Hindi, and English,
    and advances through a natural sales progression without repeating previous questions.
    """
    product_ctx = get_product_context(product_name, custom_product)
    p_name = product_ctx["name"]
    features = product_ctx.get("features", [])
    f1 = features[0] if len(features) > 0 else f"{p_name} capabilities"
    f2 = features[1] if len(features) > 1 else f"{p_name} integrations"
    benefits = product_ctx.get("benefits", "streamlines team operations and eliminates manual bottlenecks")
    pricing = product_ctx.get("pricing", "flexible tiered subscription options")

    is_hi = language.startswith("hi")
    is_te = language.startswith("te")

    past_assistant_texts = [
        str(m.get("text") or m.get("content") or "").strip()
        for m in history
        if m.get("role") in ("assistant", "ai")
    ]
    past_assistant_lower = " ".join(t.lower() for t in past_assistant_texts)

    low = user_text.lower().strip()

    # 1. Closing / Wrap-up / No further questions
    is_closing = any(x in low for x in [
        "that's all", "thats all", "nothing else", "no other questions", "no more questions", "all good", "thank you that is all", "have a good day", "we are good", "all set", "goodbye", "bye",
        "बस इतना ही", "कोई सवाल नहीं", "कोई अन्य प्रश्न नहीं", "धन्यवाद बस इतना ही", "अलविदा", "सब ठीक है",
        "ఇంకేమీ లేవు", "ఇంకేమీ ప్రశ్నలు లేవు", "ఏమీ లేవు", "అంతే", "చాలు", "ధన్యవాదాలు అంతే", "మంచి రోజు", "సెలవు"
    ])

    # 2. Not Interested / Opt-out
    is_not_interested = any(x in low for x in [
        "not interested", "no thanks", "stop", "don't call", "leave me alone", "busy", "don't need", "not looking",
        "नहीं चाहिए", "रुचि नहीं", "बंद करो", "व्यस्त", "कॉल मत करो", "समय नहीं",
        "వద్దు", "ఆసక్తి లేదు", "చేయవద్దు", "కాల్ చేయవద్దు", "ఫోన్ పెట్టేయండి", "సమయం లేదు", "ఆపండి", "అవసరం లేదు"
    ])

    # 3. Demo / Walkthrough / Scheduling Request
    is_demo = any(x in low for x in [
        "demo", "walkthrough", "show me", "presentation", "trial", "schedule", "book", "meeting", "see it", "test drive",
        "डेमो", "दिखाइए", "प्रदर्शन", "ट्रायल", "मीटिंग", "दिखाएं", "शेड्यूल", "लाइव",
        "డెమో", "డేమో", "చూపించండి", "టూర్", "ట్రయల్", "చూడాలి", "మీటింగ్", "షెడ్యూల్", "లైవ్", "పరిశీలన"
    ])

    # 4. Cost Objection / Budget Concern
    is_pricing_objection = any(x in low for x in [
        "too expensive", "expensive", "too much", "costly", "high price", "out of budget", "tight budget", "can't afford", "cheaper", "discount",
        "महंगा", "बहुत ज्यादा", "बजट ज्यादा", "बजट नहीं", "कम कीमत", "सस्ता", "खर्च ज्यादा", "ज्यादा है",
        "ఖరీదైనది", "ధర ఎక్కువ", "ఖర్చు ఎక్కువ", "బడ్జెట్ ఎక్కువ", "బడ్జెట్ లేదు", "రేటు ఎక్కువ", "భరించలేము", "తక్కువ", "ఎక్కువ"
    ])

    # 5. Pricing / Cost Inquiry
    is_pricing_inquiry = any(x in low for x in [
        "price", "pricing", "cost", "how much", "rate", "subscription", "plans", "fee", "tier", "quote", "charge",
        "कीमत", "दाम", "लागत", "फीस", "खर्च", "रेट", "मूल्य", "प्लान", "सब्सक्रिप्शन", "कितने का", "पैकेज",
        "ధర", "ఖర్చు", "ఫీజు", "రేటు", "ధరలు", "ప్లాన్", "ఎంత", "సబ్‌స్క్రిప్షన్", "ఖరీదు", "ప్యాకేజ్"
    ])

    # 6. Features / Capabilities Inquiry (prioritized above generic problem keywords)
    is_features_inquiry = any(x in low for x in [
        "feature", "features", "benefit", "benefits", "capabilities", "what does", "how does it work", "how do you help", "how can it help", "advantage", "use case", "works",
        "सुविधा", "फीचर", "विशेषताएं", "फायदे", "लाभ", "कैसे काम करता है", "क्या करता है", "मदद", "उपयोग", "क्षमता",
        "ఫీచర్లు", "ఫీచర్", "ప్రయోజనాలు", "ప్రయోజనం", "లాభాలు", "ఎలా పనిచేస్తుంది", "ఎలా ఉపయోగపడుతుంది", "ఏం చేస్తుంది", "సహాయం", "విశేషాలు", "సామర్థ్యం", "వివరాలు", "ముఖ్యమైన", "లక్షణాలు"
    ])

    # 7. Meta / Flow Correction
    is_meta_correction = any(x in low for x in [
        "already answered", "already told", "next question", "listen to me", "understand my answer", "repeat", "don't ask again",
        "पहले ही बता दिया", "उत्तर दे दिया", "अगला सवाल", "सुनिए", "दोबारा मत पूछिए", "आगे बढ़िए",
        "సమాధానం చెప్పాను", "చెప్పాను", "ముందే చెప్పాను", "తదుపరి ప్రశ్న", "వినండి", "మళ్లీ అడగవద్దు", "నెక్స్ట్"
    ])

    # 8. Positive Agreement / General Interest / Confirmation
    is_positive = any(x in low for x in [
        "yes", "sure", "sounds good", "okay", "tell me more", "interested", "proceed", "go ahead", "definitely", "perfect", "ok", "works", "fine", "yeah", "sounds fine",
        "हाँ", "बिल्कुल", "अच्छा है", "बताइए", "रुचि है", "आगे बढ़िए", "ठीक है", "ज़रूर", "सही है", "हाँ चलेगा",
        "అవును", "సరే", "చెప్పండి", "బాగుంది", "ఆసక్తిగా ఉంది", "ముందుకు", "తప్పకుండా", "మంచిది", "ఓకే", "సరే సమయం బాగుంది", "సరిపోతుంది"
    ])

    # 9. Problem Statement / Workflow Challenge (specific pain point keywords without generic single words)
    is_problem_statement = any(x in low for x in [
        "too much time", "wasting time", "waste time", "time consuming", "manual", "slow", "audit", "security flaw", "soc2", "compliance", "aws", "azure", "salesforce", "slack", "drift", "bottleneck", "hours wasted", "struggling",
        "समय बर्बाद", "ज्यादा समय", "समय लगता है", "मैन्युअल", "सुरक्षा खामी", "ऑडिट रिपोर्ट", "दिक्कत", "परेशानी", "चुनौती",
        "సమయం వృధా", "ఎక్కువ సమయం", "సమయం పడుతుంది", "మాన్యువల్", "ఆడిట్", "సెక్యూరిటీ", "సమస్య", "సవాలు", "కష్టం", "ఇబ్బంది", "ఆధారాలు సేకరించడం"
    ])

    # Stage awareness across past assistant turns
    has_demo_in_past = ("demo" in past_assistant_lower or "డెమో" in past_assistant_lower or "డేమో" in past_assistant_lower or "डेमो" in past_assistant_lower)
    has_pricing_in_past = ("price" in past_assistant_lower or "pricing" in past_assistant_lower or "$" in past_assistant_lower or "ధర" in past_assistant_lower or "कीमत" in past_assistant_lower or "प्लान" in past_assistant_lower or "ప్లాన్" in past_assistant_lower)
    has_features_in_past = ("feature" in past_assistant_lower or "ఆటోమేట్" in past_assistant_lower or "సహాయం" in past_assistant_lower or "సువిధ" in past_assistant_lower or "सुविधा" in past_assistant_lower or "specialize" in past_assistant_lower)

    if is_te:
        if is_closing:
            reply = f"ఈ రోజు మాట్లాడినందుకు చాలా ధన్యవాదాలు, {prospect_name}. రేపటి మా సమావేశంలో మిమ్మల్ని కలవడానికి ఎదురుచూస్తున్నాము. మీకు మంచి రోజు కావాలని కోరుకుంటున్నాను!"
        elif is_not_interested:
            reply = f"నేను మీ నిర్ణయాన్ని పూర్తిగా గౌరవిస్తున్నాను, {prospect_name}. నాతో మాట్లాడినందుకు చాలా ధన్యవాదాలు. మీకు మంచి రోజు కావాలని కోరుకుంటున్నాను!"
        elif is_meta_correction:
            reply = f"నన్ను క్షమించండి, {prospect_name}. మీ పరిస్థితిని నేను స్పష్టంగా అర్థం చేసుకున్నాను. తదుపరి దశగా, {p_name} యొక్క 20 నిమిషాల ప్రత్యక్ష డెమోను షెడ్యూల్ చేయమంటారా లేదా ధర వివరాలను అందించమంటారా?"
        elif is_features_inquiry:
            reply = f"{p_name} ప్రధానంగా {f1} మరియు {f2} అందిస్తుంది. దీని ద్వారా {benefits}. మీ బృందంలో ఈ ఫీచర్లను ప్రత్యక్ష డెమోలో చూడటానికి ఆసక్తిగా ఉన్నారా?"
        elif is_pricing_inquiry:
            reply = f"{p_name} కోసం, {pricing}. ఇది మీ బృందం బడ్జెట్‌కు మరియు ప్రణాళికలకు అనుకూలంగా ఉందా?"
        elif is_pricing_objection:
            reply = f"మీ బడ్జెట్ ఆందోళనను నేను అర్థం చేసుకోగలను, {prospect_name}. అయితే {p_name} ఉపయోగించడం వల్ల {benefits}, తద్వారా మొదటి నెలలోనే పూర్తి ROI లభిస్తుంది. మీరు దీని లైవ్ డెమోను 20 నిమిషాల్లో పరిశీలించగలరా?"
        elif is_demo:
            reply = f"చాలా సంతోషం, {prospect_name}! {p_name} ప్రత్యక్ష డెమోను ఏర్పాటు చేయడానికి నేను సిద్ధంగా ఉన్నాను. రేపు మధ్యాహ్నం 2:00 గంటలకు 20 నిమిషాల డెమో కోసం మీకు సమయం అనుకూలంగా ఉంటుందా?"
        elif is_positive:
            if not has_pricing_in_past:
                reply = f"చాలా సంతోషం, {prospect_name}! {p_name} కోసం {pricing}. ఈ ప్లాన్ మీ బడ్జెట్‌కు అనుకూలంగా ఉంటుందా?"
            elif not has_demo_in_past:
                reply = f"అద్భుతం! {p_name} మీ వర్క్‌ఫ్లోను ఎలా సులభతరం చేస్తుందో 20 నిమిషాల లైవ్ డెమోలో చూపిస్తాను. రేపు మధ్యాహ్నం 2:00 గంటల సమయం మీకు అనుకూలమా?"
            else:
                reply = f"చాలా ధన్యవాదాలు, {prospect_name}. మీ డెమో వివరాలు మరియు ప్రాధాన్యతలు నమోదు చేయబడ్డాయి. మా టెక్నికల్ బృందం వెంటనే మీతో కనెక్ట్ అవుతుంది. మీకు ఏవైనా ఇతర ప్రశ్నలు ఉన్నాయా?"
        elif is_problem_statement:
            reply = f"ఖచ్చితంగా అర్థం చేసుకున్నాను, {prospect_name}. మీ సమస్యను పరిష్కరించడానికి {p_name} ప్రత్యేకంగా రూపొందించబడింది. ఇది {f1} ద్వారా {benefits}. మీరు దీనిని 20 నిమిషాల లైవ్ డెమోలో చూడాలనుకుంటున్నారా లేదా ముందుగా ధరల వివరాలు తెలుసుకోవాలనుకుంటున్నారా?"
        else:
            if not has_features_in_past:
                reply = f"వివరాలు తెలిపినందుకు ధన్యవాదాలు, {prospect_name}. {p_name} ప్రధానంగా {f1} తో సహాయపడుతుంది మరియు {benefits}. మీరు దీని పనితీరును డెమోలో చూడాలనుకుంటున్నారా?"
            elif not has_pricing_in_past:
                reply = f"{p_name} ధరల వివరాలు చూస్తే, {pricing}. ఇది మీ బడ్జెట్‌కు సరిపోతుందా?"
            elif not has_demo_in_past:
                reply = f"ధన్యవాదాలు, {prospect_name}. మీ బృందానికి తగిన విధంగా {p_name}ను 20 నిమిషాల ప్రత్యక్ష డెమోలో పరిశీలించడానికి రేపు మీకు సమయం అనుకూలంగా ఉంటుందా?"
            else:
                reply = f"ధన్యవాదాలు, {prospect_name}. మీ అవసరాలను మేము ఖచ్చితంగా తీర్చగలము. మా టెక్నికల్ స్పెషలిస్ట్ మీతో సమన్వయం చేసుకుంటారు. మీకు ఇంకా ఏవైనా ప్రశ్నలు ఉన్నాయా?"

    elif is_hi:
        if is_closing:
            reply = f"आज बात करने के लिए बहुत-बहुत धन्यवाद, {prospect_name}। हम हमारे लाइव डेमो सत्र में आपसे मिलने के लिए उत्सुक हैं। आपका दिन शुभ हो!"
        elif is_not_interested:
            reply = f"मैं आपकी बात पूरी तरह समझता हूँ और आपके समय का सम्मान करता हूँ, {prospect_name}। आज बात करने के लिए धन्यवाद। आपका दिन शुभ हो!"
        elif is_meta_correction:
            reply = f"क्षमा करें, {prospect_name}। मैं आपकी बात समझ गया हूँ। अगले कदम के तौर पर, क्या आप {p_name} का 20 मिनट का लाइव डेमो देखना चाहेंगे या कीमत की जानकारी?"
        elif is_features_inquiry:
            reply = f"{p_name} मुख्य रूप से {f1} और {f2} की सुविधा प्रदान करता है, जिससे {benefits}। क्या आप इसे 20 मिनट के लाइव डेमो में देखना चाहेंगे?"
        elif is_pricing_inquiry:
            reply = f"{p_name} के लिए, {pricing}। क्या यह आपकी टीम की योजना और बजट के अनुरूप है?"
        elif is_pricing_objection:
            reply = f"मैं आपकी बजट संबंधी चिंता को समझता हूँ, {prospect_name}। {p_name} की मदद से {benefits}, जिससे पहले ही महीने में पूरा ROI मिल जाता है। क्या आप इसका 20 मिनट का लाइव डेमो देखना चाहेंगे?"
        elif is_demo:
            reply = f"बहुत बढ़िया, {prospect_name}! मुझे आपके लिए {p_name} का 20 मिनट का लाइव डेमो शेड्यूल करने में खुशी होगी। क्या कल दोपहर 2:00 बजे का समय आपके लिए सही रहेगा?"
        elif is_positive:
            if not has_pricing_in_past:
                reply = f"यह जानकर बहुत अच्छा लगा, {prospect_name}! {p_name} के लिए {pricing}। क्या यह आपके बजट के अनुरूप है?"
            elif not has_demo_in_past:
                reply = f"शानदार! मैं आपके लिए {p_name} का 20 मिनट का लाइव डेमो दिखाना चाहूँगा। क्या कल दोपहर 2:00 बजे का समय आपके लिए सुविधाजनक रहेगा?"
            else:
                reply = f"धन्यवाद, {prospect_name}। आपकी डेमो रिक्वेस्ट दर्ज कर ली गई है और तकनीकी टीम आपसे संपर्क करेगी। क्या आपके कोई अन्य प्रश्न हैं?"
        elif is_problem_statement:
            reply = f"मैं आपकी समस्या को पूरी तरह समझ गया, {prospect_name}। {p_name} विशेष रूप से इस चुनौती को हल करने के लिए बना है। यह {f1} के जरिए {benefits}। क्या आप 20 मिनट का लाइव डेमो देखना चाहेंगे या पहले कीमत जानना चाहेंगे?"
        else:
            if not has_features_in_past:
                reply = f"जानकारी के लिए धन्यवाद, {prospect_name}। {p_name} मुख्य रूप से {f1} उपलब्ध कराता है और {benefits}। क्या आप इसका लाइव प्रदर्शन देखना चाहेंगे?"
            elif not has_pricing_in_past:
                reply = f"{p_name} के मूल्य निर्धारण की बात करें तो, {pricing}। क्या यह आपकी योजना में फिट बैठता है?"
            elif not has_demo_in_past:
                reply = f"धन्यवाद, {prospect_name}। क्या आप कल दोपहर 20 मिनट के लाइव डेमो में {p_name} का व्यावहारिक प्रदर्शन देखना पसंद करेंगे?"
            else:
                reply = f"धन्यवाद, {prospect_name}। हमारी टीम आपके सभी वर्कफ़्लो को सुव्यवस्थित करने के लिए तैयार है। क्या कोई अन्य विवरण है जो आप साझा करना चाहेंगे?"

    else:
        if is_closing:
            reply = f"Thank you so much for your time today, {prospect_name}. We look forward to meeting with you for our walkthrough. Have a fantastic day ahead!"
        elif is_not_interested:
            reply = f"I completely understand and respect your time, {prospect_name}. Thank you for taking a moment to speak with me today. Have a wonderful day!"
        elif is_meta_correction:
            reply = f"My apologies for that, {prospect_name}. I have noted your response. As a next step, would you prefer to schedule a quick 20-minute product demo, or review our pricing options?"
        elif is_features_inquiry:
            reply = f"{p_name} specializes in {f1} and {f2}, which directly {benefits}. Would you like to explore this in a quick 20-minute live demo?"
        elif is_pricing_inquiry:
            reply = f"For {p_name}, {pricing}. Does that align with what your team has planned for this initiative?"
        elif is_pricing_objection:
            reply = f"I completely understand your budget consideration, {prospect_name}. Most teams find that {p_name} delivers rapid ROI by {benefits}. Would you like to review the exact ROI breakdown in a 20-minute demo?"
        elif is_demo:
            reply = f"I would be delighted to schedule a live 20-minute guided demonstration of {p_name} for you, {prospect_name}. Would tomorrow afternoon around 2:00 PM work for a quick walkthrough?"
        elif is_positive:
            if not has_pricing_in_past:
                reply = f"That is great to hear, {prospect_name}! For {p_name}, {pricing}. Does that tier align with your current budget?"
            elif not has_demo_in_past:
                reply = f"Wonderful! I would love to show you a quick 20-minute live demo of {p_name}. Would tomorrow afternoon around 2:00 PM work for your calendar?"
            else:
                reply = f"Thank you, {prospect_name}. Your demo request has been logged and our product specialist will reach out shortly. Do you have any other questions today?"
        elif is_problem_statement:
            reply = f"I completely understand, {prospect_name}. {p_name} was purpose-built to solve that exact bottleneck by providing {f1}, which {benefits}. Would you like to see how that works in a quick 20-minute demo, or discuss pricing options first?"
        else:
            if not has_features_in_past:
                reply = f"Thanks for sharing that, {prospect_name}. {p_name} provides {f1} and {benefits}. Would you like to see a brief walkthrough?"
            elif not has_pricing_in_past:
                reply = f"Regarding investment, {p_name} offers {pricing}. Does that align with your team's budget?"
            elif not has_demo_in_past:
                reply = f"Thank you, {prospect_name}. Would tomorrow afternoon work well for a quick 20-minute live demo to see {p_name} in action?"
            else:
                reply = f"Thank you for sharing that context, {prospect_name}. {p_name} is ready to support your team. Would you like to reserve a 20-minute calendar slot with our engineering specialist?"

    # Anti-repetition guard: ensure the candidate reply NEVER duplicates or re-asks any prior assistant turn in this call
    if is_repetitive_response(reply, past_assistant_texts, language):
        alternatives = [
            f"ధన్యవాదాలు, {prospect_name}. {p_name} యొక్క ఆర్కిటెక్చర్ మరియు ఇంటిగ్రేషన్ల గురించి మా సీనియర్ ఇంజనీర్ వివరంగా వివరించడానికి రేపు 20 నిమిషాల సమయం కేటాయించగలరా?" if is_te else (
                f"धन्यवाद, {prospect_name}। {p_name} के आर्किटेक्चर और मौजूदा सिस्टम के साथ एकीकरण को समझने के लिए क्या कल 20 मिनट का समय उपयुक्त रहेगा?" if is_hi else
                f"Thank you, {prospect_name}. Could we arrange a brief technical consultation tomorrow to walk through how {p_name} connects directly with your existing infrastructure?"
            ),
            f"ఖచ్చితంగా, {prospect_name}. మీ వర్క్‌ఫ్లోకు తగిన విధంగా {p_name} ఎలా ఆటోమేట్ చేస్తుందో లైవ్ టెస్ట్ రన్ ద్వారా పరిశీలించవచ్చు. దీనిపై మీకు ఇంకా ఏవైనా సందేహాలు ఉన్నాయా?" if is_te else (
                f"बिल्कुल, {prospect_name}। हम आपकी टीम के लिए {p_name} का एक अनुकूलित लाइव डेमो तैयार कर सकते हैं। क्या इस बारे में आपका कोई और प्रश्न है?" if is_hi else
                f"Certainly, {prospect_name}. We can tailor a hands-on live test environment to demonstrate how {p_name} automates your exact workflow. Do you have any additional questions on that?"
            ),
            f"మీ బృందం అవసరాలకు అనుగుణంగా అనుకూలమైన ROI బ్రేక్‌డౌన్ మరియు ప్రణాళికను మేము రూపొందించగలము. దీనిని తదుపరి దశలో సమీక్షిద్దామా?" if is_te else (
                f"हम आपकी आवश्यकताओं के अनुसार एक विस्तृत ROI और लागत विश्लेषण साझा करने के लिए तैयार हैं। क्या हम इसे अगले चरण में शामिल करें?" if is_hi else
                f"We can provide a customized ROI analysis and implementation roadmap tailored for your team. Would you like to include that in our follow-up?"
            ),
            f"చాలా మంచిది, {prospect_name}. మీ ప్రాధాన్యతలను మేము నోట్ చేసుకున్నాము. మా ప్రొడక్ట్ స్పెషలిస్ట్ మీకు అనుకూలమైన సమయంలో నేరుగా కనెక్ట్ అవుతారు. ధన్యవాదాలు!" if is_te else (
                f"बहुत अच्छा, {prospect_name}। मैंने आपकी सभी प्राथमिकताएं नोट कर ली हैं। हमारी तकनीकी टीम जल्द ही आपसे जुड़ेगी। आपका दिन शुभ हो!" if is_hi else
                f"Excellent, {prospect_name}. I have logged your notes and requirements. Our product specialist will be in touch with next steps shortly. Have a wonderful day!"
            )
        ]
        chosen = None
        for alt in alternatives:
            if not is_repetitive_response(alt, past_assistant_texts, language):
                chosen = alt
                break
        if chosen:
            reply = chosen
        else:
            turn_idx = len(past_assistant_texts) + 1
            if is_te:
                reply = f"ధన్యవాదాలు {prospect_name}. {p_name} పై మీ వివరాలు (విభాగం {turn_idx}) నమోదు చేయబడ్డాయి. మనం డెమో షెడ్యూల్‌ను ఖరారు చేయవచ్చా?"
            elif is_hi:
                reply = f"धन्यवाद {prospect_name}। {p_name} के संबंध में आपकी जानकारी (चरण {turn_idx}) नोट कर ली गई है। क्या हम डेमो शेड्यूल फाइनल करें?"
            else:
                reply = f"Thank you {prospect_name}. Your requirements for {p_name} (step {turn_idx}) have been noted. Would you like to finalize our consultation time?"

    return reply


@app.post("/api/chat")
@app.post("/api/respond")
def respond():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"error": "Message is empty"}), 400

    product_name = (data.get("product") or "FlowPilot").strip()[:120]
    language = data.get("language", "en-US")
    prospect_name = (data.get("customer") or data.get("prospect_name") or "Guest").strip()[:120]
    company = (data.get("company") or "").strip()[:120]
    notes = (data.get("notes") or "").strip()[:1000]
    history = data.get("history") or []
    custom_product = data.get("custom_product")

    # Authoritative history normalization
    normalized_history = []
    past_assistant_texts = []
    for item in history:
        role = "assistant" if item.get("role") in ("assistant", "ai") else "user"
        content = str(item.get("text") or item.get("content") or "").strip()[:1500]
        if content:
            if role == "assistant":
                past_assistant_texts.append(content)
            # Avoid duplicate consecutive identical messages
            if not normalized_history or normalized_history[-1].get("content") != content:
                normalized_history.append({"role": role, "content": content})

    # Ensure the latest customer message is present at the end of history exactly once
    if not normalized_history or normalized_history[-1].get("content") != user_text:
        normalized_history.append({"role": "user", "content": user_text[:2000]})

    system_prompt = build_system_prompt(
        product_name=product_name,
        language_code=language,
        prospect_name=prospect_name,
        company=company,
        notes=notes,
        custom_details=custom_product,
        past_assistant_texts=past_assistant_texts
    )

    gemini_client = get_gemini_client()
    if gemini_client:
        try:
            model_name = get_gemini_model()
            # Prepare multi-turn contents for Gemini SDK (alternating user and model roles)
            gemini_contents = []
            for item in normalized_history[-16:]:
                role = "user" if item.get("role") in ("user", "prospect") else "model"
                text_content = str(item.get("content", "")).strip()
                if text_content:
                    if gemini_contents and gemini_contents[-1].role == role:
                        prev_text = gemini_contents[-1].parts[0].text
                        gemini_contents[-1] = types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=f"{prev_text}\n{text_content}")]
                        )
                    else:
                        gemini_contents.append(
                            types.Content(
                                role=role,
                                parts=[types.Part.from_text(text=text_content)]
                            )
                        )

            # Ensure conversation starts with a user turn
            if gemini_contents and gemini_contents[0].role == "model":
                gemini_contents.insert(0, types.Content(role="user", parts=[types.Part.from_text(text="Hello")]))

            if gemini_contents:
                base_kwargs = {
                    "system_instruction": system_prompt,
                    "temperature": 0.6,
                    "max_output_tokens": 350
                }
                response, used_model, used_fallback = generate_content_with_quota_fallback(
                    gemini_client=gemini_client,
                    contents=gemini_contents,
                    base_config_kwargs=base_kwargs
                )
                candidate_reply = (getattr(response, "text", "") or "").strip()
                if not candidate_reply and response and getattr(response, "candidates", None):
                    parts = [p.text for c in response.candidates if c.content and c.content.parts for p in c.content.parts if getattr(p, "text", None)]
                    candidate_reply = " ".join(parts).strip()
                # Clean any markdown formatting that ruins natural TTS
                candidate_reply = re.sub(r"[\*#_`~]", "", candidate_reply).strip()

                # Strictly verify that Gemini did not repeat any prior assistant turn or re-ask greeting
                if candidate_reply and not is_repetitive_response(candidate_reply, past_assistant_texts, language):
                    return jsonify({
                        "reply": candidate_reply,
                        "mode": "ai",
                        "provider": "gemini",
                        "model": used_model,
                        "fallbackUsed": used_fallback
                    })
                else:
                    app.logger.warning(
                        f"Gemini generated empty or repetitive response ('{candidate_reply}'). "
                        f"Falling through to stateful context-aware conversational engine."
                    )
        except Exception as e:
            app.logger.exception(f"Gemini completion failed: {e}")
            # Fall through to stateful fallback response

    # Smart, stateful conversational fallback
    reply = generate_conversational_reply(
        user_text=user_text,
        history=normalized_history,
        product_name=product_name,
        language=language,
        prospect_name=prospect_name,
        company=company,
        custom_product=custom_product
    )

    return jsonify({
        "reply": reply,
        "mode": "fallback",
        "notice": "Generated using context-aware conversational engine."
    })


@app.post("/api/calls/end")
@app.post("/api/finalize")
def finalize_call():
    form = request.form
    json_data = request.get_json(silent=True) or {}

    call_id = form.get("callId") or form.get("id") or json_data.get("callId") or json_data.get("id")
    if not call_id:
        return jsonify({"error": "Missing callId parameter"}), 400

    with get_db() as conn:
        row = conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()

    if not row:
        return jsonify({"error": f"Call record '{call_id}' not found"}), 404

    # Extract metadata from form or JSON
    if form:
        transcript_raw = form.get("transcript", row["transcript"])
        duration = int(form.get("durationSeconds") or form.get("duration") or 0)
        customer = form.get("customer") or row["customer"]
        company = form.get("company") or row["company"]
        product = form.get("product") or row["product"]
        language = form.get("language") or row["language"]
        notes = form.get("notes") or row["notes"]
        audio_file = request.files.get("recording")
    else:
        transcript_raw = json_data.get("transcript", row["transcript"])
        duration = int(json_data.get("durationSeconds") or json_data.get("duration") or 0)
        customer = json_data.get("customer") or row["customer"]
        company = json_data.get("company") or row["company"]
        product = json_data.get("product") or row["product"]
        language = json_data.get("language") or row["language"]
        notes = json_data.get("notes") or row["notes"]
        audio_file = None

    # Parse transcript safely
    if isinstance(transcript_raw, str):
        try:
            transcript = json.loads(transcript_raw)
        except Exception:
            transcript = []
    elif isinstance(transcript_raw, list):
        transcript = transcript_raw
    else:
        transcript = []

    # Save audio recording if supplied
    recording_name = row["recording"]
    if audio_file and audio_file.filename:
        filename = f"{secure_filename(call_id)}.webm"
        file_path = RECORDINGS_DIR / filename
        try:
            audio_file.save(file_path)
            recording_name = filename
        except Exception as e:
            app.logger.error(f"Failed to save recording: {e}")

    ended_at = datetime.now(timezone.utc).isoformat()
    analysis_error = None
    analysis = None
    is_quota_exhausted = False
    is_fallback = False

    try:
        analysis = analyze_call_transcript(
            transcript=transcript,
            product_name=product,
            language=language,
            customer=customer,
            company=company,
            allow_fallback=False
        )
        processing_status = "completed"
        analysis_mode = "ai"
        is_fallback = False
    except Exception as e:
        app.logger.warning(f"Gemini call analysis failed for {call_id}: {e}")
        analysis_error = str(e)
        is_quota_exhausted = is_quota_exhausted_error(e)
        is_fallback = True
        processing_status = "quota_exhausted" if is_quota_exhausted else "completed_with_fallback"
        analysis_mode = "local_fallback"

        # Generate clearly labelled local fallback using saved transcript
        analysis = analyze_call_transcript(
            transcript=transcript,
            product_name=product,
            language=language,
            customer=customer,
            company=company,
            allow_fallback=True
        )

    # Save completed call, transcript, recording, and all intelligence fields to SQLite
    reqs_json = json.dumps(analysis["customer_requirements"], ensure_ascii=False) if isinstance(analysis["customer_requirements"], (list, dict)) else str(analysis["customer_requirements"])
    objs_json = json.dumps(analysis["objections"], ensure_ascii=False) if isinstance(analysis["objections"], (list, dict)) else str(analysis["objections"])
    buys_json = json.dumps(analysis["buying_signals"], ensure_ascii=False) if isinstance(analysis["buying_signals"], (list, dict)) else str(analysis["buying_signals"])

    with get_db() as conn:
        conn.execute("""
            UPDATE calls SET
                ended_at = ?,
                duration_seconds = ?,
                transcript = ?,
                summary = ?,
                customer_requirements = ?,
                objections = ?,
                buying_signals = ?,
                next_steps = ?,
                status = ?,
                sales_status = ?,
                recording = ?,
                customer = ?,
                company = ?,
                product = ?,
                language = ?,
                notes = ?,
                processing_status = ?,
                analysis_mode = ?
            WHERE id = ?
        """, (
            ended_at,
            duration,
            json.dumps(transcript, ensure_ascii=False),
            analysis["summary"],
            reqs_json,
            objs_json,
            buys_json,
            analysis["next_steps"],
            analysis["status"],
            analysis["status"],
            recording_name,
            customer,
            company,
            product,
            language,
            notes,
            processing_status,
            analysis_mode,
            call_id
        ))

    payload = {
        "success": True,
        "callId": call_id,
        "endedAt": ended_at,
        "durationSeconds": duration,
        "summary": analysis["summary"],
        "customerRequirements": analysis["customer_requirements"],
        "customer_requirements": analysis["customer_requirements"],
        "objections": analysis["objections"],
        "buyingSignals": analysis["buying_signals"],
        "buying_signals": analysis["buying_signals"],
        "nextSteps": analysis["next_steps"],
        "next_steps": analysis["next_steps"],
        "status": analysis["status"],
        "sales_status": analysis["status"],
        "processing_status": processing_status,
        "analysis_mode": analysis_mode,
        "isFallback": is_fallback,
        "quotaExhausted": is_quota_exhausted,
        "recording": recording_name,
        "recordingUrl": f"/api/calls/{call_id}/recording" if recording_name else ""
    }

    if is_fallback:
        payload["quotaNotice"] = (
            "Gemini API quota exhausted or API key unavailable. "
            "Call, transcript, and recording were safely preserved in SQLite. "
            "Generated local transcript fallback intelligence (not an AI-generated summary). "
            "You can use Retry AI Analysis after checking your Gemini API key or quota."
        )

    return jsonify(payload)


@app.post("/api/calls/<call_id>/analyze")
@app.post("/api/calls/<call_id>/retry-analysis")
def retry_call_analysis(call_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()

    if not row:
        return jsonify({"error": f"Call record '{call_id}' not found"}), 404

    try:
        transcript = json.loads(row["transcript"] or "[]")
    except Exception:
        transcript = []

    product = row["product"] or "FlowPilot"
    language = row["language"] or "en-US"
    customer = row["customer"] or "Guest"
    company = row["company"] or ""

    try:
        analysis = analyze_call_transcript(
            transcript=transcript,
            product_name=product,
            language=language,
            customer=customer,
            company=company,
            allow_fallback=False
        )
    except Exception as e:
        app.logger.warning(f"Retry analysis failed for {call_id}: {e}")
        quota_exhausted = is_quota_exhausted_error(e)
        error_msg = (
            "Gemini API quota is exhausted or API key is unavailable. "
            "Your transcript, recording, and local fallback analysis remain safely preserved. "
            "Please check your Gemini API key or quota and retry."
            if quota_exhausted else f"AI analysis failed: {str(e)}"
        )
        return jsonify({
            "success": False,
            "error": error_msg,
            "callId": call_id,
            "quota_exhausted": quota_exhausted,
            "processing_status": row["processing_status"] or "quota_exhausted",
            "analysis_mode": row["analysis_mode"] if "analysis_mode" in row.keys() else "local_fallback"
        }), 402 if quota_exhausted else 500

    # Gemini analysis succeeded! Update existing record in-place without duplicate records
    reqs_json = json.dumps(analysis["customer_requirements"], ensure_ascii=False) if isinstance(analysis["customer_requirements"], (list, dict)) else str(analysis["customer_requirements"])
    objs_json = json.dumps(analysis["objections"], ensure_ascii=False) if isinstance(analysis["objections"], (list, dict)) else str(analysis["objections"])
    buys_json = json.dumps(analysis["buying_signals"], ensure_ascii=False) if isinstance(analysis["buying_signals"], (list, dict)) else str(analysis["buying_signals"])

    with get_db() as conn:
        conn.execute("""
            UPDATE calls SET
                summary = ?,
                customer_requirements = ?,
                objections = ?,
                buying_signals = ?,
                next_steps = ?,
                status = ?,
                sales_status = ?,
                processing_status = 'completed',
                analysis_mode = 'ai'
            WHERE id = ?
        """, (
            analysis["summary"],
            reqs_json,
            objs_json,
            buys_json,
            analysis["next_steps"],
            analysis["status"],
            analysis["status"],
            call_id
        ))

    return jsonify({
        "success": True,
        "callId": call_id,
        "summary": analysis["summary"],
        "customerRequirements": analysis["customer_requirements"],
        "customer_requirements": analysis["customer_requirements"],
        "objections": analysis["objections"],
        "buyingSignals": analysis["buying_signals"],
        "buying_signals": analysis["buying_signals"],
        "nextSteps": analysis["next_steps"],
        "next_steps": analysis["next_steps"],
        "status": analysis["status"],
        "sales_status": analysis["status"],
        "processing_status": "completed",
        "analysis_mode": "ai",
        "isFallback": False,
    })


@app.post("/api/calls/<call_id>/recording")
def upload_call_recording(call_id):
    with get_db() as conn:
        row = conn.execute("SELECT id FROM calls WHERE id = ?", (call_id,)).fetchone()
    if not row:
        return jsonify({"error": "Call not found"}), 404

    audio_file = request.files.get("recording") or request.files.get("audio")
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "No audio file provided"}), 400

    filename = f"{secure_filename(call_id)}.webm"
    target_path = RECORDINGS_DIR / filename
    try:
        audio_file.save(target_path)
        with get_db() as conn:
            conn.execute("UPDATE calls SET recording = ? WHERE id = ?", (filename, call_id))
        return jsonify({"success": True, "recording": filename, "recordingUrl": f"/api/calls/{call_id}/recording"})
    except Exception as e:
        app.logger.exception("Failed to write recording")
        return jsonify({"error": f"Failed to save recording: {str(e)}"}), 500


def analyze_call_transcript(transcript, product_name, language, customer="Guest", company="", allow_fallback=True):
    """
    Produces evidence-based sales intelligence strictly based on the transcript and classifies into exactly one of:
    'Converted', 'Interested', 'Follow-up Required', 'Not Interested', 'Demo Requested', 'No Response'.
    """
    # Count customer messages
    user_turns = [m for m in transcript if m.get("role") in ("user", "prospect")]
    customer_speech = " ".join(str(m.get("text", "")).strip() for m in user_turns).strip()

    # Rule: If no customer replies captured, status is 'No Response'
    if not user_turns or len(customer_speech) < 2:
        if language.startswith("hi"):
            return {
                "summary": f"[स्थानीय प्रतिलेख विश्लेषण — नियम-आधारित फ़ॉलबैक]\n{customer} के साथ {product_name} के संबंध में प्रारंभिक बिक्री कॉल। कॉल कनेक्ट हुआ, लेकिन कोई सार्थक ग्राहक प्रतिक्रिया प्राप्त नहीं हुई। (यह स्थानीय नियम-आधारित विश्लेषण है; AI-जनरेटेड सारांश नहीं)।",
                "customer_requirements": ["कोई आवश्यकता पहचानी नहीं गई।"],
                "objections": [],
                "buying_signals": ["कोई स्पष्ट खरीदारी संकेत नहीं मिला।"],
                "next_steps": "कार्यालय समय के दौरान फोन या ईमेल द्वारा पुनः संपर्क करने का प्रयास करें।",
                "status": "No Response",
                "analysis_mode": "local_fallback",
                "is_fallback": True
            }
        elif language.startswith("te"):
            return {
                "summary": f"[లోకల్ ట్రాన్స్‌క్రిప్ట్ విశ్లేషణ — నియమాల ఆధారిత ఫాల్‌బ్యాక్]\n{customer} తో {product_name} అమ్మకాల ప్రారంభ కాల్. కాల్ కనెక్ట్ అయింది, కానీ కస్టమర్ నుండి స్పందన రాలేదు. (ఇది లోకల్ నియమాల ఆధారిత విశ్లేషణ; AI రూపొందించిన సారాంశం కాదు).",
                "customer_requirements": ["ఎటువంటి అవసరాలు గుర్తించబడలేదు."],
                "objections": [],
                "buying_signals": ["ఎటువంటి కొనుగోలు సంకేతాలు కనిపించలేదు."],
                "next_steps": "వ్యాపార వేళల్లో ఫోన్ లేదా ఇమెయిల్ ద్వారా మళ్లీ సంప్రదించడానికి ప్రయత్నించండి.",
                "status": "No Response",
                "analysis_mode": "local_fallback",
                "is_fallback": True
            }
        else:
            return {
                "summary": f"[Local Transcript Analysis — Rule-Based Fallback]\nInitial sales call with {customer} regarding {product_name}. The call connected, but no meaningful customer response was captured. (Rule-based local fallback analysis; not an AI-generated summary).",
                "customer_requirements": ["None identified."],
                "objections": [],
                "buying_signals": ["No clear buying signals detected."],
                "next_steps": "Attempt to reconnect via phone or email during standard business hours.",
                "status": "No Response",
                "analysis_mode": "local_fallback",
                "is_fallback": True
            }

    formatted_transcript = "\n".join(
        f"{('Prospect' if m.get('role') in ('user', 'prospect') else 'AI')}: {m.get('text', '')}"
        for m in transcript if m.get("text")
    )[:9000]

    lang_info = get_language_info(language)
    lang_name = lang_info["name"]

    client = get_gemini_client()
    if client:
        try:
            model_name = get_gemini_model()
            prompt = f"""You are a senior sales intelligence director analyzing an AI voice sales call transcript for the SaaS product '{product_name}'.
Prospect: {customer} {f'({company})' if company else ''}
Language: {lang_name} ({language})

CONVERSATION TRANSCRIPT:
{formatted_transcript}

TASK:
Analyze the conversation strictly and solely based on what was actually said in the transcript.

MANDATORY LANGUAGE DIRECTIVE:
The conversation language was {lang_name} ({language}).
You MUST output all textual explanation fields ('summary', 'customer_requirements', 'objections', 'buying_signals', 'next_steps') in {lang_name}.
- For Hindi (hi-IN), use Devanagari script.
- For Telugu (te-IN), use Telugu script.
- For English, use English.

STANDARDIZED STATUS FIELD:
The 'status' field MUST REMAIN STRICTLY ONE OF THE 6 CANONICAL ENGLISH STATUS VALUES:
- "Converted": Prospect EXPLICITLY agreed to purchase or subscribe. (Do NOT mark Converted without explicit evidence of a purchase or conversion).
- "Demo Requested": Prospect explicitly requested a demo, walkthrough, presentation, or screen share.
- "Follow-up Required": Prospect requested email details, asked to be called back later, or needs delayed contact.
- "Not Interested": Prospect clearly rejected the product, stated no fit, or asked to stop calling.
- "Interested": Prospect expressed genuine curiosity or positive reception without closing.
- "No Response": No meaningful customer response was received.
Do NOT translate the 'status' value itself; it must be one of the exact 6 English tokens above for database and system consistency.

CRITICAL RULES:
1. "summary": Provide a concise 2-3 sentence executive summary of the conversation in {lang_name} based strictly on the actual transcript. Never fabricate events or discussions.
2. "customer_requirements": Extract a list of actual customer requirements, pain points, workflow challenges, goals, or constraints in {lang_name}.
   - If no requirements or pain points were mentioned, return ["None identified." localized to {lang_name}].
3. "objections": Identify customer questions, reservations, pricing concerns, competing products, or implementation hesitations raised in {lang_name}, along with how the AI sales agent addressed them in {lang_name}.
   - Return a list of objects with "concern" and "response" fields.
   - Example: [{{"concern": "Concern text in {lang_name}", "response": "Response text in {lang_name}"}}]
   - If no questions or objections were raised, return []. Never invent objections that were not raised.
4. "buying_signals": Extract genuine buying signals with brief evidence from the call in {lang_name}.
   - If no buying signals were detected, return ["No clear buying signals detected." localized to {lang_name}].
5. "next_steps": Provide one actionable recommendation for the sales team based on the conversation in {lang_name}.
   - Do NOT claim that a meeting was booked, email was sent, or purchase was completed unless that action explicitly occurred in the transcript.
6. "status": Must be EXACTLY ONE of the 6 canonical values: "Converted", "Demo Requested", "Follow-up Required", "Not Interested", "Interested", "No Response".

Return ONLY a valid JSON object with EXACTLY these string keys:
"summary", "customer_requirements", "objections", "buying_signals", "next_steps", "status".
"""
            base_kwargs = {
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 1500
            }
            response, used_model, used_fallback = generate_content_with_quota_fallback(
                gemini_client=client,
                contents=prompt,
                base_config_kwargs=base_kwargs
            )

            raw = (getattr(response, "text", "") or "").strip()
            if not raw and response and getattr(response, "candidates", None):
                parts = [p.text for c in response.candidates if c.content and c.content.parts for p in c.content.parts if getattr(p, "text", None)]
                raw = " ".join(parts).strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
            parsed = json.loads(raw)

            # Summary validation
            summary = str(parsed.get("summary") or "").strip()
            if not summary:
                summary = f"Call completed with {customer} regarding {product_name}."

            # Customer requirements validation (list of strings)
            reqs_raw = parsed.get("customer_requirements")
            if isinstance(reqs_raw, list):
                customer_requirements = [str(r).strip() for r in reqs_raw if str(r).strip()]
            elif isinstance(reqs_raw, str) and reqs_raw.strip():
                customer_requirements = [r.strip() for r in reqs_raw.split("\n") if r.strip()]
            else:
                customer_requirements = ["None identified."]
            if not customer_requirements:
                customer_requirements = ["None identified."]

            # Objections validation (list of objects with concern and response)
            objs_raw = parsed.get("objections")
            objections = []
            if isinstance(objs_raw, list):
                for item in objs_raw:
                    if isinstance(item, dict):
                        concern = str(item.get("concern") or item.get("objection") or "").strip()
                        resp = str(item.get("response") or item.get("ai_response") or "").strip()
                        if concern and not re.search(r"^none(\s+raised|\s+expressed)?\.?$", concern, re.I):
                            objections.append({"concern": concern, "response": resp or "Addressed during conversation"})
                    elif isinstance(item, str) and item.strip() and not re.search(r"^none(\s+raised|\s+expressed)?\.?$", item, re.I):
                        objections.append({"concern": item.strip(), "response": "Addressed during conversation"})
            elif isinstance(objs_raw, str) and objs_raw.strip() and not re.search(r"^none(\s+raised|\s+expressed)?\.?$", objs_raw, re.I):
                objections.append({"concern": objs_raw.strip(), "response": "Addressed during conversation"})

            # Buying signals validation (list of strings)
            signals_raw = parsed.get("buying_signals")
            if isinstance(signals_raw, list):
                buying_signals = [str(s).strip() for s in signals_raw if str(s).strip()]
            elif isinstance(signals_raw, str) and signals_raw.strip():
                buying_signals = [s.strip() for s in signals_raw.split("\n") if s.strip()]
            else:
                buying_signals = ["No clear buying signals detected."]
            if not buying_signals:
                buying_signals = ["No clear buying signals detected."]

            # Next steps
            next_steps = str(parsed.get("next_steps") or "Review transcript and confirm next milestone.").strip()

            # Status validation
            status = str(parsed.get("status") or "").strip()
            if status not in STATUSES:
                matched = next((s for s in STATUSES if s.lower() == status.lower()), None)
                status = matched or "Interested"

            # Anti-hallucination check for Converted
            if status == "Converted":
                low_speech = customer_speech.lower()
                purchase_terms = ["buy", "purchase", "sign up", "sign me up", "card", "subscribed", "subscribe", "invoice", "payment", "take my card"]
                if not any(t in low_speech for t in purchase_terms):
                    status = "Interested"

            return {
                "summary": summary,
                "customer_requirements": customer_requirements,
                "objections": objections,
                "buying_signals": buying_signals,
                "next_steps": next_steps,
                "status": status,
                "analysis_mode": "ai",
                "model": used_model,
                "fallback_model_used": used_fallback,
                "is_fallback": False
            }
        except Exception as e:
            app.logger.exception(f"Gemini call analysis failed: {e}")
            if not allow_fallback:
                if is_quota_exhausted_error(e):
                    raise GeminiQuotaExhaustedError(f"Gemini quota exhausted: {str(e)}")
                raise GeminiAnalysisError(f"Gemini analysis failed: {str(e)}")

    if not client and not allow_fallback:
        raise GeminiAnalysisError("Gemini API key is not configured. Please add GEMINI_API_KEY in .env to generate AI insights.")

    # Rule-based fallback analysis (used when allow_fallback=True in offline or unit test mode)
    text_lower = customer_speech.lower()
    is_hi = language.startswith("hi")
    is_te = language.startswith("te")

    is_not_interested = any(k in text_lower for k in [
        "not interested", "no thanks", "stop calling", "don't call", "remove me",
        "hate this", "never", "not for us", "waste of time",
        "नहीं चाहिए", "रुचि नहीं", "फोन मत करो", "बंद करो",
        "వద్దు", "ఆసక్తి లేదు", "కాల్ చేయవద్దు", "ఆపండి"
    ])
    is_demo = any(k in text_lower for k in [
        "book a demo", "schedule a demo", "want a demo", "show me a demo",
        "demo please", "see a demo", "product tour", "walkthrough",
        "डेमो", "दिखाइए", "प्रदर्शन",
        "డెమో", "చూపించండి", "టూర్"
    ])
    is_converted = any(k in text_lower for k in [
        "i will buy", "sign me up", "send invoice", "ready to purchase",
        "purchasing now", "start subscription", "take my card",
        "खरीदना है", "सब्सक्राइब", "बिल भेजो",
        "కొంటాను", "సబ్‌స్క్రైబ్", "ఇన్‌వాయిస్ పంపండి"
    ])
    is_follow_up = any(k in text_lower for k in [
        "call me next week", "call me tomorrow", "send me an email", "send details",
        "follow up", "reach out later", "busy right now", "talk to my boss", "discuss with team",
        "कल फोन करो", "ईमेल भेजो", "बाद में बात", "टीम से बात",
        "రేపు కాల్ చేయండి", "ఇమెయిల్ పంపండి", "తర్వాత మాట్లాడదాం", "టీమ్‌తో మాట్లాడాలి"
    ])
    is_interested = any(k in text_lower for k in [
        "interested", "sounds good", "tell me more", "how much", "what features", "pricing",
        "रुचि", "अच्छा है", "और बताओ", "कीमत",
        "ఫీచర్లు", "ధర", "బాగుంది", "ఆసక్తి"
    ])

    if is_not_interested:
        status = "Not Interested"
        if is_hi:
            objections = [{"concern": "ग्राहक ने उत्पाद में सीधी अरुचि व्यक्त की", "response": "निर्णय को विनम्रता से स्वीकार किया गया"}]
            next_steps = "CRM में लीड को 'Not Interested' के रूप में बंद करें।"
            buying_signals = ["कोई स्पष्ट खरीदारी संकेत नहीं मिला।"]
            requirements = ["कोई आवश्यकता पहचानी नहीं गई।"]
        elif is_te:
            objections = [{"concern": "కస్టమర్ నేరుగా ఆసక్తి లేదని తెలిపారు", "response": "నిర్ణయాన్ని గౌరవపూర్వకంగా స్వీకరించడం జరిగింది"}]
            next_steps = "CRMలో లీడ్‌ను 'Not Interested'గా ముగించండి."
            buying_signals = ["ఎటువంటి కొనుగోలు సంకేతాలు కనిపించలేదు."]
            requirements = ["ఎటువంటి అవసరాలు గుర్తించబడలేదు."]
        else:
            objections = [{"concern": "Prospect expressed direct disinterest in the product", "response": "Acknowledged and respected decision politely"}]
            next_steps = "Close lead as not interested / disqualified in CRM."
            buying_signals = ["No clear buying signals detected."]
            requirements = ["None identified."]

    elif is_demo:
        status = "Demo Requested"
        objections = []
        if is_hi:
            next_steps = "संभावित ग्राहक के लिए 20 मिनट का लाइव उत्पाद प्रदर्शन (डेमो) व्यवस्थित करें।"
            buying_signals = ["ग्राहक ने स्पष्ट रूप से उत्पाद प्रदर्शन का अनुरोध किया।"]
            requirements = [user_turns[-1].get("text", "सुविधाओं को देखने में रुचि").strip()]
        elif is_te:
            next_steps = "కస్టమర్ కోసం 20 నిమిషాల ప్రత్యక్ష ప్రదర్శన (డెమో) ఏర్పాటు చేయండి."
            buying_signals = ["కస్టమర్ స్పష్టంగా ఉత్పత్తి డెమోని అభ్యర్థించారు."]
            requirements = [user_turns[-1].get("text", "ఫీచర్లు చూడటానికి ఆసక్తి").strip()]
        else:
            next_steps = "Arrange a 20-minute live demonstration for the prospect."
            buying_signals = ["Prospect explicitly requested a product demonstration."]
            requirements = [user_turns[-1].get("text", "Interested in seeing capabilities").strip()]

    elif is_converted:
        status = "Converted"
        objections = []
        if is_hi:
            next_steps = "ग्राहक ऑनबोर्डिंग प्रक्रिया शुरू करें और भुगतान रसीद भेजें।"
            buying_signals = ["ग्राहक ने स्पष्ट रूप से सदस्यता खरीदने पर सहमति व्यक्त की।"]
            requirements = ["पूर्ण उत्पाद परिनियोजन (Full deployment)।"]
        elif is_te:
            next_steps = "కస్టమర్ ఆన్‌బోర్డింగ్ ప్రక్రియను ప్రారంభించి రసీదు పంపండి."
            buying_signals = ["కస్టమర్ కొనుగోలు చేయడానికి స్పష్టంగా అంగీకరించారు."]
            requirements = ["పూర్తి ప్రొడక్ట్ డిప్లాయ్‌మెంట్."]
        else:
            next_steps = "Trigger customer onboarding sequence and send payment receipt."
            buying_signals = ["Prospect explicitly agreed to purchase or subscribe."]
            requirements = ["Full product deployment."]

    elif is_follow_up:
        status = "Follow-up Required"
        if is_hi:
            objections = [{"concern": "टीम से परामर्श की आवश्यकता या बाद में संपर्क का अनुरोध", "response": "सुविधाजनक समय पर फॉलो-अप की पेशकश की"}]
            next_steps = "ईमेल के माध्यम से अवलोकन विवरण भेजें और फॉलो-अप कॉल शेड्यूल करें।"
            buying_signals = ["ग्राहक ने अतिरिक्त जानकारी और निर्धारित फॉलो-अप का अनुरोध किया।"]
            requirements = ["आगे बढ़ने से पहले आंतरिक टीम से चर्चा आवश्यक।"]
        elif is_te:
            objections = [{"concern": "బృందాన్ని సంప్రదించాలి లేదా తర్వాత మాట్లాడాలని కోరారు", "response": "అనుకూలమైన సమయంలో మాట్లాడటానికి అంగీకరించారు"}]
            next_steps = "ఇమెయిల్ ద్వారా సమాచారాన్ని పంపి ఫాలో-అప్ కాల్ షెడ్యూల్ చేయండి."
            buying_signals = ["కస్టమర్ వివరాలు మరియు తదుపరి సంభాషణను కోరారు."]
            requirements = ["అంతర్గత బృందం అనుమతి అవసరం."]
        else:
            objections = [{"concern": "Needs to consult team or requests delayed contact", "response": "Offered to follow up at a convenient time"}]
            next_steps = "Send overview information via email and schedule a follow-up call."
            buying_signals = ["Customer requested further information and scheduled follow-up."]
            requirements = ["Needs internal team alignment before moving forward."]

    elif is_interested:
        status = "Interested"
        objections = []
        if is_hi:
            next_steps = "मूल्य निर्धारण जानकारी भेजें और प्रारंभिक परिचयात्मक बैठक तय करें।"
            buying_signals = ["ग्राहक ने सुविधाओं और मूल्य निर्धारण के बारे में पूछा।"]
            requirements = [user_turns[-1].get("text", "वर्कफ़्लो स्वचालन क्षमताओं की खोज").strip()]
        elif is_te:
            next_steps = "ధర వివరాలు పంపి పరిచయ సమావేశాన్ని షెడ్యూల్ చేయండి."
            buying_signals = ["కస్టమర్ ఫీచర్లు మరియు ధర వివరాల గురించి అడిగారు."]
            requirements = [user_turns[-1].get("text", "వర్క్‌ఫ్లో ఆటోమేషన్ ఫీచర్లను పరిశీలిస్తున్నారు").strip()]
        else:
            next_steps = "Send pricing information and schedule introductory discovery session."
            buying_signals = ["Prospect asked about features and pricing tiers."]
            requirements = [user_turns[-1].get("text", "Exploring workflow automation capabilities").strip()]

    else:
        status = "Interested"
        objections = []
        if is_hi:
            next_steps = "बातचीत की समीक्षा करें और अगले मील के पत्थर की पुष्टि करें।"
            buying_signals = ["संवादात्मक बातचीत में शामिल रहे।"]
            requirements = ["उत्पाद क्षमताओं का सामान्य अन्वेषण।"]
        elif is_te:
            next_steps = "సంభాషణను సమీక్షించి తదుపరి దశను నిర్ధారించండి."
            buying_signals = ["సంభాషణలో చురుకుగా పాల్గొన్నారు."]
            requirements = ["సాధారణ ప్రొడక్ట్ అన్వేషణ."]
        else:
            next_steps = "Review conversation transcript and verify next contact milestone."
            buying_signals = ["Engaged in conversational dialogue."]
            requirements = ["General exploration of product capabilities."]

    if is_hi:
        summary = (
            f"[स्थानीय प्रतिलेख विश्लेषण — नियम-आधारित फ़ॉलबैक]\n"
            f"{customer} के साथ {product_name} के संबंध में बिक्री चर्चा। "
            f"ग्राहक ने {len(user_turns)} बार बातचीत में भाग लिया। "
            f"प्रतिलेख साक्ष्य के आधार पर स्थिति '{status}' निर्धारित की गई। "
            f"(यह स्थानीय नियम-आधारित विश्लेषण है; AI-जनरेटेड सारांश नहीं)।"
        )
    elif is_te:
        summary = (
            f"[లోకల్ ట్రాన్స్‌క్రిప్ట్ విశ్లేషణ — నియమాల ఆధారిత ఫాల్‌బ్యాక్]\n"
            f"{customer} తో {product_name} గురించి అమ్మకాల సంభాషణ. "
            f"కస్టమర్ {len(user_turns)} సార్లు స్పందించారు. "
            f"ట్రాన్స్‌క్రిప్ట్ ఆధారంగా సేల్స్ స్థితి '{status}'గా వర్గీకరించబడింది. "
            f"(ఇది లోకల్ నియమాల ఆధారిత విశ్లేషణ; AI రూపొందించిన సారాంశం కాదు)."
        )
    else:
        summary = (
            f"[Local Transcript Analysis — Rule-Based Fallback]\n"
            f"Discussion with {customer} regarding {product_name}. "
            f"The prospect participated in {len(user_turns)} conversational turn(s). "
            f"Sales status classified as '{status}' based on dialogue evidence. "
            f"(Rule-based local fallback analysis; not an AI-generated summary)."
        )

    return {
        "summary": summary,
        "customer_requirements": requirements,
        "objections": objections,
        "buying_signals": buying_signals,
        "next_steps": next_steps,
        "status": status,
        "analysis_mode": "local_fallback",
        "is_fallback": True
    }


@app.get("/api/history")
def history():
    search_query = request.args.get("q", "").strip().lower()
    status_filter = request.args.get("status", "").strip()

    sql = "SELECT * FROM calls WHERE 1=1"
    params = []

    if status_filter and status_filter in STATUSES:
        sql += " AND sales_status = ?"
        params.append(status_filter)

    if search_query:
        sql += " AND (LOWER(customer) LIKE ? OR LOWER(company) LIKE ? OR LOWER(product) LIKE ?)"
        pattern = f"%{search_query}%"
        params.extend([pattern, pattern, pattern])

    sql += " ORDER BY started_at DESC LIMIT 150"

    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()

    items = []
    for r in rows:
        d = dict(r)
        try:
            d["transcript"] = json.loads(d["transcript"] or "[]")
        except Exception:
            d["transcript"] = []

        for key in ("customer_requirements", "objections", "buying_signals"):
            val = d.get(key) or ""
            if isinstance(val, str) and (val.strip().startswith("[") or val.strip().startswith("{")):
                try:
                    d[key] = json.loads(val)
                except Exception:
                    pass

        # Standardize property names for frontend convenience
        d["customerRequirements"] = d.get("customer_requirements")
        d["buyingSignals"] = d.get("buying_signals")
        d["nextSteps"] = d.get("next_steps")
        d["status"] = d.get("sales_status", "No Response")
        d["salesStatus"] = d.get("sales_status", "No Response")
        d["processingStatus"] = d.get("processing_status", "completed")
        d["analysisMode"] = d.get("analysis_mode", "ai")
        d["analysis_mode"] = d.get("analysis_mode", "ai")
        d["isFallback"] = (d.get("analysis_mode") == "local_fallback" or d.get("processing_status") == "quota_exhausted")
        d["quotaExhausted"] = (d.get("processing_status") == "quota_exhausted")
        d["recordingUrl"] = f"/api/calls/{d['id']}/recording" if d.get("recording") else ""
        items.append(d)

    return jsonify(items)


@app.get("/api/calls/<call_id>")
def get_call(call_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()

    if not row:
        return jsonify({"error": "Call record not found"}), 404

    d = dict(row)
    try:
        d["transcript"] = json.loads(d["transcript"] or "[]")
    except Exception:
        d["transcript"] = []

    for key in ("customer_requirements", "objections", "buying_signals"):
        val = d.get(key) or ""
        if isinstance(val, str) and (val.strip().startswith("[") or val.strip().startswith("{")):
            try:
                d[key] = json.loads(val)
            except Exception:
                pass

    d["customerRequirements"] = d.get("customer_requirements")
    d["buyingSignals"] = d.get("buying_signals")
    d["nextSteps"] = d.get("next_steps")
    d["status"] = d.get("sales_status", "No Response")
    d["salesStatus"] = d.get("sales_status", "No Response")
    d["processingStatus"] = d.get("processing_status", "completed")
    d["analysisMode"] = d.get("analysis_mode", "ai")
    d["analysis_mode"] = d.get("analysis_mode", "ai")
    d["isFallback"] = (d.get("analysis_mode") == "local_fallback" or d.get("processing_status") == "quota_exhausted")
    d["quotaExhausted"] = (d.get("processing_status") == "quota_exhausted")
    d["recordingUrl"] = f"/api/calls/{call_id}/recording" if d.get("recording") else ""
    return jsonify(d)


def _delete_call_record(conn, call_id):
    """
    Safely deletes a call record from SQLite and unlinks its associated audio recording file if present.
    Returns True if the record was found and deleted, False if not found.
    """
    row = conn.execute("SELECT recording FROM calls WHERE id = ?", (call_id,)).fetchone()
    if not row:
        return False

    rec = row["recording"]
    if rec:
        safe_rec = secure_filename(os.path.basename(rec))
        if safe_rec:
            target = RECORDINGS_DIR / safe_rec
            if target.exists():
                try:
                    target.unlink()
                except Exception as e:
                    app.logger.warning(f"Could not delete recording file '{safe_rec}': {e}")

    conn.execute("DELETE FROM calls WHERE id = ?", (call_id,))
    return True


@app.route("/api/calls/bulk-delete", methods=["POST", "DELETE"])
def bulk_delete_calls():
    data = request.get_json(silent=True) or {}
    call_ids = data.get("call_ids")
    if call_ids is None:
        call_ids = data.get("ids")

    if not isinstance(call_ids, list) or len(call_ids) == 0:
        return jsonify({
            "success": False,
            "error": "No call IDs provided for deletion",
            "deleted_count": 0,
            "deleted_ids": []
        }), 400

    # Sanitize and deduplicate while preserving order
    valid_ids = []
    seen = set()
    for cid in call_ids:
        if isinstance(cid, str):
            cid_clean = cid.strip()
            if cid_clean and cid_clean not in seen:
                seen.add(cid_clean)
                valid_ids.append(cid_clean)

    if not valid_ids:
        return jsonify({
            "success": False,
            "error": "No valid call IDs provided for deletion",
            "deleted_count": 0,
            "deleted_ids": []
        }), 400

    deleted_ids = []
    not_found_ids = []

    with get_db() as conn:
        for cid in valid_ids:
            if _delete_call_record(conn, cid):
                deleted_ids.append(cid)
            else:
                not_found_ids.append(cid)

    if not deleted_ids:
        return jsonify({
            "success": False,
            "error": "None of the specified calls were found",
            "deleted_count": 0,
            "deleted_ids": [],
            "not_found_ids": not_found_ids
        }), 404

    return jsonify({
        "success": True,
        "message": f"Successfully deleted {len(deleted_ids)} call(s).",
        "deleted_count": len(deleted_ids),
        "deleted_ids": deleted_ids,
        "not_found_ids": not_found_ids
    }), 200


@app.delete("/api/calls/<call_id>")
def delete_call(call_id):
    with get_db() as conn:
        if not _delete_call_record(conn, call_id):
            return jsonify({"error": "Call not found"}), 404

    return jsonify({"success": True, "message": f"Call {call_id} deleted."})


@app.get("/api/calls/<call_id>/recording")
def stream_call_recording(call_id):
    with get_db() as conn:
        row = conn.execute("SELECT recording FROM calls WHERE id = ?", (call_id,)).fetchone()

    if not row or not row["recording"]:
        return jsonify({"error": "No recording associated with this call"}), 404

    safe_name = secure_filename(os.path.basename(row["recording"]))
    return send_from_directory(RECORDINGS_DIR, safe_name, as_attachment=False, mimetype="audio/webm")


@app.get("/recordings/<path:filename>")
def legacy_recording(filename):
    safe_name = secure_filename(os.path.basename(filename))
    return send_from_directory(RECORDINGS_DIR, safe_name, as_attachment=False, mimetype="audio/webm")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
