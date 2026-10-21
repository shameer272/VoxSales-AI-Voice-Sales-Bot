# VoxSales — AI Voice Sales Agent
### Internship Assignment for Sankar Group (48-Hour Challenge)

VoxSales is a production-minded, full-stack AI voice sales agent built for modern SaaS products. It engages in real-time, bidirectional voice conversations with prospective customers, dynamically discovers business pain points, presents product value propositions, handles objections respectfully, captures audio recordings with explicit consent, and automatically synthesizes call intelligence into executive summaries and sales status classifications.

---

## Key Capabilities & Features

1. **Autonomous Real-Time Voice Conversation**
   - Live speech recognition via Web Speech API (`SpeechRecognition` / `webkitSpeechRecognition`).
   - Dynamic AI responses powered by the official **OpenAI Python SDK** (`gpt-4o-mini`).
   - Natural spoken responses via browser `window.speechSynthesis` with language matching.
   - Intelligent conversation loop with mutual exclusion to avoid transcribing self-spoken audio.
   - Multilingual support for **English (US & India)**, **Hindi (हिंदी)**, and **Telugu (తెలుగు)**.
   - Full keyboard text input fallback mode for browsers without speech recognition.

2. **Configurable SaaS Product Profiles**
   - Pre-configured SaaS catalog:
     - **FlowPilot**: Workflow automation & AI orchestration platform.
     - **CloudShield**: Autonomous cloud security & CSPM compliance auditing.
     - **DataPulse**: Real-time sales intelligence & pipeline forecasting.
   - In-app product specification viewer detailing description, features, pricing, target audience, and limitations.
   - Strict system prompt guardrails preventing the AI from hallucinating unsupported features or fake pricing.

3. **Consent-Gated Audio Recording**
   - Transparent, affirmative recording consent workflow (calls cannot begin without explicit user consent).
   - Real audio capture using browser `MediaRecorder` (`audio/webm;codecs=opus`).
   - Safe server-side uploads to `/recordings/` with sanitized filenames and path-traversal safeguards.
   - Inline audio player in the dashboard for post-call playback and review.

4. **Automated Call Intelligence & Sales Status**
   - Post-call analysis strictly classifies each call into one of the **six canonical statuses**:
     - `Converted`: Explicit agreement to purchase or sign up.
     - `Interested`: Genuine interest without final transaction commitment.
     - `Demo Requested`: Explicit request for a product walkthrough or live demonstration.
     - `Follow-up Required`: Request for email documentation or scheduled callback.
     - `Not Interested`: Clear disinterest or rejection.
     - `No Response`: No meaningful customer dialogue captured.
   - Structured extraction of **Executive Summary**, **Discovered Requirements**, **Objections Handled**, **Buying Signals**, and **Recommended Next Steps**.

5. **Persistent SQLite Call History**
   - Calls, transcripts, summaries, and recording metadata persist across server restarts.
   - Real-time search by prospect name, company, or product.
   - Filter history by sales qualification status.
   - One-click inspection to load past calls into the active transcript and audio player.

---

## Architecture & Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | Semantic HTML5, Vanilla CSS3 (Dark Navy SaaS Design System), Vanilla JS (ES6+) |
| **Backend** | Python 3.10+, Flask, Gunicorn WSGI |
| **Database** | SQLite (Parameterized SQL queries, automated schema migrations) |
| **AI Engine** | OpenAI API via official `openai>=1.0.0` Python SDK (`gpt-4o-mini`) |
| **Speech-to-Text** | Browser Web Speech API (`webkitSpeechRecognition`) + Accessible Text Fallback |
| **Text-to-Speech** | Browser Web SpeechSynthesis API |
| **Audio Capture** | Browser MediaRecorder API (`audio/webm`) |
| **Configuration** | `python-dotenv` with `.env` isolation |

---

## Local Setup Instructions (Windows PowerShell)

Follow these step-by-step instructions to run VoxSales on Windows:

### 1. Open Windows PowerShell in the Project Directory
```powershell
cd c:\Users\Shameer\Downloads\ai-voice-sales-bot\ai-voice-sales-bot
```

### 2. Create and Activate Virtual Environment
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an ExecutionPolicy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first)*.

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` in any text editor and configure your OpenAI API key:
```ini
OPENAI_API_KEY=your_actual_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
PORT=5000
```

> **Important Note on API Billing:**
> An OpenAI API key is separate from a consumer ChatGPT Plus subscription. Real AI generation requires an active OpenAI platform project with API usage credits. If no API key is supplied, VoxSales automatically engages its built-in rule-based fallback engine so you can still demonstrate the full UI/voice workflow offline.

### 5. Run Automated Tests
```powershell
python -m pytest test_app.py -v
```

### 6. Start the Application
```powershell
python app.py
```

### 7. Access the Dashboard
Open **Google Chrome** or **Microsoft Edge** and navigate to:
```
http://127.0.0.1:5000
```
Allow microphone permissions when prompted by your browser.

---

## Demonstration & Video Recording Checklist (For Internship Review)

When recording your project video demonstration for the Sankar Group review:

1. **Introduction (30s)**
   - Introduce VoxSales as the AI Voice Sales Agent for Sankar Group.
   - Point out the dark navy dashboard, real-time connection badges, and the selected SaaS product (`FlowPilot`).
2. **Review Product Specs (30s)**
   - Click **View Specs** to display the modal showing FlowPilot features, pricing tiers ($49 Starter, $199 Pro), target audience, and limitations.
3. **Placing a Live Sales Call (2 mins)**
   - Enter prospect name (e.g. `Priya Sharma`) and company (`Apex Logistics`).
   - Check the **Mandatory Recording Consent** box (demonstrating compliance).
   - Click **▶ Start Call**.
   - Show the browser prompt for microphone permission.
   - Listen to the AI speak its greeting aloud: *"Hello Priya! Thanks for taking my call..."*
   - Speak into your microphone (e.g., *"We're struggling with too many manual tickets and need Slack automation. What are your pricing plans?"*).
   - Highlight the live transcript appearing in real-time, the animated voice wave visualizer, and the AI's spoken reply.
   - Ask: *"Can you schedule a demo for our team tomorrow?"*
   - Listen to the AI confirm the demo request.
4. **Call Finalization & Intelligence (1 min)**
   - Click **■ End Call**.
   - Watch the Call Intelligence section populate automatically:
     - Sales Status badge: `DEMO REQUESTED` (in purple).
     - Executive Summary synthesized by AI.
     - Discovered Customer Requirements.
     - Objections Handled.
     - Buying Signals.
     - Recommended Next Steps.
   - Play back the call audio recording using the inline audio player.
5. **Call History & Search (30s)**
   - Inspect the Call History panel.
   - Filter by status dropdown or search by prospect name.
   - Click a historical call to demonstrate persistent transcript reload.
6. **Closing (15s)**
   - Highlight the codebase structure, test suite results (all tests passing), and Render deployment readiness.

---

## Deployment Instructions (Render)

1. **Push to GitHub**
   - Ensure `.env`, `salesbot.db`, `.venv/`, and `recordings/` are excluded by `.gitignore`.
   ```bash
   git add .
   git commit -m "feat: complete VoxSales AI Voice Sales Agent"
   git push origin main
   ```
2. **Create Web Service on Render**
   - Connect your GitHub repository.
   - Environment: `Python`.
   - Build Command: `pip install -r requirements.txt`.
   - Start Command: `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`.
3. **Configure Environment Secrets**
   - Add `OPENAI_API_KEY` under Environment Variables in the Render dashboard.
   - Add `OPENAI_MODEL` = `gpt-4o-mini`.
4. **Deploy and Verify**
   - Render will deploy the service and run health checks at `/api/health`.
   - Open your public Render HTTPS URL in Chrome or Edge.
   - Note: Because browser microphone access requires a secure context, HTTPS provided by Render is required in production.

---

## Security & Reliability Considerations

- **Credential Isolation:** API keys are never passed to the frontend or exposed in `/api/config`.
- **Path Traversal Protection:** Audio file streaming enforces `secure_filename` and basename checks.
- **Fail-Safe Fallback:** If OpenAI network timeouts occur, the conversation falls back gracefully without breaking the user experience.
- **Microphone Fallback:** If a client browser lacks Web Speech API support, the interactive text bar enables complete conversational testing.
