"""
Unit and integration tests for VoxSales AI Voice Sales Bot backend.
"""

import json
import pytest
import io
from unittest.mock import MagicMock, patch
from app import (
    app,
    STATUSES,
    analyze_call_transcript,
    RECORDINGS_DIR,
    get_db,
    get_gemini_model,
    get_gemini_fallback_model,
    get_gemini_fallback_models,
    generate_content_with_quota_fallback,
    GeminiQuotaExhaustedError,
    GeminiAnalysisError
)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def default_mock_gemini_client(monkeypatch, request):
    """
    Ensure automated unit tests do not consume the user's Gemini API quota or rate limits.
    Only test_gemini_live_smoke_test uses the real Gemini API client.
    """
    if "live_smoke" in request.node.name or "client_initialization" in request.node.name:
        return
    monkeypatch.setattr("app.get_gemini_client", lambda: None)


def make_mock_gemini_response(content):
    """Helper to return a mock Gemini generate_content response."""
    mock_resp = MagicMock()
    if isinstance(content, (dict, list)):
        mock_resp.text = json.dumps(content)
    else:
        mock_resp.text = str(content)
    # Add backward compatibility fields
    mock_choice = MagicMock()
    mock_choice.message.content = mock_resp.text
    mock_resp.choices = [mock_choice]
    return mock_resp


def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"


def test_config_endpoint(client):
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.get_json()
    assert "aiConfigured" in data
    assert "model" in data
    assert data.get("aiProvider") == "gemini"
    assert "products" in data
    assert "FlowPilot" in data["products"]
    assert "statuses" in data
    assert len(data["statuses"]) == 6
    # Ensure no secret API keys leaked
    assert "GEMINI_API_KEY" not in str(data)
    assert "OPENAI_API_KEY" not in str(data)
    assert "sk-" not in str(data)


def test_call_creation_requires_consent(client):
    # Without consent should fail
    res = client.post("/api/calls/start", json={
        "customer": "Test User",
        "consent": False
    })
    assert res.status_code == 400
    assert "consent is required" in res.get_json()["error"]

    # With consent should succeed
    res2 = client.post("/api/calls/start", json={
        "customer": "Valid User",
        "company": "Test Co",
        "product": "FlowPilot",
        "consent": True
    })
    assert res2.status_code == 200
    data = res2.get_json()
    assert "id" in data
    assert data["customer"] == "Valid User"


def test_chat_endpoint(client):
    # Test valid message
    res = client.post("/api/chat", json={
        "message": "What is the pricing for FlowPilot?",
        "product": "FlowPilot",
        "language": "en-US",
        "customer": "Test Prospect"
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "reply" in data
    assert len(data["reply"]) > 5

    # Test empty message fails cleanly
    res_empty = client.post("/api/chat", json={"message": ""})
    assert res_empty.status_code == 400


def test_status_classifications():
    """Verify that status validation and heuristic fallback maps to all 6 required statuses."""
    # 1. No Response
    res1 = analyze_call_transcript([], "FlowPilot", "en-US", "Alice", allow_fallback=True)
    assert res1["status"] == "No Response"
    assert "None identified." in res1["customer_requirements"]

    # 2. Not Interested
    res2 = analyze_call_transcript([
        {"role": "assistant", "text": "Hello, how can FlowPilot help?"},
        {"role": "user", "text": "Not interested at all, stop calling us."}
    ], "FlowPilot", "en-US", "Bob", allow_fallback=True)
    assert res2["status"] == "Not Interested"

    # 3. Demo Requested
    res3 = analyze_call_transcript([
        {"role": "assistant", "text": "Would you like to see how it works?"},
        {"role": "user", "text": "Yes, can we schedule a demo for tomorrow?"}
    ], "FlowPilot", "en-US", "Charlie", allow_fallback=True)
    assert res3["status"] == "Demo Requested"

    # 4. Converted
    res4 = analyze_call_transcript([
        {"role": "assistant", "text": "Ready to get started?"},
        {"role": "user", "text": "Yes, sign me up right now, I will buy the Pro plan."}
    ], "FlowPilot", "en-US", "Diana", allow_fallback=True)
    assert res4["status"] == "Converted"

    # 5. Follow-up Required
    res5 = analyze_call_transcript([
        {"role": "assistant", "text": "Does this look good?"},
        {"role": "user", "text": "I need to discuss with team, please send me an email and call me next week."}
    ], "FlowPilot", "en-US", "Edward", allow_fallback=True)
    assert res5["status"] == "Follow-up Required"

    # 6. Interested
    res6 = analyze_call_transcript([
        {"role": "assistant", "text": "We automate workflow tickets."},
        {"role": "user", "text": "That sounds very interesting, tell me more about features."}
    ], "FlowPilot", "en-US", "Fiona", allow_fallback=True)
    assert res6["status"] == "Interested"


def test_full_call_lifecycle_and_history(client):
    mock_ai_analysis = {
        "summary": "Alex discussed migrating cloud assets from AWS to Azure and evaluating automated SOC2 compliance evidence collection.",
        "customer_requirements": ["Automated SOC2 Type II audit report generation", "Real-time AWS and Azure asset discovery"],
        "objections": [{"concern": "Concerned about configuration drift in legacy accounts", "response": "Demonstrated autonomous drift remediation capabilities"}],
        "buying_signals": ["Inquired about growth tier pricing and requested onboarding timeline"],
        "next_steps": "Arrange a 20-minute technical demonstration with cloud security team.",
        "status": "Demo Requested"
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = make_mock_gemini_response(mock_ai_analysis)

    with patch("app.get_gemini_client", return_value=mock_client):
        # 1. Start call
        start_res = client.post("/api/calls/start", json={
            "customer": "Lifecycle Test User",
            "company": "Global Tech",
            "product": "CloudShield",
            "language": "en-US",
            "consent": True
        })
        assert start_res.status_code == 200
        call_id = start_res.get_json()["id"]

        # 2. End call with transcript
        end_res = client.post("/api/calls/end", json={
            "callId": call_id,
            "durationSeconds": 45,
            "transcript": [
                {"role": "assistant", "text": "Welcome to CloudShield. What are your main security pain points?"},
                {"role": "user", "text": "We need help passing SOC2 Type II audit automation and evaluating pricing."}
            ]
        })
        assert end_res.status_code == 200
        analysis = end_res.get_json()
        assert analysis["success"] is True
        assert analysis["status"] == "Demo Requested"
        assert len(analysis["summary"]) > 0

        # 3. Retrieve call from /api/calls/<id>
        get_res = client.get(f"/api/calls/{call_id}")
        assert get_res.status_code == 200
        call_data = get_res.get_json()
        assert call_data["customer"] == "Lifecycle Test User"
        assert call_data["duration_seconds"] == 45
        assert len(call_data["transcript"]) == 2
        assert call_data["status"] == "Demo Requested"
        assert isinstance(call_data["customer_requirements"], list)

        # 4. Search call history
        hist_res = client.get("/api/history?q=Lifecycle")
        assert hist_res.status_code == 200
        items = hist_res.get_json()
        assert any(i["id"] == call_id for i in items)

        # 5. Delete call
        del_res = client.delete(f"/api/calls/{call_id}")
        assert del_res.status_code == 200
        assert del_res.get_json()["success"] is True

        # 6. Verify deleted
        get_after = client.get(f"/api/calls/{call_id}")
        assert get_after.status_code == 404


def test_call_intelligence_structured_fields(client):
    """Verify that intelligence returned by /api/calls/end has structured requirements, objections, signals, next steps."""
    mock_ai_analysis = {
        "summary": "Sarah from Apex Dynamics explored workflow automation to eliminate 15 hours of manual data entry between Slack and Salesforce.",
        "customer_requirements": [
            "15 hours/week lost to manual copying between Slack and Salesforce",
            "Needs reliable pre-built CRM and communication connectors"
        ],
        "objections": [
            {
                "concern": "Wondered if the Pro tier cost aligns with team budget",
                "response": "Explained the $199/month tier and ROI savings"
            }
        ],
        "buying_signals": [
            "Asked specifically about Pro tier pricing",
            "Requested a personalized product demo"
        ],
        "next_steps": "Send Pro tier pricing sheet and coordinate calendar for product demonstration.",
        "status": "Demo Requested"
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = make_mock_gemini_response(mock_ai_analysis)

    with patch("app.get_gemini_client", return_value=mock_client):
        start_res = client.post("/api/calls/start", json={
            "customer": "Sarah Jenkins",
            "company": "Apex Dynamics",
            "product": "FlowPilot",
            "consent": True
        })
        call_id = start_res.get_json()["id"]

        transcript = [
            {"role": "assistant", "text": "Hello Sarah, thanks for taking my call! What is your biggest workflow challenge right now?"},
            {"role": "user", "text": "Our team spends 15 hours a week manually copying leads between Slack and Salesforce."},
            {"role": "assistant", "text": "FlowPilot has over 150 prebuilt enterprise connectors that automate that completely."},
            {"role": "user", "text": "How much does the Pro tier cost, and can we schedule a demo?"}
        ]

        end_res = client.post("/api/calls/end", json={
            "callId": call_id,
            "durationSeconds": 92,
            "transcript": transcript
        })
        assert end_res.status_code == 200
        data = end_res.get_json()

        assert data["success"] is True
        assert "summary" in data and len(data["summary"]) > 0
        assert "customerRequirements" in data
        assert isinstance(data["customerRequirements"], list)
        assert len(data["customerRequirements"]) == 2
        assert "objections" in data
        assert isinstance(data["objections"], list)
        assert len(data["objections"]) == 1
        assert "concern" in data["objections"][0]
        assert "response" in data["objections"][0]
        assert "buyingSignals" in data
        assert isinstance(data["buyingSignals"], list)
        assert len(data["buyingSignals"]) == 2
        assert "nextSteps" in data and len(data["nextSteps"]) > 0
        assert data["status"] in STATUSES
        assert data["status"] == "Demo Requested"

        # Verify persistence in SQLite
        get_res = client.get(f"/api/calls/{call_id}")
        assert get_res.status_code == 200
        saved = get_res.get_json()
        assert saved["summary"] == data["summary"]
        assert saved["status"] == data["status"]
        assert isinstance(saved["customer_requirements"], list)
        assert saved["processing_status"] == "completed"

        # Test retry analysis endpoint
        retry_res = client.post(f"/api/calls/{call_id}/analyze")
        assert retry_res.status_code == 200
        retried = retry_res.get_json()
        assert retried["success"] is True
        assert retried["callId"] == call_id
        assert retried["status"] == "Demo Requested"


def test_quota_exhaustion_saves_call_transcript_recording_and_fallback(client):
    """
    Verify that when OpenAI returns insufficient_quota or credit_balance_exhausted:
    - The completed call, transcript, and recording are still saved in SQLite.
    - Local fallback for summary, customer requirements, objections, buying signals,
      recommended next action, and sales status are generated using the saved transcript.
    - The fallback is clearly labelled and does NOT claim to be an AI-generated summary.
    - processing_status is marked as 'quota_exhausted' and analysis_mode is 'local_fallback'.
    """
    start_res = client.post("/api/calls/start", json={
        "customer": "Ravi Kumar",
        "company": "Hyderabad Tech Labs",
        "product": "FlowPilot",
        "language": "en-US",
        "consent": True
    })
    assert start_res.status_code == 200
    call_id = start_res.get_json()["id"]

    transcript = [
        {"role": "assistant", "text": "Hello Ravi, welcome to FlowPilot! How can I assist with your workflow automation today?"},
        {"role": "user", "text": "Hi, what is the pricing for FlowPilot?"},
        {"role": "assistant", "text": "FlowPilot pricing starts at $49/month for Starter, and $199/month for Pro."},
        {"role": "user", "text": "That sounds good, but is it too expensive for an early stage startup?"},
        {"role": "assistant", "text": "We offer a 14-day full trial and scalable tiers to fit every budget."},
        {"role": "user", "text": "Great, can we schedule a demo for my team tomorrow?"}
    ]

    # Simulate Gemini 429 quota exhaustion (RESOURCE_EXHAUSTED / quota exceeded)
    mock_quota_error = Exception(
        "429 RESOURCE_EXHAUSTED: You have exceeded your current quota. Check your Gemini API plan and billing details."
    )
    mock_quota_client = MagicMock()
    mock_quota_client.models.generate_content.side_effect = mock_quota_error

    audio_data = io.BytesIO(b"RIFFmockwebmdataforaudiorecording")

    with patch("app.get_gemini_client", return_value=mock_quota_client):
        end_res = client.post(
            "/api/calls/end",
            data={
                "callId": call_id,
                "durationSeconds": 68,
                "transcript": json.dumps(transcript),
                "customer": "Ravi Kumar",
                "company": "Hyderabad Tech Labs",
                "product": "FlowPilot",
                "language": "en-US",
                "recording": (audio_data, "call_test.webm")
            },
            content_type="multipart/form-data"
        )

        assert end_res.status_code == 200
        data = end_res.get_json()
        assert data["success"] is True
        assert data["callId"] == call_id
        assert data["durationSeconds"] == 68
        assert data["processing_status"] == "quota_exhausted"
        assert data["analysis_mode"] == "local_fallback"
        assert data["isFallback"] is True
        assert data["quotaExhausted"] is True

        # Verify summary is present, clearly labelled as local fallback, and NOT claiming to be AI
        summary = data["summary"]
        assert len(summary) > 20
        assert "Local" in summary or "Fallback" in summary or "Rule-Based" in summary
        assert "not an ai-generated summary" in summary.lower()

        # Verify all intelligence fields are populated from the saved transcript
        assert "customerRequirements" in data
        assert isinstance(data["customerRequirements"], list)
        assert len(data["customerRequirements"]) > 0

        assert "objections" in data
        assert isinstance(data["objections"], list)

        assert "buyingSignals" in data
        assert isinstance(data["buyingSignals"], list)
        assert len(data["buyingSignals"]) > 0

        assert "nextSteps" in data and len(data["nextSteps"]) > 0
        assert data["status"] in STATUSES
        assert data["status"] == "Demo Requested"

        # Verify audio recording was saved
        assert data["recording"]
        assert f"/api/calls/{call_id}/recording" in data["recordingUrl"]

        # Verify persistence in SQLite
        get_res = client.get(f"/api/calls/{call_id}")
        assert get_res.status_code == 200
        saved = get_res.get_json()
        assert saved["id"] == call_id
        assert saved["processing_status"] == "quota_exhausted"
        assert saved["analysis_mode"] == "local_fallback"
        assert saved["isFallback"] is True
        assert len(saved["transcript"]) == 6
        assert saved["recording"] != ""
        assert saved["status"] == "Demo Requested"
        assert "not an ai-generated summary" in saved["summary"].lower()


def test_api_failure_and_retry_behavior(client):
    """
    Verify full quota-exhaustion and retry flow:
    - Quota exhaustion saves call, transcript, recording, and local fallback without data loss.
    - Preserves call history and prevents duplicate records.
    - When API credits become available, retrying AI analysis updates the record in-place to 'completed'.
    """
    start_res = client.post("/api/calls/start", json={
        "customer": "Failure Test User",
        "company": "Beta Corp",
        "product": "FlowPilot",
        "consent": True
    })
    call_id = start_res.get_json()["id"]

    transcript = [
        {"role": "assistant", "text": "Welcome to FlowPilot."},
        {"role": "user", "text": "Tell me about pricing options."}
    ]

    # 1. Simulate Gemini API quota exhaustion
    mock_failing_client = MagicMock()
    mock_failing_client.models.generate_content.side_effect = Exception(
        "429 RESOURCE_EXHAUSTED: Rate limit exceeded or quota exhausted"
    )

    with patch("app.get_gemini_client", return_value=mock_failing_client):
        end_res = client.post("/api/calls/end", json={
            "callId": call_id,
            "durationSeconds": 30,
            "transcript": transcript
        })
        assert end_res.status_code == 200
        data = end_res.get_json()
        assert data["success"] is True
        assert data["processing_status"] == "quota_exhausted"
        assert data["analysis_mode"] == "local_fallback"
        assert data["isFallback"] is True
        assert "not an ai-generated summary" in data["summary"].lower()

        # Verify call and transcript are preserved in SQLite
        get_res = client.get(f"/api/calls/{call_id}")
        assert get_res.status_code == 200
        saved = get_res.get_json()
        assert saved["processing_status"] == "quota_exhausted"
        assert saved["analysis_mode"] == "local_fallback"
        assert len(saved["transcript"]) == 2

    # Verify call history count before retry to prevent duplicates
    history_before = client.get("/api/history").get_json()
    count_before = len([item for item in history_before if item["id"] == call_id])
    assert count_before == 1

    # 2. Simulate Retry when credits are still exhausted
    with patch("app.get_gemini_client", return_value=mock_failing_client):
        still_failing_retry = client.post(f"/api/calls/{call_id}/retry-analysis")
        assert still_failing_retry.status_code == 402
        retry_err = still_failing_retry.get_json()
        assert retry_err["success"] is False
        assert retry_err["quota_exhausted"] is True
        assert "quota is exhausted" in retry_err["error"].lower()

        # Verify SQLite still has the preserved local fallback and transcript intact
        get_res_still = client.get(f"/api/calls/{call_id}")
        assert get_res_still.status_code == 200
        assert get_res_still.get_json()["processing_status"] == "quota_exhausted"
        assert len(get_res_still.get_json()["transcript"]) == 2

    # 3. Simulate Retry after AI credits become available
    mock_recovered_client = MagicMock()
    mock_recovered_client.models.generate_content.return_value = make_mock_gemini_response({
        "summary": "Prospect inquired about available pricing tiers for FlowPilot.",
        "customer_requirements": ["Exploring affordable entry pricing options"],
        "objections": [],
        "buying_signals": ["Inquired directly about pricing"],
        "next_steps": "Send official pricing schedule.",
        "status": "Interested"
    })

    with patch("app.get_gemini_client", return_value=mock_recovered_client):
        retry_res = client.post(f"/api/calls/{call_id}/retry-analysis")
        assert retry_res.status_code == 200
        retry_data = retry_res.get_json()
        assert retry_data["success"] is True
        assert retry_data["processing_status"] == "completed"
        assert retry_data["analysis_mode"] == "ai"
        assert retry_data["isFallback"] is False
        assert retry_data["status"] == "Interested"
        assert "Prospect inquired about available pricing tiers" in retry_data["summary"]

        # Verify SQLite now reflects 'completed' status and 'ai' analysis_mode
        get_res2 = client.get(f"/api/calls/{call_id}")
        assert get_res2.status_code == 200
        saved2 = get_res2.get_json()
        assert saved2["processing_status"] == "completed"
        assert saved2["analysis_mode"] == "ai"
        assert saved2["isFallback"] is False
        assert saved2["status"] == "Interested"

    # Verify strictly no duplicate records created in history
    history_after = client.get("/api/history").get_json()
    matches_after = [item for item in history_after if item["id"] == call_id]
    assert len(matches_after) == 1


def test_successful_ai_analysis_scenario(client):
    """
    Verify successful AI analysis scenario when OpenAI credits are available:
    - Finalizing call generates genuine AI summary, requirements, objections, signals, and status.
    - processing_status is marked as 'completed' and analysis_mode is 'ai'.
    - isFallback is False.
    - No local fallback disclaimer is attached.
    """
    start_res = client.post("/api/calls/start", json={
        "customer": "Sita Sharma",
        "company": "Zenith AI",
        "product": "CloudShield",
        "consent": True
    })
    call_id = start_res.get_json()["id"]

    transcript = [
        {"role": "assistant", "text": "Hello Sita, welcome to CloudShield! How can we help secure your cloud infrastructure?"},
        {"role": "user", "text": "We need SOC2 compliance automation for AWS and GCP."},
        {"role": "assistant", "text": "CloudShield automates SOC2 Type II evidence collection continuously across both AWS and GCP."},
        {"role": "user", "text": "We are ready to start subscription immediately, please send invoice."}
    ]

    mock_ai_client = MagicMock()
    mock_ai_client.models.generate_content.return_value = make_mock_gemini_response({
        "summary": "Customer requested immediate SOC2 compliance onboarding for AWS and GCP infrastructure.",
        "customer_requirements": ["SOC2 Type II compliance automation for AWS and GCP"],
        "objections": [],
        "buying_signals": ["Stated ready to start subscription immediately and requested invoice"],
        "next_steps": "Send subscription invoice and initiate cloud security setup.",
        "status": "Converted"
    })

    with patch("app.get_gemini_client", return_value=mock_ai_client):
        end_res = client.post("/api/calls/end", json={
            "callId": call_id,
            "durationSeconds": 92,
            "transcript": transcript
        })
        assert end_res.status_code == 200
        data = end_res.get_json()
        assert data["success"] is True
        assert data["processing_status"] == "completed"
        assert data["analysis_mode"] == "ai"
        assert data["isFallback"] is False
        assert data["status"] == "Converted"
        assert "Customer requested immediate SOC2" in data["summary"]
        assert "not an ai-generated summary" not in data["summary"].lower()

        # Check database persistence
        get_res = client.get(f"/api/calls/{call_id}")
        assert get_res.status_code == 200
        saved = get_res.get_json()
        assert saved["processing_status"] == "completed"
        assert saved["analysis_mode"] == "ai"
        assert saved["isFallback"] is False
        assert saved["status"] == "Converted"


def test_duplicate_prevention_across_multiple_retries(client):
    """
    Verify that multiple retries, whether failing due to quota or succeeding,
    preserve call history and never insert duplicate rows into SQLite.
    """
    start_res = client.post("/api/calls/start", json={
        "customer": "Deduplication Test Prospect",
        "company": "Unique Co",
        "product": "DataPulse",
        "consent": True
    })
    call_id = start_res.get_json()["id"]

    transcript = [
        {"role": "assistant", "text": "Welcome to DataPulse."},
        {"role": "user", "text": "Tell me about sales intelligence forecasting."}
    ]

    # Finalize with quota exhaustion
    mock_exhausted_client = MagicMock()
    mock_exhausted_client.models.generate_content.side_effect = Exception(
        "429 RESOURCE_EXHAUSTED: credit_balance_exhausted"
    )

    with patch("app.get_gemini_client", return_value=mock_exhausted_client):
        end_res = client.post("/api/calls/end", json={
            "callId": call_id,
            "durationSeconds": 45,
            "transcript": transcript
        })
        assert end_res.status_code == 200

    # Retry 3 times while quota exhausted
    with patch("app.get_gemini_client", return_value=mock_exhausted_client):
        for _ in range(3):
            retry_res = client.post(f"/api/calls/{call_id}/retry-analysis")
            assert retry_res.status_code == 402

    # Retry once after recovery
    mock_ok_client = MagicMock()
    mock_ok_client.models.generate_content.return_value = make_mock_gemini_response({
        "summary": "AI revenue forecasting discussion with DataPulse.",
        "customer_requirements": ["Pipeline forecasting"],
        "objections": [],
        "buying_signals": ["Explored forecasting"],
        "next_steps": "Schedule product demo.",
        "status": "Interested"
    })

    with patch("app.get_gemini_client", return_value=mock_ok_client):
        retry_res_ok = client.post(f"/api/calls/{call_id}/retry-analysis")
        assert retry_res_ok.status_code == 200

    # Ensure strictly 1 call record exists with this ID in SQLite and /api/history
    history_res = client.get("/api/history")
    matching = [c for c in history_res.get_json() if c["id"] == call_id]
    assert len(matching) == 1
    assert matching[0]["processingStatus"] == "completed"
    assert matching[0]["analysisMode"] == "ai"


def test_retry_endpoint_nonexistent_call(client):
    """Verify that retrying a non-existent call returns 404."""
    res = client.post("/api/calls/non-existent-uuid-12345/analyze")
    assert res.status_code == 404
    assert "not found" in res.get_json()["error"]


def test_call_detail_endpoint_by_id_and_parameterized_query(client):
    """Verify that /api/calls/<call_id> returns the exact record by ID with parameterized queries."""
    # 1. Fetch history to find call records
    history_res = client.get("/api/history")
    assert history_res.status_code == 200
    items = history_res.get_json()
    assert len(items) > 0

    elena = next((item for item in items if "elena" in (item.get("customer") or "").lower()), None)
    recorded_item = next((item for item in items if item.get("recording")), None)

    if elena:
        res = client.get(f"/api/calls/{elena['id']}")
        assert res.status_code == 200
        data = res.get_json()
        assert data["id"] == elena["id"]
        assert "Elena" in data["customer"]
        assert data["product"] == "CloudShield"
        assert data["company"] == "FinSecure Labs"
        assert isinstance(data["transcript"], list)

    if recorded_item:
        res = client.get(f"/api/calls/{recorded_item['id']}")
        assert res.status_code == 200
        data = res.get_json()
        assert data["id"] == recorded_item["id"]
        assert data["customer"] == recorded_item["customer"]
        assert data["product"] == recorded_item["product"]
        assert data["company"] == recorded_item["company"]
        assert isinstance(data["transcript"], list)
        assert data["recording"] != ""
        assert f"/api/calls/{recorded_item['id']}/recording" in data["recordingUrl"]

    # Parameterized SQL safety: test special characters and SQL injection attempts
    injection_res = client.get("/api/calls/' OR 1=1 --")
    assert injection_res.status_code == 404

    not_found_res = client.get("/api/calls/00000000-0000-0000-0000-000000000000")
    assert not_found_res.status_code == 404


def test_historical_selections_never_modify_saved_records(client):
    """Verify that inspecting historical records is strictly read-only."""
    history_res = client.get("/api/history")
    items = history_res.get_json()
    record = items[0] if items else None
    if not record:
        start_res = client.post("/api/calls/start", json={
            "customer": "Read-Only Test User",
            "product": "FlowPilot",
            "consent": True
        })
        record = start_res.get_json()

    res_before = client.get(f"/api/calls/{record['id']}").get_json()

    # Read the call detail multiple times
    for _ in range(3):
        client.get(f"/api/calls/{record['id']}")

    res_after = client.get(f"/api/calls/{record['id']}").get_json()

    assert res_before["transcript"] == res_after["transcript"]
    assert res_before["summary"] == res_after["summary"]
    assert res_before["customer"] == res_after["customer"]


def test_call_creation_validation_missing_fields(client):
    """Verify that backend rejects starting a call if required fields are missing."""
    # 1. Missing prospect name
    res1 = client.post("/api/calls/start", json={
        "customer": "",
        "product": "FlowPilot",
        "consent": True
    })
    assert res1.status_code == 400
    assert "Prospect name is required" in res1.get_json()["error"]

    # 2. Missing SaaS product
    res2 = client.post("/api/calls/start", json={
        "customer": "Jordan Lee",
        "product": "",
        "consent": True
    })
    assert res2.status_code == 400
    assert "SaaS product selection is required" in res2.get_json()["error"]

    # 3. Missing consent
    res3 = client.post("/api/calls/start", json={
        "customer": "Jordan Lee",
        "product": "FlowPilot",
        "consent": False
    })
    assert res3.status_code == 400
    assert "consent is required" in res3.get_json()["error"]

    # 4. Success with empty company and notes
    res4 = client.post("/api/calls/start", json={
        "customer": "Jordan Lee",
        "company": "",
        "product": "FlowPilot",
        "notes": "",
        "consent": True
    })
    assert res4.status_code == 200
    data4 = res4.get_json()
    assert data4["customer"] == "Jordan Lee"
    assert data4["product"] == "FlowPilot"


def test_branding_and_initial_blank_defaults(client):
    """Verify that internship branding is removed and HTML has blank initial inputs."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # 1. Check internship branding removed
    assert "Sankar Group" not in html
    assert "Internship Project" not in html
    assert "AI-Powered Voice Sales Agent" in html

    # 2. Check no pre-filled fictional defaults
    assert 'value="Alex Morgan"' not in html
    assert 'value="Apex Logistics"' not in html
    assert 'value="Interested in reducing manual ops time' not in html

    # 3. Check correct neutral placeholders
    assert 'placeholder="Enter prospect name"' in html
    assert 'placeholder="Enter company name (optional)"' in html
    assert 'placeholder="Enter customer needs, pain points, or sales notes (optional)"' in html
    assert 'Select a SaaS product' in html
    assert 'Select a language' in html


def test_language_options_and_config(client):
    """Verify that all 4 required languages are available in configuration."""
    res = client.get("/api/config")
    assert res.status_code == 200
    languages = res.get_json()["languages"]
    codes = [l["code"] for l in languages]
    assert "en-US" in codes
    assert "en-IN" in codes
    assert "hi-IN" in codes
    assert "te-IN" in codes


def test_system_prompt_language_directives():
    """Verify that backend build_system_prompt includes strict language directives."""
    from app import build_system_prompt

    # 1. Hindi Prompt
    hi_prompt = build_system_prompt("FlowPilot", "hi-IN", "राजेश शर्मा", "टेक सॉल्यूशंस")
    assert "MANDATORY LANGUAGE DIRECTIVE: HINDI (हिंदी)" in hi_prompt
    assert "Devanagari script" in hi_prompt
    assert "NEVER switch to English" in hi_prompt

    # 2. Telugu Prompt
    te_prompt = build_system_prompt("CloudShield", "te-IN", "శ్రీనివాస్ రావు", "హైదరాబాద్ టెక్")
    assert "MANDATORY LANGUAGE DIRECTIVE: TELUGU (తెలుగు)" in te_prompt
    assert "Telugu script" in te_prompt
    assert "NEVER switch to English" in te_prompt

    # 3. English (India) Prompt
    en_in_prompt = build_system_prompt("DataPulse", "en-IN", "Aditi Rao")
    assert "MANDATORY LANGUAGE DIRECTIVE: ENGLISH (INDIA)" in en_in_prompt

    # 4. English (US) Prompt
    en_us_prompt = build_system_prompt("FlowPilot", "en-US", "Alex Morgan")
    assert "MANDATORY LANGUAGE DIRECTIVE: ENGLISH (US)" in en_us_prompt


def test_call_creation_localized_greetings(client):
    """Verify that starting a call returns an initial greeting matching the selected language."""
    # 1. Hindi call start
    res_hi = client.post("/api/calls/start", json={
        "customer": "विक्रम",
        "product": "FlowPilot",
        "language": "hi-IN",
        "consent": True
    })
    assert res_hi.status_code == 200
    data_hi = res_hi.get_json()
    assert data_hi["language"] == "hi-IN"
    assert "नमस्ते विक्रम!" in data_hi["greeting"]
    assert "AI सेल्स विशेषज्ञ" in data_hi["greeting"]

    # 2. Telugu call start
    res_te = client.post("/api/calls/start", json={
        "customer": "వెంకట్",
        "product": "CloudShield",
        "language": "te-IN",
        "consent": True
    })
    assert res_te.status_code == 200
    data_te = res_te.get_json()
    assert data_te["language"] == "te-IN"
    assert "నమస్కారం వెంకట్!" in data_te["greeting"]
    assert "AI నిపుణుడిని" in data_te["greeting"]

    # 3. English US call start
    res_en = client.post("/api/calls/start", json={
        "customer": "Sarah",
        "product": "DataPulse",
        "language": "en-US",
        "consent": True
    })
    assert res_en.status_code == 200
    data_en = res_en.get_json()
    assert data_en["language"] == "en-US"
    assert "Hello Sarah!" in data_en["greeting"]


def test_chat_endpoint_multilingual_responses(client):
    """Verify that chat responses are returned in the selected language."""
    # 1. Hindi chat pricing inquiry
    res_hi = client.post("/api/chat", json={
        "message": "इसकी कीमत क्या है?",
        "product": "FlowPilot",
        "language": "hi-IN",
        "customer": "राहुल"
    })
    assert res_hi.status_code == 200
    reply_hi = res_hi.get_json()["reply"]
    assert "बजट" in reply_hi or "उपलब्ध" in reply_hi or "FlowPilot" in reply_hi

    # 2. Telugu chat demo inquiry
    res_te = client.post("/api/chat", json={
        "message": "నాకు డెమో చూపించండి",
        "product": "FlowPilot",
        "language": "te-IN",
        "customer": "సురేష్"
    })
    assert res_te.status_code == 200
    reply_te = res_te.get_json()["reply"]
    assert "డెమో" in reply_te or "సమయం" in reply_te

    # 3. English chat
    res_en = client.post("/api/chat", json={
        "message": "Can I see a demo?",
        "product": "FlowPilot",
        "language": "en-US",
        "customer": "David"
    })
    assert res_en.status_code == 200
    assert "demo" in res_en.get_json()["reply"].lower()


def test_call_intelligence_in_selected_languages():
    """Verify that analyze_call_transcript returns localized summaries & explanations while keeping standard status tokens."""
    from app import analyze_call_transcript

    # 1. Hindi Demo Request
    hi_transcript = [
        {"role": "assistant", "text": "नमस्ते अमित! मैं FlowPilot से बोल रहा हूँ।"},
        {"role": "user", "text": "मुझे इसका एक लाइव डेमो दिखाइए, हम ऑटोमेशन चाहते हैं।"}
    ]
    analysis_hi = analyze_call_transcript(hi_transcript, "FlowPilot", "hi-IN", "अमित", allow_fallback=True)
    assert analysis_hi["status"] == "Demo Requested"
    assert "डेमो" in analysis_hi["next_steps"] or "प्रदर्शन" in analysis_hi["next_steps"]
    assert "अमित" in analysis_hi["summary"]

    # 2. Telugu Follow-up Required
    te_transcript = [
        {"role": "assistant", "text": "నమస్కారం రమేష్! నేను CloudShield నుండి మాట్లాడుతున్నాను."},
        {"role": "user", "text": "నాకు ఇమెయిల్ పంపండి, రేపు కాల్ చేయండి నేను టీమ్‌తో మాట్లాడాలి."}
    ]
    analysis_te = analyze_call_transcript(te_transcript, "CloudShield", "te-IN", "రమేష్", allow_fallback=True)
    assert analysis_te["status"] == "Follow-up Required"
    assert "ఇమెయిల్" in analysis_te["next_steps"] or "షెడ్యూల్" in analysis_te["next_steps"]
    assert "రమేష్" in analysis_te["summary"]

    # 3. Hindi No Response
    analysis_no_resp = analyze_call_transcript([], "FlowPilot", "hi-IN", "अनिल", allow_fallback=True)
    assert analysis_no_resp["status"] == "No Response"
    assert "कॉल कनेक्ट हुआ" in analysis_no_resp["summary"]
    assert "कोई आवश्यकता पहचानी नहीं गई।" in analysis_no_resp["customer_requirements"]


def test_historical_call_preserves_and_restores_language(client):
    """Verify that finalized calls preserve their selected language in SQLite and retrieve it accurately."""
    mock_telugu_analysis = {
        "summary": "కిరణ్ వర్మ తో DataPulse గురించి అమ్మకాల సంభాషణ. సంభాషణలో 1 సార్లు స్పందించారు.",
        "customer_requirements": ["ధర వివరాలు చెప్పండి, చాలా ఆసక్తిగా ఉంది."],
        "objections": [],
        "buying_signals": ["కస్టమర్ ధర వివరాలు అడిగారు."],
        "next_steps": "ధర వివరాలు పంపి పరిచయ సమావేశాన్ని షెడ్యూల్ చేయండి.",
        "status": "Interested"
    }
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = make_mock_gemini_response(mock_telugu_analysis)

    with patch("app.get_gemini_client", return_value=mock_client):
        # 1. Create call in Telugu
        start_res = client.post("/api/calls/start", json={
            "customer": "కిరణ్ వర్మ",
            "company": "ఆంధ్ర టెక్",
            "product": "DataPulse",
            "language": "te-IN",
            "consent": True
        })
        call_id = start_res.get_json()["id"]

        # 2. Finalize call
        end_res = client.post("/api/calls/end", json={
            "callId": call_id,
            "durationSeconds": 45,
            "language": "te-IN",
            "transcript": [
                {"role": "assistant", "text": "నమస్కారం కిరణ్ గారు!"},
                {"role": "user", "text": "నాకు ధర వివరాలు చెప్పండి, చాలా ఆసక్తిగా ఉంది."}
            ]
        })
        assert end_res.status_code == 200
        intel = end_res.get_json()
        assert intel["status"] == "Interested"

        # 3. Retrieve historical detail
        detail_res = client.get(f"/api/calls/{call_id}")
        assert detail_res.status_code == 200
        detail = detail_res.get_json()
        assert detail["language"] == "te-IN"
        assert detail["customer"] == "కిరణ్ వర్మ"
        assert "కిరణ్" in detail["summary"]


def test_tts_voices_list(client):
    """Verify that /api/tts/voices returns verified voice specifications for all supported languages."""
    res = client.get("/api/tts/voices")
    assert res.status_code == 200
    data = res.get_json()
    assert "voices" in data
    assert "provider" in data
    voices = data["voices"]
    codes = [v["code"] for v in voices]
    assert "te-IN" in codes
    assert "hi-IN" in codes
    assert "en-IN" in codes
    assert "en-US" in codes
    te_voice = next(v for v in voices if v["code"] == "te-IN")
    assert te_voice["voiceId"] == "google-te-standard"
    assert te_voice["supported"] is True


def test_tts_endpoint_telugu(client):
    """Verify that POST /api/tts generates genuine MP3 audio for Telugu text."""
    telugu_text = "నమస్కారం! నేను FlowPilot సేల్స్ నిపుణుడిని. మీ ప్రస్తుత వర్క్‌ఫ్లోలో మీరు ఎదుర్కొంటున్న సమస్య ఏమిటి?"
    res = client.post("/api/tts", json={
        "text": telugu_text,
        "language": "te-IN"
    })
    assert res.status_code == 200
    assert res.headers.get("Content-Type") == "audio/mpeg"
    assert len(res.data) > 500  # Non-trivial MP3 audio binary payload


def test_tts_endpoint_hindi(client):
    """Verify that POST /api/tts generates genuine MP3 audio for Hindi text."""
    hindi_text = "नमस्ते! मैं FlowPilot के लिए AI सेल्स विशेषज्ञ हूँ। क्या आप एक लाइव डेमो देखना चाहेंगे?"
    res = client.post("/api/tts", json={
        "text": hindi_text,
        "language": "hi-IN"
    })
    assert res.status_code == 200
    assert res.headers.get("Content-Type") == "audio/mpeg"
    assert len(res.data) > 500


def test_tts_endpoint_english(client):
    """Verify that POST /api/tts generates genuine MP3 audio for English (US & India)."""
    # 1. English US
    res_us = client.post("/api/tts", json={
        "text": "Hello, thank you for taking my call today. How can FlowPilot help your team?",
        "language": "en-US"
    })
    assert res_us.status_code == 200
    assert res_us.headers.get("Content-Type") == "audio/mpeg"
    assert len(res_us.data) > 500

    # 2. English India
    res_in = client.post("/api/tts", json={
        "text": "Hello, thank you for taking my call. FlowPilot automates manual workflow tasks.",
        "language": "en-IN"
    })
    assert res_in.status_code == 200
    assert res_in.headers.get("Content-Type") == "audio/mpeg"
    assert len(res_in.data) > 500


def test_tts_validation_and_errors(client):
    """Verify that /api/tts properly validates empty input and unsupported languages."""
    # 1. Empty text
    res_empty = client.post("/api/tts", json={
        "text": "   ",
        "language": "te-IN"
    })
    assert res_empty.status_code == 400
    assert "Text is required" in res_empty.get_json()["error"]

    # 2. Unsupported language
    res_unsupported = client.post("/api/tts", json={
        "text": "Hello world",
        "language": "xx-YY"
    })
    assert res_unsupported.status_code == 400
    err_json = res_unsupported.get_json()
    assert "not supported" in err_json["error"]
    assert "supportedLanguages" in err_json
    assert "te-IN" in err_json["supportedLanguages"]


def test_tts_direct_generation_helper():
    """Verify the generate_tts_audio helper directly produces valid MP3 bytes for Telugu."""
    from app import generate_tts_audio
    audio_bytes, mimetype = generate_tts_audio("ధన్యవాదాలు!", "te-IN")
    assert mimetype == "audio/mpeg"
    assert isinstance(audio_bytes, (bytes, bytearray))
    assert len(audio_bytes) > 200


def test_multiturn_conversation_telugu(client):
    """
    Verify multi-turn sales conversation in Telugu:
    1. Features inquiry -> AI explains features without repeating greeting.
    2. Pricing inquiry -> AI answers pricing without repeating features.
    3. Cost objection -> AI addresses budget concern without repeating pricing.
    4. Demo request -> AI confirms demo schedule without restarting call.
    Verify all AI responses are distinct, in Telugu, and role order is strictly preserved.
    """
    transcript = [
        {"role": "assistant", "text": "నమస్కారం Rahul! కాల్ మాట్లాడినందుకు ధన్యవాదాలు. నేను CloudShield సేల్స్ AI నిపుణుడిని. మీ ప్రస్తుత వర్క్‌ఫ్లోలో మీరు ఎదుర్కొంటున్న అతిపెద్ద సవాలు ఏమిటి?"}
    ]

    # Turn 1: Customer asks about features & benefits
    t1_text = "మీ క్లౌడ్ శీల్ ప్రోడక్ట్ మా కంపెనీకి ఎలా ఉపయోగపడుతుంది దాని ముఖ్యమైన ప్రయోజనాలు ఏమిటి"
    res1 = client.post("/api/chat", json={
        "message": t1_text,
        "product": "CloudShield",
        "language": "te-IN",
        "customer": "Rahul",
        "history": transcript
    })
    assert res1.status_code == 200
    reply1 = res1.get_json()["reply"]
    assert len(reply1) > 10
    # Must be in Telugu
    assert any(c in reply1 for c in ["ఆటోమేట్", "క్లౌడ్", "సెక్యూరిటీ", "సహాయం", "ప్రయోజనాలు", "ఫీచర్", "CloudShield"])
    # Must NOT repeat opening question
    assert "అతిపెద్ద సవాలు ఏమిటి" not in reply1
    assert reply1 != transcript[0]["text"]

    transcript.append({"role": "user", "text": t1_text})
    transcript.append({"role": "assistant", "text": reply1})

    # Turn 2: Customer asks about pricing
    t2_text = "దీని ధర ఎంత అవుతుంది మరియు ఏ ప్లాన్లు ఉన్నాయి?"
    res2 = client.post("/api/chat", json={
        "message": t2_text,
        "product": "CloudShield",
        "language": "te-IN",
        "customer": "Rahul",
        "history": transcript
    })
    assert res2.status_code == 200
    reply2 = res2.get_json()["reply"]
    assert len(reply2) > 10
    # Must mention pricing
    assert "$" in reply2 or "ధర" in reply2 or "ప్లాన్" in reply2 or "299" in reply2
    # Must NOT repeat Turn 1 reply or greeting
    assert reply2 != reply1
    assert reply2 != transcript[0]["text"]

    transcript.append({"role": "user", "text": t2_text})
    transcript.append({"role": "assistant", "text": reply2})

    # Turn 3: Customer raises cost objection
    t3_text = "ఇది మా స్టార్టప్ బడ్జెట్ కంటే చాలా ఎక్కువ ఖరీదైనది"
    res3 = client.post("/api/chat", json={
        "message": t3_text,
        "product": "CloudShield",
        "language": "te-IN",
        "customer": "Rahul",
        "history": transcript
    })
    assert res3.status_code == 200
    reply3 = res3.get_json()["reply"]
    assert len(reply3) > 10
    # Must address budget / ROI / value
    assert "బడ్జెట్" in reply3 or "ROI" in reply3 or "డెమో" in reply3 or "ఖర్చు" in reply3
    # Must NOT repeat Turn 2, Turn 1, or greeting
    assert reply3 != reply2
    assert reply3 != reply1
    assert reply3 != transcript[0]["text"]

    transcript.append({"role": "user", "text": t3_text})
    transcript.append({"role": "assistant", "text": reply3})

    # Turn 4: Customer requests a demo
    t4_text = "సరే నాకు రేపు 20 నిమిషాల లైవ్ డెమో చూపించండి"
    res4 = client.post("/api/chat", json={
        "message": t4_text,
        "product": "CloudShield",
        "language": "te-IN",
        "customer": "Rahul",
        "history": transcript
    })
    assert res4.status_code == 200
    reply4 = res4.get_json()["reply"]
    assert len(reply4) > 10
    # Must acknowledge demo schedule
    assert "డెమో" in reply4 or "సమయం" in reply4 or "షెడ్యూల్" in reply4 or "రేపు" in reply4
    # Must NOT repeat any earlier reply
    assert reply4 not in [reply1, reply2, reply3, transcript[0]["text"]]

    transcript.append({"role": "user", "text": t4_text})
    transcript.append({"role": "assistant", "text": reply4})

    # Verify transcript integrity
    assert len(transcript) == 9  # 1 greeting + 4 pairs of user-assistant turns
    roles = [m["role"] for m in transcript]
    expected_roles = ["assistant", "user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant"]
    assert roles == expected_roles

    # Verify all assistant responses are unique
    assistant_texts = [m["text"] for m in transcript if m["role"] == "assistant"]
    assert len(assistant_texts) == len(set(assistant_texts))


def test_multiturn_conversation_hindi(client):
    """
    Verify multi-turn sales conversation in Hindi:
    Features -> Pricing -> Cost Objection -> Demo Request.
    """
    transcript = [
        {"role": "assistant", "text": "नमस्ते अमित! मैं FlowPilot से बात कर रहा हूँ। वर्तमान में आपके वर्कफ़्लो में सबसे बड़ी चुनौती क्या आ रही है?"}
    ]

    # Turn 1: Features
    res1 = client.post("/api/chat", json={
        "message": "FlowPilot की मुख्य विशेषताएं और क्या लाभ हैं?",
        "product": "FlowPilot",
        "language": "hi-IN",
        "customer": "अमित",
        "history": transcript
    })
    assert res1.status_code == 200
    reply1 = res1.get_json()["reply"]
    assert "FlowPilot" in reply1 or "स्वचालित" in reply1 or "सुविधा" in reply1 or "वर्कफ़्लो" in reply1
    assert reply1 != transcript[0]["text"]
    transcript.extend([{"role": "user", "text": "FlowPilot की मुख्य विशेषताएं और क्या लाभ हैं?"}, {"role": "assistant", "text": reply1}])

    # Turn 2: Pricing
    res2 = client.post("/api/chat", json={
        "message": "इसके प्रो प्लान की कीमत कितनी है?",
        "product": "FlowPilot",
        "language": "hi-IN",
        "customer": "अमित",
        "history": transcript
    })
    assert res2.status_code == 200
    reply2 = res2.get_json()["reply"]
    assert "$" in reply2 or "कीमत" in reply2 or "प्लान" in reply2 or "199" in reply2 or "49" in reply2
    assert reply2 != reply1
    transcript.extend([{"role": "user", "text": "इसके प्रो प्लान की कीमत कितनी है?"}, {"role": "assistant", "text": reply2}])

    # Turn 3: Cost Objection
    res3 = client.post("/api/chat", json={
        "message": "यह हमारी टीम के बजट से थोड़ा महंगा लग रहा है",
        "product": "FlowPilot",
        "language": "hi-IN",
        "customer": "अमित",
        "history": transcript
    })
    assert res3.status_code == 200
    reply3 = res3.get_json()["reply"]
    assert "बजट" in reply3 or "ROI" in reply3 or "डेमो" in reply3 or "बचत" in reply3
    assert reply3 not in [reply1, reply2]
    transcript.extend([{"role": "user", "text": "यह हमारी टीम के बजट से थोड़ा महंगा लग रहा है"}, {"role": "assistant", "text": reply3}])

    # Turn 4: Demo Request
    res4 = client.post("/api/chat", json={
        "message": "क्या हम कल दोपहर एक लाइव डेमो शेड्यूल कर सकते हैं?",
        "product": "FlowPilot",
        "language": "hi-IN",
        "customer": "अमित",
        "history": transcript
    })
    assert res4.status_code == 200
    reply4 = res4.get_json()["reply"]
    assert "डेमो" in reply4 or "समय" in reply4 or "कल" in reply4 or "शेड्यूल" in reply4
    assert reply4 not in [reply1, reply2, reply3, transcript[0]["text"]]

    # Verify all assistant responses are unique and distinct
    assistant_texts = [m["text"] for m in transcript if m["role"] == "assistant"]
    assert len(assistant_texts) == len(set(assistant_texts))


def test_multiturn_conversation_english(client):
    """
    Verify multi-turn sales conversation in English:
    Features -> Pricing -> Cost Objection -> Demo Request.
    """
    transcript = [
        {"role": "assistant", "text": "Hello Alex! Thanks for taking my call. What is your biggest challenge with your current workflow right now?"}
    ]

    # Turn 1: Features
    res1 = client.post("/api/chat", json={
        "message": "Can you explain the key features and capabilities of DataPulse?",
        "product": "DataPulse",
        "language": "en-US",
        "customer": "Alex",
        "history": transcript
    })
    assert res1.status_code == 200
    reply1 = res1.get_json()["reply"]
    assert "datapulse" in reply1.lower() or "pipeline" in reply1.lower() or "forecast" in reply1.lower() or "crm" in reply1.lower()
    assert reply1 != transcript[0]["text"]
    transcript.extend([{"role": "user", "text": "Can you explain the key features and capabilities of DataPulse?"}, {"role": "assistant", "text": reply1}])

    # Turn 2: Pricing
    res2 = client.post("/api/chat", json={
        "message": "How much does it cost?",
        "product": "DataPulse",
        "language": "en-US",
        "customer": "Alex",
        "history": transcript
    })
    assert res2.status_code == 200
    reply2 = res2.get_json()["reply"]
    assert "$" in reply2 or "pricing" in reply2.lower() or "tier" in reply2.lower() or "month" in reply2.lower()
    assert reply2 != reply1
    transcript.extend([{"role": "user", "text": "How much does it cost?"}, {"role": "assistant", "text": reply2}])

    # Turn 3: Cost Objection
    res3 = client.post("/api/chat", json={
        "message": "That sounds a bit too expensive for our team right now",
        "product": "DataPulse",
        "language": "en-US",
        "customer": "Alex",
        "history": transcript
    })
    assert res3.status_code == 200
    reply3 = res3.get_json()["reply"]
    assert "budget" in reply3.lower() or "roi" in reply3.lower() or "demo" in reply3.lower() or "value" in reply3.lower()
    assert reply3 not in [reply1, reply2]
    transcript.extend([{"role": "user", "text": "That sounds a bit too expensive for our team right now"}, {"role": "assistant", "text": reply3}])

    # Turn 4: Demo Request
    res4 = client.post("/api/chat", json={
        "message": "Could we schedule a 20-minute live demo for tomorrow afternoon?",
        "product": "DataPulse",
        "language": "en-US",
        "customer": "Alex",
        "history": transcript
    })
    assert res4.status_code == 200
    reply4 = res4.get_json()["reply"]
    assert "demo" in reply4.lower() or "tomorrow" in reply4.lower() or "walkthrough" in reply4.lower()
    assert reply4 not in [reply1, reply2, reply3, transcript[0]["text"]]

    # Verify all assistant responses are unique
    assistant_texts = [m["text"] for m in transcript if m["role"] == "assistant"]
    assert len(assistant_texts) == len(set(assistant_texts))


def test_gemini_conversation_history_deduplication(client):
    """
    Verify that when Gemini client is used, the history normalization prevents
    duplicate consecutive user messages or role scrambling.
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = make_mock_gemini_response(
        "I understand your need for automation. FlowPilot integrates with over 150 enterprise connectors."
    )

    with patch("app.get_gemini_client", return_value=mock_client):
        # Simulate frontend sending state.transcript where user message was already pushed
        user_msg = "How does FlowPilot connect to Salesforce?"
        history = [
            {"role": "assistant", "text": "Hello, how can I help?"},
            {"role": "user", "text": user_msg}
        ]

        res = client.post("/api/chat", json={
            "message": user_msg,
            "product": "FlowPilot",
            "language": "en-US",
            "customer": "Jordan",
            "history": history
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["mode"] == "ai"
        assert data["provider"] == "gemini"

        # Check call arguments passed to Gemini generate_content
        call_kwargs = mock_client.models.generate_content.call_args[1]
        sent_contents = call_kwargs["contents"]
        assert len(sent_contents) > 0
        assert sent_contents[-1].role == "user"
        assert user_msg in sent_contents[-1].parts[0].text


def test_gemini_repetition_rejected_and_falls_back(client):
    """
    Verify that if Gemini generate_content returns an assistant response that was
    already delivered in an earlier turn, the server rejects it and falls back
    to a unique, progressive conversational response.
    """
    duplicated_text = "వివరాలు తెలిపినందుకు ధన్యవాదాలు, Rahul. CloudShield మీ పని సామర్థ్యాన్ని పెంచడానికి రూపొందించబడింది. ప్రస్తుతం మీ వ్యాపారానికి ఏది అత్యంత అవసరం?"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = make_mock_gemini_response(duplicated_text)

    with patch("app.get_gemini_client", return_value=mock_client):
        history = [
            {"role": "assistant", "text": "నమస్కారం Rahul! కాల్ మాట్లాడినందుకు ధన్యవాదాలు. నేను CloudShield సేల్స్ AI నిపుణుడిని. మీ ప్రస్తుత వర్క్‌ఫ్లోలో మీరు ఎదుర్కొంటున్న అతిపెద్ద సవాలు ఏమిటి?"},
            {"role": "user", "text": "నమస్కారం మీ క్లౌడ్ శీల్ ప్రోడక్ట్ మా కంపెనీకి ఎలా ఉపయోగపడుతుంది"},
            {"role": "assistant", "text": duplicated_text}
        ]

        # Customer sends next message
        res = client.post("/api/chat", json={
            "message": "మా కంపెనీలో ఏడబ్ల్యూఎస్ ఆడిట్ కోసం ఆధారాలను సేకరించడానికి చాలా సమయం పడుతుంది",
            "product": "CloudShield",
            "language": "te-IN",
            "customer": "Rahul",
            "history": history
        })
        assert res.status_code == 200
        reply = res.get_json()["reply"]

        # The duplicate Gemini text MUST NOT be returned
        assert reply != duplicated_text
        assert "అత్యంత అవసరం" not in reply
        # Must be in Telugu and distinct from all past turns
        assert len(reply) > 10
        assert reply not in [h["text"] for h in history]


def test_gemini_reask_greeting_rejected(client):
    """
    Verify that if Gemini generate_content returns a response that re-asks the greeting question,
    it is rejected in favor of a response that advances the sales dialogue.
    """
    reask_text = "What is your biggest challenge with your current workflow right now?"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = make_mock_gemini_response(reask_text)

    with patch("app.get_gemini_client", return_value=mock_client):
        history = [
            {"role": "assistant", "text": "Hello Alex! Thanks for taking my call. What is your biggest challenge with your current workflow right now?"},
            {"role": "user", "text": "We spend 20 hours a week on manual reporting."}
        ]

        res = client.post("/api/chat", json={
            "message": "We spend 20 hours a week on manual reporting.",
            "product": "DataPulse",
            "language": "en-US",
            "customer": "Alex",
            "history": history
        })
        assert res.status_code == 200
        reply = res.get_json()["reply"]
        assert reply != reask_text
        assert "biggest challenge" not in reply.lower()


def test_extended_multiturn_telugu_non_repetition(client):
    """
    Verify a full 6-turn sales conversation in Telugu:
    1. Features -> 2. Pricing -> 3. Cost objection -> 4. Demo request -> 5. Positive time confirmation -> 6. Closing wrap-up.
    All 7 assistant turns (including greeting) must be 100% unique and distinct.
    """
    transcript = [
        {"role": "assistant", "text": "నమస్కారం Rahul! కాల్ మాట్లాడినందుకు ధన్యవాదాలు. నేను CloudShield సేల్స్ AI నిపుణుడిని. మీ ప్రస్తుత వర్క్‌ఫ్లోలో మీరు ఎదుర్కొంటున్న అతిపెద్ద సవాలు ఏమిటి?"}
    ]

    turns = [
        "మీ క్లౌడ్ శీల్ ప్రోడక్ట్ మా కంపెనీకి ఎలా ఉపయోగపడుతుంది దాని ముఖ్యమైన ప్రయోజనాలు ఏమిటి",
        "దీని ధర ఎంత అవుతుంది మరియు ఏ ప్లాన్లు ఉన్నాయి?",
        "ఇది మా స్టార్టప్ బడ్జెట్ కంటే చాలా ఎక్కువ ఖరీదైనది",
        "సరే నాకు రేపు 20 నిమిషాల లైవ్ డెమో చూపించండి",
        "అవును రేపు 2 గంటల సమయం బాగుంది",
        "ఇంకేమీ ప్రశ్నలు లేవు ధన్యవాదాలు"
    ]

    for t in turns:
        res = client.post("/api/chat", json={
            "message": t,
            "product": "CloudShield",
            "language": "te-IN",
            "customer": "Rahul",
            "history": transcript
        })
        assert res.status_code == 200
        reply = res.get_json()["reply"]
        assert len(reply) > 5
        transcript.append({"role": "user", "text": t})
        transcript.append({"role": "assistant", "text": reply})

    assistant_texts = [m["text"] for m in transcript if m["role"] == "assistant"]
    assert len(assistant_texts) == 7
    assert len(assistant_texts) == len(set(assistant_texts))


def test_extended_multiturn_hindi_non_repetition(client):
    """
    Verify a full 6-turn sales conversation in Hindi:
    All 7 assistant turns (including greeting) must be 100% unique and distinct.
    """
    transcript = [
        {"role": "assistant", "text": "नमस्ते अमित! मैं FlowPilot से बात कर रहा हूँ। वर्तमान में आपके वर्कफ़्लो में सबसे बड़ी चुनौती क्या आ रही है?"}
    ]

    turns = [
        "FlowPilot की मुख्य विशेषताएं और क्या लाभ हैं?",
        "इसके प्रो प्लान की कीमत कितनी है?",
        "यह हमारी टीम के बजट से थोड़ा महंगा लग रहा है",
        "क्या हम कल दोपहर एक लाइव डेमो शेड्यूल कर सकते हैं?",
        "हाँ कल 2 बजे का समय सही रहेगा",
        "बस इतना ही धन्यवाद"
    ]

    for t in turns:
        res = client.post("/api/chat", json={
            "message": t,
            "product": "FlowPilot",
            "language": "hi-IN",
            "customer": "अमित",
            "history": transcript
        })
        assert res.status_code == 200
        reply = res.get_json()["reply"]
        assert len(reply) > 5
        transcript.append({"role": "user", "text": t})
        transcript.append({"role": "assistant", "text": reply})

    assistant_texts = [m["text"] for m in transcript if m["role"] == "assistant"]
    assert len(assistant_texts) == 7
    assert len(assistant_texts) == len(set(assistant_texts))


def test_extended_multiturn_english_non_repetition(client):
    """
    Verify a full 6-turn sales conversation in English:
    All 7 assistant turns (including greeting) must be 100% unique and distinct.
    """
    transcript = [
        {"role": "assistant", "text": "Hello Alex! Thanks for taking my call. What is your biggest challenge with your current workflow right now?"}
    ]

    turns = [
        "Can you explain the key features and capabilities of DataPulse?",
        "How much does it cost?",
        "That sounds a bit too expensive for our team right now",
        "Could we schedule a 20-minute live demo for tomorrow afternoon?",
        "Yes 2 PM works great for me",
        "No other questions, that is all thank you"
    ]

    for t in turns:
        res = client.post("/api/chat", json={
            "message": t,
            "product": "DataPulse",
            "language": "en-US",
            "customer": "Alex",
            "history": transcript
        })
        assert res.status_code == 200
        reply = res.get_json()["reply"]
        assert len(reply) > 5
        transcript.append({"role": "user", "text": t})
        transcript.append({"role": "assistant", "text": reply})

    assistant_texts = [m["text"] for m in transcript if m["role"] == "assistant"]
    assert len(assistant_texts) == 7
    assert len(assistant_texts) == len(set(assistant_texts))


def test_repetition_detection_helpers():
    """
    Unit test for the speech normalization and repetition detection helper functions.
    """
    from app import normalize_speech_text, is_greeting_reask, is_repetitive_response

    # Normalization
    assert normalize_speech_text("Hello, World!!") == "hello world"
    assert normalize_speech_text("  నమస్కారం!  Rahul.. ") == "నమస్కారం rahul"

    # Greeting re-ask detection
    assert is_greeting_reask("What is your biggest challenge with your current workflow right now?", "en-US")
    assert is_greeting_reask("మీ ప్రస్తుత వర్క్‌ఫ్లోలో మీరు ఎదుర్కొంటున్న అతిపెద్ద సవాలు ఏమిటి?", "te-IN")
    assert is_greeting_reask("वर्तमान में आपके वर्कफ़्लो में सबसे बड़ी चुनौती क्या आ रही है?", "hi-IN")
    assert not is_greeting_reask("Our Pro plan starts at $299 per month.", "en-US")

    # Repetition checks
    past = ["Hello there! How can I help you today?"]
    assert is_repetitive_response("Hello there! How can I help you today?", past, "en-US")
    assert is_repetitive_response("hello there how can i help you today", past, "en-US")  # normalized match
    assert not is_repetitive_response("We offer 24/7 dedicated support.", past, "en-US")


def test_gemini_client_initialization_and_configuration():
    """Verify Gemini client initialization, environment variable handling, and model defaults."""
    import os
    from app import get_gemini_client, get_gemini_model

    # 1. Without API key
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        assert get_gemini_client() is None

    # 2. With mock valid key
    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeTestKey12345"}):
        with patch("google.genai.Client") as mock_genai_cls:
            mock_genai_cls.return_value = MagicMock()
            client = get_gemini_client()
            assert client is not None
            mock_genai_cls.assert_called_once_with(api_key="AIzaSyFakeTestKey12345")

    # 3. Model defaults and normalization
    with patch.dict(os.environ, {"GEMINI_MODEL": ""}):
        assert get_gemini_model() == "gemini-3.6-flash"

    # Custom supported model name
    with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-pro"}):
        assert get_gemini_model() == "gemini-2.5-pro"


def test_gemini_missing_and_invalid_api_keys(client):
    """
    Verify behavior when GEMINI_API_KEY is missing or invalid:
    - Never crashes or exposes internal secrets.
    - Conversational chat falls back to context-aware local engine.
    - Call intelligence falls back to clearly labeled rule-based analysis.
    """
    # 1. Missing key
    with patch("app.get_gemini_client", return_value=None):
        res = client.post("/api/chat", json={
            "message": "Can you tell me about FlowPilot pricing?",
            "product": "FlowPilot",
            "language": "en-US",
            "customer": "Maya",
            "history": []
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["mode"] == "fallback"
        assert "pricing" in data["reply"].lower() or "$49" in data["reply"]

    # 2. Invalid API key error (e.g. 400 or 403 API key not valid)
    invalid_key_client = MagicMock()
    invalid_key_client.models.generate_content.side_effect = Exception(
        "400 INVALID_ARGUMENT: API key not valid. Please pass a valid API key."
    )

    with patch("app.get_gemini_client", return_value=invalid_key_client):
        # /api/chat falls back safely without unhandled 500 error
        res = client.post("/api/chat", json={
            "message": "What integrations do you support?",
            "product": "FlowPilot",
            "language": "en-US",
            "customer": "Maya",
            "history": []
        })
        assert res.status_code == 200
        assert res.get_json()["mode"] == "fallback"

        # analyze_call_transcript without fallback raises GeminiAnalysisError
        with pytest.raises(Exception) as exc_info:
            analyze_call_transcript(
                transcript=[{"role": "user", "text": "Hello"}],
                product_name="FlowPilot",
                language="en-US",
                customer="Maya",
                allow_fallback=False
            )
        assert "Gemini analysis failed" in str(exc_info.value) or "API key not valid" in str(exc_info.value)


def test_gemini_successful_conversation_responses(client):
    """
    Verify that when Gemini client is active:
    - /api/chat calls Gemini generate_content and returns response in 'ai' mode.
    - provider is reported as 'gemini' and model is reported correctly.
    - Markdown formatting like asterisks and bold markers are stripped for clean audio TTS.
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = make_mock_gemini_response(
        "**FlowPilot** integrates with over 150 enterprise tools, saving *significant* engineering time."
    )

    with patch("app.get_gemini_client", return_value=mock_client):
        res = client.post("/api/chat", json={
            "message": "What integrations do you have?",
            "product": "FlowPilot",
            "language": "en-US",
            "customer": "Sarah",
            "history": []
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["mode"] == "ai"
        assert data["provider"] == "gemini"
        assert "gemini" in data["model"].lower()
        # Ensure markdown symbols are stripped
        assert "**" not in data["reply"]
        assert "*" not in data["reply"]
        assert "FlowPilot integrates with over 150 enterprise tools" in data["reply"]


def test_gemini_selected_language_system_prompt_directives():
    """Verify system prompts generated for Gemini strictly mandate selected languages (Hindi, Telugu, English)."""
    from app import build_system_prompt

    # Hindi directive
    hindi_prompt = build_system_prompt(product_name="FlowPilot", language_code="hi-IN", prospect_name="विकास")
    assert "MANDATORY LANGUAGE DIRECTIVE: HINDI (हिंदी)" in hindi_prompt
    assert "Devanagari script" in hindi_prompt
    assert "NEVER switch to English" in hindi_prompt

    # Telugu directive
    telugu_prompt = build_system_prompt(product_name="CloudShield", language_code="te-IN", prospect_name="రమేష్")
    assert "MANDATORY LANGUAGE DIRECTIVE: TELUGU (తెలుగు)" in telugu_prompt
    assert "Telugu script" in telugu_prompt
    assert "NEVER switch to English" in telugu_prompt

    # English (India) directive
    indian_en_prompt = build_system_prompt(product_name="DataPulse", language_code="en-IN", prospect_name="Ananya")
    assert "ENGLISH (INDIA)" in indian_en_prompt


def test_gemini_genuine_structured_call_intelligence():
    """
    Verify genuine Gemini-generated call analysis with all 8 requested fields:
    1. Executive summary
    2. Customer requirements
    3. Objections with concern and response
    4. How AI responded to objections
    5. Buying signals
    6. Status classification
    7. Recommended next action (next_steps)
    8. Analysis mode and fallback flag
    """
    mock_gemini_payload = {
        "summary": "David evaluated DataPulse enterprise sales intelligence for pipeline forecasting.",
        "customer_requirements": ["Automated pipeline revenue prediction", "Salesforce CRM sync"],
        "objections": [
            {
                "concern": "Expressed concern about onboarding duration for 50 sales reps",
                "response": "Clarified that automated bulk sync completes in under 2 hours with guided assistance"
            }
        ],
        "buying_signals": ["Inquired about Enterprise tier contract and confirmed budget authorization"],
        "next_steps": "Send enterprise pricing agreement and schedule onboarding kickoff call.",
        "status": "Interested"
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = make_mock_gemini_response(mock_gemini_payload)

    with patch("app.get_gemini_client", return_value=mock_client):
        analysis = analyze_call_transcript(
            transcript=[
                {"role": "assistant", "text": "Welcome to DataPulse. How can we support your team?"},
                {"role": "user", "text": "We need automated pipeline revenue prediction and Salesforce sync."}
            ],
            product_name="DataPulse",
            language="en-US",
            customer="David",
            allow_fallback=False
        )

        assert analysis["analysis_mode"] == "ai"
        assert analysis["is_fallback"] is False
        assert "David evaluated DataPulse" in analysis["summary"]
        assert "Automated pipeline revenue prediction" in analysis["customer_requirements"]
        assert len(analysis["objections"]) == 1
        assert analysis["objections"][0]["concern"] == "Expressed concern about onboarding duration for 50 sales reps"
        assert "guided assistance" in analysis["objections"][0]["response"]
        assert "confirmed budget authorization" in analysis["buying_signals"][0]
        assert analysis["status"] == "Interested"
        assert "Send enterprise pricing agreement" in analysis["next_steps"]


def test_gemini_malformed_response_handling(client):
    """
    Verify that when Gemini returns malformed or non-JSON text during post-call analysis,
    the application gracefully falls back to local rule-based intelligence without crashing.
    """
    start_res = client.post("/api/calls/start", json={
        "customer": "Malformed Test User",
        "product": "FlowPilot",
        "consent": True
    })
    call_id = start_res.get_json()["id"]

    transcript = [
        {"role": "assistant", "text": "Hello, how can I help?"},
        {"role": "user", "text": "We are interested in automated workflow routing."}
    ]

    # Return broken non-JSON text
    mock_bad_client = MagicMock()
    mock_bad_client.models.generate_content.return_value = make_mock_gemini_response(
        "I am an AI assistant and here is your summary: {invalid_json: true, summary: "
    )

    with patch("app.get_gemini_client", return_value=mock_bad_client):
        end_res = client.post("/api/calls/end", json={
            "callId": call_id,
            "durationSeconds": 40,
            "transcript": transcript
        })
        assert end_res.status_code == 200
        data = end_res.get_json()
        assert data["success"] is True
        assert data["isFallback"] is True
        assert data["analysis_mode"] == "local_fallback"
        assert "not an ai-generated summary" in data["summary"].lower()


def test_gemini_transient_error_retry():
    """
    Verify that execute_with_transient_retry:
    - Retries transient errors (e.g., 503 Service Unavailable, timeout) using exponential backoff.
    - Successfully returns the result if a subsequent attempt succeeds.
    - Immediately raises permanent errors (e.g., 400 Invalid Argument, 429 Quota Exhausted) without retrying.
    """
    from app import execute_with_transient_retry

    # 1. Transient error recovering on 2nd attempt
    attempts = [0]
    def flaky_transient_operation():
        attempts[0] += 1
        if attempts[0] == 1:
            raise Exception("503 Service Unavailable: Backed off")
        return "success_recovered"

    result = execute_with_transient_retry(flaky_transient_operation, max_retries=2, initial_delay=0.01)
    assert result == "success_recovered"
    assert attempts[0] == 2

    # 2. Permanent error (400) fails immediately without repeated retries
    perm_attempts = [0]
    def permanent_error_operation():
        perm_attempts[0] += 1
        raise Exception("400 INVALID_ARGUMENT: API key not valid")

    with pytest.raises(Exception) as exc_info:
        execute_with_transient_retry(permanent_error_operation, max_retries=3, initial_delay=0.01)
    assert "API key not valid" in str(exc_info.value)
    assert perm_attempts[0] == 1  # Strictly 1 attempt, zero retries for permanent errors!


def test_gemini_live_smoke_test():
    """
    Separate real API smoke test using actual GEMINI_API_KEY from environment if present.
    Verifies that the application can reach the Gemini API and safely fails over to the
    fallback model (e.g. gemini-3.8-flash) if the primary model (gemini-3.6-flash) is quota exhausted.
    Skipped automatically when GEMINI_API_KEY is not configured, ensuring zero test failures in offline environments.
    """
    import os
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("GEMINI_API_KEY environment variable is not configured. Skipping live API smoke test.")

    from google import genai
    client = genai.Client(api_key=api_key)
    primary_model = get_gemini_model()
    fallback_models = get_gemini_fallback_models()

    response = None
    try:
        response, used_model, used_fallback = generate_content_with_quota_fallback(
            gemini_client=client,
            contents="Say 'VoxSales Gemini online' in exactly 3 words.",
            base_config_kwargs={"max_output_tokens": 300},
            primary_model=primary_model,
            fallback_model=fallback_models
        )
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            pytest.skip(f"Gemini API quota currently exhausted across all configured models: {str(e)[:120]}. Skipping live smoke test.")
        raise

    # Extract text safely from response or candidate parts
    text = getattr(response, "text", "") or ""
    if not text and response and response.candidates:
        parts = [p.text for c in response.candidates if c.content and c.content.parts for p in c.content.parts if getattr(p, "text", None)]
        text = " ".join(parts).strip()
    assert len(text.strip()) > 0
    assert used_model in ([primary_model] + fallback_models)


# ==============================================================================
# Deletion & Bulk Deletion Tests
# ==============================================================================

def test_individual_call_deletion_with_recording_file(client):
    """Verify that deleting an individual call removes the database record and unlinks its audio recording file."""
    start_res = client.post("/api/calls/start", json={
        "customer": "Individual Delete Test",
        "company": "Acme Corp",
        "product": "flowpilot",
        "language": "en",
        "consent": True
    })
    assert start_res.status_code == 200
    call_id = start_res.get_json()["callId"]

    # Create dummy recording file in RECORDINGS_DIR
    dummy_filename = f"{call_id}.webm"
    dummy_file = RECORDINGS_DIR / dummy_filename
    dummy_file.write_bytes(b"dummy audio data for testing")
    assert dummy_file.exists()

    with get_db() as conn:
        conn.execute("UPDATE calls SET recording = ? WHERE id = ?", (dummy_filename, call_id))

    # Perform individual DELETE
    del_res = client.delete(f"/api/calls/{call_id}")
    assert del_res.status_code == 200
    assert del_res.get_json()["success"] is True

    # Verify recording file unlinked
    assert not dummy_file.exists()

    # Verify database record deleted
    get_res = client.get(f"/api/calls/{call_id}")
    assert get_res.status_code == 404


def test_individual_call_deletion_missing_recording_file(client):
    """Verify that deleting a call whose recording file does not exist on disk still succeeds safely without error."""
    start_res = client.post("/api/calls/start", json={
        "customer": "Missing Audio Delete Test",
        "product": "flowpilot",
        "language": "en",
        "consent": True
    })
    assert start_res.status_code == 200
    call_id = start_res.get_json()["callId"]

    with get_db() as conn:
        conn.execute("UPDATE calls SET recording = ? WHERE id = ?", ("non_existent_file_xyz.webm", call_id))

    # Deletion should safely succeed despite missing recording
    del_res = client.delete(f"/api/calls/{call_id}")
    assert del_res.status_code == 200
    assert del_res.get_json()["success"] is True

    get_res = client.get(f"/api/calls/{call_id}")
    assert get_res.status_code == 404


def test_individual_call_deletion_invalid_id(client):
    """Verify that deleting a nonexistent call ID returns a 404 error."""
    del_res = client.delete("/api/calls/non_existent_random_id_99999")
    assert del_res.status_code == 404
    assert "error" in del_res.get_json()


def test_bulk_delete_calls_success(client):
    """Verify that bulk deletion deletes multiple database records and unlinks their recording files, while preserving unselected calls."""
    # Create 3 calls
    call_ids = []
    dummy_files = []
    for i in range(3):
        res = client.post("/api/calls/start", json={
            "customer": f"Bulk Prospect {i}",
            "product": "flowpilot",
            "language": "en",
            "consent": True
        })
        cid = res.get_json()["callId"]
        call_ids.append(cid)
        # Create audio file
        d_file = RECORDINGS_DIR / f"{cid}.webm"
        d_file.write_bytes(f"audio content {i}".encode("utf-8"))
        dummy_files.append(d_file)
        with get_db() as conn:
            conn.execute("UPDATE calls SET recording = ? WHERE id = ?", (f"{cid}.webm", cid))

    assert all(f.exists() for f in dummy_files)

    # Bulk delete the first two calls
    to_delete = [call_ids[0], call_ids[1]]
    bulk_res = client.post("/api/calls/bulk-delete", json={"call_ids": to_delete})
    assert bulk_res.status_code == 200
    data = bulk_res.get_json()
    assert data["success"] is True
    assert data["deleted_count"] == 2
    assert set(data["deleted_ids"]) == set(to_delete)

    # First two recording files should be deleted
    assert not dummy_files[0].exists()
    assert not dummy_files[1].exists()

    # Third call's recording file should STILL exist
    assert dummy_files[2].exists()

    # First two calls should return 404
    assert client.get(f"/api/calls/{call_ids[0]}").status_code == 404
    assert client.get(f"/api/calls/{call_ids[1]}").status_code == 404

    # Third call should remain intact
    third_res = client.get(f"/api/calls/{call_ids[2]}")
    assert third_res.status_code == 200
    assert third_res.get_json()["customer"] == "Bulk Prospect 2"

    # Cleanup third call
    client.delete(f"/api/calls/{call_ids[2]}")


def test_bulk_delete_via_delete_http_method(client):
    """Verify that the bulk delete endpoint accepts the DELETE HTTP method in addition to POST."""
    call_ids = []
    for i in range(2):
        res = client.post("/api/calls/start", json={
            "customer": f"Method Prospect {i}",
            "product": "flowpilot",
            "language": "en",
            "consent": True
        })
        call_ids.append(res.get_json()["callId"])

    del_res = client.delete("/api/calls/bulk-delete", json={"call_ids": call_ids})
    assert del_res.status_code == 200
    data = del_res.get_json()
    assert data["success"] is True
    assert data["deleted_count"] == 2

    for cid in call_ids:
        assert client.get(f"/api/calls/{cid}").status_code == 404


def test_bulk_delete_validation_empty_or_invalid_payload(client):
    """Verify error responses for missing, empty, or malformed bulk delete payloads."""
    # Empty JSON body
    res1 = client.post("/api/calls/bulk-delete", json={})
    assert res1.status_code == 400
    assert res1.get_json()["success"] is False

    # Empty list
    res2 = client.post("/api/calls/bulk-delete", json={"call_ids": []})
    assert res2.status_code == 400

    # Non-list value
    res3 = client.post("/api/calls/bulk-delete", json={"call_ids": "not_a_list"})
    assert res3.status_code == 400

    # Whitespace-only strings
    res4 = client.post("/api/calls/bulk-delete", json={"call_ids": ["  ", ""]})
    assert res4.status_code == 400


def test_bulk_delete_nonexistent_ids_returns_404(client):
    """Verify that providing only nonexistent IDs returns a 404 status."""
    res = client.post("/api/calls/bulk-delete", json={"call_ids": ["ghost_id_1", "ghost_id_2"]})
    assert res.status_code == 404
    data = res.get_json()
    assert data["success"] is False
    assert data["deleted_count"] == 0
    assert set(data["not_found_ids"]) == {"ghost_id_1", "ghost_id_2"}


def test_bulk_delete_partial_valid_and_invalid_ids(client):
    """Verify that bulk delete removes valid IDs and reports invalid IDs without failing."""
    start_res = client.post("/api/calls/start", json={
        "customer": "Partial Match Test",
        "product": "flowpilot",
        "language": "en",
        "consent": True
    })
    valid_id = start_res.get_json()["callId"]
    invalid_id = "ghost_partial_123"

    res = client.post("/api/calls/bulk-delete", json={"call_ids": [valid_id, invalid_id]})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["deleted_count"] == 1
    assert data["deleted_ids"] == [valid_id]
    assert data["not_found_ids"] == [invalid_id]

    assert client.get(f"/api/calls/{valid_id}").status_code == 404


def test_bulk_delete_missing_recording_files(client):
    """Verify bulk delete succeeds smoothly when calls reference recording files that do not exist on disk."""
    call_ids = []
    for i in range(2):
        res = client.post("/api/calls/start", json={
            "customer": f"Missing Audio Bulk {i}",
            "product": "flowpilot",
            "language": "en",
            "consent": True
        })
        cid = res.get_json()["callId"]
        call_ids.append(cid)
        with get_db() as conn:
            conn.execute("UPDATE calls SET recording = ? WHERE id = ?", (f"missing_disk_{cid}.webm", cid))

    res = client.post("/api/calls/bulk-delete", json={"call_ids": call_ids})
    assert res.status_code == 200
    data = res.get_json()
    assert data["deleted_count"] == 2

    for cid in call_ids:
        assert client.get(f"/api/calls/{cid}").status_code == 404


def test_bulk_delete_cleans_ai_analysis_data(client):
    """Verify that bulk delete completely purges saved AI analysis data (requirements, objections, signals, summary)."""
    start_res = client.post("/api/calls/start", json={
        "customer": "AI Intelligence Test Prospect",
        "company": "DeepTech",
        "product": "flowpilot",
        "language": "en",
        "consent": True
    })
    cid = start_res.get_json()["callId"]

    # Insert rich AI analysis data
    with get_db() as conn:
        conn.execute("""
            UPDATE calls SET
                summary = 'Executive summary of discussion',
                customer_requirements = '["Automated workflow", "CRM sync"]',
                objections = '["Security compliance questions"]',
                buying_signals = '["Budget approved for Q4"]',
                next_steps = '["Send contract"]',
                sales_status = 'Interested',
                status = 'Interested',
                processing_status = 'completed',
                analysis_mode = 'ai'
            WHERE id = ?
        """, (cid,))

    # Confirm it can be retrieved with all intelligence fields
    get_res = client.get(f"/api/calls/{cid}")
    assert get_res.status_code == 200
    intel = get_res.get_json()
    assert intel["summary"] == "Executive summary of discussion"
    assert "CRM sync" in intel["customerRequirements"]

    # Bulk delete the call
    del_res = client.post("/api/calls/bulk-delete", json={"call_ids": [cid]})
    assert del_res.status_code == 200

    # Ensure it is completely gone
    assert client.get(f"/api/calls/{cid}").status_code == 404
    hist_res = client.get("/api/history?q=DeepTech")
    assert hist_res.status_code == 200
    assert not any(item["id"] == cid for item in hist_res.get_json())


# ==============================================================================
# Model Quota & Fallback Tests
# ==============================================================================

def test_quota_fallback_chat_endpoint_success(client):
    """
    Verify that when the primary model encounters a 429 quota exhaustion error,
    /api/chat automatically switches to the fallback model and returns an AI response.
    """
    mock_client = MagicMock()

    def side_effect(model, contents, config):
        if "3.6" in model:
            raise Exception("429 RESOURCE_EXHAUSTED: You have exceeded your current quota for gemini-3.6-flash.")
        # Fallback model succeeds
        return make_mock_gemini_response("FlowPilot automates sales follow-ups and connects to your CRM seamlessly.")

    mock_client.models.generate_content.side_effect = side_effect

    with patch("app.get_gemini_client", return_value=mock_client):
        res = client.post("/api/chat", json={
            "message": "What does FlowPilot do?",
            "product": "FlowPilot",
            "language": "en-US",
            "customer": "Sarah",
            "history": []
        })

        assert res.status_code == 200
        data = res.get_json()
        assert data["mode"] == "ai"
        assert data["provider"] == "gemini"
        assert data["model"] == "gemini-3.8-flash"
        assert data["fallbackUsed"] is True
        assert "FlowPilot" in data["reply"]
        assert mock_client.models.generate_content.call_count == 2


def test_quota_fallback_analysis_success():
    """
    Verify that when the primary model has quota exhaustion during call intelligence analysis,
    analyze_call_transcript safely executes using the fallback model.
    """
    mock_client = MagicMock()
    mock_payload = {
        "summary": "Customer discussed automated sales pipeline management.",
        "customer_requirements": ["Automated email cadence", "HubSpot integration"],
        "objections": [{"concern": "Data residency in EU", "response": "FlowPilot offers EU-based data centers"}],
        "buying_signals": ["Requested pricing schedule"],
        "next_steps": "Send proposal and schedule walkthrough.",
        "status": "Interested"
    }

    def side_effect(model, contents, config):
        if "3.6" in model:
            raise Exception("429 RESOURCE_EXHAUSTED: Quota limit reached for model gemini-3.6-flash.")
        return make_mock_gemini_response(mock_payload)

    mock_client.models.generate_content.side_effect = side_effect

    transcript = [
        {"role": "assistant", "text": "Hello, how can FlowPilot help?"},
        {"role": "user", "text": "We need EU data centers and HubSpot integration."}
    ]

    with patch("app.get_gemini_client", return_value=mock_client):
        analysis = analyze_call_transcript(
            transcript=transcript,
            product_name="FlowPilot",
            language="en-US",
            customer="European Client",
            allow_fallback=False
        )

        assert analysis["analysis_mode"] == "ai"
        assert analysis["model"] == "gemini-3.8-flash"
        assert analysis["fallback_model_used"] is True
        assert analysis["is_fallback"] is False
        assert "EU-based data centers" in str(analysis["objections"])
        assert mock_client.models.generate_content.call_count == 2


def test_quota_fallback_both_fail_triggers_local_fallback(client):
    """
    Verify behavior when BOTH primary and fallback models encounter quota exhaustion:
    1. /api/chat falls through to stateful local conversational engine.
    2. analyze_call_transcript(allow_fallback=False) raises GeminiQuotaExhaustedError.
    3. analyze_call_transcript(allow_fallback=True) falls through to rule-based Local Fallback Analysis.
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "429 RESOURCE_EXHAUSTED: All Gemini quota exhausted."
    )

    transcript = [
        {"role": "assistant", "text": "Welcome to FlowPilot!"},
        {"role": "user", "text": "Can you provide custom pricing for our 50-person team?"}
    ]

    with patch("app.get_gemini_client", return_value=mock_client):
        # 1. Chat fallback to local engine
        chat_res = client.post("/api/chat", json={
            "message": "Can you provide custom pricing for our 50-person team?",
            "product": "FlowPilot",
            "language": "en-US",
            "customer": "Alex",
            "history": []
        })
        assert chat_res.status_code == 200
        assert chat_res.get_json()["mode"] == "fallback"

        # 2. Strict analysis raises GeminiQuotaExhaustedError
        with pytest.raises(GeminiQuotaExhaustedError):
            analyze_call_transcript(
                transcript=transcript,
                product_name="FlowPilot",
                language="en-US",
                customer="Alex",
                allow_fallback=False
            )

        # 3. Permissive analysis produces safe Local Fallback Analysis
        fb_analysis = analyze_call_transcript(
            transcript=transcript,
            product_name="FlowPilot",
            language="en-US",
            customer="Alex",
            allow_fallback=True
        )
        assert fb_analysis["analysis_mode"] == "local_fallback"
        assert fb_analysis["is_fallback"] is True
        assert "Rule-Based Fallback" in fb_analysis["summary"]


def test_non_quota_error_does_not_trigger_fallback():
    """
    Verify that non-quota errors (such as 400 Invalid Argument or authentication errors)
    do NOT attempt the fallback model and fail immediately without wasteful retries.
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "400 INVALID_ARGUMENT: Invalid parameter specified."
    )

    transcript = [{"role": "user", "text": "Hello"}]

    with patch("app.get_gemini_client", return_value=mock_client):
        with pytest.raises(GeminiAnalysisError) as exc_info:
            analyze_call_transcript(
                transcript=transcript,
                product_name="FlowPilot",
                language="en-US",
                customer="Alex",
                allow_fallback=False
            )

        assert "Invalid parameter" in str(exc_info.value)
        # Verify fallback was NOT attempted (strictly 1 attempt on primary model)
        assert mock_client.models.generate_content.call_count == 1


def test_fallback_disabled_configuration(monkeypatch):
    """
    Verify that when GEMINI_FALLBACK_MODEL is configured to 'none', 'disabled', or empty,
    fallback is skipped and quota exhaustion is raised directly.
    """
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "none")
    assert get_gemini_fallback_model() is None

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Quota exhausted.")

    with patch("app.get_gemini_client", return_value=mock_client):
        with pytest.raises(GeminiQuotaExhaustedError):
            analyze_call_transcript(
                transcript=[{"role": "user", "text": "Hi"}],
                product_name="FlowPilot",
                language="en-US",
                customer="Alex",
                allow_fallback=False
            )
        # Exactly 1 call attempted because fallback is disabled
        assert mock_client.models.generate_content.call_count == 1


def test_gemini_fallback_model_comma_separated_parsing(monkeypatch):
    """
    Verify that GEMINI_FALLBACK_MODEL=gemini-3.8-flash,gemini-3-flash-preview
    is parsed correctly as two separate model names with whitespace trimmed.
    """
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash,gemini-3-flash-preview")
    models = get_gemini_fallback_models()
    assert models == ["gemini-3.8-flash", "gemini-3-flash-preview"]
    assert len(models) == 2
    assert models[0] == "gemini-3.8-flash"
    assert models[1] == "gemini-3-flash-preview"

    # Also test whitespace and empty item trimming
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "  gemini-3.8-flash  , , gemini-3-flash-preview  ")
    assert get_gemini_fallback_models() == ["gemini-3.8-flash", "gemini-3-flash-preview"]


def test_sequential_quota_fallback_chat_endpoint(client, monkeypatch):
    """
    Verify that when primary model (3.6) and first fallback (3.8) both hit quota exhaustion (429),
    /api/chat sequentially attempts the second fallback (gemini-3-flash-preview) and returns
    the successful AI chat reply with model='gemini-3-flash-preview' and fallbackUsed=True.
    """
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash,gemini-3-flash-preview")
    called_models = []

    mock_client = MagicMock()
    def side_effect(model, contents, config):
        called_models.append(model)
        if "3.6" in model:
            raise Exception("429 RESOURCE_EXHAUSTED: gemini-3.6-flash quota exhausted.")
        if "3.8" in model:
            raise Exception("429 RESOURCE_EXHAUSTED: gemini-3.8-flash quota exhausted.")
        return make_mock_gemini_response("FlowPilot boosts sales outreach with automated workflows.")

    mock_client.models.generate_content.side_effect = side_effect

    with patch("app.get_gemini_client", return_value=mock_client):
        res = client.post("/api/chat", json={
            "message": "Tell me about FlowPilot features.",
            "product": "FlowPilot",
            "language": "en-US",
            "customer": "Jordan",
            "history": []
        })

        assert res.status_code == 200
        data = res.get_json()
        assert data["mode"] == "ai"
        assert data["provider"] == "gemini"
        assert data["model"] == "gemini-3-flash-preview"
        assert data["fallbackUsed"] is True
        assert "FlowPilot" in data["reply"]
        # Verify sequential order of attempts
        assert called_models == ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3-flash-preview"]


def test_sequential_quota_fallback_call_analysis(monkeypatch):
    """
    Verify that when primary model (3.6) and first fallback (3.8) both encounter quota exhaustion,
    analyze_call_transcript sequentially attempts the second fallback (gemini-3-flash-preview)
    and succeeds with an AI analysis payload.
    """
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash,gemini-3-flash-preview")
    called_models = []

    mock_client = MagicMock()
    mock_payload = {
        "summary": "Customer discussed enterprise CRM data synchronization with FlowPilot.",
        "customer_requirements": ["Automated sync", "SOC-2 Type II"],
        "objections": [{"concern": "Data residency in UK", "response": "FlowPilot supports UK cloud regions"}],
        "buying_signals": ["Inquired about enterprise onboarding schedule"],
        "next_steps": "Send SOC-2 report and setup proof-of-concept tenant.",
        "status": "Demo Requested"
    }

    def side_effect(model, contents, config):
        called_models.append(model)
        if "3.6" in model:
            raise Exception("429 RESOURCE_EXHAUSTED: Primary quota limit hit.")
        if "3.8" in model:
            raise Exception("429 RESOURCE_EXHAUSTED: First fallback quota limit hit.")
        return make_mock_gemini_response(mock_payload)

    mock_client.models.generate_content.side_effect = side_effect

    transcript = [
        {"role": "assistant", "text": "Welcome to FlowPilot! How can we assist?"},
        {"role": "user", "text": "We need UK data residency and automated CRM sync for our sales team."}
    ]

    with patch("app.get_gemini_client", return_value=mock_client):
        analysis = analyze_call_transcript(
            transcript=transcript,
            product_name="FlowPilot",
            language="en-US",
            customer="Jordan",
            allow_fallback=False
        )

        assert analysis["analysis_mode"] == "ai"
        assert analysis["model"] == "gemini-3-flash-preview"
        assert analysis["fallback_model_used"] is True
        assert analysis["is_fallback"] is False
        assert analysis["status"] == "Demo Requested"
        assert "enterprise CRM data synchronization" in analysis["summary"]
        assert called_models == ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3-flash-preview"]










