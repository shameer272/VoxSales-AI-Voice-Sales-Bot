/**
 * VoxSales — AI Voice Sales Agent
 * Frontend Client Application
 */

const $ = id => document.getElementById(id);

// Application State
const state = {
  callId: null,
  viewingCallId: null,
  targetCallId: null,
  failedCallId: null,
  historyAbortController: null,
  historySelectionSeq: 0,
  selectedCallIds: new Set(),
  currentHistoryItems: [],
  isActive: false,
  isMuted: false,
  isSpeaking: false,
  isProcessing: false,
  isFinalizing: false,
  startedAt: 0,
  timerInterval: null,
  transcript: [],
  recordedChunks: [],
  mediaStream: null,
  mediaRecorder: null,
  recognition: null,
  interimNode: null,
  availableProducts: {},
  currentProductSpecs: null,
  currentAudio: null,
  audioCtx: null,
  mixDestination: null,
  turnCounter: 0,
  lastSubmittedText: '',
  lastSubmittedTime: 0,
  lastSpokenText: ''
};

// Language map for Web Speech API and synthesis
const langMapping = {
  'en-US': { name: 'English (US)', speech: 'en-US', native: 'English (US)', prefix: 'en' },
  'en-IN': { name: 'English (India)', speech: 'en-IN', native: 'English (India)', prefix: 'en' },
  'hi-IN': { name: 'Hindi', speech: 'hi-IN', native: 'हिंदी', prefix: 'hi' },
  'te-IN': { name: 'Telugu', speech: 'te-IN', native: 'తెలుగు', prefix: 'te' }
};

// Update active language badge and input prompts
function updateActiveLanguageUI(langCode) {
  const pill = $('activeLangPill');
  if (pill) {
    if (!langCode || !langMapping[langCode]) {
      pill.textContent = '🌐 Select Language';
      pill.classList.add('unselected');
      pill.title = 'No call language selected';
    } else {
      const info = langMapping[langCode];
      pill.textContent = `🌐 ${info.name}`;
      pill.classList.remove('unselected');
      pill.title = `Active Call Language: ${info.name} (${info.native}) [${info.speech}]`;
    }
  }

  const input = $('textFallbackInput');
  if (input && !state.viewingCallId) {
    if (langCode && langMapping[langCode]) {
      const info = langMapping[langCode];
      input.placeholder = `Type message in ${info.name} (${info.native}) or speak…`;
    } else {
      input.placeholder = 'Type customer message here (or speak into your microphone)…';
    }
  }
}

// Show a temporary toast notification
function showToast(message, duration = 3500) {
  const toast = $('toast');
  if (!toast) return;
  toast.textContent = message;
  toast.style.display = 'block';
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => {
    toast.style.display = 'none';
  }, duration);
}

// Format duration in MM:SS
function formatDuration(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

// Update UI call status and banner
function setCallStatus(statusText, visualWaveState = 'idle') {
  const badge = $('callStatusPill');
  const wave = $('voiceWave');
  const banner = $('voiceStateBanner');
  const text = $('voiceStateText');

  if (badge) badge.textContent = statusText.toUpperCase();
  if (text) text.textContent = statusText;

  if (wave) {
    if (visualWaveState === 'active') {
      wave.classList.add('active');
    } else {
      wave.classList.remove('active');
    }
  }
}

// Update elapsed call timer
function tickTimer() {
  const elapsed = Math.floor((Date.now() - state.startedAt) / 1000);
  const timerElem = $('callDuration');
  if (timerElem) timerElem.textContent = formatDuration(elapsed);
  return elapsed;
}

// Append a message to the transcript view
function appendMessage(role, text, isInterim = false, customSenderName = null, timestampStr = null) {
  const transcriptBox = $('conversationBox');
  const emptyPlaceholder = $('emptyStatePlaceholder');
  if (emptyPlaceholder) emptyPlaceholder.remove();

  const isUser = (role === 'user' || role === 'prospect');
  const msgElem = document.createElement('div');
  msgElem.className = `chat-msg ${isUser ? 'user' : 'ai'}${isInterim ? ' interim' : ''}`;

  let timeDisplay = '';
  if (isInterim) {
    timeDisplay = '• listening';
  } else if (timestampStr) {
    try {
      timeDisplay = '• ' + new Date(timestampStr).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      timeDisplay = '• ' + timestampStr;
    }
  } else {
    timeDisplay = '• ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  const defaultUser = state.viewingCallId ? 'Prospect' : ($('prospectName').value || 'Prospect');
  const label = isUser ? (customSenderName || defaultUser) : 'VoxSales AI';

  msgElem.innerHTML = `
    <div class="msg-avatar">${isUser ? 'YOU' : 'AI'}</div>
    <div class="msg-content-wrapper">
      <div class="msg-meta">
        <span class="msg-sender-name">${escapeHtml(label)}</span>
        <span class="msg-time">${timeDisplay}</span>
      </div>
      <div class="msg-bubble">${escapeHtml(text)}</div>
    </div>
  `;

  transcriptBox.appendChild(msgElem);
  transcriptBox.scrollTop = transcriptBox.scrollHeight;
  return msgElem;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// Find compatible installed browser voice
function findCompatibleVoice(langCode) {
  if (!('speechSynthesis' in window)) return null;
  const voices = window.speechSynthesis.getVoices();
  if (!voices || voices.length === 0) return null;

  const target = (langCode || 'en-US').toLowerCase();
  const primaryPrefix = target.split('-')[0]; // 'hi', 'te', 'en'

  // 1. Exact match e.g. 'te-IN' === 'te-in' or 'te_in'
  let voice = voices.find(v => v.lang.replace('_', '-').toLowerCase() === target);
  if (voice) return voice;

  // 2. Prefix match e.g. starts with 'te-' or 'hi-' or matches 'te'
  voice = voices.find(v => {
    const l = v.lang.replace('_', '-').toLowerCase();
    return l.startsWith(primaryPrefix + '-') || l === primaryPrefix;
  });
  if (voice) return voice;

  // 3. Name hints for Indian language voices (e.g. Google हिन्दी, Microsoft Mohan, Swara, Chitra, etc.)
  const nameKeywords = {
    'hi-in': ['hindi', 'हिन्दी', 'kalpana', 'hemant', 'swara', 'madhur'],
    'te-in': ['telugu', 'తెలుగు', 'chitra', 'mohan']
  };
  const keywords = nameKeywords[target] || [];
  for (const kw of keywords) {
    voice = voices.find(v => v.name.toLowerCase().includes(kw));
    if (voice) return voice;
  }

  return null;
}

// Stop all active speech (browser synthesis and server audio playback)
function stopAllSpeech() {
  if (state.currentAudio) {
    try {
      state.currentAudio.pause();
      state.currentAudio.currentTime = 0;
    } catch (e) {}
    state.currentAudio = null;
  }
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
  state.isSpeaking = false;
}

// Attach playing audio element to recording mixer so AI responses are included in call recording
function attachAudioToMixer(audio) {
  if (!state.audioCtx || !state.mixDestination) return;
  try {
    if (!audio._sourceNode) {
      audio.crossOrigin = 'anonymous';
      const source = state.audioCtx.createMediaElementSource(audio);
      source.connect(state.audioCtx.destination);
      source.connect(state.mixDestination);
      audio._sourceNode = source;
    }
  } catch (err) {
    console.warn('Could not attach audio element to mixer:', err);
  }
}

// Play speech via native browser SpeechSynthesis (Option 1)
function speakWithBrowserSpeech(text, langCode, matchedVoice) {
  return new Promise((resolve, reject) => {
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = matchedVoice ? matchedVoice.lang : langCode;
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      if (matchedVoice) utterance.voice = matchedVoice;

      state.isSpeaking = true;
      setCallStatus('AI Speaking…', 'active');

      utterance.onend = () => {
        state.isSpeaking = false;
        resolve();
      };
      utterance.onerror = (e) => {
        state.isSpeaking = false;
        reject(e);
      };
      window.speechSynthesis.speak(utterance);
    } catch (err) {
      state.isSpeaking = false;
      reject(err);
    }
  });
}

// Play speech via Server-Side TTS Fallback (Option 2 - supports Telugu te-IN, Hindi hi-IN, English)
async function playServerTTS(text, langCode) {
  stopAllSpeech();

  state.isSpeaking = true;
  setCallStatus('AI Speaking…', 'active');

  const langInfo = langMapping[langCode] || { name: langCode, native: langCode };

  try {
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        language: langCode || $('callLanguage').value || 'en-US'
      })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `TTS server responded with status ${res.status}`);
    }

    const audioBlob = await res.blob();
    if (!audioBlob || audioBlob.size < 50) {
      throw new Error('Received invalid audio from TTS provider');
    }

    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio();
    audio.src = audioUrl;
    state.currentAudio = audio;

    // Route into Web Audio mixer if available
    attachAudioToMixer(audio);

    await new Promise((resolve, reject) => {
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        state.currentAudio = null;
        resolve();
      };
      audio.onerror = (e) => {
        URL.revokeObjectURL(audioUrl);
        state.currentAudio = null;
        console.warn('Audio playback error:', e);
        reject(new Error('Audio playback failed in browser'));
      };
      audio.play().catch(playErr => {
        URL.revokeObjectURL(audioUrl);
        state.currentAudio = null;
        console.warn('Audio play() was interrupted or blocked:', playErr);
        reject(playErr);
      });
    });

  } catch (err) {
    console.warn(`Server TTS playback failed for ${langInfo.name}:`, err);
    showToast(`Voice playback unavailable for ${langInfo.name}. Response displayed in text.`, 4000);
  } finally {
    state.isSpeaking = false;
  }
}

// Primary Speech Dispatcher:
// 1. Tries installed browser voice first if a compatible one exists.
// 2. Automatically falls back to server-side TTS (which explicitly supports Telugu, Hindi, English).
async function speakText(text, langCode) {
  if (!text || !text.trim()) return;

  const selectedLang = langCode || $('callLanguage').value || 'en-US';
  state.lastSpokenText = text.trim();
  state.isSpeaking = true;

  // Stop any overlapping speech first
  stopAllSpeech();
  state.isSpeaking = true;

  try {
    // Option 1: Browser SpeechSynthesis if genuinely compatible installed voice exists
    const matchedVoice = findCompatibleVoice(selectedLang);
    if (matchedVoice && 'speechSynthesis' in window) {
      try {
        console.log(`Using browser voice: ${matchedVoice.name} (${matchedVoice.lang}) for ${selectedLang}`);
        await speakWithBrowserSpeech(text, selectedLang, matchedVoice);
        return;
      } catch (browserTtsErr) {
        console.warn('Browser speech synthesis error, falling back to server TTS:', browserTtsErr);
      }
    }

    // Option 2: Fall back to Server-Side TTS provider (supports Telugu, Hindi, and English without browser voice packs)
    console.log(`Using server-side TTS fallback for language: ${selectedLang}`);
    await playServerTTS(text, selectedLang);
  } finally {
    state.isSpeaking = false;
  }
}

// Localized initial greeting helper
function getFallbackGreeting(lang, name, product) {
  const pName = name ? ' ' + name : '';
  switch (lang) {
    case 'hi-IN':
      return `नमस्ते${pName}! कॉल पर बात करने के लिए धन्यवाद। मैं ${product} के लिए AI सेल्स विशेषज्ञ हूँ। वर्तमान में आपके वर्कफ़्लो में सबसे बड़ी चुनौती क्या आ रही है?`;
    case 'te-IN':
      return `నమస్కారం${pName}! కాల్ మాట్లాడినందుకు ధన్యవాదాలు. నేను ${product} సేల్స్ AI నిపుణుడిని. మీ ప్రస్తుత వర్క్‌ఫ్లోలో మీరు ఎదుర్కొంటున్న అతిపెద్ద సవాలు ఏమిటి?`;
    case 'en-IN':
      return `Hello${pName}, thank you for taking my call. I am the AI sales specialist for ${product}. What is the biggest bottleneck your team is facing in your workflow today?`;
    case 'en-US':
    default:
      return `Hello${pName}! Thanks for taking my call. I am the AI sales specialist for ${product}. What is your biggest challenge with your current workflow right now?`;
  }
}

// Initialize Web Speech Recognition
function setupSpeechRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    showToast('Browser speech recognition not supported. You can use the text input below to converse.');
    const speechPill = $('speechSupportPill');
    if (speechPill) {
      speechPill.innerHTML = '<span class="pulse-dot amber"></span>Text Mode Fallback';
    }
    return false;
  }

  const rec = new SpeechRec();
  const selectedLang = $('callLanguage').value || 'en-US';
  rec.lang = selectedLang;
  rec.continuous = true;
  rec.interimResults = true;

  rec.onresult = async (event) => {
    if (!state.isActive || state.isMuted || state.isSpeaking || state.isProcessing || state.isFinalizing) return;

    let interimText = '';
    let finalText = '';

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const trans = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalText += trans + ' ';
      } else {
        interimText += trans;
      }
    }

    if (interimText.trim() && !state.isSpeaking && !state.isProcessing) {
      if (!state.interimNode) {
        state.interimNode = appendMessage('user', interimText, true);
      } else {
        const bubble = state.interimNode.querySelector('.msg-bubble');
        if (bubble) bubble.textContent = interimText;
        $('conversationBox').scrollTop = $('conversationBox').scrollHeight;
      }
    }

    if (finalText.trim()) {
      if (state.interimNode) {
        state.interimNode.remove();
        state.interimNode = null;
      }
      const cleanUserText = finalText.trim();
      // Guard against microphone picking up recent AI speech (audio echo from speaker)
      const normUser = cleanUserText.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, '').replace(/\s+/g, ' ').trim();
      const normSpoken = (state.lastSpokenText || '').toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, '').replace(/\s+/g, ' ').trim();
      if (normSpoken && (normUser === normSpoken || (normUser.length > 15 && normSpoken.includes(normUser)) || (normSpoken.length > 15 && normUser.includes(normSpoken)))) {
        console.log('Ignored speech recognition echo of AI speech:', cleanUserText);
        return;
      }
      handleUserSubmission(cleanUserText);
    }
  };

  rec.onerror = (event) => {
    if (event.error === 'not-allowed') {
      showToast('Microphone access was denied in browser settings.');
    } else if (event.error === 'language-not-supported') {
      const langName = (langMapping[$('callLanguage').value] && langMapping[$('callLanguage').value].name) || $('callLanguage').value;
      showToast(`Speech recognition is not supported for ${langName} in this browser. Please use the text input below.`, 5000);
      const speechPill = $('speechSupportPill');
      if (speechPill) {
        speechPill.innerHTML = '<span class="pulse-dot amber"></span>Text Mode Fallback';
      }
    } else if (event.error !== 'no-speech' && event.error !== 'aborted') {
      console.warn('Speech recognition warning:', event.error);
    }
  };

  rec.onend = () => {
    // Automatically restart listening if call is active and AI isn't speaking
    if (state.isActive && !state.isMuted && !state.isSpeaking && !state.isProcessing) {
      try { rec.start(); } catch (e) {}
    }
  };

  state.recognition = rec;
  return true;
}

// Safely switch conversation language on the fly during an active call (Requirement 8)
function handleActiveLanguageSwitch(newLang) {
  if (!newLang || !state.isActive) return;

  const langInfo = langMapping[newLang] || { name: newLang, speech: newLang };
  console.log(`Active call language switched to: ${langInfo.name} (${newLang})`);

  // 1. Safely stop current speech synthesis, audio playback, and recognition
  stopAllSpeech();

  if (state.recognition) {
    state.recognition.onend = null; // Prevent immediate restart with old language
    try { state.recognition.stop(); } catch (e) {}
    state.recognition.lang = newLang;
    state.recognition.onend = () => {
      if (state.isActive && !state.isMuted && !state.isSpeaking && !state.isProcessing) {
        try { state.recognition.start(); } catch (e) {}
      }
    };
  }

  // 2. Update active language badge and input placeholder
  updateActiveLanguageUI(newLang);

  // 3. User notification
  showToast(`Active language switched to ${langInfo.name}. Voice & text will use ${langInfo.name}.`);

  // 5. Restart recognition safely if active and unmuted
  if (state.isActive && !state.isMuted && !state.isSpeaking && !state.isProcessing && state.recognition) {
    setTimeout(() => {
      if (state.isActive && !state.isMuted && !state.isSpeaking && !state.isProcessing) {
        try { state.recognition.start(); } catch (e) {}
        setCallStatus('Listening…', 'idle');
      }
    }, 200);
  }
}

// Central handler for user turns (from mic or text box)
async function handleUserSubmission(text) {
  const cleanText = (text || '').trim();
  if (!cleanText || !state.isActive || !state.callId || state.isProcessing || state.isFinalizing) return;

  // Prevent duplicate submissions of the identical text within 2.5 seconds
  const now = Date.now();
  if (state.lastSubmittedText === cleanText && (now - state.lastSubmittedTime) < 2500) {
    console.log('Ignoring duplicate utterance within 2.5s:', cleanText);
    return;
  }
  state.lastSubmittedText = cleanText;
  state.lastSubmittedTime = now;

  state.isProcessing = true;
  const currentTurnId = ++state.turnCounter;

  // Disable submission controls while processing
  if ($('fallbackSendBtn')) $('fallbackSendBtn').disabled = true;
  if ($('textFallbackInput')) $('textFallbackInput').disabled = true;

  // Temporarily stop microphone recognition so it does not capture ambient noise or AI audio
  if (state.recognition) {
    state.recognition.onend = null;
    try { state.recognition.stop(); } catch (e) {}
  }

  // Save customer turn to transcript exactly once
  state.transcript.push({
    role: 'user',
    text: cleanText,
    timestamp: new Date().toISOString()
  });
  appendMessage('user', cleanText);

  setCallStatus('AI Thinking…', 'active');

  try {
    const payload = {
      message: cleanText,
      product: $('saasProductSelect').value,
      language: $('callLanguage').value,
      customer: $('prospectName').value || 'Guest',
      company: $('prospectCompany').value || '',
      notes: $('prospectNotes').value || '',
      history: state.transcript
    };

    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'AI generation failed');

    // Stale turn check
    if (state.turnCounter !== currentTurnId || !state.isActive) {
      console.warn('Discarding response for stale turn', currentTurnId);
      return;
    }

    const reply = data.reply || 'Thank you for sharing that.';

    // Save AI reply to transcript exactly once
    state.transcript.push({
      role: 'assistant',
      text: reply,
      timestamp: new Date().toISOString()
    });
    appendMessage('assistant', reply);

    // Speak aloud in selected language
    await speakText(reply, $('callLanguage').value);

  } catch (err) {
    console.error('Error generating AI response:', err);
    showToast('Failed to reach AI service: ' + err.message);
  } finally {
    state.isProcessing = false;
    if (state.isActive) {
      if ($('fallbackSendBtn')) $('fallbackSendBtn').disabled = false;
      if ($('textFallbackInput')) $('textFallbackInput').disabled = false;
      if (!state.isMuted) {
        setCallStatus('Listening…', 'idle');
        setupSpeechRecognition();
        if (state.recognition) {
          try { state.recognition.start(); } catch (e) {}
        }
      }
    }
  }
}

// Form Validation Error Helpers
function clearFieldErrors() {
  ['prospectName', 'saasProductSelect', 'callLanguage'].forEach(id => {
    const el = $(id);
    if (el) el.classList.remove('field-invalid');
    const err = $(id + 'Error');
    if (err) {
      err.textContent = '';
      err.classList.remove('visible');
    }
  });
  const consentBox = $('consentBox');
  if (consentBox) consentBox.classList.remove('field-invalid');
  const consentErr = $('recordingConsentError');
  if (consentErr) {
    consentErr.textContent = '';
    consentErr.classList.remove('visible');
  }
}

function setFieldError(fieldId, errorId, message) {
  const el = $(fieldId);
  if (el) el.classList.add('field-invalid');
  const err = $(errorId);
  if (err) {
    err.textContent = message;
    err.classList.add('visible');
  }
}

function clearFieldError(fieldId, errorId) {
  const el = $(fieldId);
  if (el) el.classList.remove('field-invalid');
  const err = $(errorId);
  if (err) {
    err.textContent = '';
    err.classList.remove('visible');
  }
}

// Start Call Workflow
async function startCall() {
  if (state.isActive) return;

  clearFieldErrors();
  let hasError = false;
  let firstInvalid = null;

  const prospectNameVal = $('prospectName').value.trim();
  if (!prospectNameVal) {
    setFieldError('prospectName', 'prospectNameError', 'Prospect name is required');
    hasError = true;
    if (!firstInvalid) firstInvalid = $('prospectName');
  }

  const productVal = $('saasProductSelect').value.trim();
  if (!productVal) {
    setFieldError('saasProductSelect', 'saasProductSelectError', 'Please select a SaaS product');
    hasError = true;
    if (!firstInvalid) firstInvalid = $('saasProductSelect');
  }

  const langVal = $('callLanguage').value.trim();
  if (!langVal) {
    setFieldError('callLanguage', 'callLanguageError', 'Please select a language');
    hasError = true;
    if (!firstInvalid) firstInvalid = $('callLanguage');
  }

  const consentChecked = $('recordingConsentCheck').checked;
  if (!consentChecked) {
    const box = $('consentBox');
    if (box) box.classList.add('field-invalid');
    const err = $('recordingConsentError');
    if (err) {
      err.textContent = 'Recording consent is required before placing call';
      err.classList.add('visible');
    }
    hasError = true;
    if (!firstInvalid) firstInvalid = $('recordingConsentCheck');
  }

  if (hasError) {
    showToast('Please complete all required fields to start the call.');
    if (firstInvalid) firstInvalid.focus();
    return;
  }

  // Clear any historical call viewing state
  state.viewingCallId = null;
  state.targetCallId = null;
  state.failedCallId = null;
  document.querySelectorAll('.history-item').forEach(el => el.classList.remove('selected'));
  const newCallBtn = $('newCallBtn');
  if (newCallBtn) newCallBtn.style.display = 'none';
  $('startCallBtn').style.display = 'inline-flex';

  try {
    setCallStatus('Connecting…', 'active');

    // Request microphone access
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        state.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (micErr) {
        console.warn('Physical mic access unavailable, using synthetic stream fallback:', micErr);
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) {
          const dummyCtx = new AudioCtx();
          const osc = dummyCtx.createOscillator();
          const dst = dummyCtx.createMediaStreamDestination();
          osc.connect(dst);
          osc.start();
          state.mediaStream = dst.stream;
        } else {
          throw micErr;
        }
      }
    } else {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        const dummyCtx = new AudioCtx();
        const osc = dummyCtx.createOscillator();
        const dst = dummyCtx.createMediaStreamDestination();
        osc.connect(dst);
        osc.start();
        state.mediaStream = dst.stream;
      } else {
        throw new Error('Audio recording is not supported in this browser.');
      }
    }

    // Call backend to create call record
    const startPayload = {
      customer: prospectNameVal,
      company: $('prospectCompany').value.trim(),
      product: productVal,
      language: langVal,
      notes: $('prospectNotes').value.trim(),
      consent: true
    };

    const response = await fetch('/api/calls/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(startPayload)
    });

    const initData = await response.json();
    if (!response.ok) throw new Error(initData.error || 'Could not initiate call');

    state.callId = initData.id;
    state.transcript = [];
    state.recordedChunks = [];
    state.isActive = true;
    state.isMuted = false;
    state.isFinalizing = false;
    state.startedAt = Date.now();
    state.turnCounter = 0;
    state.lastSubmittedText = '';
    state.lastSubmittedTime = 0;
    state.lastSpokenText = '';

    // Set Call Intelligence panel into In Progress state
    setIntelState('in-progress');

    // Setup MediaRecorder with Web Audio mixer if supported
    let recordStream = state.mediaStream;
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        state.audioCtx = new AudioContextClass();
        const micSource = state.audioCtx.createMediaStreamSource(state.mediaStream);
        state.mixDestination = state.audioCtx.createMediaStreamDestination();
        micSource.connect(state.mixDestination);
        recordStream = state.mixDestination.stream;
      }
    } catch (mixErr) {
      console.warn('AudioContext mixer setup warning, using mic stream directly:', mixErr);
      recordStream = state.mediaStream;
    }

    try {
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
      state.mediaRecorder = new MediaRecorder(recordStream, { mimeType: mime });
      state.mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          state.recordedChunks.push(e.data);
        }
      };
      state.mediaRecorder.start(600);
    } catch (recErr) {
      console.warn('MediaRecorder error:', recErr);
    }

    // Lock configuration fields but keep callLanguage enabled for active switching
    setFormControlsDisabled(true, true);
    $('startCallBtn').disabled = true;
    $('endCallBtn').disabled = false;
    $('muteBtn').disabled = false;
    $('fallbackSendBtn').disabled = false;
    $('textFallbackInput').disabled = false;

    // Start timer
    state.timerInterval = setInterval(tickTimer, 1000);

    // Clear transcript UI
    $('conversationBox').innerHTML = '';

    // AI Initial Spoken Greeting in the selected language
    const prospectName = $('prospectName').value.trim();
    const productName = $('saasProductSelect').value;
    const greeting = initData.greeting || getFallbackGreeting(langVal, prospectName, productName);

    state.transcript.push({
      role: 'assistant',
      text: greeting,
      timestamp: new Date().toISOString()
    });
    appendMessage('assistant', greeting);

    updateActiveLanguageUI(langVal);
    await speakText(greeting, langVal);

    // Initialize and start speech recognition only AFTER greeting has finished speaking
    if (state.isActive) {
      setupSpeechRecognition();
      if (state.recognition && !state.isMuted) {
        try { state.recognition.start(); } catch (e) {}
      }
      if (!state.isMuted) {
        setCallStatus('Listening…', 'idle');
      }
    }

  } catch (err) {
    console.error('Call start failed:', err);
    showToast(err.message || 'Microphone access denied or error starting call.');
    cleanupMedia();
    setCallStatus('Ready', 'idle');
    setIntelState('awaiting');
    setFormControlsDisabled(false);
    $('startCallBtn').disabled = false;
    $('endCallBtn').disabled = true;
  }
}

// End Call Workflow
async function endCall() {
  if (!state.isActive || !state.callId || state.isFinalizing) return;
  state.isFinalizing = true;
  state.isActive = false;

  clearInterval(state.timerInterval);
  const durationSec = tickTimer();
  setCallStatus('Processing Analysis…', 'active');
  setIntelState('analyzing');

  $('startCallBtn').disabled = true;
  $('endCallBtn').disabled = true;
  $('muteBtn').disabled = true;
  $('fallbackSendBtn').disabled = true;
  $('textFallbackInput').disabled = true;

  try {
    // Stop recognition & synthesis
    if (state.recognition) {
      state.recognition.onend = null;
      try { state.recognition.stop(); } catch (e) {}
    }
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }

    // Stop MediaRecorder and wait for completion
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
      await new Promise(resolve => {
        state.mediaRecorder.onstop = resolve;
        state.mediaRecorder.stop();
      });
    }

    // Stop tracks
    cleanupMedia();

    // Create recording blob
    const audioBlob = new Blob(state.recordedChunks, { type: 'audio/webm' });

    // Send to backend via FormData
    const formData = new FormData();
    formData.append('callId', state.callId);
    formData.append('durationSeconds', durationSec);
    formData.append('transcript', JSON.stringify(state.transcript));
    formData.append('customer', $('prospectName').value || 'Guest');
    formData.append('company', $('prospectCompany').value || '');
    formData.append('product', $('saasProductSelect').value || 'FlowPilot');
    formData.append('language', $('callLanguage').value || 'en-US');
    formData.append('notes', $('prospectNotes').value || '');

    if (audioBlob.size > 0) {
      formData.append('recording', audioBlob, `${state.callId}.webm`);
    }

    const res = await fetch('/api/calls/end', {
      method: 'POST',
      body: formData
    });

    const result = await res.json();
    if (!res.ok || result.success === false) {
      throw new Error(result.error || 'Failed to finalize call analysis');
    }

    // Display Intelligence
    displayIntelligence(result);
    showToast('Call analysis complete and saved to history!');
    await loadHistory();

  } catch (err) {
    console.error('Failed to end call:', err);
    setIntelState('failed', err.message, state.callId);
    showToast('Call ended, but analysis failed: ' + err.message);
    await loadHistory();
  } finally {
    state.isFinalizing = false;
    setCallStatus('Completed', 'idle');
    setFormControlsDisabled(false);
    $('startCallBtn').disabled = false;
    $('startCallBtn').style.display = 'inline-flex';
    $('endCallBtn').disabled = true;
    $('muteBtn').disabled = true;
    $('textFallbackInput').disabled = true;
    $('fallbackSendBtn').disabled = true;
  }
}

// Manage dynamic Call Intelligence panel UI states
function setIntelState(mode, errorMsg = '', targetCallId = null) {
  const badge = $('intelSalesStatusBadge');
  const loadingBanner = $('intelLoadingBanner');
  const errorCard = $('intelErrorCard');
  const fallbackBanner = $('intelFallbackBanner');
  const errorMsgElem = $('intelErrorMessage');
  const summaryText = $('intelSummaryText');
  const reqText = $('intelRequirementsText');
  const objText = $('intelObjectionsText');
  const buyText = $('intelSignalsText');
  const nextText = $('intelNextStepsText');
  const audioWrap = $('audioPlayerCard');

  if (targetCallId) {
    state.targetCallId = targetCallId;
  }

  if (mode === 'awaiting') {
    if (loadingBanner) loadingBanner.style.display = 'none';
    if (errorCard) errorCard.style.display = 'none';
    if (fallbackBanner) fallbackBanner.style.display = 'none';
    if (audioWrap) audioWrap.style.display = 'none';
    if (badge) {
      badge.textContent = 'AWAITING CALL';
      badge.className = 'status-badge awaiting-call';
    }
    if (summaryText) summaryText.textContent = 'Complete a sales call to generate insights.';
    if (reqText) reqText.innerHTML = '<span class="intel-placeholder">No customer requirements recorded yet.</span>';
    if (objText) objText.innerHTML = '<span class="intel-placeholder">No questions or objections recorded yet.</span>';
    if (buyText) buyText.innerHTML = '<span class="intel-placeholder">No buying signals recorded yet.</span>';
    if (nextText) nextText.innerHTML = '<span class="intel-placeholder">No next action recommended yet.</span>';
  } else if (mode === 'in-progress') {
    if (loadingBanner) loadingBanner.style.display = 'none';
    if (errorCard) errorCard.style.display = 'none';
    if (fallbackBanner) fallbackBanner.style.display = 'none';
    if (audioWrap) audioWrap.style.display = 'none';
    if (badge) {
      badge.textContent = 'IN PROGRESS';
      badge.className = 'status-badge in-progress';
    }
    if (summaryText) summaryText.textContent = 'Conversation in progress. Insights will be generated when the call ends.';
    if (reqText) reqText.innerHTML = '<span class="intel-placeholder">Listening for customer needs and pain points…</span>';
    if (objText) objText.innerHTML = '<span class="intel-placeholder">Tracking customer questions and objections in real time…</span>';
    if (buyText) buyText.innerHTML = '<span class="intel-placeholder">Monitoring customer purchasing signals…</span>';
    if (nextText) nextText.innerHTML = '<span class="intel-placeholder">Next action will be recommended upon call completion.</span>';
  } else if (mode === 'analyzing') {
    if (loadingBanner) loadingBanner.style.display = 'flex';
    if (errorCard) errorCard.style.display = 'none';
    if (fallbackBanner) fallbackBanner.style.display = 'none';
    if (badge) {
      badge.textContent = 'ANALYZING…';
      badge.className = 'status-badge analyzing';
    }
    if (summaryText) summaryText.textContent = 'Analyzing conversation and generating sales insights…';
  } else if (mode === 'failed') {
    if (loadingBanner) loadingBanner.style.display = 'none';
    if (errorCard) errorCard.style.display = 'flex';
    if (fallbackBanner) fallbackBanner.style.display = 'none';
    if (errorMsgElem) {
      errorMsgElem.textContent = errorMsg || 'An error occurred while communicating with the AI service. Your transcript and call recording have been safely preserved.';
    }
    if (badge) {
      badge.textContent = 'ANALYSIS FAILED';
      badge.className = 'status-badge failed';
    }
  }
}

// Render customer requirements as readable bullet points
function renderRequirements(reqs) {
  if (!reqs) return '<span class="intel-placeholder">None identified.</span>';
  let items = [];
  if (Array.isArray(reqs)) {
    items = reqs.map(r => String(r).trim()).filter(Boolean);
  } else if (typeof reqs === 'string') {
    const trimmed = reqs.trim();
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) items = parsed.map(r => String(r).trim()).filter(Boolean);
      } catch (e) {
        items = [trimmed];
      }
    } else {
      items = trimmed.split('\n').map(r => r.replace(/^[-*•\d.]+\s*/, '').trim()).filter(Boolean);
    }
  }

  const validItems = items.filter(it => it && !/^none(\s+identified|\s+noted|\s+captured)?\.?$/i.test(it.trim()));
  if (validItems.length === 0) {
    return '<span class="intel-placeholder">None identified.</span>';
  }
  return `<ul class="intel-bullets">${validItems.map(it => `<li>${escapeHtml(it)}</li>`).join('')}</ul>`;
}

// Render customer questions and objections with AI response pairing
function renderObjections(objs) {
  if (!objs) return '<span class="intel-placeholder">No questions or objections were raised.</span>';
  let items = [];
  if (Array.isArray(objs)) {
    items = objs;
  } else if (typeof objs === 'string') {
    const trimmed = objs.trim();
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) items = parsed;
      } catch (e) {
        items = [trimmed];
      }
    } else {
      items = trimmed.split('\n').map(o => o.trim()).filter(Boolean);
    }
  }

  const rendered = items.map(obj => {
    if (typeof obj === 'object' && obj !== null) {
      const concern = (obj.concern || obj.objection || obj.question || '').trim();
      const response = (obj.response || obj.answer || obj.ai_response || '').trim();
      if (!concern || /^none(\s+raised|\s+expressed|\s+noted)?\.?$/i.test(concern)) {
        return '';
      }
      return `
        <div class="intel-objection-card">
          <div class="intel-objection-concern">
            <span class="intel-tag tag-concern">Customer Concern</span>
            <span>${escapeHtml(concern)}</span>
          </div>
          ${response && !/^n\/?a$/i.test(response) && !/^none(\s+raised|\s+expressed)?\.?$/i.test(response) ? `
            <div class="intel-objection-response">
              <span class="intel-tag tag-response">AI Response</span>
              <span>${escapeHtml(response)}</span>
            </div>
          ` : ''}
        </div>
      `;
    } else {
      const str = String(obj).trim();
      if (!str || /^none(\s+raised|\s+expressed|\s+noted)?\.?$/i.test(str)) {
        return '';
      }
      return `
        <div class="intel-objection-card">
          <div class="intel-objection-concern">
            <span class="intel-tag tag-concern">Question / Objection</span>
            <span>${escapeHtml(str)}</span>
          </div>
        </div>
      `;
    }
  }).filter(Boolean);

  if (rendered.length === 0) {
    return '<span class="intel-placeholder">No questions or objections were raised.</span>';
  }
  return `<div class="intel-objections-list">${rendered.join('')}</div>`;
}

// Render buying signals with brief evidence
function renderBuyingSignals(sigs) {
  if (!sigs) return '<span class="intel-placeholder">No clear buying signals detected.</span>';
  let items = [];
  if (Array.isArray(sigs)) {
    items = sigs.map(s => String(s).trim()).filter(Boolean);
  } else if (typeof sigs === 'string') {
    const trimmed = sigs.trim();
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) items = parsed.map(s => String(s).trim()).filter(Boolean);
      } catch (e) {
        items = [trimmed];
      }
    } else {
      items = trimmed.split('\n').map(s => s.replace(/^[-*•\d.]+\s*/, '').trim()).filter(Boolean);
    }
  }

  const validItems = items.filter(it => it && !/^no(\s+clear)?\s+buying\s+signals?(\s+detected|\s+identified|\s+recorded)?\.?$/i.test(it.trim()));
  if (validItems.length === 0) {
    return '<span class="intel-placeholder">No clear buying signals detected.</span>';
  }
  return `<ul class="intel-bullets signals-bullets">${validItems.map(it => `<li>${escapeHtml(it)}</li>`).join('')}</ul>`;
}

// Display Intelligence in Right Panel
function displayIntelligence(data) {
  if (!data) return;

  const loadingBanner = $('intelLoadingBanner');
  const errorCard = $('intelErrorCard');
  const fallbackBanner = $('intelFallbackBanner');
  if (loadingBanner) loadingBanner.style.display = 'none';
  if (errorCard) errorCard.style.display = 'none';

  const isFallback = (
    data.isFallback === true ||
    data.is_fallback === true ||
    data.analysis_mode === 'local_fallback' ||
    data.analysisMode === 'local_fallback' ||
    data.processing_status === 'quota_exhausted' ||
    data.processingStatus === 'quota_exhausted'
  );

  if (fallbackBanner) {
    if (isFallback) {
      fallbackBanner.style.display = 'flex';
      state.targetCallId = data.callId || data.id || state.targetCallId || state.callId;
      const noticeElem = $('fallbackBannerNotice');
      if (noticeElem && data.quotaNotice) {
        noticeElem.textContent = data.quotaNotice;
      }
    } else {
      fallbackBanner.style.display = 'none';
    }
  }

  const statusBadge = $('intelSalesStatusBadge');
  const summaryText = $('intelSummaryText');
  const reqText = $('intelRequirementsText');
  const objText = $('intelObjectionsText');
  const buyText = $('intelSignalsText');
  const nextText = $('intelNextStepsText');
  const audioWrap = $('audioPlayerCard');
  const audioElem = $('callAudioPlayer');

  const rawStatus = data.status || data.sales_status || 'No Response';
  if (statusBadge) {
    statusBadge.textContent = isFallback ? `${rawStatus} • LOCAL FALLBACK` : rawStatus;
    const statusClass = rawStatus.toLowerCase().replace(/\s+/g, '-');
    statusBadge.className = 'status-badge ' + statusClass + (isFallback ? ' fallback-mode' : '');
  }

  if (summaryText) {
    summaryText.textContent = data.summary ? data.summary : 'No executive summary available for this call.';
  }

  if (reqText) {
    reqText.innerHTML = renderRequirements(data.customerRequirements || data.customer_requirements);
  }

  if (objText) {
    objText.innerHTML = renderObjections(data.objections);
  }

  if (buyText) {
    buyText.innerHTML = renderBuyingSignals(data.buyingSignals || data.buying_signals);
  }

  if (nextText) {
    const nextVal = data.nextSteps || data.next_steps || '';
    if (nextVal && !/^none\.?$/i.test(nextVal.trim())) {
      nextText.textContent = nextVal;
    } else {
      nextText.innerHTML = '<span class="intel-placeholder">No next action recommended yet.</span>';
    }
  }

  if (data.recordingUrl) {
    audioWrap.style.display = 'flex';
    audioElem.src = data.recordingUrl + (data.recordingUrl.includes('?') ? '&' : '?') + 't=' + Date.now();
  } else {
    audioWrap.style.display = 'none';
    audioElem.removeAttribute('src');
    try { audioElem.pause(); } catch (e) {}
  }
}

// Retry Analysis Workflow
async function retryAnalysis() {
  const callId = state.targetCallId || state.failedCallId || state.viewingCallId || state.callId;
  if (!callId) {
    showToast('No call selected to retry analysis.');
    return;
  }

  const retryBtn = $('retryAnalysisBtn');
  const retryFallbackBtn = $('retryFallbackAnalysisBtn');
  if (retryBtn) retryBtn.disabled = true;
  if (retryFallbackBtn) retryFallbackBtn.disabled = true;
  setIntelState('analyzing');

  try {
    const res = await fetch(`/api/calls/${encodeURIComponent(callId)}/retry-analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });

    const data = await res.json();
    if (!res.ok || data.success === false) {
      throw new Error(data.error || 'Retry analysis failed');
    }

    displayIntelligence(data);
    showToast('AI analysis regenerated and saved successfully!');
    await loadHistory();
  } catch (err) {
    console.error('Retry analysis failed:', err);
    // Restore existing preserved record and local fallback
    try {
      const callRes = await fetch(`/api/calls/${encodeURIComponent(callId)}`);
      if (callRes.ok) {
        const callData = await callRes.json();
        displayIntelligence(callData);
      } else {
        setIntelState('failed', err.message, callId);
      }
    } catch (e) {
      setIntelState('failed', err.message, callId);
    }
    showToast(err.message || 'Analysis retry failed', 6000);
  } finally {
    if (retryBtn) retryBtn.disabled = false;
    if (retryFallbackBtn) retryFallbackBtn.disabled = false;
  }
}

// Toggle Microphone Mute
function toggleMute() {
  if (!state.isActive) return;
  state.isMuted = !state.isMuted;

  const muteBtn = $('muteBtn');
  if (state.isMuted) {
    muteBtn.classList.add('muted');
    muteBtn.innerHTML = '🔇 Unmute';
    setCallStatus('Mic Muted', 'idle');
    if (state.recognition) {
      try { state.recognition.stop(); } catch (e) {}
    }
  } else {
    muteBtn.classList.remove('muted');
    muteBtn.innerHTML = '🎤 Mute';
    setCallStatus('Listening…', 'idle');
    if (state.recognition) {
      try { state.recognition.start(); } catch (e) {}
    }
  }
}

// Clean up media streams and audio contexts
function cleanupMedia() {
  stopAllSpeech();
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach(track => track.stop());
    state.mediaStream = null;
  }
  if (state.audioCtx) {
    try { state.audioCtx.close(); } catch (e) {}
    state.audioCtx = null;
    state.mixDestination = null;
  }
}

// Enable/Disable form controls
// If isLiveCall is true, callLanguage is kept editable to allow on-the-fly language switching (Requirement 8)
function setFormControlsDisabled(disabled, isLiveCall = false) {
  $('prospectName').disabled = disabled;
  $('prospectCompany').disabled = disabled;
  $('saasProductSelect').disabled = disabled;
  $('callLanguage').disabled = isLiveCall ? false : disabled;
  $('prospectNotes').disabled = disabled;
  $('recordingConsentCheck').disabled = disabled;
}

// Load Call History with Search & Filter
async function loadHistory() {
  try {
    const q = $('historySearchInput').value.trim();
    const status = $('historyStatusFilter').value.trim();

    const params = new URLSearchParams();
    if (q) params.set('q', q);
    if (status) params.set('status', status);

    const res = await fetch('/api/history?' + params.toString());
    const items = await res.json();

    state.currentHistoryItems = Array.isArray(items) ? items : [];

    const list = $('historyList');
    list.innerHTML = '';

    if (!state.currentHistoryItems || state.currentHistoryItems.length === 0) {
      list.innerHTML = '<div class="empty-state" style="padding: 24px;">No call records found.</div>';
      updateBulkActionsUI();
      return;
    }

    state.currentHistoryItems.forEach(item => {
      const itemNode = document.createElement('div');
      itemNode.className = 'history-item';
      itemNode.dataset.id = item.id;
      if (item.id === state.viewingCallId || item.id === state.callId) {
        itemNode.classList.add('selected');
      }

      const isChecked = state.selectedCallIds.has(item.id);
      if (isChecked) {
        itemNode.classList.add('checked');
      }

      const dateStr = item.started_at ? new Date(item.started_at).toLocaleString([], {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
      }) : 'Recent';

      const statusClass = (item.status || 'no-response').toLowerCase().replace(/\s+/g, '-');
      const isFallback = (
        item.processingStatus === 'quota_exhausted' ||
        item.processing_status === 'quota_exhausted' ||
        item.analysis_mode === 'local_fallback' ||
        item.analysisMode === 'local_fallback' ||
        item.isFallback === true
      );

      itemNode.innerHTML = `
        <div class="history-item-checkbox-wrap">
          <input type="checkbox" class="call-checkbox call-select-checkbox" data-id="${item.id}" ${isChecked ? 'checked' : ''} aria-label="Select call with ${escapeHtml(item.customer || 'Guest')}">
        </div>
        <div class="history-item-content">
          <div class="hist-top">
            <span class="hist-name">${escapeHtml(item.customer || 'Guest')}</span>
            <div style="display: flex; gap: 5px; align-items: center;">
              ${isFallback ? '<span class="status-badge quota-pill" title="Local Fallback (Gemini Quota or Key Unavailable)">Local Fallback</span>' : ''}
              <span class="status-badge ${statusClass}">${item.status || 'No Response'}</span>
            </div>
          </div>
          <div class="hist-sub">
            ${escapeHtml(item.company ? item.company + ' • ' : '')}${escapeHtml(item.product || 'FlowPilot')}
          </div>
          <div class="hist-bottom">
            <span>⏱ ${formatDuration(item.duration_seconds || 0)} • ${dateStr}</span>
            <div>
              ${item.recording ? '<span title="Recording Available">🎙</span> ' : ''}
              <button class="delete-btn" title="Delete record" data-id="${item.id}">✕</button>
            </div>
          </div>
        </div>
      `;

      const checkbox = itemNode.querySelector('.call-select-checkbox');
      if (checkbox) {
        checkbox.addEventListener('click', (e) => {
          e.stopPropagation();
        });
        checkbox.addEventListener('change', (e) => {
          e.stopPropagation();
          if (checkbox.checked) {
            state.selectedCallIds.add(item.id);
            itemNode.classList.add('checked');
          } else {
            state.selectedCallIds.delete(item.id);
            itemNode.classList.remove('checked');
          }
          updateBulkActionsUI();
        });
      }

      itemNode.addEventListener('click', (e) => {
        if (e.target.closest('.delete-btn')) {
          e.stopPropagation();
          deleteCall(item.id);
          return;
        }
        if (e.target.closest('.history-item-checkbox-wrap') || e.target.classList.contains('call-checkbox')) {
          return;
        }
        loadCallDetails(item.id);
      });

      list.appendChild(itemNode);
    });

    updateBulkActionsUI();

  } catch (err) {
    console.error('Failed to load history:', err);
  }
}

/**
 * Loads complete saved call record by its unique call ID.
 * Protects against race conditions / stale results using an AbortController and sequence counter.
 */
async function loadCallDetails(callId) {
  if (!callId) return;

  // Abort any prior pending call detail fetch
  if (state.historyAbortController) {
    state.historyAbortController.abort();
  }
  state.historyAbortController = new AbortController();
  const currentSeq = ++state.historySelectionSeq;

  // If a live call is active, prompt confirmation before leaving
  if (state.isActive) {
    if (!confirm('A call is currently in progress. Switching to historical call view will end monitoring. Proceed?')) {
      return;
    }
    cleanupMedia();
    state.isActive = false;
    clearInterval(state.timerInterval);
  }

  // Update selected history item highlight immediately in UI
  document.querySelectorAll('.history-item').forEach(el => {
    if (el.dataset.id === callId) {
      el.classList.add('selected');
    } else {
      el.classList.remove('selected');
    }
  });

  try {
    const res = await fetch(`/api/calls/${encodeURIComponent(callId)}`, {
      signal: state.historyAbortController.signal
    });

    if (!res.ok) {
      throw new Error(`Failed to load call details (${res.status})`);
    }

    const call = await res.json();

    // Stale check: discard if another call was selected while this request was in flight
    if (currentSeq !== state.historySelectionSeq) {
      return;
    }

    applyHistoricalCallToUI(call);

  } catch (err) {
    if (err.name === 'AbortError') {
      // Intentionally aborted by newer selection, ignore silently
      return;
    }
    console.error('Error loading call details:', err);
    showToast('Failed to load call details: ' + err.message);
  }
}

/**
 * Renders complete historical call record into UI:
 * - Updates inputs (prospect, company, product, language, notes)
 * - Updates duration and historical status badge
 * - Replaces transcript completely with caller's messages (no leftover Alex messages)
 * - Updates Call Intelligence and audio recording player
 * - Disables live call actions and shows "+ New Call" button
 */
function applyHistoricalCallToUI(call) {
  // Mark state as viewing historical call and clear live call ID
  state.viewingCallId = call.id;
  state.callId = null;
  state.failedCallId = call.id;
  state.targetCallId = call.id;

  clearFieldErrors();

  // 1. Update prospect / call setup inputs
  $('prospectName').value = call.customer || '';
  $('prospectCompany').value = call.company || '';
  if (call.product && $('saasProductSelect').querySelector(`option[value="${call.product}"]`)) {
    $('saasProductSelect').value = call.product;
  } else {
    $('saasProductSelect').value = '';
  }
  if (call.language && $('callLanguage').querySelector(`option[value="${call.language}"]`)) {
    $('callLanguage').value = call.language;
  } else {
    $('callLanguage').value = '';
  }
  updateActiveLanguageUI(call.language || '');
  $('prospectNotes').value = call.notes || '';

  // Disable form controls while viewing history
  setFormControlsDisabled(true, false);

  // 2. Duration & Status
  $('callDuration').textContent = formatDuration(call.duration_seconds || 0);

  const statusPill = $('callStatusPill');
  if (statusPill) {
    statusPill.textContent = 'HISTORICAL CALL';
    statusPill.className = 'call-status-pill historical';
  }
  const bannerText = $('voiceStateText');
  if (bannerText) {
    bannerText.textContent = `Viewing Record: ${call.customer || 'Historical Call'}`;
  }
  const wave = $('voiceWave');
  if (wave) wave.classList.remove('active');

  // Controls: Disable live call controls and show "+ New Call"
  $('startCallBtn').style.display = 'none';
  $('startCallBtn').disabled = true;
  $('endCallBtn').disabled = true;
  $('muteBtn').disabled = true;
  $('textFallbackInput').disabled = true;
  $('textFallbackInput').placeholder = 'Viewing historical call (chat input disabled)';
  $('fallbackSendBtn').disabled = true;

  const newCallBtn = $('newCallBtn');
  if (newCallBtn) {
    newCallBtn.style.display = 'inline-flex';
    newCallBtn.disabled = false;
  }

  // 3. Replace entire transcript with the selected record's transcript
  const box = $('conversationBox');
  box.innerHTML = '';

  const transcript = Array.isArray(call.transcript) ? call.transcript : [];
  if (transcript.length === 0) {
    box.innerHTML = '<div class="empty-state" style="padding: 30px;">No transcript was recorded for this call.</div>';
  } else {
    const customerName = call.customer || 'Prospect';
    transcript.forEach(msg => {
      const isUser = (msg.role === 'user' || msg.role === 'prospect');
      appendMessage(
        msg.role,
        msg.text || '',
        false,
        isUser ? customerName : 'VoxSales AI',
        msg.timestamp
      );
    });
  }

  // 4. Update Call Intelligence panel and Recording Player
  const procStatus = call.processing_status || call.processingStatus;
  if (procStatus === 'failed') {
    setIntelState('failed', 'AI analysis failed for this call.', call.id);
  } else {
    displayIntelligence(call);
  }

  showToast(`Loaded details for call with ${call.customer || 'Guest'}.`);
}

/**
 * Restores UI and state back to fresh Live Call preparation mode
 */
function resetToNewCallState() {
  state.viewingCallId = null;
  state.callId = null;
  state.transcript = [];
  state.recordedChunks = [];
  state.targetCallId = null;
  state.failedCallId = null;

  // Unselect any history item in list
  document.querySelectorAll('.history-item').forEach(el => el.classList.remove('selected'));

  // Reset inputs to blank defaults
  $('prospectName').value = '';
  $('prospectCompany').value = '';
  $('saasProductSelect').value = '';
  $('callLanguage').value = '';
  updateActiveLanguageUI('');
  $('prospectNotes').value = '';
  $('recordingConsentCheck').checked = false;
  clearFieldErrors();

  setFormControlsDisabled(false);

  // Reset duration and status
  $('callDuration').textContent = '00:00';
  const statusPill = $('callStatusPill');
  if (statusPill) {
    statusPill.textContent = 'READY';
    statusPill.className = 'call-status-pill';
  }
  const bannerText = $('voiceStateText');
  if (bannerText) bannerText.textContent = 'Agent idle. Ready to place call.';
  const wave = $('voiceWave');
  if (wave) wave.classList.remove('active');

  // Controls
  $('startCallBtn').style.display = 'inline-flex';
  $('startCallBtn').disabled = false;
  $('endCallBtn').disabled = true;
  $('muteBtn').disabled = true;
  $('muteBtn').classList.remove('muted');
  $('muteBtn').innerHTML = '🎤 Mute';

  const newCallBtn = $('newCallBtn');
  if (newCallBtn) newCallBtn.style.display = 'none';

  $('textFallbackInput').disabled = true;
  $('textFallbackInput').placeholder = 'Type customer message here (or speak into your microphone)…';
  $('fallbackSendBtn').disabled = true;

  // Clear conversation box and restore default placeholder
  const box = $('conversationBox');
  box.innerHTML = `
    <div class="empty-state" id="emptyStatePlaceholder">
      <div class="empty-icon">🎙</div>
      <h3 style="font-size: 15px; margin-bottom: 6px; color: #e2e8f0;">Voice Session Inactive</h3>
      <p style="font-size: 12.5px;">Click <strong>Start Call</strong> to initiate the AI voice agent. The AI will introduce the product, discover requirements, and speak dynamically.</p>
    </div>
  `;

  // Reset Call Intelligence to awaiting
  setIntelState('awaiting');

  showToast('Ready for new call.');
}

// Bulk selection UI updater
function updateBulkActionsUI() {
  const totalVisible = state.currentHistoryItems ? state.currentHistoryItems.length : 0;
  const selectedCount = state.selectedCallIds.size;
  const selectAllCb = $('selectAllCalls');
  const delBtn = $('deleteSelectedBtn');
  const countSpan = $('deleteSelectedCount');
  const badge = $('selectedCallsCount');

  if (countSpan) {
    countSpan.textContent = selectedCount;
  }

  if (badge) {
    badge.textContent = `${selectedCount} selected`;
    if (selectedCount > 0) {
      badge.classList.add('has-selected');
    } else {
      badge.classList.remove('has-selected');
    }
  }

  if (delBtn) {
    delBtn.disabled = (selectedCount === 0);
    delBtn.setAttribute('aria-disabled', selectedCount === 0 ? 'true' : 'false');
  }

  if (selectAllCb) {
    if (totalVisible === 0) {
      selectAllCb.checked = false;
      selectAllCb.indeterminate = false;
      selectAllCb.disabled = true;
    } else {
      selectAllCb.disabled = false;
      const allVisibleSelected = totalVisible > 0 && state.currentHistoryItems.every(i => state.selectedCallIds.has(i.id));
      const someVisibleSelected = state.currentHistoryItems.some(i => state.selectedCallIds.has(i.id));
      selectAllCb.checked = allVisibleSelected;
      selectAllCb.indeterminate = !allVisibleSelected && someVisibleSelected;
    }
  }
}

// Select / Unselect all visible history items
function handleSelectAllChange() {
  const selectAllCb = $('selectAllCalls');
  if (!selectAllCb || !state.currentHistoryItems) return;
  const isChecked = selectAllCb.checked;

  state.currentHistoryItems.forEach(item => {
    if (isChecked) {
      state.selectedCallIds.add(item.id);
    } else {
      state.selectedCallIds.delete(item.id);
    }
  });

  document.querySelectorAll('.call-select-checkbox').forEach(cb => {
    const id = cb.dataset.id;
    cb.checked = state.selectedCallIds.has(id);
    const parentItem = cb.closest('.history-item');
    if (parentItem) {
      if (cb.checked) {
        parentItem.classList.add('checked');
      } else {
        parentItem.classList.remove('checked');
      }
    }
  });

  updateBulkActionsUI();
}

// Bulk delete selected call records
async function bulkDeleteCalls() {
  const selectedIds = Array.from(state.selectedCallIds);
  if (selectedIds.length === 0) {
    showToast('Please select at least one call to delete.');
    return;
  }

  const count = selectedIds.length;
  const confirmText = count === 1
    ? 'Are you sure you want to delete this selected call record and its recording? This action cannot be undone.'
    : `Are you sure you want to delete ${count} selected call records and their recordings? This action cannot be undone.`;

  if (!confirm(confirmText)) {
    return;
  }

  const delBtn = $('deleteSelectedBtn');
  if (delBtn) delBtn.disabled = true;

  try {
    const res = await fetch('/api/calls/bulk-delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ call_ids: selectedIds })
    });

    const data = await res.json();
    if (res.ok && data.success) {
      const deletedCount = data.deleted_count || count;
      showToast(`Successfully deleted ${deletedCount} call${deletedCount === 1 ? '' : 's'}.`);

      if (state.viewingCallId && selectedIds.includes(state.viewingCallId)) {
        resetToNewCallState();
      }

      state.selectedCallIds.clear();
      updateBulkActionsUI();
      await loadHistory();
    } else {
      showToast(data.error || 'Failed to delete selected calls.');
      updateBulkActionsUI();
    }
  } catch (err) {
    console.error('Bulk delete failed:', err);
    showToast('Failed to delete selected calls: ' + err.message);
    updateBulkActionsUI();
  }
}

// Delete call record
async function deleteCall(id) {
  if (!confirm('Are you sure you want to delete this call record and its recording? This action cannot be undone.')) return;
  try {
    const res = await fetch(`/api/calls/${id}`, { method: 'DELETE' });
    if (res.ok) {
      showToast('Call deleted.');
      if (state.viewingCallId === id) {
        resetToNewCallState();
      }
      state.selectedCallIds.delete(id);
      updateBulkActionsUI();
      await loadHistory();
    } else {
      const data = await res.json().catch(() => ({}));
      showToast(data.error || 'Failed to delete call.');
    }
  } catch (e) {
    showToast('Failed to delete call.');
  }
}

// View Product Specs Modal
function openProductModal() {
  const currentKey = $('saasProductSelect').value;
  const product = state.availableProducts[currentKey];
  if (!product) {
    showToast('Please select a SaaS product first to view its specifications.');
    return;
  }

  $('modalProductName').textContent = product.name;
  $('modalTagline').textContent = product.tagline;
  $('modalDescription').textContent = product.description;

  const featuresList = $('modalFeaturesList');
  featuresList.innerHTML = '';
  (product.features || []).forEach(f => {
    const li = document.createElement('li');
    li.textContent = f;
    featuresList.appendChild(li);
  });

  $('modalPricing').textContent = product.pricing;
  $('modalAudience').textContent = product.target_audience;
  $('modalLimitations').textContent = product.limitations;

  $('productModalOverlay').classList.add('open');
}

function closeProductModal() {
  $('productModalOverlay').classList.remove('open');
}

// Initialize Configuration and Bind Events
async function initApp() {
  try {
    const res = await fetch('/api/config');
    const config = await res.json();

    // AI status badge
    const aiPill = $('aiConnectionPill');
    if (aiPill) {
      if (config.aiConfigured) {
        aiPill.innerHTML = `<span class="pulse-dot"></span>AI Active (${config.model || 'gemini-3.6-flash'})`;
      } else {
        aiPill.innerHTML = `<span class="pulse-dot amber"></span>Fallback Engine (Add API Key)`;
        aiPill.title = 'Add GEMINI_API_KEY in .env for genuine Gemini responses';
      }
    }

    // Populate products
    if (config.products) {
      state.availableProducts = config.products;
      const select = $('saasProductSelect');
      const currentVal = select.value;
      select.innerHTML = '<option value="" disabled selected>Select a SaaS product</option>';
      Object.keys(config.products).forEach(key => {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = `${config.products[key].name} — ${config.products[key].tagline.slice(0, 32)}…`;
        select.appendChild(opt);
      });
      if (currentVal && config.products[currentVal]) {
        select.value = currentVal;
      } else {
        select.value = '';
      }
    }

  } catch (err) {
    console.error('Failed to load initial configuration:', err);
  }

  // Event Listeners
  $('startCallBtn').addEventListener('click', startCall);
  $('endCallBtn').addEventListener('click', endCall);
  $('muteBtn').addEventListener('click', toggleMute);
  const newCallBtn = $('newCallBtn');
  if (newCallBtn) newCallBtn.addEventListener('click', resetToNewCallState);
  $('viewSpecsBtn').addEventListener('click', openProductModal);
  $('closeModalBtn').addEventListener('click', closeProductModal);
  $('productModalOverlay').addEventListener('click', (e) => {
    if (e.target === $('productModalOverlay')) closeProductModal();
  });

  // Real-time validation clearance and language switching listeners
  $('prospectName').addEventListener('input', () => {
    if ($('prospectName').value.trim()) clearFieldError('prospectName', 'prospectNameError');
  });
  $('saasProductSelect').addEventListener('change', () => {
    if ($('saasProductSelect').value) clearFieldError('saasProductSelect', 'saasProductSelectError');
  });
  $('callLanguage').addEventListener('change', () => {
    const val = $('callLanguage').value;
    if (val) {
      clearFieldError('callLanguage', 'callLanguageError');
      updateActiveLanguageUI(val);
      if (state.isActive) {
        handleActiveLanguageSwitch(val);
      }
    } else {
      updateActiveLanguageUI('');
    }
  });

  // Listen for dynamic voice loading in browser
  if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => {
      updateActiveLanguageUI($('callLanguage').value);
    };
  }
  $('recordingConsentCheck').addEventListener('change', () => {
    if ($('recordingConsentCheck').checked) {
      const box = $('consentBox');
      if (box) box.classList.remove('field-invalid');
      const err = $('recordingConsentError');
      if (err) {
        err.textContent = '';
        err.classList.remove('visible');
      }
    }
  });

  // Fallback text input send
  const sendFallback = () => {
    const input = $('textFallbackInput');
    const val = input.value.trim();
    if (!val) return;
    input.value = '';
    handleUserSubmission(val);
  };

  $('fallbackSendBtn').addEventListener('click', sendFallback);
  $('textFallbackInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendFallback();
  });

  // History search and filter
  $('historySearchInput').addEventListener('input', () => {
    clearTimeout(state.searchDebounce);
    state.searchDebounce = setTimeout(loadHistory, 300);
  });
  $('historyStatusFilter').addEventListener('change', loadHistory);

  // Bulk Selection and Deletion Listeners
  const selectAllCb = $('selectAllCalls');
  if (selectAllCb) {
    selectAllCb.addEventListener('change', handleSelectAllChange);
  }
  const deleteSelBtn = $('deleteSelectedBtn');
  if (deleteSelBtn) {
    deleteSelBtn.addEventListener('click', bulkDeleteCalls);
  }

  // Bind Retry Analysis buttons
  const retryBtn = $('retryAnalysisBtn');
  if (retryBtn) {
    retryBtn.addEventListener('click', retryAnalysis);
  }
  const retryFallbackBtn = $('retryFallbackAnalysisBtn');
  if (retryFallbackBtn) {
    retryFallbackBtn.addEventListener('click', retryAnalysis);
  }

  // Set initial Call Intelligence state to neutral awaiting state
  setIntelState('awaiting');

  // Initial history load
  await loadHistory();
}

window.addEventListener('DOMContentLoaded', initApp);
