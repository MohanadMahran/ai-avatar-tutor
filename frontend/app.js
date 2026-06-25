/**
 * AI Avatar Tutor - Frontend Application
 *
 * Handles voice recording, API communication, chat display,
 * avatar video playback, and document management.
 */
// ============================================
// Application State
// ============================================
const state = {
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    audioStream: null,
    audioContext: null,
    analyser: null,
    animationFrame: null,
    isProcessing: false,
    conversationHistory: [],
};
// ============================================
// DOM Elements
// ============================================
const elements = {
    recordBtn: document.getElementById('recordBtn'),
    recordingIndicator: document.getElementById('recordingIndicator'),
    audioVisualizer: document.getElementById('audioVisualizer'),
    audioCanvas: document.getElementById('audioCanvas'),
    chatMessages: document.getElementById('chatMessages'),
    avatarVideo: document.getElementById('avatarVideo'),
    avatarPlaceholder: document.getElementById('avatarPlaceholder'),
    avatarContainer: document.getElementById('avatarContainer'),
    pipelineStatus: document.getElementById('pipelineStatus'),
    loadingBarFill: document.getElementById('loadingBarFill'),
    statusIndicator: document.getElementById('statusIndicator'),
    confidenceBadge: document.getElementById('confidenceBadge'),
    confidenceScore: document.getElementById('confidenceScore'),
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    uploadProgress: document.getElementById('uploadProgress'),
    uploadProgressFill: document.getElementById('uploadProgressFill'),
    uploadProgressText: document.getElementById('uploadProgressText'),
    docList: document.getElementById('docList'),
    clearConversationBtn: document.getElementById('clearConversationBtn'),
    refreshDocsBtn: document.getElementById('refreshDocsBtn'),
    stepTranscribe: document.getElementById('stepTranscribe'),
    stepThinking: document.getElementById('stepThinking'),
    stepAvatar: document.getElementById('stepAvatar'),
};
// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initializeRecording();
    initializeUpload();
    initializeEventListeners();
    loadDocumentList();
    checkHealth();
});
function initializeEventListeners() {
    elements.clearConversationBtn.addEventListener('click', clearConversation);
    elements.refreshDocsBtn.addEventListener('click', loadDocumentList);
}
// ============================================
// Audio Recording
// ============================================
function initializeRecording() {
    const btn = elements.recordBtn;
    // Mouse events
    btn.addEventListener('mousedown', startRecording);
    btn.addEventListener('mouseup', stopRecording);
    btn.addEventListener('mouseleave', stopRecording);
    // Touch events for mobile
    btn.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startRecording();
    });
    btn.addEventListener('touchend', (e) => {
        e.preventDefault();
        stopRecording();
    });
}
async function startRecording() {
    if (state.isRecording || state.isProcessing) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });
        state.audioStream = stream;
        state.audioChunks = [];
        state.isRecording = true;
        // Setup MediaRecorder
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : 'audio/webm';
        state.mediaRecorder = new MediaRecorder(stream, { mimeType });
        state.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                state.audioChunks.push(event.data);
            }
        };
        state.mediaRecorder.onstop = () => {
            processRecording();
        };
        state.mediaRecorder.start(100); // Collect data every 100ms
        // Update UI
        elements.recordBtn.classList.add('recording');
        elements.recordBtn.querySelector('.record-text').textContent = 'Recording...';
        elements.recordingIndicator.style.display = 'flex';
        // Start audio visualization
        startAudioVisualization(stream);
        updateStatus('Recording...', 'recording');
    } catch (error) {
        console.error('Microphone access error:', error);
        showError('Microphone access denied. Please allow microphone access in your browser settings.');
    }
}
function stopRecording() {
    if (!state.isRecording) return;
    state.isRecording = false;
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
    }
    if (state.audioStream) {
        state.audioStream.getTracks().forEach(track => track.stop());
        state.audioStream = null;
    }
    // Stop visualization
    stopAudioVisualization();
    // Update UI
    elements.recordBtn.classList.remove('recording');
    elements.recordBtn.querySelector('.record-text').textContent = 'Hold to Speak';
    elements.recordingIndicator.style.display = 'none';
}
async function processRecording() {
    if (state.audioChunks.length === 0) {
        showError('No audio captured. Please try again.');
        return;
    }
    const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
    if (audioBlob.size < 1000) {
        showError('Recording too short. Please hold the button longer.');
        return;
    }
    state.isProcessing = true;
    showPipelineStatus(true);
    setActiveStep('transcribe');
    try {
        // Use streaming endpoint for real-time response
        await processWithStreaming(audioBlob);
    } catch (error) {
        console.error('Processing error:', error);
        // Fallback to non-streaming
        await processWithoutStreaming(audioBlob);
    } finally {
        state.isProcessing = false;
        showPipelineStatus(false);
        updateStatus('Ready', 'ready');
    }
}
async function processWithStreaming(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    const response = await fetch('/api/interact-stream', {
        method: 'POST',
        body: formData,
    });
    if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantMessage = null;
    let fullResponse = '';
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        const lines = text.split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                try {
                    const data = JSON.parse(line.slice(6));
                    switch (data.type) {
                        case 'transcription':
                            addMessage('user', data.content);
                            setActiveStep('thinking');
                            break;
                        case 'metadata':
                            if (data.metadata) {
                                updateConfidence(data.metadata.confidence);
                            }
                            break;
                        case 'token':
                            if (!assistantMessage) {
                                assistantMessage = addStreamingMessage('assistant', '');
                            }
                            fullResponse += data.content;
                            updateStreamingMessage(assistantMessage, fullResponse);
                            break;
                        case 'complete':
                            if (assistantMessage) {
                                finalizeStreamingMessage(assistantMessage, data.content);
                            }
                            setActiveStep('avatar');
                            break;
                        case 'video':
                            playAvatarVideo(data.content);
                            break;
                        case 'done':
                            setActiveStep('done');
                            break;
                        case 'error':
                            showError(data.content);
                            break;
                    }
                } catch (e) {
                    // Skip malformed JSON lines
                }
            }
        }
    }
}
async function processWithoutStreaming(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    formData.append('generate_video', 'true');
    updateStatus('Processing...', 'processing');
    const response = await fetch('/api/interact', {
        method: 'POST',
        body: formData,
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Request failed');
    }
    const result = await response.json();
    // Display transcription
    if (result.transcription) {
        addMessage('user', result.transcription);
    }
    // Display response
    if (result.response_text) {
        addMessage('assistant', result.response_text, result.sources);
    }
    // Update confidence
    updateConfidence(result.confidence);
    // Play avatar video
    if (result.video_url) {
        playAvatarVideo(result.video_url);
    }
}
// ============================================
// Audio Visualization
// ============================================
function startAudioVisualization(stream) {
    elements.audioVisualizer.style.display = 'block';
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 256;
    const source = state.audioContext.createMediaStreamSource(stream);
    source.connect(state.analyser);
    drawVisualization();
}
function drawVisualization() {
    if (!state.isRecording) return;
    const canvas = elements.audioCanvas;
    const ctx = canvas.getContext('2d');
    const bufferLength = state.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    state.analyser.getByteFrequencyData(dataArray);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const barWidth = (canvas.width / bufferLength) * 2;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        const gradient = ctx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height);
        gradient.addColorStop(0, '#6366f1');
        gradient.addColorStop(1, '#8b5cf6');
        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
        x += barWidth;
    }
    state.animationFrame = requestAnimationFrame(drawVisualization);
}
function stopAudioVisualization() {
    if (state.animationFrame) {
        cancelAnimationFrame(state.animationFrame);
        state.animationFrame = null;
    }
    if (state.audioContext) {
        state.audioContext.close();
        state.audioContext = null;
    }
    elements.audioVisualizer.style.display = 'none';
}
// ============================================
// Chat Messages
// ============================================
function addMessage(role, content, sources = []) {
    // Remove welcome message if present
    const welcome = elements.chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `<div class="message-meta"> Sources: ${sources.join(', ')}</div>`;
    }
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    messageDiv.innerHTML = `
        <div class="message-content">${formatContent(content)}</div>
        ${sourcesHtml}
        <div class="message-meta">${role === 'user' ? 'You' : 'Tutor'} • ${timeStr}</div>
    `;
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    return messageDiv;
}
function addStreamingMessage(role, content) {
    const welcome = elements.chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.innerHTML = `
        <div class="message-content">${formatContent(content)}<span class="streaming-cursor"></span></div>
        <div class="message-meta">Tutor • typing...</div>
    `;
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    return messageDiv;
}
function updateStreamingMessage(messageDiv, content) {
    const contentEl = messageDiv.querySelector('.message-content');
    contentEl.innerHTML = `${formatContent(content)}<span class="streaming-cursor"></span>`;
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}
function finalizeStreamingMessage(messageDiv, content) {
    const contentEl = messageDiv.querySelector('.message-content');
    contentEl.innerHTML = formatContent(content);
    const metaEl = messageDiv.querySelector('.message-meta');
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    metaEl.textContent = `Tutor • ${timeStr}`;
}
function formatContent(content) {
    // Simple markdown-like formatting
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}
// ============================================
// Avatar Video
// ============================================
function playAvatarVideo(url) {
    if (!url) return;
    elements.avatarPlaceholder.style.display = 'none';
    elements.avatarVideo.style.display = 'block';
    elements.avatarVideo.src = url;
    elements.avatarVideo.load();
    elements.avatarVideo.play().catch(e => {
        console.warn('Auto-play blocked:', e);
    });
}
// ============================================
// Document Management
// ============================================
function initializeUpload() {
    const uploadArea = elements.uploadArea;
    const fileInput = elements.fileInput;
    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadFiles(e.target.files);
        }
    });
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            uploadFiles(e.dataTransfer.files);
        }
    });
}
async function uploadFiles(files) {
    const formData = new FormData();
    let validFiles = 0;
    for (const file of files) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (['pdf', 'txt', 'md'].includes(ext)) {
            formData.append('files', file);
            validFiles++;
        }
    }
    if (validFiles === 0) {
        showError('No valid files selected. Supported formats: PDF, TXT, MD');
        return;
    }
    // Show progress
    elements.uploadProgress.style.display = 'block';
    elements.uploadProgressFill.style.width = '30%';
    elements.uploadProgressText.textContent = `Uploading ${validFiles} file(s)...`;
    try {
        const response = await fetch('/api/upload-docs', {
            method: 'POST',
            body: formData,
        });
        elements.uploadProgressFill.style.width = '70%';
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        const result = await response.json();
        elements.uploadProgressFill.style.width = '100%';
        elements.uploadProgressText.textContent = ` ${result.message}`;
        // Refresh document list
        await loadDocumentList();
        setTimeout(() => {
            elements.uploadProgress.style.display = 'none';
            elements.uploadProgressFill.style.width = '0%';
        }, 3000);
    } catch (error) {
        console.error('Upload error:', error);
        elements.uploadProgressText.textContent = ` Error: ${error.message}`;
        elements.uploadProgressFill.style.background = 'var(--error)';
        setTimeout(() => {
            elements.uploadProgress.style.display = 'none';
            elements.uploadProgressFill.style.width = '0%';
            elements.uploadProgressFill.style.background = '';
        }, 5000);
    }
    // Reset file input
    elements.fileInput.value = '';
}
async function loadDocumentList() {
    try {
        const response = await fetch('/api/docs-list');
        const result = await response.json();
        const docList = elements.docList;
        if (result.documents && result.documents.length > 0) {
            docList.innerHTML = result.documents.map(doc => `
                <div class="doc-item">
                    <span class="doc-item-name" title="${doc}">[Dokument] ${doc}</span>
                    <button class="doc-item-delete" onclick="deleteDocument('${doc}')" title="Delete"></button>
                </div>
            `).join('');
        } else {
            docList.innerHTML = '<p class="no-docs">No documents indexed yet.</p>';
        }
    } catch (error) {
        console.error('Failed to load document list:', error);
    }
}
async function deleteDocument(sourceName) {
    if (!confirm(`Delete "${sourceName}" from the index?`)) return;
    try {
        const response = await fetch('/api/delete-doc', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_name: sourceName }),
        });
        const result = await response.json();
        if (result.success) {
            await loadDocumentList();
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Failed to delete document: ' + error.message);
    }
}
// ============================================
// Conversation Management
// ============================================
async function clearConversation() {
    try {
        const response = await fetch('/api/conversation', { method: 'DELETE' });
        const result = await response.json();
        if (result.success) {
            elements.chatMessages.innerHTML = `
                <div class="welcome-message">
                    <div class="welcome-icon"></div>
                    <h3>Conversation Cleared</h3>
                    <p>Start a new conversation by pressing the microphone button.</p>
                </div>
            `;
            elements.confidenceBadge.style.display = 'none';
            elements.avatarVideo.style.display = 'none';
            elements.avatarPlaceholder.style.display = 'block';
        }
    } catch (error) {
        showError('Failed to clear conversation: ' + error.message);
    }
}
// ============================================
// UI Helpers
// ============================================
function showPipelineStatus(show) {
    elements.pipelineStatus.style.display = show ? 'block' : 'none';
    if (!show) {
        elements.loadingBarFill.style.width = '0%';
        resetSteps();
    }
}
function setActiveStep(step) {
    resetSteps();
    switch (step) {
        case 'transcribe':
            elements.stepTranscribe.classList.add('active');
            elements.loadingBarFill.style.width = '25%';
            updateStatus('Transcribing...', 'processing');
            break;
        case 'thinking':
            elements.stepTranscribe.classList.add('done');
            elements.stepThinking.classList.add('active');
            elements.loadingBarFill.style.width = '50%';
            updateStatus('Thinking...', 'processing');
            break;
        case 'avatar':
            elements.stepTranscribe.classList.add('done');
            elements.stepThinking.classList.add('done');
            elements.stepAvatar.classList.add('active');
            elements.loadingBarFill.style.width = '75%';
            updateStatus('Generating avatar...', 'processing');
            break;
        case 'done':
            elements.stepTranscribe.classList.add('done');
            elements.stepThinking.classList.add('done');
            elements.stepAvatar.classList.add('done');
            elements.loadingBarFill.style.width = '100%';
            updateStatus('Ready', 'ready');
            break;
    }
}
function resetSteps() {
    ['stepTranscribe', 'stepThinking', 'stepAvatar'].forEach(id => {
        elements[id].classList.remove('active', 'done');
    });
}
function updateStatus(text, state) {
    const statusEl = elements.statusIndicator;
    const dot = statusEl.querySelector('.status-dot');
    const textEl = statusEl.querySelector('.status-text');
    textEl.textContent = text;
    switch (state) {
        case 'ready':
            dot.style.background = 'var(--success)';
            break;
        case 'recording':
            dot.style.background = 'var(--error)';
            break;
        case 'processing':
            dot.style.background = 'var(--warning)';
            break;
    }
}
function updateConfidence(score) {
    if (score > 0) {
        elements.confidenceBadge.style.display = 'inline-flex';
        elements.confidenceScore.textContent = `${Math.round(score * 100)}%`;
    }
}
function showError(message) {
    addMessage('assistant', `[Warnung] ${message}`);
}
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const result = await response.json();
        if (result.status === 'healthy') {
            updateStatus('Ready', 'ready');
        } else {
            updateStatus('Degraded', 'processing');
        }
    } catch (error) {
        updateStatus('Offline', 'processing');
    }
}