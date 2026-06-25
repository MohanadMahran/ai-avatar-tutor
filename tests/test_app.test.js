/**
 * JS Tests for AI Avatar Tutor Frontend Application
 */

// Set up mock DOM elements before loading app.js
document.body.innerHTML = `
    <button id="recordBtn"><span class="record-text">Hold to Speak</span></button>
    <div id="recordingIndicator" style="display: none;"></div>
    <div id="audioVisualizer"></div>
    <canvas id="audioCanvas"></canvas>
    <div id="chatMessages"></div>
    <video id="avatarVideo" style="display: none;"></video>
    <div id="avatarPlaceholder"></div>
    <div id="avatarContainer"></div>
    <div id="pipelineStatus" style="display: none;"></div>
    <div id="loadingBarFill" style="width: 0%;"></div>
    <div id="statusIndicator">
        <span class="status-dot"></span>
        <span class="status-text">Ready</span>
    </div>
    <span id="confidenceBadge" style="display: none;"></span>
    <span id="confidenceScore">0%</span>
    <div id="uploadArea"></div>
    <input type="file" id="fileInput" />
    <div id="uploadProgress" style="display: none;">
        <div id="uploadProgressFill"></div>
        <span id="uploadProgressText"></span>
    </div>
    <div id="docList"></div>
    <button id="clearConversationBtn"></button>
    <button id="refreshDocsBtn"></button>
    <div id="stepTranscribe"></div>
    <div id="stepThinking"></div>
    <div id="stepAvatar"></div>
`;

// Override status-dot style.backgroundColor to bypass JSDOM CSS validation
const dotEl = document.querySelector('.status-dot');
let statusDotBgColor = '';
Object.defineProperty(dotEl.style, 'backgroundColor', {
    get: () => statusDotBgColor,
    set: (val) => { statusDotBgColor = val; }
});

// Mock global browser APIs not present or needed in jsdom environment
global.navigator.mediaDevices = {
    getUserMedia: jest.fn().mockImplementation(() => Promise.resolve({
        getTracks: () => [{ stop: jest.fn() }]
    }))
};

class MockMediaRecorder {
    constructor(stream, options) {
        this.stream = stream;
        this.options = options;
        this.state = 'inactive';
    }
    start() { this.state = 'recording'; }
    stop() { this.state = 'inactive'; if (this.onstop) this.onstop(); }
}
global.MediaRecorder = MockMediaRecorder;
global.MediaRecorder.isTypeSupported = jest.fn().mockReturnValue(true);

// Now load the application code
const app = require('../frontend/app.js');

describe('AI Avatar Tutor Frontend Application Logic', () => {
    
    beforeEach(() => {
        // Reset state & DOM elements between tests
        app.state.isRecording = false;
        app.state.isProcessing = false;
        app.state.audioChunks = [];
        app.state.audioStream = null;
        
        // Reset DOM styles & classes
        app.elements.pipelineStatus.style.display = 'none';
        app.elements.loadingBarFill.style.width = '0%';
        app.elements.confidenceBadge.style.display = 'none';
        app.elements.confidenceScore.textContent = '0%';
        
        const dot = app.elements.statusIndicator.querySelector('.status-dot');
        const text = app.elements.statusIndicator.querySelector('.status-text');
        dot.style.backgroundColor = '';
        text.textContent = 'Ready';
        
        ['stepTranscribe', 'stepThinking', 'stepAvatar'].forEach(id => {
            app.elements[id].classList.remove('active', 'done');
        });
    });

    test('updateStatus updates the status label text and indicator dot background color', () => {
        // Ready state (success)
        app.updateStatus('Ready', 'ready');
        expect(app.elements.statusIndicator.querySelector('.status-text').textContent).toBe('Ready');
        expect(app.elements.statusIndicator.querySelector('.status-dot').style.backgroundColor).toBe('var(--success)');

        // Recording state (error/red)
        app.updateStatus('Recording...', 'recording');
        expect(app.elements.statusIndicator.querySelector('.status-text').textContent).toBe('Recording...');
        expect(app.elements.statusIndicator.querySelector('.status-dot').style.backgroundColor).toBe('var(--error)');

        // Processing state (warning/yellow)
        app.updateStatus('Thinking...', 'processing');
        expect(app.elements.statusIndicator.querySelector('.status-text').textContent).toBe('Thinking...');
        expect(app.elements.statusIndicator.querySelector('.status-dot').style.backgroundColor).toBe('var(--warning)');
    });

    test('updateConfidence updates confidence display score and makes badge visible', () => {
        app.updateConfidence(0.856);
        expect(app.elements.confidenceBadge.style.display).toBe('inline-flex');
        expect(app.elements.confidenceScore.textContent).toBe('86%');
    });

    test('showPipelineStatus resets steps and updates display of pipeline panel', () => {
        // Show pipeline
        app.showPipelineStatus(true);
        expect(app.elements.pipelineStatus.style.display).toBe('block');

        // Hide pipeline (should reset step classes and loading bar width)
        app.elements.loadingBarFill.style.width = '50%';
        app.elements.stepTranscribe.classList.add('active');
        
        app.showPipelineStatus(false);
        expect(app.elements.pipelineStatus.style.display).toBe('none');
        expect(app.elements.loadingBarFill.style.width).toBe('0%');
        expect(app.elements.stepTranscribe.classList.contains('active')).toBe(false);
    });

    test('setActiveStep transitions step element classes and adjusts loading bar width', () => {
        app.setActiveStep('transcribe');
        expect(app.elements.stepTranscribe.classList.contains('active')).toBe(true);
        expect(app.elements.loadingBarFill.style.width).toBe('25%');
        expect(app.elements.statusIndicator.querySelector('.status-text').textContent).toBe('Transcribing...');

        app.setActiveStep('thinking');
        expect(app.elements.stepTranscribe.classList.contains('done')).toBe(true);
        expect(app.elements.stepThinking.classList.contains('active')).toBe(true);
        expect(app.elements.loadingBarFill.style.width).toBe('50%');
        expect(app.elements.statusIndicator.querySelector('.status-text').textContent).toBe('Thinking...');

        app.setActiveStep('avatar');
        expect(app.elements.stepThinking.classList.contains('done')).toBe(true);
        expect(app.elements.stepAvatar.classList.contains('active')).toBe(true);
        expect(app.elements.loadingBarFill.style.width).toBe('75%');
        expect(app.elements.statusIndicator.querySelector('.status-text').textContent).toBe('Generating avatar...');

        app.setActiveStep('done');
        expect(app.elements.stepAvatar.classList.contains('done')).toBe(true);
        expect(app.elements.loadingBarFill.style.width).toBe('100%');
        expect(app.elements.statusIndicator.querySelector('.status-text').textContent).toBe('Ready');
    });
});
