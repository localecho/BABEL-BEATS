// Complete JavaScript for BABEL-BEATS frontend

// Music player completion
function toggleMusic() {
    if (!currentAudio) {
        // Create mock audio for demo
        currentAudio = new Audio();
        // Using a data URI for a simple tone
        currentAudio.src = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBkKw7+2hXRIHUK/n77FVER9Mo+7kujsvFCh51urlsl4IA0m+8+6sVBYCSb/041sPBCiV0OXLhy0bjNPu8faVUBQAS6rx7blmFwY7k9n1';
        
        currentAudio.addEventListener('timeupdate', updateProgress);
        currentAudio.addEventListener('ended', () => {
            document.getElementById('playIcon').textContent = '▶️';
        });
    }

    if (currentAudio.paused) {
        currentAudio.play();
        document.getElementById('playIcon').textContent = '⏸️';
    } else {
        currentAudio.pause();
        document.getElementById('playIcon').textContent = '▶️';
    }
}

function updateProgress() {
    if (currentAudio && !isNaN(currentAudio.duration)) {
        const progress = (currentAudio.currentTime / currentAudio.duration) * 100;
        document.getElementById('progressFill').style.width = progress + '%';
        
        const currentMinutes = Math.floor(currentAudio.currentTime / 60);
        const currentSeconds = Math.floor(currentAudio.currentTime % 60).toString().padStart(2, '0');
        const durationMinutes = Math.floor(currentAudio.duration / 60);
        const durationSeconds = Math.floor(currentAudio.duration % 60).toString().padStart(2, '0');
        
        document.getElementById('timeDisplay').textContent = 
            `${currentMinutes}:${currentSeconds} / ${durationMinutes}:${durationSeconds}`;
    }
}

function seekMusic(event) {
    if (currentAudio && !isNaN(currentAudio.duration)) {
        const rect = event.currentTarget.getBoundingClientRect();
        const percent = (event.clientX - rect.left) / rect.width;
        currentAudio.currentTime = percent * currentAudio.duration;
    }
}

// Exercise functions
function startExercise(type) {
    document.getElementById('practiceMode').style.display = 'block';
    document.getElementById('learningExercises').style.display = 'none';
    
    switch(type) {
        case 'rhythm':
            document.getElementById('practiceTitle').textContent = 'Rhythm Training';
            startRhythmExercise();
            break;
        case 'tone':
            document.getElementById('practiceTitle').textContent = 'Tone Practice';
            startToneExercise();
            break;
        case 'pronunciation':
            document.getElementById('practiceTitle').textContent = 'Pronunciation Drill';
            startPronunciationExercise();
            break;
    }
    
    // Smooth scroll to practice mode
    document.getElementById('practiceMode').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function startRhythmExercise() {
    // Adjust metronome speed based on target language
    const metronomeArm = document.querySelector('.metronome-arm');
    let animationDuration = '1s';
    
    if (selectedLanguage === 'ja-JP') {
        animationDuration = '0.8s'; // Faster for Japanese
    } else if (selectedLanguage === 'es-ES') {
        animationDuration = '1.2s'; // Slower for Spanish
    }
    
    metronomeArm.style.animationDuration = animationDuration;
}

function startToneExercise() {
    // Animate pitch line for tone practice
    animatePitchLine();
}

function startPronunciationExercise() {
    // Show phoneme guides
    console.log('Starting pronunciation exercise');
}

function endPractice() {
    document.getElementById('practiceMode').style.display = 'none';
    document.getElementById('learningExercises').style.display = 'block';
}

// Waveform animation
let waveformInterval;

function animateWaveform() {
    const waveform = document.getElementById('waveform');
    waveform.innerHTML = '';
    
    // Create waveform bars
    for (let i = 0; i < 50; i++) {
        const bar = document.createElement('div');
        bar.className = 'waveform-bar';
        bar.style.height = '10px';
        waveform.appendChild(bar);
    }
    
    // Animate bars
    waveformInterval = setInterval(() => {
        const bars = waveform.querySelectorAll('.waveform-bar');
        bars.forEach(bar => {
            const height = Math.random() * 80 + 20;
            bar.style.height = height + 'px';
        });
    }, 100);
}

function stopWaveformAnimation() {
    clearInterval(waveformInterval);
    
    // Fade out waveform
    const bars = document.querySelectorAll('.waveform-bar');
    bars.forEach((bar, index) => {
        setTimeout(() => {
            bar.style.height = '10px';
        }, index * 10);
    });
}

// Pitch line animation
function animatePitchLine() {
    const pitchLine = document.getElementById('pitchLine');
    let position = 0;
    
    const pitchInterval = setInterval(() => {
        position += 2;
        if (position > 100) position = 0;
        
        const frequency = Math.sin(position * 0.1) * 30 + 50;
        pitchLine.style.bottom = frequency + '%';
        pitchLine.style.left = position + '%';
    }, 50);
    
    // Store interval ID for cleanup
    pitchLine.dataset.intervalId = pitchInterval;
}

// Achievement system
function unlockAchievement(achievementId) {
    const achievement = document.getElementById(achievementId);
    if (achievement && !achievement.classList.contains('unlocked')) {
        achievement.classList.add('unlocked');
        
        // Save to localStorage
        const achievements = JSON.parse(localStorage.getItem('babelBeatsAchievements') || '[]');
        if (!achievements.includes(achievementId)) {
            achievements.push(achievementId);
            localStorage.setItem('babelBeatsAchievements', JSON.stringify(achievements));
        }
        
        // Show notification
        showNotification(`Achievement Unlocked: ${achievement.querySelector('div:last-child').textContent}`);
    }
}

function loadAchievements() {
    const achievements = JSON.parse(localStorage.getItem('babelBeatsAchievements') || '[]');
    achievements.forEach(achievementId => {
        const achievement = document.getElementById(achievementId);
        if (achievement) {
            achievement.classList.add('unlocked');
        }
    });
}

// Utility functions
function showError(message) {
    const errorElement = document.getElementById('errorMessage');
    errorElement.textContent = message;
    errorElement.style.display = 'block';
    
    setTimeout(() => {
        errorElement.style.display = 'none';
    }, 5000);
}

function showNotification(message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 8px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// API Integration (for production)
async function analyzeWithAPI(audioData, language) {
    const response = await fetch('http://localhost:8000/api/v1/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer YOUR_API_KEY'
        },
        body: JSON.stringify({
            audio_data: audioData,
            language: language,
            analysis_type: 'comprehensive'
        })
    });
    
    if (!response.ok) {
        throw new Error('API request failed');
    }
    
    return response.json();
}

async function generateMusicWithAPI(analysisId, preferences) {
    const response = await fetch('http://localhost:8000/api/v1/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer YOUR_API_KEY'
        },
        body: JSON.stringify({
            analysis_id: analysisId,
            music_style: preferences.style || 'classical',
            duration: 30,
            focus_areas: ['rhythm', 'tone', 'pronunciation']
        })
    });
    
    if (!response.ok) {
        throw new Error('Music generation failed');
    }
    
    return response.json();
}

// Export functions for use in HTML
window.toggleMusic = toggleMusic;
window.seekMusic = seekMusic;
window.startExercise = startExercise;
window.endPractice = endPractice;
window.animateWaveform = animateWaveform;
window.stopWaveformAnimation = stopWaveformAnimation;
window.unlockAchievement = unlockAchievement;
window.loadAchievements = loadAchievements;
window.showError = showError;
window.showNotification = showNotification;