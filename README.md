# XTTS V2 Voice Cloning - Universal GPU Support

## 📋 What is this?

Professional voice cloning script using Coqui XTTS V2, with **universal GPU acceleration** (NVIDIA, AMD, Apple Silicon) and advanced features that make it superior to using raw XTTS V2.

## ✨ Why use this script instead of raw Coqui XTTS V2?

### 🎯 Advanced Features Not Available in Standard XTTS V2:

1. **Intelligent Linguistic Segmentation**
   - Automatically splits text at natural punctuation points (`.`, `?`, `!`, `;`, `:`, `,`)
   - Respects sentence boundaries for natural speech flow
   - Prevents memory errors on long texts (15+ minute audio)

2. **Automatic Voice Calibration** 
   - Analyzes your voice with advanced algorithms (YIN pitch detection, spectral analysis, MFCC)
   - Auto-optimizes TTS parameters for maximum fidelity
   - No manual parameter tuning needed

3. **Sentiment-Aware Processing**
   - Detects emotions in text (joy, anger, sadness, surprise, fear, neutral)
   - Adjusts intonation and expressiveness automatically
   - Natural-sounding emotional delivery

4. **Linking Word Optimization**
   - Intelligent handling of conjunctions and transitions ("and", "but", "however", "therefore")
   - Preserves natural speech rhythm and flow
   - Prevents robotic-sounding pauses

5. **Universal GPU Acceleration**
   - **NVIDIA GPUs**: CUDA acceleration (3-5x faster than CPU)
   - **AMD GPUs**: ROCm support (3-5x faster than CPU)
   - **Apple Silicon**: Native MPS acceleration (3-5x faster than CPU)
   - Automatic GPU detection and optimal device selection

6. **Professional Audio Processing**
   - Native WAV concatenation without quality loss
   - Automatic segment blending for seamless output
   - No external dependencies (ffmpeg) needed

**Bottom line**: Raw XTTS V2 = basic TTS. This script = production-ready voice cloning system.

---

## 🖥️ System Requirements

- **Python**: 3.11 (recommended) - other versions (3.10-3.12) may work but 3.11 is optimal
- **OS**: macOS 14+, Windows 10+, or Linux (Ubuntu 20.04+)
- **RAM**: Minimum 8GB, recommended 16GB
- **Storage**: ~6GB free space (4GB models + 2GB dependencies)
- **GPU** (optional): NVIDIA (CUDA), AMD (ROCm), or Apple Silicon (MPS) - **3-5x faster than CPU**

---

---

## ⚠️ 🔴 **CRITICAL REQUIREMENT - READ THIS FIRST!** 🔴

> **Before running this script, you MUST have your own voice recording file in the project folder!**
> 
> - **Required file**: `speaker.wav` (or any audio file with your voice)
> - **Location**: Must be in the project directory (`Coqui-XTTS-V2-Enhanced-AppleSilicon-Nvidia-AMD-CPU/`)
> - **Without this file, the script CANNOT clone your voice!**
> 
> 👉 **See the [Speaker.wav Requirements](#️-requirements-for-speakerwav-your-voice-file) section below for recording guidelines.**

---

## 🎙️ Requirements for `speaker.wav` (Your Voice File)

### ✅ Critical Requirements for High-Quality Cloning:

1. **Duration**: 10-30 seconds (optimal: 15-20 seconds)
   - Too short (< 10s): Insufficient vocal characteristics
   - Too long (> 30s): Unnecessary and slower processing

2. **Content**: Natural, expressive speech
   - Read 2-3 complete sentences
   - Include varied intonation (not monotone)
   - Speak at normal conversational pace

3. **Audio Quality**:
   - **Format**: WAV, MP3, or FLAC
   - **Sample rate**: 22050 Hz or higher
   - **Bit depth**: 16-bit minimum
   - **Background noise**: Minimal to none (record in quiet room)
   - **Clarity**: No distortion, clipping, or echoes

4. **Recording Tips**:
   - Use a decent microphone (built-in Mac mic is acceptable)
   - Maintain consistent distance from microphone (15-20 cm)
   - Avoid pops and sibilance (use pop filter if available)
   - Record in quiet environment (no fans, traffic, or background voices)
   - Speak naturally - don't shout or whisper

### ⚠️ What to Avoid:
- ❌ Music or sound effects in background
- ❌ Multiple speakers
- ❌ Heavy compression or audio processing
- ❌ Phone call quality recordings
- ❌ Recordings with echo or reverb

**Example good recording**: "Hello, this is my natural speaking voice. I'm recording this sample for voice cloning purposes. The quality of this recording will determine how accurate the cloned voice sounds."

---

## 📦 Installation (5 minutes)

```bash
# 1. Navigate to project folder
cd XTTS_Voice_Cloning

# 2. Create Python virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Upgrade pip (important!)
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt
```

### GPU Setup (Optional - for NVIDIA/AMD users):

**NVIDIA GPU:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**AMD GPU (Linux only):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

**Apple Silicon:** No extra setup needed - MPS support is built-in.

**Note**: First run will download XTTS V2 models (~2GB) - this is normal and only happens once.

---

## 🚀 Usage

### Basic Command Structure

```bash
# First, activate the virtual environment
source venv/bin/activate

# Basic syntax
python3 generate_audio.py <voice_file> <text_file> -o <output_file>

# Example 1: Using default files (speaker.wav + sample_text_for_clone.txt)
python3 generate_audio.py speaker.wav sample_text_for_clone.txt -o output.wav

# Example 2: Custom files
python3 generate_audio.py my_voice.wav my_script.txt -o tutorial.wav

# Example 3: With speed adjustment
python3 generate_audio.py speaker.wav script.txt --speed 1.1 -o fast_tutorial.wav

# Example 4: All custom parameters
python3 generate_audio.py voice.wav text.txt -o output.wav --speed 1.05 --temperature 0.75
```

---

### Available Arguments:

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `speaker_wav` | Your voice file (positional) | `speaker.wav` | `my_voice.wav` |
| `text_file` | Text to clone (positional) | `sample_text_for_clone.txt` | `script.txt` |
| `-o`, `--output` | Output audio file | `cloned_<text_file>.wav` | `tutorial.wav` |
| `--speed` | Speech speed (0.5-2.0) | Auto-calibrated | `1.1` |
| `--temperature` | Voice variation (0.1-1.0) | Auto-calibrated | `0.75` |
| `--top_p` | Sampling diversity (0.1-1.0) | Auto-calibrated | `0.85` |
| `--repetition_penalty` | Avoid repetition (1.0-5.0) | Auto-calibrated | `2.5` |
| `--top_k` | Vocabulary sampling (1-100) | Auto-calibrated | `50` |

**💡 Tip**: For most users, no parameters are needed - auto-calibration handles everything!

---

### Complete Command Examples:

```bash
# 1. Using default files with default output name
python3 generate_audio.py speaker.wav sample_text_for_clone.txt -o cloned_output.wav

# 2. Custom files
python3 generate_audio.py my_voice.wav my_text.txt -o result.wav

# 3. Custom output name
python3 generate_audio.py speaker.wav script.txt -o my_tutorial.wav

# 4. Faster speech for YouTube tutorials
python3 generate_audio.py speaker.wav tutorial.txt --speed 1.15 -o youtube_tutorial.wav

# 5. Manual parameter control (advanced)
python3 generate_audio.py voice.wav text.txt -o custom.wav \
  --speed 1.05 \
  --temperature 0.78 \
  --top_p 0.89 \
  --repetition_penalty 2.7 \
  --top_k 45
```

**⚠️ Important**: You must specify both the voice file and text file - there are no defaults!

---

## 📊 What Happens During Processing?

```
1. 🔍  Detecting GPU (NVIDIA/AMD/Apple Silicon) or using CPU
2. 🎙️  Loading voice file (speaker.wav)
3. �  Analyzing vocal characteristics (pitch, timbre, spectral features)
4. ⚙️   Auto-calibrating TTS parameters
5. 📝  Loading and cleaning text
6. 🎭  Analyzing sentiment and emotions
7. ✂️   Intelligent segmentation (respecting sentence boundaries)
8. 🎵  Generating audio for each segment
9. 🔗  Concatenating segments seamlessly
10. 💾  Saving final audio file
11. ✅  Done!
```

**Typical processing time**: 
- **With GPU**: 3-5 minutes for 2000-3000 characters
- **CPU only**: 15-20 minutes for 2000-3000 characters

---

## 🛠️ Troubleshooting

### "ModuleNotFoundError"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### GPU Not Detected

**Check GPU availability:**
```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

**NVIDIA GPU not detected:**
- Install CUDA-enabled PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- Verify NVIDIA drivers are installed

**AMD GPU not detected (Linux only):**
- Install ROCm-enabled PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7`
- Verify ROCm drivers are installed

**Apple Silicon MPS not detected:**
- Update to macOS 12.3 or later
- Restart terminal after macOS update

**Note:** Script works on CPU if no GPU is detected - just slower (3-5x).

### First run very slow
Normal - downloading XTTS V2 models (~2GB). Subsequent runs are much faster.

### Poor voice quality
1. Check `speaker.wav` quality (see requirements above)
2. Ensure 15-20 second duration
3. Verify quiet recording environment
4. Try re-recording with better microphone

### Script crashes on long text
Should not happen - intelligent segmentation prevents this. If it does, report the issue.

---

## 📁 Project Files

```
XTTS_Voice_Cloning/
├── generate_audio.py              # Main script (universal GPU support)
├── universal_voice_calibrator.py  # Auto-calibration system
├── voice_analyzer.py              # Voice analysis module
├── speaker.wav                    # Your reference voice (replace this!)
├── sample_text_for_clone.txt      # Example text
├── requirements.txt               # Python dependencies
└── venv/                          # Virtual environment (created during install)
```

---

## 💡 Tips for Best Results

1. **Voice Recording**: 
   - Record multiple samples, choose the best one
   - Natural, conversational tone works best
   - Include varied intonation (not monotone)

2. **Text Preparation**:
   - Use proper punctuation for natural pauses
   - Break very long sentences with commas
   - Avoid excessive capitalization or special characters

3. **Speed Settings**:
   - Default (auto-calibrated): Most natural
   - `--speed 1.05-1.1`: Good for tutorials/presentations
   - `--speed 1.15-1.2`: Fast-paced content
   - `--speed 0.9-0.95`: Slower, more deliberate

4. **For YouTube Tutorials**:
   - Use `--speed 1.08` for natural but efficient pacing
   - Record speaker.wav in same environment as final use
   - Keep text well-punctuated for natural pauses

---

## 🆘 Support

For issues or questions:
1. Check Python version: `python3 --version` (should be 3.11.x)
2. Verify virtual environment is activated: `which python3` (should show venv path)
3. Ensure all dependencies installed: `pip list | grep -i tts`
4. Check GPU detection: See troubleshooting section above

---

## 📝 License

Educational voice cloning system. Ensure you have rights to clone any voice you use.

---

**Version**: 3.0 (Universal GPU Support)  
**Last Updated**: November 18, 2025  
**GPU Support**: NVIDIA CUDA, AMD ROCm, Apple Silicon MPS, CPU fallback

