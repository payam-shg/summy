# 🎬 SUMMY - AI-Powered Video Processing Suite

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, modular video processing pipeline that transcribes, summarizes, translates, and generates audio from video content. Built with flexibility in mind, SUMMY allows you to choose from multiple AI services for each processing stage.

## ✨ Features

### 🎙️ **Multi-Service Audio Transcription**
- **Local Whisper** - Privacy-focused, offline transcription
- **Groq API** - Lightning-fast cloud transcription
- **AssemblyAI** - Enterprise-grade accuracy

### 📝 **Intelligent Summarization**
- **Local Ollama** - Run powerful LLMs locally
- **Groq API** - Ultra-fast cloud inference
- **OpenRouter** - Access to multiple premium models

### 🌍 **Flexible Translation Engine**
Choose your preferred translation service:
- **Google Translate** - Free, fast, and reliable (via deep-translator)
- **Local Ollama** - Context-aware AI translation
- **Groq API** - High-speed neural translation
- **OpenRouter** - Premium model translation

**Supported Languages:**
- 🇮🇷 Persian (Farsi)
- 🇮🇹 Italian
- 🇪🇸 Spanish
- 🇫🇷 French
- 🇵🇹 Portuguese
- 🇷🇺 Russian
- 🇨🇳 Chinese

### 🎵 **Text-to-Speech Generation**
Generate natural-sounding audio from summaries using **ElevenLabs**:
- 8 professional voice options (male & female)
- Multilingual support
- High-quality MP3 output

### 📄 **Additional Features**
- **PDF Export** - Professional transcript documents
- **Smart Caching** - Resumes from existing files
- **Batch Processing** - Handle multiple videos automatically
- **Low-Bitrate Audio** - Optimized for API usage
- **Progress Tracking** - Real-time processing updates

## 🎬 Demo

```
Welcome to the Video Processing Script!

Please select the Audio-to-Text service:
  1. Local Whisper
  2. Groq API (requires GROQ_API_KEY)
  3. AssemblyAI API (requires ASSEMBLYAI_API_KEY)
Enter the number of your choice: 1

Please select the Text Processing service (for summarization):
  1. Local Ollama
  2. Groq API (requires GROQ_API_KEY)
  3. OpenRouter API (requires OPENROUTER_API_KEY)
Enter the number of your choice: 2

Do you want to generate audio from summaries using ElevenLabs? (y/n): y

Please select the ElevenLabs voice for audio generation:
  1. Rachel - Calm, Natural Female Voice
  2. Bella - Soft, Friendly Female Voice
  ...
Enter the number of your choice: 1

Do you want to translate the transcript and summary? (y/n): y

Please select the target language for translation:
  1. Persian
  2. Italian
  ...
Enter the number of your choice: 1

Please select the Translation service:
  1. Google Translate (Free, Fast)
  2. Local Ollama
  3. Groq API
  4. OpenRouter API
Enter the number of your choice: 1

🔍 Found 3 videos for processing.
🔊 Audio-to-Text Service Selected: Local Whisper
✍️ Text Processing Service (Summarization): Groq API
🎙️ Audio Generation: Enabled (Voice: Rachel - Calm, Natural Female Voice)
🌍 Translation: Enabled
   └─ Target Language: Persian
   └─ Translation Service: Google Translate (Free, Fast)

Processing videos...
```

## 📋 Requirements

- Python 3.8 or higher
- FFmpeg (for audio extraction)
- API keys for cloud services (optional)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/summy.git
cd summy

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# API Keys (add only the services you plan to use)
GROQ_API_KEY=your_groq_api_key_here
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Model Configurations
LOCAL_WHISPER_MODEL=turbo
GROQ_TRANSCRIPTION_MODEL=whisper-large-v3-turbo
GROQ_CHAT_MODEL=moonshotai/kimi-k2-instruct-0905
OLLAMA_MODEL=llama3.1:8b
OPENROUTER_CHAT_MODEL=x-ai/grok-4-fast:free

# PDF Font Settings (for non-Latin scripts)
PDF_FONT_NAME=Tahoma
PDF_FONT_PATH=tahoma.ttf
```

### 3. Usage

```bash
# Place your video files in the input_videos folder
# Supported formats: .mp4, .mov, .avi, .mkv, .webm

# Run the script
python app.py
```

The interactive CLI will guide you through:
1. Selecting audio transcription service
2. Choosing summarization service
3. Enabling audio generation (optional)
4. Configuring translation (optional)

## 📁 Project Structure

```
summy/
├── input_videos/          # Place your video files here
├── audio_files/           # Extracted audio (original + low-bitrate)
├── transcripts/           # Text transcripts (.txt + .pdf)
├── summaries/             # Generated summaries
├── summary_audio_files/   # Text-to-speech audio files
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── .env                   # Configuration (create this)
└── README.md             # This file
```

## 🎯 Use Cases

- **Content Creators** - Generate transcripts and summaries for YouTube videos
- **Researchers** - Process interview recordings and lectures
- **Educators** - Create multilingual educational content
- **Businesses** - Transcribe meetings and generate action items
- **Accessibility** - Make video content accessible in multiple formats

## 🔧 Advanced Configuration

### Local Whisper Models
Available models: `tiny`, `base`, `small`, `medium`, `large`, `turbo`
- Smaller models = faster, less accurate
- Larger models = slower, more accurate

### Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.1:8b
```

### ElevenLabs Voices
The script includes 8 pre-configured voices:
- **Female**: Rachel, Bella, Domi
- **Male**: Antoni, Arnold, Adam, Sam, Charlie

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Groq](https://groq.com/) - Fast inference
- [AssemblyAI](https://www.assemblyai.com/) - Audio intelligence
- [ElevenLabs](https://elevenlabs.io/) - Text-to-speech
- [Ollama](https://ollama.com/) - Local LLM runtime
- [Deep Translator](https://github.com/nidhaloff/deep-translator) - Translation services

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for the open-source community**
