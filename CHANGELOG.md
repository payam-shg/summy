# Changelog

All notable changes to SUMMY will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-12

### Added
- Initial release of SUMMY
- Multi-service audio transcription (Whisper, Groq, AssemblyAI)
- Intelligent summarization with multiple LLM options
- Flexible translation engine with 4 service options
- Support for 7 languages (Persian, Italian, Spanish, French, Portuguese, Russian, Chinese)
- Text-to-speech generation with ElevenLabs (8 voice options)
- PDF export for transcripts
- Smart caching system to resume processing
- Batch video processing
- Low-bitrate audio optimization for API usage
- Interactive CLI with progress tracking
- Comprehensive error handling

### Features by Component

#### Transcription
- Local Whisper support (offline, privacy-focused)
- Groq API integration (fast cloud transcription)
- AssemblyAI integration (enterprise-grade accuracy)
- Automatic low-bitrate audio creation for API services

#### Summarization
- Local Ollama support
- Groq API integration
- OpenRouter API integration
- Configurable models via .env

#### Translation
- Google Translate via deep-translator (free & fast)
- Local Ollama translation
- Groq API translation
- OpenRouter API translation
- Support for 7 target languages
- Automatic text chunking for long content

#### Audio Generation
- ElevenLabs TTS integration
- 8 professional voice options (4 female, 4 male)
- Multilingual audio generation
- High-quality MP3 output

#### File Management
- Automatic folder structure creation
- Smart file existence checking
- PDF generation with custom font support
- Organized output structure

### Technical
- Python 3.8+ compatibility
- Modular architecture
- Environment-based configuration
- Comprehensive error handling
- Progress tracking with tqdm
- UTF-8 encoding support

### Documentation
- Comprehensive README with examples
- Contributing guidelines
- MIT License
- .env.example template
- .gitignore for clean repository

## [Unreleased]

### Planned Features
- Web interface
- Docker support
- Additional language support
- More TTS providers
- Subtitle generation
- Video chapter detection
- Batch configuration presets
- API endpoint mode
- Database integration for history
- Advanced filtering options

---

For more details, see the [GitHub repository](https://github.com/yourusername/summy).
