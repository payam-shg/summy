# Changelog

All notable changes to SUMMY will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-11-12

### Added
- **Separate Translation Service Selection** - Users can now choose different services for summarization and translation
- **Google Translate Integration** - Free and fast translation via deep-translator library
- **ElevenLabs Text-to-Speech** - Generate audio from summaries with 8 professional voice options
- **Voice Selection** - Choose from 8 different voices (4 female, 4 male) for audio generation
- **Automatic Text Chunking** - Smart handling of long texts for translation (splits at 4500 chars)
- **Enhanced User Experience** - Improved CLI with better service descriptions and visual hierarchy

### Changed
- **Translation Architecture** - Separated translation service from summarization service
- **Library Migration** - Replaced `googletrans` with `deep-translator` for better stability and no dependency conflicts
- **Service Selection Flow** - Added dedicated translation service selection step
- **Output Display** - Improved formatting to show all selected services clearly

### Fixed
- **Google Translator Initialization** - Fixed global variable issues causing initialization failures
- **Dependency Conflicts** - Resolved httpx version conflicts by switching to deep-translator
- **Translation Service Key** - Now properly passes translation service key instead of text service key

### Documentation
- **Comprehensive README** - Complete rewrite with professional formatting, badges, and examples
- **Contributing Guidelines** - Added CONTRIBUTING.md with clear guidelines for contributors
- **License** - Added MIT License file
- **Environment Template** - Added .env.example with detailed comments
- **Changelog** - Added this changelog to track version history
- **.gitignore** - Added proper gitignore for Python projects

## [1.0.0] - 2025-05-27

### Added
- Initial release of SUMMY
- Multi-service audio transcription (Whisper, Groq, AssemblyAI)
- Intelligent summarization with multiple LLM options (Ollama, Groq, OpenRouter)
- Basic translation support (using text processing service)
- Support for 7 languages (Persian, Italian, Spanish, French, Portuguese, Russian, Chinese)
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
