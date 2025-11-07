# SUMMY Free Video Transcript, Summarization, and Translation 🎬✍️🌍

This script transcribes English videos, summarizes the content, and translates both the transcript and summary into one of several languages:

Persian
Italian
Spanish
French
Portuguese
Russian
Chinese

### How to Use

Place Videos: Add your .mp4, .mov, .avi, .mkv, or .webm video files into the input_videos folder.

Configure Environment:

Modify the `.env` file to add your API keys (for services like Groq, OpenRouter, AssemblyAI) and to customize model preferences or other settings as needed.

Run the Script:

`python app.py`

The script will then guide you through selecting processing services and translation options. Output files (transcripts, summaries) will be saved in the transcripts and summaries folders.

## Features

Multiple Audio-to-Text Options: Choose between local Whisper, Groq API, or AssemblyAI.

Multiple Text Processing Options: Select from local Ollama, Groq API, or OpenRouter for summarization and translation.

Multi-Language Translation: Supports translation to Persian, Italian, Spanish, French, Portuguese, Russian, and Chinese.

PDF Transcripts: Generates PDF versions of English transcripts.

File Management: Automatically organizes input and output files into dedicated folders.

Resumes Processing: Checks for existing files to avoid re-processing.
