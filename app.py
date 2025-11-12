import os
import time
import requests
import json
from dotenv import load_dotenv

# Import service-specific libraries
try:
    import whisper
except ImportError:
    print("Local Whisper library not found. Please install with 'pip install openai-whisper'")
    whisper = None

try:
    from moviepy import VideoFileClip, AudioFileClip
except ImportError:
    print("MoviePy library not found. Please install with 'pip install moviepy'")
    VideoFileClip = None
    AudioFileClip = None

try:
    from tqdm import tqdm
except ImportError:
    print("TQDM library not found. Please install with 'pip install tqdm'")
    def tqdm(iterable, *args, **kwargs):
        for item in iterable:
            if 'desc' in kwargs:
                print(f"{kwargs['desc']}: Processing item...")
            yield item
    tqdm.write = print

try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("Deep Translator library not found. Please install with 'pip install deep-translator'")
    GoogleTranslator = None


# PDF Generation Libraries
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_LEFT
except ImportError:
    print("ReportLab library not found. PDF generation will be disabled. Please install with 'pip install reportlab'")
    SimpleDocTemplate = Paragraph = getSampleStyleSheet = pdfmetrics = TTFont = None
    TA_LEFT = 0

load_dotenv()

# ========== API Keys & Model Configurations ========== #
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "turbo")
GROQ_TRANSCRIPTION_MODEL = os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "google/gemini-2.0-flash-exp:free")

OPENROUTER_SITE_URL = "http://localhost"
OPENROUTER_APP_NAME = "VideoProcessorScript"


# ========== Folder Settings ========== #
INPUT_FOLDER = "./input_videos"
AUDIO_FOLDER = "./audio_files"
TEXT_FOLDER = "./transcripts"
SUMMARY_FOLDER = "./summaries"
SUMMARY_AUDIO_FOLDER = "./summary_audio_files"

# ========== Font Settings for PDF ========== #
PDF_FONT_NAME = os.getenv("PDF_FONT_NAME", "Tahoma")
PDF_FONT_PATH = os.getenv("PDF_FONT_PATH", "tahoma.ttf")

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_AUDIO_FOLDER, exist_ok=True)

if TTFont:
    try:
        pdfmetrics.registerFont(TTFont(PDF_FONT_NAME, PDF_FONT_PATH))
        print(f"Font '{PDF_FONT_NAME}' loaded successfully for PDF generation.")
    except Exception as e:
        print(f"Error loading font '{PDF_FONT_NAME}' from path '{PDF_FONT_PATH}'. Using default 'Helvetica'. Error: {e}")
        PDF_FONT_NAME = "Helvetica"
else:
    print("ReportLab not available, PDF font registration skipped.")

class TranscriptionResult:
    def __init__(self, text, status="completed", error_message=None):
        self.text = text
        self.status = status if not error_message else "error"
        self.error_message = error_message

groq_client = None
assemblyai_transcriber = None
elevenlabs_client = None
google_translator = None

def initialize_google_translator():
    global google_translator
    if GoogleTranslator:
        try:
            # Test initialization with a simple translation
            test = GoogleTranslator(source='en', target='es')
            google_translator = True  # Just a flag to indicate it's available
            print("Google Translator initialized.")
            return True
        except Exception as e:
            print(f"Failed to initialize Google Translator: {e}")
            google_translator = None
            return False
    else:
        print("Deep Translator library not available. Please install with 'pip install deep-translator'")
        google_translator = None
        return False

def initialize_elevenlabs_client():
    global elevenlabs_client
    if ELEVENLABS_API_KEY and ELEVENLABS_API_KEY != "your_elevenlabs_api_key_here":
        try:
            from elevenlabs import ElevenLabs
            elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            print("ElevenLabs client initialized.")
            return True
        except ImportError:
            print("ElevenLabs library not found. Please install with 'pip install elevenlabs'")
            elevenlabs_client = None
            return False
        except Exception as e:
            print(f"Failed to initialize ElevenLabs client: {e}")
            elevenlabs_client = None
            return False
    else:
        print("ELEVENLABS_API_KEY not found or not set in .env. ElevenLabs services will be unavailable.")
        elevenlabs_client = None
        return False

def initialize_selected_clients(audio_service_key, text_service_key):
    global groq_client, assemblyai_transcriber
    if audio_service_key == "groq" or text_service_key == "groq":
        if GROQ_API_KEY:
            try:
                from groq import Groq
                groq_client = Groq(api_key=GROQ_API_KEY)
                print("Groq client initialized.")
            except ImportError: print("Groq library not found. Please install with 'pip install groq'"); groq_client = None
            except Exception as e: print(f"Failed to initialize Groq client: {e}"); groq_client = None
        else: print("GROQ_API_KEY not found in .env. Groq services will be unavailable if selected."); groq_client = None

    if audio_service_key == "assemblyai":
        if ASSEMBLYAI_API_KEY:
            try:
                import assemblyai
                assemblyai.settings.api_key = ASSEMBLYAI_API_KEY
                assemblyai_transcriber = assemblyai.Transcriber()
                print("AssemblyAI client initialized.")
            except ImportError: print("AssemblyAI library not found. Please install with 'pip install assemblyai'"); assemblyai_transcriber = None
            except Exception as e: print(f"Failed to initialize AssemblyAI client: {e}"); assemblyai_transcriber = None
        else: print("ASSEMBLYAI_API_KEY not found in .env. AssemblyAI services will be unavailable if selected."); assemblyai_transcriber = None

def get_user_choice(prompt_message, options_dict):
    print(f"\n{prompt_message}")
    for key, (value, description) in options_dict.items():
        print(f"  {key}. {description} (Service Code: {key if isinstance(value, str) else value.get('code', value)})") # Adjusted for new structure
    while True:
        choice_num = input(f"Enter the number of your choice: ")
        if choice_num in options_dict:
            return options_dict[choice_num][0]
        else:
            print("Invalid choice. Please enter a number from the list above.")

def get_yes_no_choice(prompt_message):
    while True:
        choice = input(f"\n{prompt_message} (y/n): ").lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")

def get_video_files():
    supported_ext = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    return [f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in supported_ext]

def extract_audio(video_path, audio_output):
    if not VideoFileClip: tqdm.write("MoviePy is not available. Cannot extract audio."); return False
    try:
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_output, codec='mp3')
        video_clip.close()
        return True
    except Exception as e: tqdm.write(f"Error extracting original audio: {str(e)}"); return False

def create_low_bitrate_audio(source_audio_path, output_audio_path_low, bitrate="64k"):
    if not AudioFileClip: tqdm.write("MoviePy (AudioFileClip) is not available. Cannot create low-bitrate audio."); return False
    try:
        tqdm.write(f"Creating low-bitrate audio ({bitrate}) at {output_audio_path_low} from {source_audio_path}...")
        audio_clip = AudioFileClip(source_audio_path)
        audio_clip.write_audiofile(output_audio_path_low, bitrate=bitrate, codec='mp3')
        audio_clip.close()
        tqdm.write(f"🎧 Low-bitrate audio saved: {output_audio_path_low}")
        return True
    except Exception as e:
        tqdm.write(f"❌ Error creating low-bitrate audio: {str(e)}")
        return False

def save_transcript_as_pdf(text_content, pdf_path, font_to_use):
    if not SimpleDocTemplate: tqdm.write("ReportLab not available. Cannot save PDF."); return False
    current_font_to_use = font_to_use
    try:
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        try:
            pdfmetrics.getFont(current_font_to_use)
        except KeyError:
            tqdm.write(f"Font '{current_font_to_use}' not found by ReportLab, falling back to 'Helvetica' for PDF: {pdf_path}")
            current_font_to_use = "Helvetica"
        
        style = styles["Normal"]; style.fontName = current_font_to_use; style.fontSize = 10; style.leading = 14; style.alignment = TA_LEFT
        formatted_text = text_content.replace('\n', '<br/>')
        story = [Paragraph(formatted_text, style)]
        doc.build(story)
        tqdm.write(f"📄 English transcript successfully saved as PDF: {pdf_path} (Font: {current_font_to_use})")
        return True
    except Exception as e: tqdm.write(f"❌ Error saving PDF for '{pdf_path}': {str(e)}"); return False

# --- Audio to Text Implementations --- #
def transcribe_audio_whisper_local(audio_path_to_use):
    if not whisper: return TranscriptionResult(None, error_message="Local Whisper library not available.")
    try:
        tqdm.write(f"Loading local Whisper '{LOCAL_WHISPER_MODEL}' model...")
        model = whisper.load_model(LOCAL_WHISPER_MODEL)
        tqdm.write(f"Transcribing {audio_path_to_use} with local Whisper...")
        result = model.transcribe(audio_path_to_use)
        return TranscriptionResult(result["text"])
    except Exception as e: tqdm.write(f"Error in local Whisper: {str(e)}"); return TranscriptionResult(None, error_message=str(e))

def transcribe_audio_groq(audio_path_to_use):
    global groq_client
    if not groq_client: return TranscriptionResult(None, error_message="Groq client not initialized.")
    try:
        tqdm.write(f"Transcribing {audio_path_to_use} with Groq (Model: {GROQ_TRANSCRIPTION_MODEL})...")
        with open(audio_path_to_use, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(file=(audio_path_to_use, file.read()), model=GROQ_TRANSCRIPTION_MODEL)
        return TranscriptionResult(transcription.text)
    except Exception as e: tqdm.write(f"Error in Groq transcription: {str(e)}"); return TranscriptionResult(None, error_message=str(e))

def transcribe_audio_assemblyai(audio_path_to_use):
    global assemblyai_transcriber
    if not assemblyai_transcriber: return TranscriptionResult(None, error_message="AssemblyAI client not initialized.")
    try:
        tqdm.write(f"Uploading {audio_path_to_use} and transcribing with AssemblyAI...")
        import assemblyai 
        config = assemblyai.TranscriptionConfig()
        transcript = assemblyai_transcriber.transcribe(audio_path_to_use, config=config)
        if transcript.status == assemblyai.TranscriptStatus.error: return TranscriptionResult(None, error_message=transcript.error)
        if transcript.text: return TranscriptionResult(transcript.text)
        return TranscriptionResult(None, error_message="AssemblyAI: No text and no explicit error.")
    except Exception as e: tqdm.write(f"Error in AssemblyAI transcription: {str(e)}"); return TranscriptionResult(None, error_message=str(e))

def transcribe_audio_dispatcher(original_audio_path, service_key):
    target_audio_path = original_audio_path
    if service_key == "groq" or service_key == "assemblyai":
        base_name_no_ext = os.path.splitext(os.path.basename(original_audio_path))[0]
        audio_dir = os.path.dirname(original_audio_path)
        if base_name_no_ext.endswith("_api"): 
            base_name_no_ext = base_name_no_ext[:-4]
        low_bitrate_audio_path = os.path.join(audio_dir, f"{base_name_no_ext}_api.mp3")
        if os.path.exists(low_bitrate_audio_path):
            target_audio_path = low_bitrate_audio_path
            tqdm.write(f"Using low-bitrate audio for {service_key}: {target_audio_path}")
        else:
            tqdm.write(f"Warning: Low-bitrate audio '{low_bitrate_audio_path}' not found for {service_key}. Using original audio: {original_audio_path}")

    tqdm.write(f"Attempting transcription with {service_key} using audio: {target_audio_path}")
    if service_key == "whisper_local": return transcribe_audio_whisper_local(target_audio_path)
    elif service_key == "groq": return transcribe_audio_groq(target_audio_path)
    elif service_key == "assemblyai": return transcribe_audio_assemblyai(target_audio_path)
    tqdm.write(f"Unknown audio service key: {service_key}. Defaulting to local Whisper."); return transcribe_audio_whisper_local(original_audio_path)


# --- Text Processing Implementations --- #
def process_text_ollama_local(prompt, task):
    data = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        tqdm.write(f"Processing text with local Ollama (Model: {OLLAMA_MODEL}) for {task}...")
        r = requests.post(OLLAMA_API_URL, json=data, timeout=300); r.raise_for_status(); return r.json()['response']
    except Exception as e: tqdm.write(f"Error during {task} with Ollama: {e}"); return None

def process_text_groq(prompt, task):
    global groq_client
    if not groq_client: tqdm.write(f"Groq client not initialized for {task}."); return None
    try:
        tqdm.write(f"Processing text with Groq (Model: {GROQ_CHAT_MODEL}) for {task}...")
        cc = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=GROQ_CHAT_MODEL)
        return cc.choices[0].message.content
    except Exception as e: tqdm.write(f"Error during {task} with Groq: {e}"); return None

def process_text_openrouter(prompt, task):
    if not OPENROUTER_API_KEY: tqdm.write(f"OPENROUTER_API_KEY not found for {task}."); return None
    h = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json", "HTTP-Referer": OPENROUTER_SITE_URL, "X-Title": OPENROUTER_APP_NAME}
    d = {"model": OPENROUTER_CHAT_MODEL, "messages": [{"role": "user", "content": prompt}]}
    try:
        tqdm.write(f"Processing text with OpenRouter (Model: {OPENROUTER_CHAT_MODEL}) for {task}...")
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=h, json=d, timeout=300); r.raise_for_status(); return r.json()['choices'][0]['message']['content']
    except Exception as e: tqdm.write(f"Error during {task} with OpenRouter: {e}"); return None

def summarize_text_dispatcher(text, key):
    p = f"Summarize the following English text, highlighting key points. Output only the summary:\n\n{text}"
    tqdm.write(f"Attempting summarization using service key: {key}")
    if key == "ollama_local": return process_text_ollama_local(p, "Summarization")
    elif key == "groq": return process_text_groq(p, "Summarization")
    elif key == "openrouter": return process_text_openrouter(p, "Summarization")
    tqdm.write(f"Unknown text service key for summarization: {key}. Defaulting to Ollama."); return process_text_ollama_local(p, "Summarization")

# MODIFIED: Generalized translation function with Google Translate support
def translate_text_google(text, target_language_code):
    """Translate text using Google Translate via deep-translator"""
    global google_translator
    if not google_translator or not GoogleTranslator:
        tqdm.write("❌ Google Translator not initialized.")
        return None
    
    try:
        tqdm.write(f"Translating with Google Translate to language code: {target_language_code}...")
        translator = GoogleTranslator(source='en', target=target_language_code)
        
        # Split text into chunks if it's too long (Google Translate has a 5000 char limit)
        max_length = 4500
        if len(text) <= max_length:
            return translator.translate(text)
        else:
            # Split by paragraphs and translate in chunks
            chunks = []
            current_chunk = ""
            for paragraph in text.split('\n'):
                if len(current_chunk) + len(paragraph) + 1 <= max_length:
                    current_chunk += paragraph + '\n'
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = paragraph + '\n'
            if current_chunk:
                chunks.append(current_chunk)
            
            translated_chunks = [translator.translate(chunk) for chunk in chunks]
            return '\n'.join(translated_chunks)
    except Exception as e:
        tqdm.write(f"❌ Error during Google Translate: {e}")
        return None

def translate_text_dispatcher(text, target_language_llm_name, target_language_code, service_key):
    task_name = f"{target_language_llm_name} Translation"
    tqdm.write(f"Attempting {task_name} using service key: {service_key}")
    
    if service_key == "google_translate":
        return translate_text_google(text, target_language_code)
    else:
        # Use LLM-based translation
        p = f"Translate the following English text to {target_language_llm_name}. Provide only the {target_language_llm_name} translation without any introductory phrases, explanations, or quotation marks around the translation:\n\n{text}"
        if service_key == "ollama_local": return process_text_ollama_local(p, task_name)
        elif service_key == "groq": return process_text_groq(p, task_name)
        elif service_key == "openrouter": return process_text_openrouter(p, task_name)
        tqdm.write(f"Unknown text service key for translation: {service_key}. Defaulting to Ollama."); return process_text_ollama_local(p, task_name)

# --- ElevenLabs Audio Generation --- #
def generate_audio_elevenlabs(text_content, output_audio_path, voice_id="21m00Tcm4TlvDq8ikWAM"):
    """
    Generate audio from text using ElevenLabs API
    Default voice: Rachel (21m00Tcm4TlvDq8ikWAM)
    """
    global elevenlabs_client
    if not elevenlabs_client:
        tqdm.write("❌ ElevenLabs client not initialized. Cannot generate audio.")
        return False
    
    try:
        tqdm.write(f"🎙️ Generating audio with ElevenLabs (Voice ID: {voice_id})...")
        
        # Generate audio using ElevenLabs
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            text=text_content,
            model_id="eleven_multilingual_v2"
        )
        
        # Save the audio file
        with open(output_audio_path, "wb") as audio_file:
            for chunk in audio_generator:
                audio_file.write(chunk)
        
        tqdm.write(f"🎵 Audio successfully generated: {output_audio_path}")
        return True
        
    except Exception as e:
        tqdm.write(f"❌ Error generating audio with ElevenLabs: {str(e)}")
        return False

# ========== Main Execution ========== #
if __name__ == "__main__":
    if not VideoFileClip or not tqdm:
        print("One or more essential libraries (MoviePy, TQDM) are missing. Please install them. Exiting.")
        exit()

    AUDIO_SERVICE_OPTIONS = {
        "1": ("whisper_local", "Local Whisper"),
        "2": ("groq", "Groq API (requires GROQ_API_KEY)"),
        "3": ("assemblyai", "AssemblyAI API (requires ASSEMBLYAI_API_KEY)")
    }
    TEXT_SERVICE_OPTIONS = {
        "1": ("ollama_local", "Local Ollama"),
        "2": ("groq", "Groq API (requires GROQ_API_KEY)"),
        "3": ("openrouter", "OpenRouter API (requires OPENROUTER_API_KEY)")
    }

    TRANSLATION_LANGUAGE_OPTIONS = {
        "1": ({"code": "fa", "llm_name": "Persian", "display": "Persian", "emoji": "🇮🇷"}, "Persian"),
        "2": ({"code": "it", "llm_name": "Italian", "display": "Italian", "emoji": "🇮🇹"}, "Italian"),
        "3": ({"code": "es", "llm_name": "Spanish", "display": "Spanish", "emoji": "🇪🇸"}, "Spanish"),
        "4": ({"code": "fr", "llm_name": "French", "display": "French", "emoji": "🇫🇷"}, "French"),
        "5": ({"code": "pt", "llm_name": "Portuguese", "display": "Portuguese", "emoji": "🇵🇹"}, "Portuguese"),
        "6": ({"code": "ru", "llm_name": "Russian", "display": "Russian", "emoji": "🇷🇺"}, "Russian"),
        "7": ({"code": "zh", "llm_name": "Chinese", "display": "Chinese", "emoji": "🇨🇳"}, "Chinese"),
    }

    ELEVENLABS_VOICE_OPTIONS = {
        "1": ("21m00Tcm4TlvDq8ikWAM", "Rachel - Calm, Natural Female Voice"),
        "2": ("EXAVITQu4vr4xnSDxMaL", "Bella - Soft, Friendly Female Voice"),
        "3": ("AZnzlk1XvdvUeBnXmlld", "Domi - Strong, Confident Female Voice"),
        "4": ("ErXwobaYiN019PkySvjV", "Antoni - Well-Rounded Male Voice"),
        "5": ("VR6AewLTigWG4xSOukaG", "Arnold - Crisp, Professional Male Voice"),
        "6": ("pNInz6obpgDQGcFmaJgB", "Adam - Deep, Authoritative Male Voice"),
        "7": ("yoZ06aMxZJJ28mfd3POQ", "Sam - Dynamic, Energetic Male Voice"),
        "8": ("IKne3meq5aSn9XLyUdCD", "Charlie - Casual, Conversational Male Voice"),
    }

    TRANSLATION_SERVICE_OPTIONS = {
        "1": ("google_translate", "Google Translate (Free, Fast)"),
        "2": ("ollama_local", "Local Ollama"),
        "3": ("groq", "Groq API (requires GROQ_API_KEY)"),
        "4": ("openrouter", "OpenRouter API (requires OPENROUTER_API_KEY)")
    }

    print("Welcome to the Video Processing Script!")
    chosen_audio_service_key = get_user_choice("Please select the Audio-to-Text service:", AUDIO_SERVICE_OPTIONS)
    chosen_text_service_key = get_user_choice("Please select the Text Processing service (for summarization):", TEXT_SERVICE_OPTIONS)

    perform_audio_generation = get_yes_no_choice("Do you want to generate audio from summaries using ElevenLabs?")
    
    chosen_voice_id = None
    if perform_audio_generation:
        if not initialize_elevenlabs_client():
            print("⚠️ ElevenLabs initialization failed. Audio generation will be skipped.")
            perform_audio_generation = False
        else:
            chosen_voice_id = get_user_choice("Please select the ElevenLabs voice for audio generation:", ELEVENLABS_VOICE_OPTIONS)

    perform_translation = get_yes_no_choice("Do you want to translate the transcript and summary?")
    
    chosen_translation_lang_details = None
    chosen_translation_service_key = None
    if perform_translation:
        chosen_translation_lang_details = get_user_choice("Please select the target language for translation:", TRANSLATION_LANGUAGE_OPTIONS)
        chosen_translation_service_key = get_user_choice("Please select the Translation service:", TRANSLATION_SERVICE_OPTIONS)
        
        # Initialize Google Translator if selected
        if chosen_translation_service_key == "google_translate":
            if not initialize_google_translator():
                print("⚠️ Google Translator initialization failed. Please select another translation service.")
                chosen_translation_service_key = get_user_choice("Please select an alternative Translation service:", TRANSLATION_SERVICE_OPTIONS)

    initialize_selected_clients(chosen_audio_service_key, chosen_text_service_key)
    video_files = get_video_files()

    if not video_files: print("❌ No video files found in the input folder.")
    else:
        audio_service_desc = next((desc for val, desc in AUDIO_SERVICE_OPTIONS.values() if val == chosen_audio_service_key), chosen_audio_service_key)
        text_service_desc = next((desc for val, desc in TEXT_SERVICE_OPTIONS.values() if val == chosen_text_service_key), chosen_text_service_key)
        
        print(f"\n🔍 Found {len(video_files)} videos for processing.")
        print(f"🔊 Audio-to-Text Service Selected: {audio_service_desc}")
        print(f"✍️ Text Processing Service (Summarization): {text_service_desc}")
        if perform_audio_generation and chosen_voice_id:
            voice_desc = next((desc for vid, desc in ELEVENLABS_VOICE_OPTIONS.values() if vid == chosen_voice_id), "Unknown Voice")
            print(f"�️ Audio Generation: Enabled (Voice: {voice_desc})")
        else:
            print(f"🎙️ Audio Generation: Skipped")
        if perform_translation and chosen_translation_lang_details and chosen_translation_service_key:
            translation_service_desc = next((desc for val, desc in TRANSLATION_SERVICE_OPTIONS.values() if val == chosen_translation_service_key), chosen_translation_service_key)
            print(f"🌍 Translation: Enabled")
            print(f"   └─ Target Language: {chosen_translation_lang_details['display']}")
            print(f"   └─ Translation Service: {translation_service_desc}")
        else:
            print("🌍 Translation: Skipped")
        print("")


        for video_file in tqdm(video_files, desc="Overall Video Processing Progress"):
            base_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(INPUT_FOLDER, video_file)
            
            original_audio_path = os.path.join(AUDIO_FOLDER, f"{base_name}.mp3")
            low_bitrate_audio_path = os.path.join(AUDIO_FOLDER, f"{base_name}_api.mp3")
            
            text_path_txt_en = os.path.join(TEXT_FOLDER, f"{base_name}.txt")
            text_path_pdf_en = os.path.join(TEXT_FOLDER, f"{base_name}.pdf")
            summary_path_en = os.path.join(SUMMARY_FOLDER, f"{base_name}_summary.txt")

            text_path_txt_translated = None
            summary_path_translated = None
            current_lang_code = None
            current_lang_llm_name = None
            current_lang_display_name = None
            current_lang_emoji = "🌐"


            if perform_translation and chosen_translation_lang_details:
                current_lang_code = chosen_translation_lang_details["code"]
                current_lang_llm_name = chosen_translation_lang_details["llm_name"]
                current_lang_display_name = chosen_translation_lang_details["display"]
                current_lang_emoji = chosen_translation_lang_details["emoji"]
                text_path_txt_translated = os.path.join(TEXT_FOLDER, f"{base_name}_{current_lang_code}.txt")
                summary_path_translated = os.path.join(SUMMARY_FOLDER, f"{base_name}_summary_{current_lang_code}.txt")

            tqdm.write(f"\n{'='*10} Processing: {video_file} {'='*10}")

            if not os.path.exists(original_audio_path):
                tqdm.write(f"🔊 Extracting original audio for {video_file}...")
                if not extract_audio(video_path, original_audio_path):
                    tqdm.write(f"❌ Original audio extraction failed for {video_file}. Skipping this file.")
                    continue
                tqdm.write(f"🎧 Original audio extracted: {original_audio_path}")
            else:
                tqdm.write(f"🔊 Original audio file '{original_audio_path}' already exists.")

            if os.path.exists(original_audio_path):
                if not os.path.exists(low_bitrate_audio_path):
                    tqdm.write(f"🔊 Creating low-bitrate audio version for API usage...")
                    if not create_low_bitrate_audio(original_audio_path, low_bitrate_audio_path, bitrate="64k"):
                        tqdm.write(f"⚠️ Failed to create low-bitrate audio. API services will use original if selected and this fails.")
                else:
                    tqdm.write(f"🔊 Low-bitrate audio file '{low_bitrate_audio_path}' already exists.")
            else:
                    tqdm.write(f"⚠️ Original audio '{original_audio_path}' does not exist. Cannot create or use low-bitrate version.")

            english_transcript_content = None
            if os.path.exists(text_path_txt_en):
                tqdm.write(f"📝 English transcript TXT '{text_path_txt_en}' exists. Loading.")
                try:
                    with open(text_path_txt_en, "r", encoding="utf-8") as f: english_transcript_content = f.read()
                    tqdm.write("📝 Loaded from existing English TXT.")
                    if os.path.exists(text_path_pdf_en): tqdm.write(f"📄 English PDF '{text_path_pdf_en}' also exists.")
                    elif english_transcript_content and SimpleDocTemplate:
                        tqdm.write(f"📄 English PDF '{text_path_pdf_en}' missing. Generating...")
                        save_transcript_as_pdf(english_transcript_content, text_path_pdf_en, PDF_FONT_NAME)
                except Exception as e: tqdm.write(f"❌ Error loading '{text_path_txt_en}': {e}. Re-transcribing.")
            
            if not english_transcript_content:
                if not os.path.exists(original_audio_path): 
                    tqdm.write(f"❌ Original audio '{original_audio_path}' not found. Cannot transcribe. Skipping {video_file}.")
                    continue
                trans_result = transcribe_audio_dispatcher(original_audio_path, chosen_audio_service_key)
                if trans_result and trans_result.status == "completed" and trans_result.text:
                    english_transcript_content = trans_result.text
                    tqdm.write("📝 English transcription successful.")
                    with open(text_path_txt_en, "w", encoding="utf-8") as f: f.write(english_transcript_content)
                    tqdm.write(f"💾 Saved English TXT: {text_path_txt_en}")
                    if SimpleDocTemplate: save_transcript_as_pdf(english_transcript_content, text_path_pdf_en, PDF_FONT_NAME)
                else:
                    err_msg = trans_result.error_message if trans_result else "Unknown transcription error."
                    tqdm.write(f"❌ Transcription failed: {err_msg}. Skipping {video_file}.")
                    continue
            
            if not english_transcript_content:
                tqdm.write(f"❌ No English transcript for {video_file} to proceed. Skipping."); continue

            if perform_translation and text_path_txt_translated and current_lang_llm_name:
                if os.path.exists(text_path_txt_translated): 
                    tqdm.write(f"{current_lang_emoji} Translated ({current_lang_display_name}) TXT '{text_path_txt_translated}' exists. Skipping.")
                else:
                    translated_transcript = translate_text_dispatcher(english_transcript_content, current_lang_llm_name, current_lang_code, chosen_translation_service_key)
                    if translated_transcript:
                        with open(text_path_txt_translated, "w", encoding="utf-8") as f: f.write(translated_transcript)
                        tqdm.write(f"{current_lang_emoji} Saved Translated ({current_lang_display_name}) TXT: {text_path_txt_translated}")
                    else: 
                        tqdm.write(f"❌ Translated ({current_lang_display_name}) transcript generation failed.")

            english_summary = None
            if os.path.exists(summary_path_en):
                tqdm.write(f"✍️ English summary '{summary_path_en}' exists. Loading.")
                try:
                    with open(summary_path_en, "r", encoding="utf-8") as f: english_summary = f.read()
                except Exception as e: tqdm.write(f"❌ Error loading '{summary_path_en}': {e}")
            elif len(english_transcript_content.split()) > 100:
                english_summary = summarize_text_dispatcher(english_transcript_content, chosen_text_service_key)
                if english_summary:
                    with open(summary_path_en, "w", encoding="utf-8") as f: f.write(english_summary)
                    tqdm.write(f"📑 Saved English summary: {summary_path_en}")
                else: tqdm.write("❌ English summarization failed.")
            else: tqdm.write("📖 Text not long enough for summarization (less than 100 words).")

            if perform_translation and summary_path_translated and current_lang_llm_name:
                if english_summary:
                    if os.path.exists(summary_path_translated): 
                        tqdm.write(f"{current_lang_emoji} Translated ({current_lang_display_name}) summary '{summary_path_translated}' exists. Skipping.")
                    else:
                        translated_summary = translate_text_dispatcher(english_summary, current_lang_llm_name, current_lang_code, chosen_translation_service_key)
                        if translated_summary:
                            with open(summary_path_translated, "w", encoding="utf-8") as f: f.write(translated_summary)
                            tqdm.write(f"{current_lang_emoji} Saved Translated ({current_lang_display_name}) summary: {summary_path_translated}")
                        else: 
                            tqdm.write(f"❌ Translated ({current_lang_display_name}) summary generation failed.")
                elif os.path.exists(summary_path_en): 
                    tqdm.write(f"ℹ️ English summary file '{summary_path_en}' exists but content not processed for translated summary.")
                else: 
                    tqdm.write(f"ℹ️ No English summary to translate to {current_lang_display_name}.")
            
            # Generate audio from summaries using ElevenLabs
            if perform_audio_generation and elevenlabs_client and chosen_voice_id:
                # Generate audio for English summary
                if english_summary:
                    audio_path_en = os.path.join(SUMMARY_AUDIO_FOLDER, f"{base_name}_summary_en.mp3")
                    if os.path.exists(audio_path_en):
                        tqdm.write(f"🎵 English summary audio '{audio_path_en}' already exists. Skipping.")
                    else:
                        generate_audio_elevenlabs(english_summary, audio_path_en, chosen_voice_id)
                
                # Generate audio for translated summary if available
                if perform_translation and summary_path_translated and current_lang_code:
                    if os.path.exists(summary_path_translated):
                        try:
                            with open(summary_path_translated, "r", encoding="utf-8") as f:
                                translated_summary_content = f.read()
                            
                            audio_path_translated = os.path.join(SUMMARY_AUDIO_FOLDER, f"{base_name}_summary_{current_lang_code}.mp3")
                            if os.path.exists(audio_path_translated):
                                tqdm.write(f"{current_lang_emoji} Translated ({current_lang_display_name}) summary audio '{audio_path_translated}' already exists. Skipping.")
                            else:
                                generate_audio_elevenlabs(translated_summary_content, audio_path_translated, chosen_voice_id)
                        except Exception as e:
                            tqdm.write(f"❌ Error reading translated summary for audio generation: {e}")
            
            time.sleep(0.2)


        print("\n✅ All video processing finished.")
