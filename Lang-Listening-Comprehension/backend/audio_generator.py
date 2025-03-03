import boto3
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
import tempfile
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSServiceError(Exception):
    """Custom exception for TTS service errors"""
    pass

class AudioGenerator:
    def __init__(self):
        self.setup_paths()
        self.setup_tts_services()
        self.cleanup_old_files()

    def setup_paths(self):
        """Setup and validate audio paths"""
        # Get paths from environment or use defaults
        base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.audio_dir = Path(os.getenv('AUDIO_STORAGE_PATH', base_dir / 'frontend/static/audio'))
        self.temp_dir = Path(os.getenv('TEMP_AUDIO_PATH', base_dir / 'data/temp_audio'))
        
        # Create directories if they don't exist
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def setup_tts_services(self):
        """Setup TTS services with proper error handling"""
        self.tts_services = []
        
        # Try to setup AWS (primary service)
        try:
            self.bedrock = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))
            self.polly = boto3.client('polly')
            self.tts_services.append('aws')
        except Exception as e:
            logger.warning(f"Failed to initialize AWS services: {str(e)}")

        # Try to setup Google Cloud (fallback)
        try:
            from google.cloud import texttospeech
            self.google_client = texttospeech.TextToSpeechClient()
            self.tts_services.append('google')
        except Exception as e:
            logger.warning(f"Failed to initialize Google Cloud TTS: {str(e)}")

        # Try to setup Azure (fallback)
        try:
            from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer
            from azure.cognitiveservices.speech.audio import AudioOutputConfig
            
            if os.getenv('AZURE_SPEECH_KEY') and os.getenv('AZURE_SPEECH_REGION'):
                self.azure_speech_config = SpeechConfig(
                    subscription=os.getenv('AZURE_SPEECH_KEY'),
                    region=os.getenv('AZURE_SPEECH_REGION')
                )
                self.tts_services.append('azure')
        except Exception as e:
            logger.warning(f"Failed to initialize Azure TTS: {str(e)}")

        if not self.tts_services:
            raise TTSServiceError("No TTS services available. Please configure at least one service.")

        # Define voices for available services
        self.voices = {
            'aws': {
                'male': ['Takumi'],
                'female': ['Kazuha'],
                'announcer': 'Takumi'
            },
            'google': {
                'male': ['ja-JP-Standard-C', 'ja-JP-Standard-D'],
                'female': ['ja-JP-Standard-A', 'ja-JP-Standard-B'],
                'announcer': 'ja-JP-Standard-D'
            },
            'azure': {
                'male': ['ja-JP-KeitaNeural', 'ja-JP-DaichiNeural'],
                'female': ['ja-JP-NanamiNeural', 'ja-JP-AoiNeural'],
                'announcer': 'ja-JP-KeitaNeural'
            }
        }

    def cleanup_old_files(self):
        """Clean up old audio files"""
        try:
            max_age = int(os.getenv('MAX_AUDIO_FILE_AGE_DAYS', 7))
            cutoff_date = datetime.now() - timedelta(days=max_age)
            
            # Clean up temporary files
            for file in self.temp_dir.glob('*.mp3'):
                if datetime.fromtimestamp(file.stat().st_mtime) < cutoff_date:
                    file.unlink()
            
            # Clean up old static files
            for file in self.audio_dir.glob('*.mp3'):
                if datetime.fromtimestamp(file.stat().st_mtime) < cutoff_date:
                    file.unlink()
        except Exception as e:
            logger.error(f"Error during file cleanup: {str(e)}")

    def generate_audio(self, question: Dict) -> Optional[str]:
        """Generate audio for a question with fallback mechanisms"""
        try:
            parts = self.parse_conversation(question)
            if not self.validate_conversation_parts(parts):
                raise ValueError("Invalid conversation parts")

            # Try each available TTS service until one succeeds
            for service in self.tts_services:
                try:
                    audio_file = self._generate_audio_with_service(parts, service)
                    if audio_file:
                        return audio_file
                except Exception as e:
                    logger.error(f"Error with {service} TTS: {str(e)}")
                    continue

            raise TTSServiceError("All TTS services failed")
            
        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            return None

    def _generate_audio_with_service(self, parts: List[Tuple[str, str, str]], service: str) -> Optional[str]:
        """Generate audio using a specific TTS service"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.audio_dir / f"question_{timestamp}.mp3"
        
        try:
            # Generate audio for each part
            audio_parts = []
            current_section = None
            
            # Generate silence files for pauses
            long_pause = self.generate_silence(2000)  # 2 second pause
            short_pause = self.generate_silence(500)  # 0.5 second pause
            
            for speaker, text, gender in parts:
                # Detect section changes and add appropriate pauses
                if speaker.lower() == 'announcer':
                    if '次の会話' in text:  # Introduction
                        if current_section is not None:
                            audio_parts.append(long_pause)
                        current_section = 'intro'
                    elif '質問' in text or '選択肢' in text:  # Question or options
                        audio_parts.append(long_pause)
                        current_section = 'question'
                elif current_section == 'intro':
                    audio_parts.append(long_pause)
                    current_section = 'conversation'
                
                # Get appropriate voice for this speaker
                voice = self.get_voice_for_gender(gender, service)
                logger.info(f"Using voice {voice} for {speaker} ({gender}) with {service}")
                
                # Generate audio for this part
                audio_file = self.generate_audio_part(text, voice, service)
                if not audio_file:
                    raise Exception("Failed to generate audio part")
                audio_parts.append(audio_file)
                
                # Add short pause between conversation turns
                if current_section == 'conversation':
                    audio_parts.append(short_pause)
            
            # Combine all parts into final audio
            if not self.combine_audio_files(audio_parts, output_file):
                raise Exception("Failed to combine audio files")
            
            return str(output_file)
            
        except Exception as e:
            # Clean up the output file if it exists
            if output_file.exists():
                output_file.unlink()
            raise Exception(f"Audio generation failed: {str(e)}")

    def parse_conversation(self, question: Dict) -> List[Tuple[str, str, str]]:
        """
        Convert question into a format for audio generation.
        Returns a list of (speaker, text, gender) tuples.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Ask Nova to parse the conversation and assign speakers and genders
                prompt = f"""
                You are a JLPT listening test audio script generator. Format the following question for audio generation.

                Rules:
                1. Introduction and Question parts:
                   - Must start with 'Speaker: Announcer (Gender: male)'
                   - Keep as separate parts

                2. Conversation parts:
                   - Name speakers based on their role (Student, Teacher, etc.)
                   - Must specify gender EXACTLY as either 'Gender: male' or 'Gender: female'
                   - Use consistent names for the same speaker
                   - Split long speeches at natural pauses

                Format each part EXACTLY like this, with no variations:
                Speaker: [name] (Gender: male)
                Text: [Japanese text]
                ---

                Example format:
                Speaker: Announcer (Gender: male)
                Text: 次の会話を聞いて、質問に答えてください。
                ---
                Speaker: Student (Gender: female)
                Text: すみません、この電車は新宿駅に止まりますか。
                ---

                Question to format:
                {json.dumps(question, ensure_ascii=False, indent=2)}

                Output ONLY the formatted parts in order: introduction, conversation, question.
                Make sure to specify gender EXACTLY as shown in the example.
                """
                
                response = self._invoke_bedrock(prompt)
                
                # Parse the response into speaker parts
                parts = []
                current_speaker = None
                current_gender = None
                current_text = None
                
                # Track speakers to maintain consistent gender
                speaker_genders = {}
                
                for line in response.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.startswith('Speaker:'):
                        # Save previous speaker's part if exists
                        if current_speaker and current_text:
                            parts.append((current_speaker, current_text, current_gender))
                        
                        # Parse new speaker and gender
                        try:
                            speaker_part = line.split('Speaker:')[1].strip()
                            current_speaker = speaker_part.split('(')[0].strip()
                            gender_part = speaker_part.split('Gender:')[1].split(')')[0].strip().lower()
                            
                            # Normalize gender
                            if '男' in gender_part or 'male' in gender_part:
                                current_gender = 'male'
                            elif '女' in gender_part or 'female' in gender_part:
                                current_gender = 'female'
                            else:
                                raise ValueError(f"Invalid gender format: {gender_part}")
                            
                            # Infer gender from speaker name for consistency
                            if current_speaker.lower() in ['female', 'woman', 'girl', 'lady', '女性']:
                                current_gender = 'female'
                            elif current_speaker.lower() in ['male', 'man', 'boy', '男性']:
                                current_gender = 'male'
                            
                            # Check for gender consistency
                            if current_speaker in speaker_genders:
                                if current_gender != speaker_genders[current_speaker]:
                                    logger.warning(f"Warning: Gender mismatch for {current_speaker}. Using previously assigned gender {speaker_genders[current_speaker]}")
                                current_gender = speaker_genders[current_speaker]
                            else:
                                speaker_genders[current_speaker] = current_gender
                        except Exception as e:
                            logger.error(f"Error parsing speaker/gender: {line}")
                            raise e
                            
                    elif line.startswith('Text:'):
                        current_text = line.split('Text:')[1].strip()
                        
                    elif line == '---' and current_speaker and current_text:
                        parts.append((current_speaker, current_text, current_gender))
                        current_speaker = None
                        current_gender = None
                        current_text = None
                
                # Add final part if exists
                if current_speaker and current_text:
                    parts.append((current_speaker, current_text, current_gender))
                
                # Validate the parsed parts
                if self.validate_conversation_parts(parts):
                    return parts
                    
                logger.error(f"Attempt {attempt + 1}: Invalid conversation format, retrying...")
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception("Failed to parse conversation after multiple attempts")
        
        raise Exception("Failed to generate valid conversation format")

    def validate_conversation_parts(self, parts: List[Tuple[str, str, str]]) -> bool:
        """
        Validate that the conversation parts are properly formatted.
        Returns True if valid, False otherwise.
        """
        if not parts:
            logger.error("Error: No conversation parts generated")
            return False
            
        # Check that we have an announcer for intro
        if not parts[0][0].lower() == 'announcer':
            logger.error("Error: First speaker must be Announcer")
            return False
            
        # Check that each part has valid content
        for i, (speaker, text, gender) in enumerate(parts):
            # Check speaker
            if not speaker or not isinstance(speaker, str):
                logger.error(f"Error: Invalid speaker in part {i+1}")
                return False
                
            # Check text
            if not text or not isinstance(text, str):
                logger.error(f"Error: Invalid text in part {i+1}")
                return False
                
            # Check gender
            if gender not in ['male', 'female']:
                logger.error(f"Error: Invalid gender in part {i+1}: {gender}")
                return False
                
            # Check text contains Japanese characters
            if not any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text):
                logger.error(f"Error: Text does not contain Japanese characters in part {i+1}")
                return False
        
        return True

    def get_voice_for_gender(self, gender: str, service: str) -> str:
        """Get an appropriate voice for the given gender and service"""
        if gender == 'male':
            return self.voices[service]['male'][0]  # Male voice
        else:
            return self.voices[service]['female'][0]  # Female voice

    def generate_audio_part(self, text: str, voice_name: str, service: str) -> Optional[str]:
        """Generate audio for a single part using the specified TTS service"""
        if service == 'aws':
            response = self.polly.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=voice_name,
                Engine='neural',
                LanguageCode='ja-JP'
            )
        elif service == 'google':
            from google.cloud import texttospeech
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code='ja-JP',
                name=voice_name
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            response = self.google_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
        elif service == 'azure':
            from azure.cognitiveservices.speech import SpeechSynthesizer
            from azure.cognitiveservices.speech.audio import AudioOutputConfig
            synthesizer = SpeechSynthesizer(
                self.azure_speech_config,
                audio_config=AudioOutputConfig(use_default_speaker=True)
            )
            result = synthesizer.speak_text_async(text).get()
            if result.reason == SpeechSynthesizerResult.Reason.SynthesizingAudioCompleted:
                response = result.audio_data
            else:
                logger.error(f"Error with Azure TTS: {result.reason}")
                return None
        else:
            logger.error(f"Unsupported TTS service: {service}")
            return None
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_file.write(response.audio_content if service != 'aws' else response['AudioStream'].read())
            return temp_file.name

    def combine_audio_files(self, audio_files: List[str], output_file: str):
        """Combine multiple audio files using ffmpeg"""
        file_list = None
        try:
            # Create file list for ffmpeg
            with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")
                file_list = f.name
            
            # Combine audio files
            subprocess.run([
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', file_list,
                '-c', 'copy',
                output_file
            ], check=True)
            
            return True
        except Exception as e:
            logger.error(f"Error combining audio files: {str(e)}")
            if os.path.exists(output_file):
                os.unlink(output_file)
            return False
        finally:
            # Clean up temporary files
            if file_list and os.path.exists(file_list):
                os.unlink(file_list)
            for audio_file in audio_files:
                if os.path.exists(audio_file):
                    try:
                        os.unlink(audio_file)
                    except Exception as e:
                        logger.error(f"Error cleaning up {audio_file}: {str(e)}")

    def generate_silence(self, duration_ms: int) -> str:
        """Generate a silent audio file of specified duration"""
        output_file = self.temp_dir / f'silence_{duration_ms}ms.mp3'
        if not output_file.exists():
            subprocess.run([
                'ffmpeg', '-f', 'lavfi', '-i',
                f'anullsrc=r=24000:cl=mono:d={duration_ms/1000}',
                '-c:a', 'libmp3lame', '-b:a', '48k',
                str(output_file)
            ])
        return str(output_file)

    def _invoke_bedrock(self, prompt: str) -> str:
        """Invoke Bedrock with the given prompt using converse API"""
        messages = [{
            "role": "user",
            "content": [{
                "text": prompt
            }]
        }]
        
        try:
            response = self.bedrock.converse(
                modelId=os.getenv('MODEL_ID', 'amazon.nova-micro-v1:0'),
                messages=messages,
                inferenceConfig={
                    "temperature": 0.3,
                    "topP": 0.95,
                    "maxTokens": 2000
                }
            )
            return response['output']['message']['content'][0]['text']
        except Exception as e:
            logger.error(f"Error in Bedrock converse: {str(e)}")
            raise e
