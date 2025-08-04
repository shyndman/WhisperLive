import os
import json
import logging
import threading
import time
import torch
import ctranslate2
from huggingface_hub import snapshot_download

from whisper_live.transcriber.transcriber_faster_whisper import WhisperModel
from whisper_live.backend.base import ServeClientBase


class ServeClientFasterWhisper(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(
        self,
        websocket,
        task="transcribe",
        device=None,
        language=None,
        client_uid=None,
        model="small.en",
        initial_prompt=None,
        vad_parameters=None,
        use_vad=True,
        single_model=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=7,
        cache_path="~/.cache/whisper-live/",
        translation_queue=None,
    ):
        """
        Initialize a ServeClient instance.
        The Whisper model is initialized based on the client's language and device availability.
        The transcription thread is started upon initialization. A "SERVER_READY" message is sent
        to the client to indicate that the server is ready.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            task (str, optional): The task type, e.g., "transcribe". Defaults to "transcribe".
            device (str, optional): The device type for Whisper, "cuda" or "cpu". Defaults to None.
            language (str, optional): The language for transcription. Defaults to None.
            client_uid (str, optional): A unique identifier for the client. Defaults to None.
            model (str, optional): The whisper model size. Defaults to 'small.en'
            initial_prompt (str, optional): Prompt for whisper inference. Defaults to None.
            single_model (bool, optional): Whether to instantiate a new model for each client connection. Defaults to False.
            send_last_n_segments (int, optional): Number of most recent segments to send to the client. Defaults to 10.
            no_speech_thresh (float, optional): Segments with no speech probability above this threshold will be discarded. Defaults to 0.45.
            clip_audio (bool, optional): Whether to clip audio with no valid segments. Defaults to False.
            same_output_threshold (int, optional): Number of repeated outputs before considering it as a valid segment. Defaults to 10.

        """
        logging.debug(f"Initializing ServeClientFasterWhisper for client {client_uid} with model {model}, task {task}, device {device}")
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
            translation_queue
        )
        self.cache_path = cache_path
        self.model_sizes = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v2", "large-v3", "distil-small.en",
            "distil-medium.en", "distil-large-v2", "distil-large-v3",
            "large-v3-turbo", "turbo"
        ]

        self.model_size_or_path = model
        self.language = "en" if self.model_size_or_path.endswith("en") else language
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"onset": 0.5}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.debug(f"Selected device: {device}")
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability(device)
            self.compute_type = "float16" if major >= 7 else "float32"
            logging.debug(f"CUDA device capability: {major}, using compute_type: {self.compute_type}")
        else:
            self.compute_type = "int8"
            logging.debug(f"Using CPU with compute_type: {self.compute_type}")

        if self.model_size_or_path is None:
            return
        logging.info(f"Using Device={device} with precision {self.compute_type}")
    
        try:
            if single_model:
                logging.debug(f"Using single model mode for client {client_uid}")
                if ServeClientFasterWhisper.SINGLE_MODEL is None:
                    logging.debug("Creating new single model instance")
                    self.create_model(device)
                    ServeClientFasterWhisper.SINGLE_MODEL = self.transcriber
                else:
                    logging.debug("Reusing existing single model instance")
                    self.transcriber = ServeClientFasterWhisper.SINGLE_MODEL
            else:
                logging.debug(f"Creating dedicated model for client {client_uid}")
                self.create_model(device)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "status": "ERROR",
                "message": f"Failed to load model: {str(self.model_size_or_path)}"
            }))
            self.websocket.close()
            return

        self.use_vad = use_vad

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.SERVER_READY,
                    "backend": "faster_whisper"
                }
            )
        )

    def create_model(self, device):
        """
        Instantiates a new model, sets it as the transcriber. If model is a huggingface model_id
        then it is automatically converted to ctranslate2(faster_whisper) format.
        """
        logging.debug(f"Creating model with reference: {self.model_size_or_path}")
        model_ref = self.model_size_or_path

        if model_ref in self.model_sizes:
            logging.debug(f"Model '{model_ref}' found in standard model sizes")
            model_to_load = model_ref
        else:
            logging.debug(f"Model '{model_ref}' not in standard model sizes, checking if custom model")
            if os.path.isdir(model_ref) and ctranslate2.contains_model(model_ref):
                logging.debug(f"Found local CTranslate2 model at: {model_ref}")
                model_to_load = model_ref
            else:
                logging.debug(f"Downloading model from HuggingFace: {model_ref}")
                local_snapshot = snapshot_download(
                    repo_id = model_ref,
                    repo_type = "model",
                )
                logging.debug(f"Downloaded model to: {local_snapshot}")
                if ctranslate2.contains_model(local_snapshot):
                    logging.debug("Downloaded model is already in CTranslate2 format")
                    model_to_load = local_snapshot
                else:
                    cache_root = os.path.expanduser(os.path.join(self.cache_path, "whisper-ct2-models/"))
                    os.makedirs(cache_root, exist_ok=True)
                    safe_name = model_ref.replace("/", "--")
                    ct2_dir = os.path.join(cache_root, safe_name)
                    logging.debug(f"CTranslate2 cache directory: {ct2_dir}")

                    if not ctranslate2.contains_model(ct2_dir):
                        logging.debug(f"Converting '{model_ref}' to CTranslate2 format at {ct2_dir}")
                        logging.info(f"Converting '{model_ref}' to CTranslate2 @ {ct2_dir}")
                        ct2_converter = ctranslate2.converters.TransformersConverter(
                            local_snapshot, 
                            copy_files=["tokenizer.json", "preprocessor_config.json"]
                        )
                        ct2_converter.convert(
                            output_dir=ct2_dir,
                            quantization=self.compute_type,
                            force=False,  # skip if already up-to-date
                        )
                        logging.debug("Model conversion completed")
                    else:
                        logging.debug("CTranslate2 model already exists in cache")
                    model_to_load = ct2_dir

        logging.info(f"Loading model: {model_to_load}")
        logging.debug(f"Model loading parameters - device: {device}, compute_type: {self.compute_type}")
        self.transcriber = WhisperModel(
            model_to_load,
            device=device,
            compute_type=self.compute_type,
            local_files_only=False,
        )
        logging.debug("Model loaded successfully")

    def set_language(self, info):
        """
        Updates the language attribute based on the detected language information.

        Args:
            info (object): An object containing the detected language and its probability. This object
                        must have at least two attributes: `language`, a string indicating the detected
                        language, and `language_probability`, a float representing the confidence level
                        of the language detection.
        """
        logging.debug(f"Language detection info - language: {info.language}, probability: {info.language_probability}")
        if info.language_probability > 0.5:
            logging.debug(f"Language probability {info.language_probability} > 0.5, setting language to {info.language}")
            self.language = info.language
            logging.info(f"Detected language {self.language} with probability {info.language_probability}")
            self.websocket.send(json.dumps(
                {"uid": self.client_uid, "language": self.language, "language_prob": info.language_probability}))
        else:
            logging.debug(f"Language probability {info.language_probability} <= 0.5, not setting language")

    def transcribe_audio(self, input_sample):
        """
        Transcribes the provided audio sample using the configured transcriber instance.

        If the language has not been set, it updates the session's language based on the transcription
        information.

        Args:
            input_sample (np.array): The audio chunk to be transcribed. This should be a NumPy
                                    array representing the audio data.

        Returns:
            The transcription result from the transcriber. The exact format of this result
            depends on the implementation of the `transcriber.transcribe` method but typically
            includes the transcribed text.
        """
        logging.debug(f"Transcribing audio sample with shape: {input_sample.shape if hasattr(input_sample, 'shape') else 'unknown'}")
        
        if ServeClientFasterWhisper.SINGLE_MODEL:
            logging.debug("Acquiring single model lock")
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.acquire()
        
        logging.debug(f"Starting transcription with language: {self.language}, task: {self.task}, vad: {self.use_vad}")
        result, info = self.transcriber.transcribe(
            input_sample,
            initial_prompt=self.initial_prompt,
            language=self.language,
            task=self.task,
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None)
        
        if ServeClientFasterWhisper.SINGLE_MODEL:
            logging.debug("Releasing single model lock")
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.release()

        logging.debug(f"Transcription completed with {len(list(result)) if result else 0} segments")
        if self.language is None and info is not None:
            self.set_language(info)
        return result

    def handle_transcription_output(self, result, duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            result (str): The result from whisper inference i.e. the list of segments.
            duration (float): Duration of the transcribed audio chunk.
        """
        logging.debug(f"Handling transcription output - duration: {duration:.2f}s, result segments: {len(result) if result else 0}")
        segments = []
        if len(result):
            logging.debug("Processing transcription segments")
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)
            logging.debug(f"Prepared {len(segments)} segments for client")

        if len(segments):
            logging.debug(f"Sending {len(segments)} segments to client")
            self.send_transcription_to_client(segments)
        else:
            logging.debug("No segments to send to client")
