# analyzers.py
import os
import math
import logging
import tempfile
import re
from typing import Dict, Tuple

import numpy as np
import cv2
import mediapipe as mp
import speech_recognition as sr
import soundfile as sf
import librosa

# PyAV: used to demux & decode audio from video without calling ffmpeg binary
try:
    import av
except ImportError:
    av = None

logger = logging.getLogger("analyzers")
logger.setLevel(logging.INFO)

class VisualAnalyzer:
    LIPS_UPPER = 13
    LIPS_LOWER = 14
    LIPS_LEFT = 61
    LIPS_RIGHT = 291
    NOSE_TIP = 1
    LEFT_EAR = 234
    RIGHT_EAR = 454

    def __init__(self, max_faces: int = 1, detection_confidence: float = 0.5, tracking_confidence: float = 0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.scores = {
            "eye_contact": [],
            "facial_expression": [],
            "lighting": [],
            "centering": [],
            "framing": [],
            "sharpness": []
        }
        self._eps = 1e-6

    @staticmethod
    def _safe_landmark(landmarks, idx):
        if 0 <= idx < len(landmarks):
            return landmarks[idx]
        return None

    def _get_sharpness_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(max(0.0, min(1.0, (variance - 50) / 150.0)))

    def _get_lighting_score(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_brightness = float(hsv[..., 2].mean())
        if avg_brightness < 50:
            return 0.4
        if avg_brightness > 220:
            return 0.6
        return 1.0

    def _analyze_face_position(self, landmarks) -> Tuple[float, float]:
        xs = np.array([p.x for p in landmarks], dtype=np.float32)
        ys = np.array([p.y for p in landmarks], dtype=np.float32)
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())
        face_center_x = (min_x + max_x) / 2.0
        dist_from_center = abs(face_center_x - 0.5)
        centering_score = max(0.0, 1.0 - (dist_from_center * 3.0))
        face_height = max_y - min_y
        if 0.3 <= face_height <= 0.6:
            framing_score = 1.0
        else:
            diff = min(abs(face_height - 0.3), abs(face_height - 0.6))
            framing_score = max(0.0, 1.0 - (diff * 3.0))
        return centering_score, framing_score

    def _check_eye_contact(self, landmarks) -> float:
        nose = self._safe_landmark(landmarks, self.NOSE_TIP)
        left_ear = self._safe_landmark(landmarks, self.LEFT_EAR)
        right_ear = self._safe_landmark(landmarks, self.RIGHT_EAR)
        if not (nose and left_ear and right_ear):
            return 0.0
        ear_dist_total = abs(nose.x - right_ear.x) + abs(nose.x - left_ear.x) + self._eps
        ratio = abs(nose.x - left_ear.x) / ear_dist_total
        deviation = abs(ratio - 0.5)
        return max(0.0, 1.0 - (deviation * 4.0))

    def _check_smile(self, landmarks) -> float:
        left = self._safe_landmark(landmarks, self.LIPS_LEFT)
        right = self._safe_landmark(landmarks, self.LIPS_RIGHT)
        top = self._safe_landmark(landmarks, self.LIPS_UPPER)
        bottom = self._safe_landmark(landmarks, self.LIPS_LOWER)
        if not (left and right and top and bottom):
            return 0.0
        w = math.hypot(left.x - right.x, left.y - right.y)
        h = math.hypot(top.x - bottom.x, top.y - bottom.y) + self._eps
        ratio = w / h
        if ratio < 2.0:
            return 0.3
        if ratio > 4.0:
            return 1.0
        return float((ratio - 2.0) / 2.0)

    def process_frame(self, frame):
        try:
            self.scores["lighting"].append(self._get_lighting_score(frame))
            self.scores["sharpness"].append(self._get_sharpness_score(frame))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if results and results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                self.scores["eye_contact"].append(self._check_eye_contact(lm))
                self.scores["facial_expression"].append(self._check_smile(lm))
                c, f = self._analyze_face_position(lm)
                self.scores["centering"].append(c)
                self.scores["framing"].append(f)
            else:
                self.scores["eye_contact"].append(0.0)
                self.scores["facial_expression"].append(0.0)
                self.scores["centering"].append(0.0)
                self.scores["framing"].append(0.0)
        except Exception as e:
            logger.exception("Visual frame processing error: %s", e)
            self.scores["eye_contact"].append(0.0)
            self.scores["facial_expression"].append(0.0)
            self.scores["centering"].append(0.0)
            self.scores["framing"].append(0.0)

    def get_averages(self) -> Dict[str, float]:
        return {k: (float(np.mean(v)) if v else 0.0) for k, v in self.scores.items()}


class AudioTextAnalyzer:
    def __init__(self):
        self._recognizer = sr.Recognizer()
        # Regex for fillers
        self._fillers = re.compile(
            r'\b(?:'
            r'um|uh|erm|er|hmm|huh|mm|mmh|ah|oh|eh|uh-huh|uhm|'
            r'like|you know|ya know|i mean|i guess|i suppose|i think|i feel|'
            r'kind of|kinda|sort of|sorta|a bit|a little|a little bit|'
            r'just|just saying|just to|just so|'
            r'well|so|right|okay|ok|alright|anyway|'
            r'basically|actually|literally|seriously|honestly|frankly|to be honest|'
            r'maybe|perhaps|possibly|probably|might be|could be|'
            r'you see|you know what i mean|you know what i\'m saying|'
            r'like i said|as i said|as i mentioned|'
            r'let me think|let\'s see|let me see|hold on|hang on|'
            r'at the end of the day|to be fair|if i\'m honest|to be honest|'
            r'and stuff|and so on|stuff like that|that sort of thing|that kind of thing|'
            r'and all that|and all|and things|and things like that|'
            r'for sure|you know what i mean|i mean|i guess so|i suppose so|'
            r'right now|sort of like|kinda like|you know what|'
            r'uh uh|mm hmm|mm-mm|ah right|oh right|'
            r'no offense|not gonna lie|ngl|to be clear|let me be clear|'
            r'if that makes sense|if that makes any sense|'
            r'that said|having said that|on the other hand|'
            r'so yeah|so anyway|so like|so basically|'
            r'that\'s kind of|that\'s sort of|that\'s basically|'
            r'you got me|you know what i mean\?'
            r')\b', flags=re.I)

        # Skills database
        self.skills_db = {
            "Business": [
                "sales", "business development", "marketing", "management", "strategy",
                "finance", "budgeting", "forecasting", "negotiation", "crm",
                "leadership", "stakeholder management", "vendor management",
                "operations", "procurement", "p&l", "mergers and acquisitions",
                "fundraising", "investor relations", "go-to-market", "pricing strategy",
                "business analysis", "product strategy", "retail", "client relations",
                "salesforce", "account management", "channel management", "distribution"
            ],
            "Technical": [
                "python", "java", "c++", "c#", "javascript", "typescript", "nodejs",
                "react", "angular", "vue", "django", "flask", "spring", "kotlin",
                "swift", "golang", "rust", "php", "laravel", ".net", "ruby on rails",
                "objective-c", "scala", "api design", "rest api", "graphql", "microservices"
            ],
            "Data & ML": [
                "data analysis", "data science", "pandas", "numpy", "scikit-learn",
                "tensorflow", "pytorch", "keras", "xgboost", "lightgbm", "spark",
                "hadoop", "feature engineering", "model deployment", "mlops",
                "statistics", "probability", "data visualization", "matplotlib",
                "tableau", "powerbi", "etl", "big data", "time series", "nlp",
                "computer vision", "modeling", "data engineering", "airflow"
            ],
            "Databases": [
                "sql", "mysql", "postgresql", "mongodb", "redis", "cassandra",
                "dynamodb", "oracle", "sqlite", "timeseries", "cockroachdb",
                "database design", "query optimization"
            ],
            "Cloud & DevOps": [
                "aws", "azure", "gcp", "kubernetes", "docker", "terraform",
                "cloudformation", "ci/cd", "jenkins", "gitlab-ci", "circleci",
                "github actions", "prometheus", "grafana", "helm", "ansible",
                "serverless", "lambda", "ecs", "ecr", "monitoring", "devops"
            ],
            "Security": [
                "cybersecurity", "encryption", "secure coding", "penetration testing",
                "vulnerability assessment", "incident response", "iam", "oauth",
                "sso", "ssl", "tls", "firewall", "siem", "threat modeling",
                "application security", "network security", "compliance", "gdpr"
            ],
            "QA & Testing": [
                "testing", "unit testing", "integration testing", "system testing",
                "automation testing", "selenium", "pytest", "jest", "mocha", "cypress",
                "tdd", "bdd", "load testing", "performance testing", "test planning"
            ],
            "Frontend / UI": [
                "html", "css", "sass", "less", "responsive design", "accessibility",
                "ux", "ui", "design systems", "storybook", "webpack", "babel",
                "user interface", "interaction design", "css3", "html5"
            ],
            "Product & Design": [
                "product management", "roadmap", "product strategy", "user research",
                "prototyping", "figma", "sketch", "adobe xd", "usability testing",
                "customer interviews", "a/b testing", "feature prioritization"
            ],
            "Analytics & BI": [
                "analytics", "cohort analysis", "attribution",
                "google analytics", "mixpanel", "amplitude", "data modeling",
                "data warehousing", "dbt", "reporting", "business intelligence"
            ],
            "Marketing & Growth": [
                "content marketing", "seo", "sem", "ppc", "social media", "email marketing",
                "growth hacking", "affiliate", "brand management",
                "performance marketing", "inbound marketing", "campaign management",
                "google ads", "facebook ads", "content strategy"
            ],
            "Soft Skills": [
                "communication", "teamwork", "collaboration", "problem solving",
                "adaptability", "time management", "public speaking", "presentation",
                "conflict resolution", "active listening", "mentoring", "coaching",
                "empathy", "critical thinking", "creativity", "decision making",
                "leadership", "influence", "feedback", "emotional intelligence"
            ],
            "Mobile": [
                "android", "ios", "react native", "flutter", "mobile development",
                "kotlin", "swiftui", "mobile testing", "app store optimization"
            ],
            "Tools": [
                "git", "github", "bitbucket", "jira", "confluence", "slack", "notion",
                "trello", "asana", "excel", "google sheets", "vscode", "intellij",
                "postman", "docker-compose"
            ],
            "Blockchain": [
                "blockchain", "web3", "solidity", "smart contracts", "ethereum",
                "nft", "defi", "cryptography", "tokenomics"
            ],
            "Other": [
                "iot", "edge computing", "robotics", "automation",
                "digital transformation", "supply chain", "logistics",
                "customer success", "hr analytics", "talent acquisition"
            ]
        }

    def _extract_audio_with_av(self, video_path: str, out_wav: str) -> None:
        if av is None:
            raise EnvironmentError(
                "PyAV (av) is not available. Install it with `pip install av`"
            )

        logger.info(f"Attempting to extract audio from {video_path} using PyAV...")
        container = av.open(video_path)
        stream = None
        for s in container.streams:
            if s.type == "audio":
                stream = s
                break
        
        if stream is None:
            logger.warning("No audio stream found in video. Skipping audio analysis.")
            sf.write(out_wav, np.zeros(16000), 16000)
            return

        frames = []
        sample_rate = stream.rate or 16000

        # MANUAL FORMAT HANDLING TO AVOID "UNEXPECTED KEYWORD ARGUMENT" ERROR
        try:
            for packet in container.demux(stream):
                for frame in packet.decode():
                    # 1. Get raw data without specifying format
                    arr = frame.to_ndarray()
                    
                    # 2. Convert to float32
                    arr = arr.astype(np.float32)
                    
                    # 3. Normalize based on native format
                    # 's16' or 's16p' -> signed 16-bit int
                    if 's16' in frame.format.name:
                        arr /= 32768.0
                    # 's32' -> signed 32-bit int
                    elif 's32' in frame.format.name:
                        arr /= 2147483648.0
                    # 'u8' -> unsigned 8-bit
                    elif 'u8' in frame.format.name:
                        arr = (arr - 128.0) / 128.0
                    
                    # 4. Handle shape (Planes vs Packed)
                    # PyAV ndarray for audio usually comes as (n_channels, n_samples) for planar
                    # or (1, n_samples) if mono planar.
                    # We standardize to (samples, channels) for easy concatenation
                    if arr.ndim == 2:
                        if arr.shape[0] < arr.shape[1]: 
                            # Shape is (Channels, Samples) -> Transpose to (Samples, Channels)
                            arr = arr.T
                    
                    frames.append(arr)

        except Exception as e:
            logger.error(f"Error during audio decoding: {e}")

        if not frames:
            logger.warning("No decoded audio frames extracted. Audio might be empty.")
            sf.write(out_wav, np.zeros(16000), 16000)
            return

        # Concatenate all frames
        audio_np = np.concatenate(frames, axis=0)
        
        # Convert to mono (average across channels if multiple exist)
        if audio_np.ndim > 1:
            audio_mono = np.mean(audio_np, axis=1)
        else:
            audio_mono = audio_np

        # Write WAV using soundfile
        sf.write(out_wav, audio_mono.astype(np.float32), int(sample_rate), subtype='PCM_16')
        logger.info(f"Audio extracted successfully to {out_wav}")

    def analyze(self, video_path: str) -> Dict:
        temp_wav = None
        data = {
            "vocal_confidence": 0.0,
            "pitch_stability": 0.0,
            "formality": 0.0,
            "grammar_clarity": 0.0,
            "transcription": "",
            "skills": {},
            "word_count": 0
        }

        try:
            # Create temp wav
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_wav = tf.name
            tf.close()

            # Extract audio
            self._extract_audio_with_av(video_path, temp_wav)

            # Load for analysis
            y, sr_rate = librosa.load(temp_wav, sr=16000, mono=True)
            
            # Check if silent
            if len(y) == 0 or np.max(np.abs(y)) < 0.001:
                logger.warning("Audio is silent or empty.")
                return data

            # Vocal confidence
            rms = librosa.feature.rms(y=y)[0]
            data["vocal_confidence"] = float(min(1.0, np.mean(rms) * 25.0))

            # Pitch stability
            try:
                f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                f0_valid = f0[~np.isnan(f0)]
                if len(f0_valid) > 10:
                    std = float(np.std(f0_valid))
                    if 15 <= std <= 40:
                        data["pitch_stability"] = 1.0
                    elif 5 < std < 15:
                        data["pitch_stability"] = 0.8
                    else:
                        data["pitch_stability"] = 0.5
            except Exception:
                data["pitch_stability"] = 0.0

            # Speech to Text
            text = ""
            with sr.AudioFile(temp_wav) as source:
                audio_data = self._recognizer.record(source)
                try:
                    text = self._recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    logger.warning("Speech Recognition could not understand audio.")
                except sr.RequestError as e:
                    logger.warning(f"Speech Recognition request failed: {e}")

            data["transcription"] = text
            
            if text:
                words = text.split()
                data["word_count"] = len(words)
                avg_len = sum(len(w) for w in words) / max(1, len(words))
                data["formality"] = float(min(1.0, (avg_len / 5.0)))
                
                filler_count = len(self._fillers.findall(text))
                penalty = min(1.0, filler_count * 0.1)
                data["grammar_clarity"] = float(max(0.0, 1.0 - penalty))

                t_lower = text.lower()
                found = {}
                for cat, kw_list in self.skills_db.items():
                    matches = []
                    for kw in kw_list:
                        if re.search(r'\b' + re.escape(kw) + r'\b', t_lower):
                            matches.append(kw)
                    if matches:
                        found[cat] = list(sorted(set(matches)))
                data["skills"] = found

        except Exception as e:
            logger.exception("AudioTextAnalyzer error: %s", e)
            
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

        return data