# main.py
import os
import time
import shutil
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException

from analyzers import VisualAnalyzer, AudioTextAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ascraa-main")

app = FastAPI(title="Ascraa Video Resume Grader (no ffmpeg binary)")

FRAME_SKIP = 5
MAX_WIDTH = 640
MAX_WORKERS = 2

def scale10(val: float) -> float:
    return round(val * 10.0, 1)

@app.post("/analyze-video-resume")
async def analyze_video(file: UploadFile = File(...)):
    start_time = time.time()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".mp4")
    try:
        with tmp as f:
            shutil.copyfileobj(file.file, f)
        temp_path = tmp.name

        visual = VisualAnalyzer()
        audio = AudioTextAnalyzer()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            audio_future = ex.submit(audio.analyze, temp_path)

            import cv2
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Cannot open video file")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_idx = 0
            
            while True:
                if total_frames > 0 and frame_idx >= total_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]
                if w > MAX_WIDTH:
                    scale = MAX_WIDTH / float(w)
                    frame = cv2.resize(frame, (MAX_WIDTH, int(h * scale)))
                
                visual.process_frame(frame)
                frame_idx += FRAME_SKIP
            
            cap.release()

            audio_data = audio_future.result(timeout=120)

        vis = visual.get_averages()

        # --- Scoring Logic ---
        
        # Body Language
        raw_face = (vis["eye_contact"] * 0.5) + (vis["facial_expression"] * 0.5)
        raw_groom = (vis["sharpness"] * 0.7) + (vis["lighting"] * 0.3)
        raw_env = (vis["framing"] * 0.5) + (vis["lighting"] * 0.25) + (vis["centering"] * 0.25)
        score_body_total = (raw_face * 0.60) + (raw_groom * 0.25) + (raw_env * 0.15)

        # Tone
        raw_conf = audio_data.get("vocal_confidence", 0.0)
        raw_pitch = audio_data.get("pitch_stability", 0.0)
        score_tone_total = (raw_conf * 0.65) + (raw_pitch * 0.35)

        # Words
        raw_form = audio_data.get("formality", 0.0)
        raw_gram = audio_data.get("grammar_clarity", 0.0)
        score_word_total = (raw_form * 0.80) + (raw_gram * 0.20)

        FINAL_SCORE = ((score_body_total * 0.55) + (score_tone_total * 0.38) + (score_word_total * 0.07)) * 10.0
        FINAL_SCORE = max(0.0, min(10.0, FINAL_SCORE))

        grade = "Excellent" if FINAL_SCORE > 8 else "Good" if FINAL_SCORE > 6 else "Needs Improvement"

        response = {
            "scores": {
                "filename": file.filename,
                "duration_processed": f"{round(time.time() - start_time, 2)}s"
            },
            "overall_performance": {
                "score": round(FINAL_SCORE, 1),
                "max_score": 10,
                "grade": grade,
                "summary": (
                    f"Your video resume scored {round(FINAL_SCORE,1)}/10. "
                    f"Body Language was {'Good' if score_body_total > 0.6 else 'Needs Improvement'}, "
                    f"Tone was {'Good' if score_tone_total > 0.6 else 'Needs Improvement'}, and "
                    f"Content was {'Good' if score_word_total > 0.6 else 'Needs Improvement'}."
                )
            },
            "breakdown": {
                "body_language_score": {
                    "score": scale10(score_body_total),
                    "details": {
                        "facial_expression_eye_contact": {
                            "value": scale10(raw_face),
                            "weight": "33% of Total",
                            "feedback": "Good engagement" if raw_face > 0.6 else "Maintain better eye contact"
                        },
                        "grooming_appearance": {
                            "value": scale10(raw_groom),
                            "weight": "13.75% of Total",
                            "feedback": "Professional appearance" if raw_groom > 0.6 else "Improve lighting or camera quality"
                        },
                        "environment_framing": {
                            "value": scale10(raw_env),
                            "weight": "8.25% of Total",
                            "components": {
                                "framing": scale10(vis["framing"]),
                                "centering": scale10(vis["centering"])
                            }
                        }
                    }
                },
                "voice_tone_score": {
                    "score": scale10(score_tone_total),
                    "details": {
                        "vocal_confidence": {"value": scale10(raw_conf), "weight": "24.7% of Total"},
                        "pitch_stability": {"value": scale10(raw_pitch), "weight": "13.3% of Total"}
                    }
                },
                "verbal_content_score": {
                    "score": scale10(score_word_total),
                    "details": {
                        "formality_vocabulary": scale10(raw_form),
                        "grammar_clarity": scale10(raw_gram)
                    }
                }
            },
            "extracted_insights": {
                "skills_identified": audio_data.get("skills", {}),
                "word_count": audio_data.get("word_count", 0),
                "full_transcription": audio_data.get("transcription", "")
            }
        }

        return response

    except Exception as e:
        logger.exception("Error analyzing video: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)