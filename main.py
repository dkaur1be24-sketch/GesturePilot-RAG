"""
╔══════════════════════════════════════════════════════════════════╗
║         GESTURE CONTROL  +  RAG SESSION Q&A                     ║
║                                                                  ║
║  GESTURES (webcam window):                                       ║
║   😊 Smile (hold)   → Play YouTube                              ║
║   ✋ Open palm       → Volume UP                                 ║
║   ✊ Closed fist     → Volume DOWN                               ║
║   ✌  V-sign (hold)  → Close tab + Exit                          ║
║                                                                  ║
║  RAG Q&A (terminal — works while webcam is live):               ║
║   Just type your question and press Enter, e.g.:                ║
║   > How many times did I raise the volume?                       ║
║   > When did YouTube open?                                       ║
║   > What was my most used gesture?                               ║
║                                                                  ║
║  KEYS: R = reset face lock  |  Q = quit                         ║
╚══════════════════════════════════════════════════════════════════╝

INSTALL (one-time):
    pip install opencv-python face_recognition mediapipe pyautogui \
                faiss-cpu sentence-transformers anthropic

SET YOUR API KEY:
    export ANTHROPIC_API_KEY="sk-ant-..."
    (or hard-code it in ANTHROPIC_API_KEY below)
"""

import cv2
import face_recognition
import mediapipe as mp
import webbrowser
import pyautogui
import time
import threading
import os
import sys
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic

# ─────────────────────────────────────────────────────────────
#  USER SETTINGS
# ─────────────────────────────────────────────────────────────
YOUTUBE_URL           = "https://www.youtube.com/watch?v=xzUVPN68Ym4"
MAR_THRESHOLD         = 1.8
SMILE_FRAMES_REQUIRED = 50
VOLUME_COOLDOWN       = 1.2
VSIGN_FRAMES_REQUIRED = 25
MATCH_TOLERANCE       = 0.5
ENCODING_INTERVAL     = 10
ACTIVE_USER_ID        = "User #1"

# RAG settings
LOG_FILE              = "gesture_log.txt"
INDEX_REFRESH_SECS    = 5       # re-embed log into FAISS every N seconds
TOP_K                 = 8       # how many log chunks to retrieve per query
ANTHROPIC_API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")   # or paste key here
ANTHROPIC_MODEL       = "claude-sonnet-4-20250514"

# ─────────────────────────────────────────────────────────────
#  MEDIAPIPE LANDMARK INDEXES
# ─────────────────────────────────────────────────────────────
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14

# ─────────────────────────────────────────────────────────────
#  GLOBALS  (shared between threads — all writes are under lock)
# ─────────────────────────────────────────────────────────────
log_lock       = threading.Lock()   # protects gesture_log.txt writes
index_lock     = threading.Lock()   # protects FAISS index reads/writes
faiss_index    = None               # FAISS flat index
faiss_chunks   = []                 # list of raw log strings (parallel to index)
shutdown_event = threading.Event()  # set → all threads exit cleanly

# ─────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────
print("[INFO] Loading models — please wait...")

# MediaPipe
mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils
hands_model  = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.6
)
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=4,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Sentence embedding model (runs locally, ~90 MB download on first run)
print("[INFO] Loading sentence embedding model (SentenceTransformer)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384   # dimension for all-MiniLM-L6-v2

# Anthropic client
if not ANTHROPIC_API_KEY:
    print("[INFO] No ANTHROPIC_API_KEY — RAG will return raw log lines (no LLM needed).")
    anthropic_client = None
else:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

print("[INFO] All models loaded.\n")

# ─────────────────────────────────────────────────────────────
#  LOGGING HELPERS
# ─────────────────────────────────────────────────────────────
def log_action(action: str):
    """Append a timestamped action line to the gesture log."""
    ts  = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {action}\n"
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    # No print here — terminal is used for Q&A


def init_log():
    """Write a session-start header to the log."""
    session_start = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== SESSION START: {session_start} ===\n")

# ─────────────────────────────────────────────────────────────
#  RAG — INDEX BUILDER  (runs in a background thread)
# ─────────────────────────────────────────────────────────────
def build_faiss_index(lines: list[str]):
    """Embed all log lines and build a fresh FAISS flat-L2 index."""
    if not lines:
        return None, []

    embeddings = embedder.encode(lines, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(embeddings)
    return index, lines


def index_updater_thread():
    """
    Background thread: every INDEX_REFRESH_SECS seconds, re-read the log
    file and rebuild the FAISS index so new gestures are immediately
    searchable.
    """
    global faiss_index, faiss_chunks
    while not shutdown_event.is_set():
        time.sleep(INDEX_REFRESH_SECS)
        try:
            with log_lock:
                if not os.path.exists(LOG_FILE):
                    continue
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]

            if not lines:
                continue

            new_index, new_chunks = build_faiss_index(lines)
            with index_lock:
                faiss_index  = new_index
                faiss_chunks = new_chunks
        except Exception as e:
            pass   # silently skip — index will refresh next cycle

# ─────────────────────────────────────────────────────────────
#  RAG — RETRIEVAL + GENERATION
# ─────────────────────────────────────────────────────────────
def retrieve_chunks(query: str, k: int = TOP_K) -> list[str]:
    """Embed the query and retrieve top-k most relevant log chunks."""
    with index_lock:
        if faiss_index is None or faiss_index.ntotal == 0:
            return []
        q_vec = embedder.encode([query], show_progress_bar=False).astype("float32")
        k_actual = min(k, faiss_index.ntotal)
        distances, indices = faiss_index.search(q_vec, k_actual)
        return [faiss_chunks[i] for i in indices[0] if i < len(faiss_chunks)]


def answer_with_rag(question: str) -> str:
    """
    RAG pipeline:
      1. Retrieve relevant log chunks via FAISS
      2. If Claude API key is set  → send to Claude for a natural answer
         If no API key            → return the retrieved log lines directly
    """
    chunks = retrieve_chunks(question)
    if not chunks:
        return "No session log yet. Perform some gestures first!"

    # ── NO API KEY: show retrieved log lines directly ──────────
    if anthropic_client is None:
        result  = "Most relevant log entries for your question:\n"
        result += "\n".join(f"  → {c}" for c in chunks)
        return result

    # ── WITH API KEY: send to Claude for a natural answer ──────
    context = "\n".join(chunks)

    prompt = f"""You are an assistant that answers questions about a gesture-control session log.
The user controlled their computer using hand gestures and facial expressions captured by a webcam.

Here are the most relevant log entries for the question:
───────────────────────────────────────────
{context}
───────────────────────────────────────────

Answer the following question concisely and accurately based ONLY on the log above.
If the answer cannot be determined from the log, say so clearly.

Question: {question}"""

    try:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"[RAG] API error: {e}"

# ─────────────────────────────────────────────────────────────
#  RAG — INTERACTIVE Q&A THREAD  (runs in terminal)
# ─────────────────────────────────────────────────────────────
def qa_thread():
    """
    Runs in a separate thread.
    Reads questions from stdin while the webcam loop runs in the main thread.
    """
    print("\n" + "═"*60)
    print("  RAG Q&A READY — type a question and press Enter")
    print("  (webcam continues running in the background)")
    print("  Examples:")
    print("    > How many times did I raise the volume?")
    print("    > When did YouTube open?")
    print("    > What was my most used gesture?")
    print("    > How long was my session?")
    print("═"*60 + "\n")

    while not shutdown_event.is_set():
        try:
            # Non-blocking on Windows/Linux via a short timeout trick
            question = input("❓ Ask about your session: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("[Q&A] Type Q in the webcam window to quit.")
            continue

        print("\n🔍 Searching logs...", flush=True)
        answer = answer_with_rag(question)
        print(f"\n💬 Answer: {answer}\n")
        print("─"*60)

# ─────────────────────────────────────────────────────────────
#  MEDIAPIPE / CV HELPERS  (unchanged from original)
# ─────────────────────────────────────────────────────────────
def mouth_aspect_ratio(landmarks, img_w, img_h):
    def px(lm_idx):
        lm = landmarks.landmark[lm_idx]
        return int(lm.x * img_w), int(lm.y * img_h)
    lx, ly = px(MOUTH_LEFT)
    rx, ry = px(MOUTH_RIGHT)
    tx, ty = px(MOUTH_TOP)
    bx, by = px(MOUTH_BOTTOM)
    width  = ((rx - lx)**2 + (ry - ly)**2)**0.5
    height = ((bx - tx)**2 + (by - ty)**2)**0.5
    return (width / height) if height != 0 else 0


def count_fingers(hand_landmarks, handedness_label):
    lm   = hand_landmarks.landmark
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers = [lm[t].y < lm[p].y for t, p in zip(tips, pips)]
    thumb_up = (lm[4].x < lm[3].x if handedness_label == "Right"
                else lm[4].x > lm[3].x)
    fingers = [thumb_up] + fingers
    return sum(fingers), fingers


def is_open_palm(fc, fingers):  return fc == 5
def is_closed_fist(fc, fingers): return not any(fingers[1:])
def is_v_sign(fc, fingers):
    _, index, middle, ring, pinky = fingers
    return index and middle and not ring and not pinky and not fingers[0]


def draw_progress_bar(frame, value, maximum, x, y, w, h, color):
    progress = int((min(value, maximum) / maximum) * w)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (200,200,200), 1)
    if progress > 0:
        cv2.rectangle(frame, (x, y), (x+progress, y+h), color, -1)


def put_label(frame, text, pos, color=(255,255,255), scale=0.6, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

# ─────────────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────────────
locked_encoding  = None
last_encoding    = None
smile_counter    = 0
youtube_opened   = False
vsign_counter    = 0
last_volume_time = 0
frame_count      = 0

# ─────────────────────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────────────────────
init_log()
log_action("=== Gesture session started ===")

# Start background threads
threading.Thread(target=index_updater_thread, daemon=True).start()
threading.Thread(target=qa_thread, daemon=True).start()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    sys.exit()

print("[INFO] Camera started. Gesture window opening...")

# ─────────────────────────────────────────────────────────────
#  MAIN LOOP  (runs in main thread — required for cv2.imshow)
# ─────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w  = frame.shape[:2]

    # ── FACE RECOGNITION ──────────────────────────────────────
    small          = cv2.resize(rgb, (0,0), fx=0.5, fy=0.5)
    face_locations = face_recognition.face_locations(small)
    all_encodings  = []

    if face_locations:
        if frame_count % ENCODING_INTERVAL == 0:
            all_encodings = face_recognition.face_encodings(small, face_locations)
            if all_encodings:
                last_encoding = all_encodings[0]
        else:
            if last_encoding is not None:
                all_encodings = [last_encoding] + [None]*(len(face_locations)-1)

    if locked_encoding is None and all_encodings and all_encodings[0] is not None:
        locked_encoding = all_encodings[0]
        msg = f"{ACTIVE_USER_ID} face locked"
        print(f"\n[INFO] {msg}")
        log_action(msg)

    # ── FACE MESH + SMILE ──────────────────────────────────────
    mesh_result      = face_mesh.process(rgb)
    active_mar       = None
    active_box_drawn = False

    if mesh_result.multi_face_landmarks:
        for idx, face_lm in enumerate(mesh_result.multi_face_landmarks):
            xs = [lm.x for lm in face_lm.landmark]
            ys = [lm.y for lm in face_lm.landmark]
            x1 = max(int(min(xs)*w), 0)
            y1 = max(int(min(ys)*h), 0)
            x2 = min(int(max(xs)*w), w)
            y2 = min(int(max(ys)*h), h)

            is_active = False
            if locked_encoding is not None and idx < len(all_encodings):
                enc = all_encodings[idx]
                if enc is not None:
                    is_active = face_recognition.compare_faces(
                        [locked_encoding], enc, tolerance=MATCH_TOLERANCE)[0]

            box_color  = (0,220,80)  if is_active else (0,0,220)
            label_text = (f"{ACTIVE_USER_ID}  [ACTIVE]" if is_active
                          else (f"Stranger #{idx+1}  [ignored]"
                                if locked_encoding is not None else "Identifying..."))

            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            pill_y1 = max(y1-th-10, 0)
            pill_y2 = max(y1-2, th+4)
            cv2.rectangle(frame, (x1, pill_y1), (x1+tw+8, pill_y2), box_color, -1)
            cv2.putText(frame, label_text, (x1+4, pill_y2-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)

            if is_active:
                # Lip dots
                for lid in [61,185,40,39,37,0,267,269,270,409,291,
                             375,321,405,314,17,84,181,91,146]:
                    lm = face_lm.landmark[lid]
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 1, (0,220,100), -1)

                # GESTURE 1 — SMILE
                mar        = mouth_aspect_ratio(face_lm, w, h)
                active_mar = mar

                if mar > MAR_THRESHOLD:
                    smile_counter += 1
                    draw_progress_bar(frame, smile_counter,
                                      SMILE_FRAMES_REQUIRED,
                                      20, 62, 220, 13, (0,255,120))
                    put_label(frame,
                              f"Smiling... ({smile_counter}/{SMILE_FRAMES_REQUIRED})",
                              (20,58), (0,255,120), 0.5)

                    if smile_counter >= SMILE_FRAMES_REQUIRED and not youtube_opened:
                        ts  = datetime.now().strftime("%H:%M:%S")
                        msg = f"Smile detected → YouTube opened ({YOUTUBE_URL})"
                        log_action(msg)
                        webbrowser.open(YOUTUBE_URL)
                        youtube_opened = True
                else:
                    smile_counter  = 0
                    youtube_opened = False

                active_box_drawn = True

        if active_box_drawn:
            put_label(frame, f"{ACTIVE_USER_ID} — controlling", (20,40), (0,220,80))
        else:
            smile_counter = 0
            put_label(frame, f"{ACTIVE_USER_ID} not in frame", (20,40), (0,180,255))

        if active_mar is not None:
            put_label(frame, f"MAR: {active_mar:.2f}  (threshold > {MAR_THRESHOLD})",
                      (20, h-15), (255,230,0), 0.45, 1)
    else:
        smile_counter = 0
        put_label(frame, "No face detected", (20,40), (0,0,255))

    # ── HAND GESTURES ──────────────────────────────────────────
    hand_result = hands_model.process(rgb)
    now         = time.time()

    if hand_result.multi_hand_landmarks:
        hand_lm    = hand_result.multi_hand_landmarks[0]
        handedness = hand_result.multi_handedness[0].classification[0].label

        mp_drawing.draw_landmarks(
            frame, hand_lm, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,200,255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(200,80,255), thickness=2)
        )

        finger_count, fingers = count_fingers(hand_lm, handedness)

        # GESTURE 2 — OPEN PALM → Volume Up
        if is_open_palm(finger_count, fingers):
            draw_progress_bar(frame, 1,1, 20,97,220,13, (0,210,255))
            put_label(frame, "Open palm — Volume UP", (20,93), (0,210,255), 0.55)
            if now - last_volume_time > VOLUME_COOLDOWN:
                log_action("Open palm → Volume UP")
                pyautogui.press('volumeup')
                pyautogui.press('volumeup')
                last_volume_time = now
            vsign_counter = 0

        # GESTURE 3 — CLOSED FIST → Volume Down
        elif is_closed_fist(finger_count, fingers):
            draw_progress_bar(frame, 1,1, 20,127,220,13, (255,140,0))
            put_label(frame, "Closed fist — Volume DOWN", (20,123), (255,140,0), 0.55)
            if now - last_volume_time > VOLUME_COOLDOWN:
                log_action("Closed fist → Volume DOWN")
                pyautogui.press('volumedown')
                pyautogui.press('volumedown')
                last_volume_time = now
            vsign_counter = 0

        # GESTURE 4 — V-SIGN → Exit
        elif is_v_sign(finger_count, fingers):
            vsign_counter += 1
            draw_progress_bar(frame, vsign_counter,
                              VSIGN_FRAMES_REQUIRED,
                              20,157,220,13, (80,80,255))
            put_label(frame,
                      f"V-sign — Hold to EXIT ({vsign_counter}/{VSIGN_FRAMES_REQUIRED})",
                      (20,153), (120,120,255), 0.5)
            if vsign_counter >= VSIGN_FRAMES_REQUIRED:
                log_action("V-sign confirmed → Session ended by user")
                pyautogui.hotkey('ctrl', 'w')
                time.sleep(0.3)
                break
        else:
            vsign_counter = 0

        put_label(frame, f"Fingers: {finger_count}  Hand: {handedness}",
                  (20, h-35), (200,200,255), 0.45, 1)
    else:
        vsign_counter = 0
        put_label(frame, "No hand detected", (w-200,40), (120,120,120), 0.5, 1)

    # ── HUD HEADER ─────────────────────────────────────────────
    # Show RAG index size in corner so user knows it's working
    with index_lock:
        idx_size = faiss_index.ntotal if faiss_index else 0

    cv2.rectangle(frame, (0,0), (w,22), (0,0,0), -1)
    put_label(frame,
              "SMILE=Play  |  Palm=Vol+  |  Fist=Vol-  |  V=Exit  |  R=Reset  Q=Quit",
              (6,16), (180,180,180), 0.38, 1)

    # RAG status badge (bottom right)
    rag_txt = f"RAG index: {idx_size} entries"
    (rw, rh), _ = cv2.getTextSize(rag_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(frame, (w-rw-12, h-22), (w, h), (20,20,20), -1)
    put_label(frame, rag_txt, (w-rw-6, h-7), (100,255,180), 0.38, 1)

    cv2.imshow("Gesture Control  [RAG-powered]", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        log_action("Session ended by user (Q key)")
        break
    elif key == ord('r'):
        locked_encoding = None
        last_encoding   = None
        smile_counter   = 0
        vsign_counter   = 0
        youtube_opened  = False
        frame_count     = 0
        log_action(f"Face lock reset — next face will become {ACTIVE_USER_ID}")
        print(f"\n[INFO] Face lock reset.")

# ─────────────────────────────────────────────────────────────
#  CLEANUP
# ─────────────────────────────────────────────────────────────
log_action("=== Session ended ===")
shutdown_event.set()

cap.release()
cv2.destroyAllWindows()
hands_model.close()
face_mesh.close()

print("\n[INFO] Camera released.")
print(f"[INFO] Session log saved to: {os.path.abspath(LOG_FILE)}")
print("\n[Q&A] You can still query your session log!")
print("      Run this command to ask questions about your session:")
print(f"      python ask_session.py\n")
