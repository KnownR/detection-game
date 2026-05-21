# pose_collector_3class.py
# Collect: fireball (1), neutral (2), shield (3)
import cv2, numpy as np, json, time
from ultralytics import YOLO
import mediapipe as mp
import pygame
from collections import deque

# ------------------- CONFIG -------------------
DATA_FILE      = 'pose_data_3class.json'
LABEL_KEYS     = {pygame.K_1: 'fireball', pygame.K_2: 'neutral', pygame.K_3: 'shield'}
COOLDOWN       = 1.2
SAVE_EVERY     = 25
MIN_CONF       = 0.5
# ---------------------------------------------

pygame.init()
screen = pygame.display.set_mode((1280, 720))
pygame.display.set_caption('3-Class Pose Collector – 1=fireball, 2=neutral, 3=shield')
font   = pygame.font.SysFont('arial', 36)
big    = pygame.font.SysFont('arial', 60)

model = YOLO('yolov8n.pt')
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pose_data      = []
last_save_time = {k: 0.0 for k in LABEL_KEYS.values()}
label_active   = None
label_buffer   = deque(maxlen=8)

print("Hold keys: 1=fireball, 2=neutral, 3=shield. ESC=quit.")

clock = pygame.time.Clock()
running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            elif e.key in LABEL_KEYS:
                label_active = LABEL_KEYS[e.key]
                print(f"Labeling: {label_active.upper()}")
        elif e.type == pygame.KEYUP:
            if e.key in LABEL_KEYS:
                label_active = None

    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (1280, 720))
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO
    dets = model(rgb, classes=0)[0].boxes.data.cpu().numpy()
    if len(dets) == 0:
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0,1))
        screen.blit(surf, (0,0))
        pygame.display.flip()
        clock.tick(30)
        continue

    person = max(dets, key=lambda x: x[4])
    if person[4] < MIN_CONF: continue
    x1,y1,x2,y2 = map(int, person[:4])
    pad = 30
    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
    x2 = min(1280, x2+pad); y2 = min(720, y2+pad)
    crop = rgb[y1:y2, x1:x2]

    # MediaPipe
    res = pose.process(crop)
    if not res.pose_landmarks: continue

    lm = res.pose_landmarks.landmark
    for l in lm:
        l.x = (l.x * (x2-x1) + x1) / 1280
        l.y = (l.y * (y2-y1) + y1) / 720

    # SAVE
    if label_active and (time.time() - last_save_time[label_active]) > COOLDOWN:
        pose_data.append({
            'move': label_active,
            'landmarks': [(l.x, l.y) for l in lm],
            'timestamp': time.time()
        })
        last_save_time[label_active] = time.time()
        label_buffer.append(label_active)

    # DRAW
    mp.solutions.drawing_utils.draw_landmarks(
        rgb, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2),
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255,0,0), thickness=2)
    )
    pygame.draw.rect(screen, (0,255,0), (x1,y1,x2-x1,y2-y1), 3)

    surf = pygame.surfarray.make_surface(rgb.swapaxes(0,1))
    screen.blit(surf, (0,0))

    # COUNTER
    counts = {k: sum(1 for p in pose_data if p['move'] == k) for k in LABEL_KEYS.values()}
    txt = font.render(f"F: {counts['fireball']} | N: {counts['neutral']} | S: {counts['shield']}  (1/2/3)", True, (255,255,255))
    screen.blit(txt, (10,10))

    # BANNER
    if label_buffer:
        current = max(set(label_buffer), key=label_buffer.count)
        color = (255,165,0) if current == 'fireball' else (100,200,255) if current == 'shield' else (200,200,200)
        banner = big.render(current.upper(), True, color)
        screen.blit(banner, (screen.get_width()//2 - banner.get_width()//2, 80))

    # AUTO-SAVE
    if len(pose_data) and len(pose_data) % SAVE_EVERY == 0:
        with open(DATA_FILE, 'w') as f:
            json.dump(pose_data, f, indent=2)
        print(f"Auto-saved {len(pose_data)} samples → {DATA_FILE}")

    pygame.display.flip()
    clock.tick(30)

# FINAL SAVE
if pose_data:
    with open(DATA_FILE, 'w') as f:
        json.dump(pose_data, f, indent=2)
    print(f"Saved {len(pose_data)} samples → {DATA_FILE}")
    for k, v in counts.items():
        print(f"   {k}: {v}")

cap.release()
pose.close()
pygame.quit()