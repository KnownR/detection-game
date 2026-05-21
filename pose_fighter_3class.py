# pose_fighter_3class_optimized.py
# Optimized: Performance improvements, better memory management, cleaner code structure
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import pygame
import torch.serialization
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

# ------------------- 3-CLASS ML CLASSIFIER -------------------
from classifier_3class import Pose3ClassClassifier
clf = Pose3ClassClassifier(model_path="pose_3class_resnet.pth")
CONF_THRESHOLD = 0.80
SMOOTH_FRAMES = 5
# -------------------------------------------------------------

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 720
    FPS = 30
    
    # Colors (as tuples for faster access)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (100, 149, 237)
    ORANGE = (255, 140, 0)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    BLACK = (0, 0, 0)
    
    # Game parameters
    MAX_HEALTH = 100
    ROUNDS_TO_WIN = 2
    ROUND_DELAY = 3.0
    FIREBALL_DAMAGE = 15
    FIREBALL_SPEED = 11
    ATTACK_COOLDOWN = 1.6
    SHIELD_DURATION = 1.8
    PARTICLES_PER_EFFECT = 20
    PARTICLE_LIFETIME = 30
    YOLO_CONF = 0.5
    IOU_THRESHOLD = 0.3
    PLAYER_STALE_TIME = 1.0
    YOLO_CONF_BUFFER_SIZE = 8
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

# ============================================================================
# PLAYER
# ============================================================================
class Player:
    __slots__ = ('side', 'bbox', 'torso_bbox', 'center', 'last_move_time', 
                 'shield_active', 'shield_end_time', 'conf_buffer', 
                 'yolo_conf_buffer', 'last_seen')
    
    def __init__(self, side: int):
        self.side = side  # 0 = left (P1), 1 = right (P2)
        self.bbox = None
        self.torso_bbox = None
        self.center = (0, 0)
        self.last_move_time = 0
        self.shield_active = False
        self.shield_end_time = 0
        self.conf_buffer = deque(maxlen=SMOOTH_FRAMES)
        self.yolo_conf_buffer = deque(maxlen=Config.YOLO_CONF_BUFFER_SIZE)
        self.last_seen = 0.0

    def update_bbox(self, bbox: Tuple[int, int, int, int], conf: float):
        """Update player bounding box and calculate torso region."""
        self.bbox = bbox
        self.yolo_conf_buffer.append(conf)
        self.last_seen = time.time()
        
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        # Calculate torso bbox (optimized with single calculation)
        torso_x1 = x1 + (w >> 2)  # Bit shift for faster division by 4
        torso_x2 = x2 - (w >> 2)
        torso_y1 = y1 + int(h * 0.2)
        torso_y2 = y1 + int(h * 0.6)
        
        self.torso_bbox = (torso_x1, torso_y1, torso_x2, torso_y2)
        self.center = ((x1 + x2) >> 1, (y1 + y2) >> 1)  # Bit shift for division

    def get_avg_conf(self) -> float:
        """Get average confidence from buffer."""
        return sum(self.conf_buffer) / len(self.conf_buffer) if self.conf_buffer else 0.0

    def add_conf(self, conf: float):
        """Add confidence value to buffer."""
        self.conf_buffer.append(conf)

    def detect_action(self, landmarks: List[Tuple[float, float]]) -> Optional[str]:
        """Detect player action from pose landmarks."""
        if not landmarks:
            return None
        
        current_time = time.time()
        if current_time - self.last_move_time < Config.ATTACK_COOLDOWN:
            return None

        pred, conf = clf.predict(landmarks)
        self.add_conf(conf)

        if conf > CONF_THRESHOLD:
            if pred == "fireball":
                self.last_move_time = current_time
                return "fireball"
            elif pred == "shield":
                self.shield_active = True
                self.shield_end_time = current_time + Config.SHIELD_DURATION
                self.last_move_time = current_time
                return "shield"
        return None

    def update_shield(self):
        """Update shield status based on time."""
        if self.shield_active and time.time() > self.shield_end_time:
            self.shield_active = False

    def is_stale(self) -> bool:
        """Check if player hasn't been seen recently."""
        return time.time() - self.last_seen > Config.PLAYER_STALE_TIME

# ============================================================================
# IOU + MATCHING (Optimized)
# ============================================================================
def iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x1, y1, x2, y2 = b1
    x1_, y1_, x2_, y2_ = b2
    
    # Calculate intersection
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    
    # Calculate union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_ - x1_) * (y2_ - y1_)
    union = area1 + area2 - inter
    
    return inter / union

def match_players(dets: np.ndarray, players: List[Player]) -> List[Player]:
    """Match detections to existing players using IOU."""
    matched = []
    screen_center = Config.SCREEN_WIDTH >> 1
    
    for det in dets:
        conf = float(det[4])
        if conf < Config.YOLO_CONF:
            continue
            
        # Convert to integers once
        x1, y1, x2, y2 = map(int, det[:4])
        bbox = (x1, y1, x2, y2)
        cx = (x1 + x2) >> 1

        # Filter valid players for matching
        valid_players = [p for p in players if p not in matched and not p.is_stale()]
        
        # Find best match using IOU
        best_player = None
        best_iou = Config.IOU_THRESHOLD
        
        for p in valid_players:
            if p.bbox:
                current_iou = iou(bbox, p.bbox)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_player = p

        if best_player:
            best_player.update_bbox(bbox, conf)
            matched.append(best_player)
        else:
            # Create new player on appropriate side
            side = 0 if cx < screen_center else 1
            new_player = Player(side)
            new_player.update_bbox(bbox, conf)
            matched.append(new_player)

    # Return only non-stale players
    all_players = players + [p for p in matched if p not in players]
    return [p for p in all_players if not p.is_stale()]

# ============================================================================
# PARTICLES & ATTACK (Optimized)
# ============================================================================
@dataclass
class Particle:
    """Particle for visual effects."""
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    size: float
    life: int
    max_life: int
    
    def update(self) -> bool:
        """Update particle position and life. Returns True if still alive."""
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.size *= 0.94
        return self.life > 0
    
    def draw(self, screen: pygame.Surface):
        """Draw particle with alpha blending."""
        if self.size < 0.5:
            return
            
        alpha = int(255 * self.life / self.max_life)
        size_int = int(self.size)
        diameter = size_int << 1
        
        surf = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*self.color, alpha), (size_int, size_int), size_int)
        screen.blit(surf, (int(self.x - self.size), int(self.y - self.size)))

class ParticleSystem:
    """Manages particle effects."""
    __slots__ = ('particles',)
    
    def __init__(self):
        self.particles = []
    
    def emit(self, x: float, y: float, color: Tuple[int, int, int], count: int = 20):
        """Emit particles from a point."""
        two_pi = 2 * math.pi
        for _ in range(count):
            angle = np.random.uniform(0, two_pi)
            speed = np.random.uniform(2, 6)
            self.particles.append(Particle(
                x, y,
                math.cos(angle) * speed,
                math.sin(angle) * speed,
                color,
                np.random.uniform(3, 6),
                Config.PARTICLE_LIFETIME,
                Config.PARTICLE_LIFETIME
            ))
    
    def update(self):
        """Update all particles and remove dead ones."""
        self.particles = [p for p in self.particles if p.update()]
    
    def draw(self, screen: pygame.Surface):
        """Draw all particles."""
        for p in self.particles:
            p.draw(screen)

class Attack:
    """Fireball attack projectile."""
    __slots__ = ('x', 'y', 'direction', 'owner', 'active', 'speed', 'size', '_rect')
    
    def __init__(self, x: float, y: float, direction: int, owner_player: Player):
        self.x = x
        self.y = y
        self.direction = direction
        self.owner = owner_player
        self.active = True
        self.speed = Config.FIREBALL_SPEED
        self.size = 22
        self._rect = None  # Cache rect

    def update(self):
        """Update attack position."""
        self.x += self.speed * self.direction
        self.active = 0 < self.x < Config.SCREEN_WIDTH
        self._rect = None  # Invalidate cached rect

    def draw(self, screen: pygame.Surface):
        """Draw fireball with glow effect."""
        pos = (int(self.x), int(self.y))
        pygame.draw.circle(screen, Config.ORANGE, pos, self.size)
        pygame.draw.circle(screen, Config.YELLOW, pos, self.size >> 1)

    def get_rect(self) -> pygame.Rect:
        """Get collision rect (cached)."""
        if self._rect is None:
            s = self.size * 0.7
            self._rect = pygame.Rect(self.x - s, self.y - s, s * 2, s * 2)
        return self._rect

class AttackSystem:
    """Manages all attacks."""
    __slots__ = ('attacks', 'particle_system')
    
    def __init__(self, particle_system: ParticleSystem):
        self.attacks = []
        self.particle_system = particle_system

    def spawn(self, player: Player):
        """Spawn new attack from player."""
        direction = 1 if player.side == 0 else -1
        attack = Attack(player.center[0], player.center[1], direction, player)
        self.attacks.append(attack)
        self.particle_system.emit(attack.x, attack.y, Config.ORANGE, 18)

    def update(self, players: List[Player]) -> Optional[Tuple[int, int]]:
        """Update all attacks and check collisions. Returns (side, damage) if hit."""
        hit_result = None
        
        for attack in self.attacks[:]:  # Iterate over copy
            attack.update()
            
            if not attack.active:
                self.attacks.remove(attack)
                continue
            
            attack_rect = attack.get_rect()
            
            # Check collision with players
            for player in players:
                if player is attack.owner or not player.torso_bbox:
                    continue
                
                torso_rect = pygame.Rect(*player.torso_bbox)
                if attack_rect.colliderect(torso_rect):
                    damage = 0 if player.shield_active else Config.FIREBALL_DAMAGE
                    hit_result = (player.side, damage)
                    color = Config.CYAN if player.shield_active else Config.RED
                    self.particle_system.emit(attack.x, attack.y, color, 30)
                    attack.active = False
                    self.attacks.remove(attack)
                    break
        
        return hit_result

    def draw(self, screen: pygame.Surface):
        """Draw all attacks."""
        for attack in self.attacks:
            attack.draw(screen)

# ============================================================================
# GAME STATE & UI (Optimized)
# ============================================================================
class GameState:
    """Manages game state and round logic."""
    __slots__ = ('p1_health', 'p2_health', 'p1_rounds', 'p2_rounds', 'round', 
                 'round_over', 'winner', 'end_time', 'game_over')
    
    def __init__(self):
        self.p1_health = Config.MAX_HEALTH
        self.p2_health = Config.MAX_HEALTH
        self.p1_rounds = 0
        self.p2_rounds = 0
        self.round = 1
        self.round_over = False
        self.winner = None
        self.end_time = None
        self.game_over = False

    def apply_damage(self, side: int, damage: int):
        """Apply damage to a player."""
        if side == 0:
            self.p1_health = max(0, self.p1_health - damage)
        else:
            self.p2_health = max(0, self.p2_health - damage)

    def check_round_end(self):
        """Check if round should end."""
        if self.round_over:
            return
        
        if self.p1_health <= 0:
            self._end_round(2)
            self.p2_rounds += 1
        elif self.p2_health <= 0:
            self._end_round(1)
            self.p1_rounds += 1
    
    def _end_round(self, winner: int):
        """Helper to end the round."""
        self.round_over = True
        self.winner = winner
        self.end_time = time.time()
        
        if self.p1_rounds >= Config.ROUNDS_TO_WIN or self.p2_rounds >= Config.ROUNDS_TO_WIN:
            self.game_over = True

def draw_health_bar(screen: pygame.Surface, x: int, y: int, w: int, h: int, 
                    health: int, color: Tuple[int, int, int]):
    """Draw a health bar."""
    # Background (red)
    pygame.draw.rect(screen, Config.RED, (x, y, w, h))
    # Current health
    health_width = int(w * health / Config.MAX_HEALTH)
    pygame.draw.rect(screen, color, (x, y, health_width, h))
    # Border
    pygame.draw.rect(screen, Config.WHITE, (x, y, w, h), 2)

def draw_shield(screen: pygame.Surface, player: Player):
    """Draw shield effect around player."""
    if not (player.shield_active and player.bbox):
        return
    
    x1, y1, x2, y2 = player.bbox
    cx = (x1 + x2) >> 1
    cy = (y1 + y2) >> 1
    radius = max(x2 - x1, y2 - y1) // 2 + 20
    
    # Animated alpha
    elapsed = time.time() - (player.shield_end_time - Config.SHIELD_DURATION)
    alpha = int(128 + 127 * math.sin(elapsed * 3))
    
    surf = pygame.Surface((radius << 1, radius << 1), pygame.SRCALPHA)
    pygame.draw.circle(surf, (*Config.CYAN, alpha), (radius, radius), radius, 5)
    screen.blit(surf, (cx - radius, cy - radius))

def draw_ui(screen: pygame.Surface, game_state: GameState, players: List[Player], 
            font: pygame.font.Font, big_font: pygame.font.Font):
    """Draw all UI elements."""
    # Health bars
    draw_health_bar(screen, 50, 50, 300, 30, game_state.p1_health, Config.BLUE)
    draw_health_bar(screen, 930, 50, 300, 30, game_state.p2_health, Config.ORANGE)
    
    # Player labels
    screen.blit(font.render('P1', True, Config.WHITE), (50, 15))
    screen.blit(font.render('P2', True, Config.WHITE), (930, 15))
    
    # Round info
    screen.blit(font.render(f'Round {game_state.round}', True, Config.WHITE), (600, 20))
    screen.blit(font.render(f'{game_state.p1_rounds} - {game_state.p2_rounds}', 
                           True, Config.YELLOW), (620, 60))
    
    # Shields
    for player in players[:2]:
        draw_shield(screen, player)
    
    # Round end overlay
    if game_state.round_over and not game_state.game_over:
        overlay = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        
        text = big_font.render(f'Player {game_state.winner} Wins Round!', True, Config.YELLOW)
        screen.blit(text, text.get_rect(center=(640, 360)))
    
    # Game over overlay
    if game_state.game_over:
        overlay = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 220))
        screen.blit(overlay, (0, 0))
        
        text = big_font.render(f'PLAYER {game_state.winner} WINS!', True, Config.YELLOW)
        screen.blit(text, text.get_rect(center=(640, 300)))
        
        text2 = font.render('SPACE restart | ESC quit', True, Config.WHITE)
        screen.blit(text2, (440, 400))

# ============================================================================
# MAIN (Optimized)
# ============================================================================
def main():
    """Main game loop."""
    # Initialize models
    model = YOLO('yolov8n.pt')
    pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity=0  # Use lighter model for better performance
    )
    draw_utils = mp.solutions.drawing_utils

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
    pygame.display.set_caption('Pose Fighter 3-Class ML (Optimized)')
    font = pygame.font.SysFont('arial', 32)
    big_font = pygame.font.SysFont('arial', 64, bold=True)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.FPS)

    # Game objects
    game_state = GameState()
    players = []
    particle_system = ParticleSystem()
    attack_system = AttackSystem(particle_system)

    clock = pygame.time.Clock()
    running = True

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and game_state.game_over:
                    # Reset game
                    game_state = GameState()
                    players = []
                    attack_system.attacks.clear()
                    particle_system.particles.clear()

        # Handle round transition
        if game_state.round_over and time.time() - game_state.end_time > Config.ROUND_DELAY:
            old_rounds = (game_state.p1_rounds, game_state.p2_rounds)
            game_state = GameState()
            game_state.p1_rounds, game_state.p2_rounds = old_rounds
            game_state.round = game_state.p1_rounds + game_state.p2_rounds + 1
            players = []

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize once
        frame = cv2.resize(frame, (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO detection
        detections = model(rgb, classes=0, verbose=False)[0].boxes.data.cpu().numpy()
        players = match_players(detections, players)
        players.sort(key=lambda p: p.center[0] if p.bbox else float('inf'))

        # Process each player
        for player in players[:2]:
            player.update_shield()
            
            if not player.bbox:
                continue
            
            # Crop around player for pose detection
            x1, y1, x2, y2 = player.bbox
            pad = 30
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(Config.SCREEN_WIDTH, x2 + pad)
            crop_y2 = min(Config.SCREEN_HEIGHT, y2 + pad)
            crop = rgb[crop_y1:crop_y2, crop_x1:crop_x2]

            # Pose detection
            results = pose.process(crop)
            if results.pose_landmarks:
                # Adjust landmarks to full frame coordinates
                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1
                
                for landmark in results.pose_landmarks.landmark:
                    landmark.x = (landmark.x * crop_w + crop_x1) / Config.SCREEN_WIDTH
                    landmark.y = (landmark.y * crop_h + crop_y1) / Config.SCREEN_HEIGHT
                
                landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]

                # Detect action
                action = player.detect_action(landmarks)
                if action == "fireball" and not game_state.round_over:
                    attack_system.spawn(player)
                elif action == "shield":
                    particle_system.emit(player.center[0], player.center[1], Config.CYAN, 25)

                # Draw pose landmarks
                draw_utils.draw_landmarks(
                    rgb, 
                    results.pose_landmarks, 
                    mp.solutions.pose.POSE_CONNECTIONS,
                    draw_utils.DrawingSpec(color=(0, 255, 0), thickness=2),
                    draw_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        # Update game logic
        if not game_state.round_over:
            hit = attack_system.update(players)
            if hit:
                game_state.apply_damage(*hit)
            game_state.check_round_end()

        particle_system.update()

        # Render
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        # Draw player bounding boxes
        for i, player in enumerate(players[:2]):
            if player.bbox:
                color = Config.BLUE if player.side == 0 else Config.ORANGE
                pygame.draw.rect(screen, color, player.bbox, 3)
                pygame.draw.rect(screen, Config.YELLOW, player.torso_bbox, 2)
                
                label = f'P{i+1} [{player.get_avg_conf():.2f}]'
                if player.shield_active:
                    label += ' [SHIELD]'
                screen.blit(font.render(label, True, color), (player.bbox[0], player.bbox[1] - 35))

        attack_system.draw(screen)
        particle_system.draw(screen)
        draw_ui(screen, game_state, players, font, big_font)

        pygame.display.flip()
        clock.tick(Config.FPS)

    # Cleanup
    cap.release()
    pose.close()
    pygame.quit()

if __name__ == "__main__":
    main()