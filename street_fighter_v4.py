#!/usr/bin/env python3
"""
Street Fighter — Gesture Edition (v4)
Two players, one camera. Mana-based combat system.
"""
from __future__ import annotations
import os, sys, math, time, threading, random
import cv2, mediapipe as mp, numpy as np, pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

os.environ.setdefault('SDL_VIDEO_CENTERED', '1')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "gesture_recognizer-2.task")

# ── Display constants (populated after pygame.init) ────────────────────────────
SCREEN_W = SCREEN_H = PANEL_W = 0
HUD_TOP  = 90        # top SF-style HUD strip height
MANA_H   = 30        # mana bar height (bottom overlay)
FPS      = 60

# ── Game constants ─────────────────────────────────────────────────────────────
HP_MAX         = 100
MANA_MAX       = 100
MANA_REGEN     = 10.0    # per second, only when idle (gesture == "none")
ROUND_TIME     = 90
GOJO_MAX_USES  = 3
CONFIRM_TIME   = 0.4     # seconds to hold gesture before move triggers
FIRE_COOLDOWN  = 1.5
SHIELD_COST    = 20      # mana upfront to activate shield
SHIELD_REDUCE  = 0.4     # 60 % damage reduction
MAX_ROUNDS     = 3

# move → {damage, mana}
ATTACK_MOVES = {
    "gun":      {"damage": 15, "mana": 20},
    "gojo":     {"damage": 40, "mana": 80},
    "heart":    {"damage":  5, "mana":  8},
}

COUNTDOWN_TOTAL = 5.5    # 5 s numbers + 0.5 s FIGHT!

# ── Colours ────────────────────────────────────────────────────────────────────
C_BG           = ( 12,   8,  22)
C_P1           = ( 60, 150, 255)
C_P2           = (255,  55,  55)
C_HP_HI        = ( 50, 220,  80)
C_HP_MID       = (230, 200,  40)
C_HP_LOW       = (220,  45,  45)
C_HP_BG        = ( 35,   8,   8)
C_MANA_FILL    = ( 50, 120, 255)
C_MANA_BG      = ( 10,  20,  55)
C_SHIELD       = ( 80, 210, 255)
C_GOLD         = (255, 215,   0)
C_ORANGE       = (255, 140,   0)
C_WHITE        = (255, 255, 255)
C_GRAY         = (130, 130, 130)
C_DARK         = ( 20,  15,  30)
C_BLACK        = (  0,   0,   0)
C_HUD_BG       = (  6,   4,  14)
C_VOID_PURPLE  = (130,   0, 200)
C_VOID_GLOW    = (160,  60, 255)

ATTACK_COLORS = {
    "gun":      (255, 230,  50),
    "gojo":     (160,  60, 255),
    "heart":    (255,  60, 160),
}
GESTURE_LABELS = {
    "gun": "GUN", "gojo": "GOJO", "heart": "♥",
}


# ── Font helper ────────────────────────────────────────────────────────────────
def _make_font(size: int, bold: bool = True) -> pygame.font.Font:
    for name in ["Impact", "Arial Black", "Arial"]:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            try:
                return pygame.font.Font(path, max(6, size))
            except Exception:
                pass
    return pygame.font.SysFont("Arial", max(6, size), bold=bold)


# ═══════════════════════════════════════════════════════════════════════════════
class GestureReader:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self._lock    = threading.Lock()
        self._gesture = "none"
        self._score   = 0.0
        self._ts      = 0
        opts = mp_vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            result_callback=self._on_result, num_hands=1)
        self._rec = mp_vision.GestureRecognizer.create_from_options(opts)

    def _on_result(self, result, _img, _ts):
        g, s = "none", 0.0
        if result.gestures:
            top = result.gestures[0][0]
            g, s = top.category_name.lower(), top.score
        if g in ("fireball", "heal"):
            g, s = "none", 0.0
        with self._lock:
            self._gesture, self._score = g, s

    def feed(self, bgr: np.ndarray):
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        now  = int(time.time() * 1000)
        if now <= self._ts: now = self._ts + 1
        self._ts = now
        self._rec.recognize_async(img, now)

    @property
    def gesture(self):
        with self._lock: return self._gesture
    @property
    def score(self):
        with self._lock: return self._score
    def close(self): self._rec.close()


# ═══════════════════════════════════════════════════════════════════════════════
class Player:
    def __init__(self, player_id: int):
        self.player_id   = player_id
        self.hp          = HP_MAX
        self.max_hp      = HP_MAX
        self.mana        = MANA_MAX
        self.max_mana    = MANA_MAX
        self.rounds_won  = 0
        self.gojo_uses   = 0          # persists across rounds

        self.gesture          = "none"
        self.confidence       = 0.0
        self.prev_gesture     = "none"
        self.gesture_hold_t   = 0.0   # seconds current gesture held

        self.fire_cd          = 0.0
        self.shield_active    = False
        self.shield_paid      = False  # mana already deducted this hold

        self.hit_flash        = 0.0
        self.hit_color        = C_HP_LOW
        self.blocked_flash    = 0.0
        self.last_fired       = None
        self.last_fired_t     = 0.0

    def reset_for_round(self):
        self.hp = HP_MAX; self.mana = MANA_MAX
        self.fire_cd = 0.0; self.hit_flash = 0.0; self.blocked_flash = 0.0
        self.last_fired = None; self.gesture_hold_t = 0.0
        self.prev_gesture = "none"
        self.shield_active = False; self.shield_paid = False

    def update(self, gesture: str, confidence: float, dt: float):
        self.gesture    = gesture
        self.confidence = confidence

        # Hold timer
        if gesture == self.prev_gesture:
            self.gesture_hold_t += dt
        else:
            self.gesture_hold_t = 0.0
            if self.prev_gesture == "shield":
                self.shield_paid = False
        self.prev_gesture = gesture

        # Timers
        self.fire_cd       = max(0.0, self.fire_cd - dt)
        self.hit_flash     = max(0.0, self.hit_flash - dt)
        self.blocked_flash = max(0.0, self.blocked_flash - dt)
        if time.time() - self.last_fired_t > 2.5:
            self.last_fired = None

        # Mana regen only when completely idle
        if gesture == "none":
            self.mana = min(self.max_mana, self.mana + MANA_REGEN * dt)

        # Shield  (15 mana upfront to activate, 50% damage reduction)
        if gesture == "shield":
            if not self.shield_paid and self.mana >= SHIELD_COST:
                self.mana       -= SHIELD_COST
                self.shield_paid = True
            self.shield_active = self.shield_paid
        else:
            self.shield_active = False
            self.shield_paid   = False

        # Attack moves (fire on confirm if enough mana)
        fired = None
        if gesture in ATTACK_MOVES and self.fire_cd <= 0.0:
            if gesture == "gojo" and self.gojo_uses >= GOJO_MAX_USES:
                pass  # exhausted
            elif self.gesture_hold_t >= CONFIRM_TIME:
                cost = ATTACK_MOVES[gesture]["mana"]
                if self.mana >= cost:
                    fired               = gesture
                    self.mana          -= cost
                    self.last_fired     = fired
                    self.last_fired_t   = time.time()
                    if fired == "gojo":
                        self.gojo_uses += 1
                    self.fire_cd        = FIRE_COOLDOWN
                    self.gesture_hold_t = 0.0
        return fired

    def take_damage(self, damage: float) -> bool:
        """Returns True when shield absorbed (50% reduction)."""
        if self.shield_active:
            self.hp            = max(0, self.hp - damage * SHIELD_REDUCE)
            self.blocked_flash = 0.4
            self.hit_flash     = 0.15
            self.hit_color     = C_SHIELD
            return True
        self.hp        = max(0, self.hp - damage)
        self.hit_flash = 0.45
        self.hit_color = C_HP_LOW
        return False

    @property
    def is_ko(self): return self.hp <= 0


# ═══════════════════════════════════════════════════════════════════════════════
class StreetFighterGame:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            sys.exit("[ERROR] Cannot open camera.")

        self.reader = [GestureReader(0), GestureReader(1)]

        pygame.init()
        info = pygame.display.Info()
        global SCREEN_W, SCREEN_H, PANEL_W
        SCREEN_W = int(info.current_w * 0.82)
        SCREEN_H = int(info.current_h * 0.82)
        PANEL_W  = SCREEN_W // 2

        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Street Fighter — Gesture Edition")
        self.clock  = pygame.time.Clock()

        sh = SCREEN_H
        self.font_giant = _make_font(max(80, sh // 6))
        self.font_huge  = _make_font(max(52, sh // 9))
        self.font_big   = _make_font(max(30, sh // 16))
        self.font_med   = _make_font(max(22, sh // 24))
        self.font_small = _make_font(max(16, sh // 36))
        self.font_tiny  = _make_font(max(13, sh // 50), bold=False)

        self.p1 = Player(0)
        self.p2 = Player(1)
        self._particles:     list[dict] = []
        self._atk_effects:   list[dict] = []
        self.round_num  = 1
        self.match_msg  = ""

        self._cam_frames: list[np.ndarray | None] = [None, None]
        self._cam_lock = threading.Lock()
        self._running  = True
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

        self._new_round()

    # ── camera ────────────────────────────────────────────────────────────────
    def _camera_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret: time.sleep(0.01); continue
            frame = cv2.flip(frame, 1)
            mid = frame.shape[1] // 2
            f1, f2 = frame[:, :mid].copy(), frame[:, mid:].copy()
            self.reader[0].feed(f1); self.reader[1].feed(f2)
            with self._cam_lock:
                self._cam_frames[0] = f1; self._cam_frames[1] = f2

    def _get_frames(self):
        with self._cam_lock: return self._cam_frames[0], self._cam_frames[1]

    # ── round / match ─────────────────────────────────────────────────────────
    def _new_round(self):
        self.p1.reset_for_round(); self.p2.reset_for_round()
        self._particles.clear(); self._atk_effects.clear()
        self.round_start     = time.time()
        self.phase           = "countdown"
        self.countdown_start = time.time()
        self.round_msg       = ""
        self.round_end_t     = 0.0

    def _end_round(self, msg: str, winner: int | None = None):
        self.phase       = "round_end"
        self.round_msg   = msg
        self.round_end_t = time.time() + 3.5
        if winner == 0: self.p1.rounds_won += 1
        elif winner == 1: self.p2.rounds_won += 1
        self.round_num += 1
        if self.round_num > MAX_ROUNDS:
            self.phase = "match_end"
            if self.p1.rounds_won > self.p2.rounds_won:
                self.match_msg = "PLAYER 1 WINS THE MATCH!"
            elif self.p2.rounds_won > self.p1.rounds_won:
                self.match_msg = "PLAYER 2 WINS THE MATCH!"
            else:
                self.match_msg = "IT'S A TIE!"

    # ── attack resolution ─────────────────────────────────────────────────────
    def _resolve_attack(self, attacker: Player, defender: Player, gesture: str):
        dmg     = ATTACK_MOVES[gesture]["damage"]
        blocked = defender.take_damage(dmg)
        col     = ATTACK_COLORS.get(gesture, C_HP_LOW)

        cx = PANEL_W // 2 + (PANEL_W if defender.player_id == 1 else 0)
        cy = HUD_TOP + (SCREEN_H - HUD_TOP) // 2
        for _ in range(28 if not blocked else 8):
            a = math.tau * random.random(); sp = random.uniform(2, 9)
            self._particles.append({
                "x": cx, "y": cy,
                "vx": math.cos(a)*sp, "vy": math.sin(a)*sp,
                "r": random.randint(4, 10),
                "color": C_SHIELD if blocked else col,
                "life": random.uniform(0.4, 0.9),
                "born": time.time(),
            })

        dur = 3.5 if gesture == "gojo" and not blocked else 1.4
        self._atk_effects.append({
            "gesture": gesture, "attacker_id": attacker.player_id,
            "defender_id": defender.player_id, "blocked": blocked,
            "born": time.time(), "duration": dur, "color": col,
        })

        if not blocked and defender.is_ko:
            self._end_round(f"PLAYER {attacker.player_id + 1} WINS!",
                            winner=attacker.player_id)

    # ── utils ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _bgr_to_surf(bgr: np.ndarray, size: tuple) -> pygame.Surface:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return pygame.image.frombuffer(
            cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR).tobytes(), size, "RGB")

    def _outlined(self, font, text, color, oc=(0,0,0), ow=2) -> pygame.Surface:
        base = font.render(text, True, color)
        w, h = base.get_size()
        out  = pygame.Surface((w + ow*2, h + ow*2), pygame.SRCALPHA)
        shad = font.render(text, True, oc)
        for dx in range(-ow, ow+1):
            for dy in range(-ow, ow+1):
                if dx or dy: out.blit(shad, (dx+ow, dy+ow))
        out.blit(base, (ow, ow))
        return out

    # ══════════════════════════════════════════════════════════════════════════
    #  SF-style top HUD
    # ══════════════════════════════════════════════════════════════════════════
    def _draw_top_hud(self, surface, time_left: float):
        # HUD background
        hud = pygame.Surface((SCREEN_W, HUD_TOP), pygame.SRCALPHA)
        hud.fill((*C_HUD_BG, 245))
        surface.blit(hud, (0, 0))
        pygame.draw.line(surface, C_GOLD, (0, HUD_TOP-1), (SCREEN_W, HUD_TOP-1), 2)

        # Center reservation width (for timer/round)
        cw = 170
        hp_x1 = 10
        hp_w  = PANEL_W - cw // 2 - 15
        hp_x2 = SCREEN_W - hp_x1 - hp_w
        hp_y  = 36
        hp_h  = 34   # big HP bars

        # ── P1 HP bar (fills left→right, drains from right) ───────────────────
        self._draw_sf_hp(surface, self.p1, hp_x1, hp_y, hp_w, hp_h, flip=False)
        # ── P2 HP bar (fills right→left, drains from left) ────────────────────
        self._draw_sf_hp(surface, self.p2, hp_x2, hp_y, hp_w, hp_h, flip=True)

        # ── Player names (below HP bar, each in their own half) ───────────────
        p1n = self._outlined(self.font_big, "PLAYER 1", C_P1)
        p2n = self._outlined(self.font_big, "PLAYER 2", C_P2)
        name_y = hp_y + hp_h + 4
        surface.blit(p1n, (hp_x1, name_y))
        surface.blit(p2n, (hp_x2 + hp_w - p2n.get_width(), name_y))

        # ── Gojo-use boxes (square checkboxes, one per use) ───────────────────
        BOX  = 26
        GAP  = 6
        gojo_y  = name_y + p1n.get_height() + 6
        total_w = GOJO_MAX_USES * BOX + (GOJO_MAX_USES - 1) * GAP
        for player in (self.p1, self.p2):
            start_x = hp_x1 if player.player_id == 0 \
                      else hp_x2 + hp_w - total_w
            for i in range(GOJO_MAX_USES):
                bx   = start_x + i * (BOX + GAP)
                by   = gojo_y
                used = i < player.gojo_uses
                # box background
                bg_col = (55, 0, 90) if used else (20, 8, 38)
                pygame.draw.rect(surface, bg_col, (bx, by, BOX, BOX))
                # box border — bright when available, dimmed when used
                bd_col = (140, 60, 200) if used else (210, 120, 255)
                pygame.draw.rect(surface, bd_col, (bx, by, BOX, BOX), 2)
                # X cross when used
                if used:
                    m = 5
                    pygame.draw.line(surface, (230, 140, 255),
                                     (bx+m, by+m), (bx+BOX-m, by+BOX-m), 3)
                    pygame.draw.line(surface, (230, 140, 255),
                                     (bx+BOX-m, by+m), (bx+m, by+BOX-m), 3)

        # ── Round + Timer (center) ─────────────────────────────────────────────
        rnd   = self._outlined(self.font_small, f"ROUND  {self.round_num}", C_GOLD)
        surface.blit(rnd, (SCREEN_W//2 - rnd.get_width()//2, 4))

        ts    = max(0, int(time_left))
        tcol  = C_GOLD if ts > 20 else C_HP_LOW
        timer = self._outlined(self.font_huge, str(ts), tcol)
        surface.blit(timer, (SCREEN_W//2 - timer.get_width()//2, 24))

        # Score (below timer, bigger, dark outline for contrast)
        sc = self._outlined(self.font_med,
            f"{self.p1.rounds_won}  ●  {self.p2.rounds_won}",
            C_WHITE, oc=(20, 10, 40), ow=2)
        surface.blit(sc, (SCREEN_W//2 - sc.get_width()//2,
                          timer.get_height() + 28))

    def _draw_sf_hp(self, surface, player, x, y, w, h, flip):
        ratio  = max(0.0, player.hp / player.max_hp)
        filled = int(w * ratio)
        col    = C_HP_HI if ratio > 0.55 else C_HP_MID if ratio > 0.28 else C_HP_LOW

        # Background (dark maroon)
        pygame.draw.rect(surface, C_HP_BG, (x, y, w, h))
        # Fill
        if filled > 0:
            fx = x if not flip else x + w - filled
            pygame.draw.rect(surface, col, (fx, y, filled, h))
            # Highlight top strip
            pygame.draw.rect(surface, tuple(min(255, c+60) for c in col), (fx, y, filled, 4))
        # Black border
        pygame.draw.rect(surface, C_BLACK, (x, y, w, h), 2)
        # HP number inside bar
        hp_txt = self.font_small.render(str(max(0, int(player.hp))), True, C_WHITE)
        tx = (x + 6) if not flip else (x + w - hp_txt.get_width() - 6)
        surface.blit(hp_txt, (tx, y + h//2 - hp_txt.get_height()//2))

    # ── Mana bar (bottom overlay per panel) ───────────────────────────────────
    def _draw_mana_bar(self, surface, player: Player, base_x: int):
        x      = base_x
        y      = SCREEN_H - MANA_H
        ratio  = player.mana / player.max_mana
        filled = int(PANEL_W * ratio)

        pygame.draw.rect(surface, C_MANA_BG, (x, y, PANEL_W, MANA_H))
        if filled > 0:
            pygame.draw.rect(surface, C_MANA_FILL, (x, y, filled, MANA_H))
            # shimmer highlight
            pygame.draw.rect(surface, (100, 170, 255), (x, y, filled, 5))
        pygame.draw.rect(surface, C_BLACK, (x, y, PANEL_W, MANA_H), 2)

        # Label: "MANA  XX" centered
        mana_str = f"MANA  {int(player.mana)}/100"
        # If holding an attack, show cost
        if player.gesture in ATTACK_MOVES:
            cost = ATTACK_MOVES[player.gesture]["mana"]
            can  = player.mana >= cost
            mana_str += f"   [-{cost}]"
            txt_col = C_WHITE if can else (255, 80, 80)
        elif player.gesture == "shield":
            mana_str += f"   [-{SHIELD_COST}]"
            txt_col = C_WHITE
        else:
            txt_col = C_WHITE

        mt = self.font_tiny.render(mana_str, True, txt_col)
        surface.blit(mt, (x + PANEL_W//2 - mt.get_width()//2,
                          y + MANA_H//2 - mt.get_height()//2))

    # ── Camera panel ──────────────────────────────────────────────────────────
    def _draw_panel(self, surface, player: Player, cam_frame, flip: bool):
        base_x = PANEL_W if flip else 0
        cam_y  = HUD_TOP
        cam_h  = SCREEN_H - HUD_TOP

        # Camera feed
        if cam_frame is not None:
            surf = self._bgr_to_surf(cam_frame, (PANEL_W, cam_h))
            surf = pygame.transform.flip(surf, True, False)
            surface.blit(surf, (base_x, cam_y))
        else:
            pygame.draw.rect(surface, C_DARK, (base_x, cam_y, PANEL_W, cam_h))
            nc = self.font_med.render("No camera", True, C_GRAY)
            surface.blit(nc, (base_x + PANEL_W//2 - nc.get_width()//2,
                               cam_y + cam_h//2))

        # Hit flash
        if player.hit_flash > 0:
            a = int(player.hit_flash / 0.45 * 130)
            s = pygame.Surface((PANEL_W, cam_h), pygame.SRCALPHA)
            s.fill((*player.hit_color, a))
            surface.blit(s, (base_x, cam_y))

        # Blocked flash
        if player.blocked_flash > 0:
            a = int(player.blocked_flash / 0.4 * 90)
            s = pygame.Surface((PANEL_W, cam_h), pygame.SRCALPHA)
            s.fill((*C_SHIELD, a))
            surface.blit(s, (base_x, cam_y))

        # Shield ring
        if player.shield_active:
            t  = time.time()
            pr = int(cam_h * 0.20) + int(10 * math.sin(t * 6))
            ss = pygame.Surface((PANEL_W, cam_h), pygame.SRCALPHA)
            pygame.draw.circle(ss, (*C_SHIELD, 100), (PANEL_W//2, cam_h//2), pr, 5)
            pygame.draw.circle(ss, (*C_SHIELD, 45),  (PANEL_W//2, cam_h//2), pr+14, 3)
            surface.blit(ss, (base_x, cam_y))

        # Confirm-charge glow (brief pulse while holding a move before it fires)
        if player.gesture in ATTACK_MOVES:
            prog = min(1.0, player.gesture_hold_t / CONFIRM_TIME)
            if 0 < prog < 1.0:
                col  = ATTACK_COLORS.get(player.gesture, C_WHITE)
                t    = time.time()
                a    = int(15 + 35 * prog * abs(math.sin(t * 8)))
                sc   = pygame.Surface((PANEL_W, cam_h), pygame.SRCALPHA)
                bw   = int(6 + 10 * prog)
                pygame.draw.rect(sc, (*col, a), (0, 0, PANEL_W, cam_h), bw)
                surface.blit(sc, (base_x, cam_y))

        # Mana bar
        self._draw_mana_bar(surface, player, base_x)

    # ── Countdown ─────────────────────────────────────────────────────────────
    def _draw_countdown(self, surface, elapsed: float):
        # Semi-dark overlay over camera
        ov = pygame.Surface((SCREEN_W, SCREEN_H - HUD_TOP), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 110))
        surface.blit(ov, (0, HUD_TOP))

        cx = SCREEN_W // 2
        cy = HUD_TOP + (SCREEN_H - HUD_TOP) // 2

        if elapsed < 5.0:
            slot  = int(elapsed)            # 0→4 → show 5→1
            num   = 5 - slot
            pt    = elapsed - slot          # 0→1 within this slot
            label = str(num)
            col   = C_GOLD

            # Scale: zooms in quickly
            if pt < 0.25:
                scale = 2.2 - 1.2 * (pt / 0.25)
            elif pt < 0.75:
                scale = 1.0
            else:
                scale = 1.0
            alpha = 255 if pt < 0.75 else int((1.0 - (pt-0.75)/0.25) * 255)

            # Screen flash at start of each number
            if pt < 0.12:
                fa = int((0.12 - pt) / 0.12 * 160)
                fs = pygame.Surface((SCREEN_W, SCREEN_H - HUD_TOP), pygame.SRCALPHA)
                fs.fill((*col, fa))
                surface.blit(fs, (0, HUD_TOP))

        else:
            label = "FIGHT!"
            col   = (255, 50, 50)
            pt    = elapsed - 5.0
            scale = 1.8 - 0.8 * min(1.0, pt / 0.2)
            alpha = 255 if pt < 0.25 else int((1.0 - (pt-0.25)/0.25) * 255)

            if pt < 0.2:
                fa = int((0.2 - pt) / 0.2 * 220)
                fs = pygame.Surface((SCREEN_W, SCREEN_H - HUD_TOP), pygame.SRCALPHA)
                fs.fill((*col, fa))
                surface.blit(fs, (0, HUD_TOP))

        txt    = self._outlined(self.font_giant, label, col, ow=3)
        tw, th = txt.get_size()
        nw, nh = max(1, int(tw * scale)), max(1, int(th * scale))
        scaled = pygame.transform.smoothscale(txt, (nw, nh))
        scaled.set_alpha(alpha)
        surface.blit(scaled, (cx - nw//2, cy - nh//2))

        # "ROUND X" in background
        rn = self._outlined(self.font_big, f"ROUND {self.round_num}", C_GRAY, ow=2)
        rn.set_alpha(160)
        surface.blit(rn, (cx - rn.get_width()//2, cy - int((SCREEN_H-HUD_TOP)*0.25)))

    # ── Particles ─────────────────────────────────────────────────────────────
    def _draw_particles(self, surface):
        now, alive = time.time(), []
        for p in self._particles:
            age   = now - p["born"]
            ratio = age / p["life"]
            if ratio >= 1.0: continue
            a  = int((1.0 - ratio) * 220)
            r  = max(1, int(p["r"] * (1.0 - ratio * 0.5)))
            px = int(p["x"] + p["vx"] * age * 30)
            py = int(p["y"] + p["vy"] * age * 30)
            s  = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p["color"], a), (r, r), r)
            surface.blit(s, (px-r, py-r))
            alive.append(p)
        self._particles = alive

    # ── Domain Expansion ──────────────────────────────────────────────────────
    def _draw_domain_expansion(self, surface, ratio, att_x):
        cx = att_x + PANEL_W // 2
        cy = HUD_TOP + (SCREEN_H - HUD_TOP) // 2

        def a_ramp(r, r0, r1, a0=255, a1=0):
            if r < r0 or r > r1: return 0
            t = (r - r0) / (r1 - r0)
            return int(a0 + (a1 - a0) * t)

        # Void overlay
        va = a_ramp(ratio, 0, 0.15, 0, 225) if ratio < 0.15 else \
             a_ramp(ratio, 0.75, 0.9, 225, 0) if ratio > 0.75 else 225
        if va:
            vs = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            vs.fill((8, 0, 22, va)); surface.blit(vs, (0, 0))

        # Grid lines
        if 0.1 < ratio < 0.88:
            ga = a_ramp(ratio, 0.1, 0.25, 0, 85) if ratio < 0.25 else \
                 a_ramp(ratio, 0.75, 0.88, 85, 0) if ratio > 0.75 else 85
            g = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            for i in range(-9, 10):
                ey = cy + i * (SCREEN_H // 14)
                pygame.draw.line(g, (*C_VOID_PURPLE, ga), (0, ey), (cx, cy), 1)
                pygame.draw.line(g, (*C_VOID_PURPLE, ga), (SCREEN_W, ey), (cx, cy), 1)
            for j in range(1, 10):
                pygame.draw.circle(g, (*C_VOID_PURPLE, ga//2),
                                   (cx, cy), j * (SCREEN_H // 9), 1)
            surface.blit(g, (0, 0))

        # Orbiting void orbs
        if 0.18 < ratio < 0.85:
            oa = a_ramp(ratio, 0.18, 0.3, 0, 180) if ratio < 0.3 else \
                 a_ramp(ratio, 0.72, 0.85, 180, 0) if ratio > 0.72 else 180
            t = time.time()
            for i in range(16):
                ang  = math.tau * i / 16 + t * 0.45
                rd   = (SCREEN_H * 0.20) + (SCREEN_H * 0.08) * math.sin(t*0.7 + i*1.3)
                ox   = int(cx + rd * math.cos(ang))
                oy   = int(cy + rd * math.sin(ang) * 0.55)
                nr   = 10 + int(4 * math.sin(t*2 + i))
                os_  = pygame.Surface((nr*4, nr*4), pygame.SRCALPHA)
                pygame.draw.circle(os_, (190, 90, 255, oa), (nr*2, nr*2), nr)
                pygame.draw.circle(os_, (255, 220, 255, oa//2), (nr*2, nr*2), nr//2)
                surface.blit(os_, (ox - nr*2, oy - nr*2))

        # "DOMAIN EXPANSION" and "INFINITE VOID" — same font, stacked, clamped
        def _clamp_x(surf, panel_x):
            return max(panel_x, min(panel_x + PANEL_W - surf.get_width(),
                                    cx - surf.get_width() // 2))

        if 0.12 < ratio < 0.82:
            da  = a_ramp(ratio, 0.12, 0.22, 0, 255) if ratio < 0.22 else \
                  a_ramp(ratio, 0.68, 0.82, 255, 0) if ratio > 0.68 else 255
            de  = self._outlined(self.font_big, "DOMAIN EXPANSION",
                                 (210, 185, 255), oc=(60, 0, 120), ow=2)
            de.set_alpha(da)
            de_y = cy - de.get_height() - 10
            surface.blit(de, (_clamp_x(de, att_x), de_y))

        if 0.25 < ratio < 0.88:
            ia  = a_ramp(ratio, 0.25, 0.38, 0, 255) if ratio < 0.38 else \
                  a_ramp(ratio, 0.74, 0.88, 255, 0) if ratio > 0.74 else 255
            iv      = self._outlined(self.font_big, "INFINITE VOID",
                                     C_VOID_GLOW, oc=(40, 0, 100), ow=2)
            iv_glow = self._outlined(self.font_big, "INFINITE VOID",
                                     C_VOID_GLOW, oc=(40, 0, 100), ow=5)
            iv_y = cy + 10
            iv_glow.set_alpha(ia // 2)
            surface.blit(iv_glow, (_clamp_x(iv_glow, att_x), iv_y))
            iv.set_alpha(ia)
            surface.blit(iv, (_clamp_x(iv, att_x), iv_y))

    # ── All attack effects ─────────────────────────────────────────────────────
    def _draw_atk_effects(self, surface):
        now, alive = time.time(), []
        for eff in self._atk_effects:
            age = now - eff["born"]
            if age >= eff["duration"]: continue
            alive.append(eff)

            ratio   = age / eff["duration"]
            col     = eff["color"]
            gesture = eff["gesture"]
            blocked = eff["blocked"]
            ax      = PANEL_W if eff["attacker_id"] == 1 else 0
            dx      = PANEL_W if eff["defender_id"]  == 1 else 0
            acx     = ax + PANEL_W // 2
            acy     = HUD_TOP + (SCREEN_H - HUD_TOP) // 2
            dcx     = dx + PANEL_W // 2
            dcy     = acy

            if gesture == "gojo" and not blocked:
                self._draw_domain_expansion(surface, ratio, ax)
                continue

            # Color burst
            if ratio < 0.25:
                ba = int((0.25-ratio)/0.25 * 200)
                bs = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
                bs.fill((*col, ba)); surface.blit(bs, (0, 0))

            # Glitch strips
            if ratio < 0.35:
                sa = int((0.35-ratio)/0.35 * 150)
                for _ in range(6):
                    sy = int(random.uniform(HUD_TOP, SCREEN_H))
                    sh = random.randint(4, 20)
                    ss = pygame.Surface((PANEL_W, sh), pygame.SRCALPHA)
                    ss.fill((*col, sa)); surface.blit(ss, (ax, sy))

            # Giant move text
            if ratio < 0.70:
                lbl  = "BLOCKED!" if blocked else GESTURE_LABELS.get(gesture, gesture.upper())
                dc   = C_SHIELD if blocked else col
                sc_  = (1.8 - 0.8*(ratio/0.4)) if ratio < 0.4 else 1.0
                ta   = 255 if ratio < 0.4 else (int((1.0-ratio)/0.3*255) if (1.0-ratio)<0.3 else 255)
                bt   = self._outlined(self.font_giant, lbl, dc, ow=3)
                tw, th = bt.get_size()
                nw, nh = max(1, int(tw*sc_)), max(1, int(th*sc_))
                st   = pygame.transform.smoothscale(bt, (nw, nh))
                gl   = pygame.transform.smoothscale(bt, (nw+18, nh+18))
                gl.set_alpha(ta//2)
                surface.blit(gl, (acx - (nw+18)//2, acy - (nh+18)//2))
                st.set_alpha(ta)
                surface.blit(st, (acx - nw//2, acy - nh//2))

            # Expanding rings on defender
            rc = C_SHIELD if blocked else col
            for ri in range(4):
                rp = ratio + ri * 0.12
                if rp >= 1.0: continue
                rr = int(rp * max(SCREEN_W, SCREEN_H) * 0.65)
                rs = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
                pygame.draw.circle(rs, (*rc, int((1-rp)*200)),
                                   (dcx, dcy), rr, max(2, int((1-rp)*8)))
                surface.blit(rs, (0, 0))

            # Lightning bolts on attacker
            if ratio < 0.3 and not blocked:
                ba = int((0.3-ratio)/0.3 * 220)
                bs = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
                for _ in range(5):
                    x0, y0 = acx, random.randint(HUD_TOP, SCREEN_H)
                    pts = [(x0, y0)]
                    for _ in range(6):
                        x0 = max(ax, min(ax+PANEL_W, x0 + random.randint(-70,70)))
                        y0 = max(HUD_TOP, min(SCREEN_H, y0 + random.randint(-70,70)))
                        pts.append((x0, y0))
                    if len(pts) > 1:
                        pygame.draw.lines(bs, (*col, ba), False, pts, 2)
                surface.blit(bs, (0, 0))

        self._atk_effects = alive

    # ── Match-end overlay ─────────────────────────────────────────────────────
    def _draw_match_end(self, surface):
        ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 200)); surface.blit(ov, (0, 0))
        msg = self._outlined(self.font_huge, self.match_msg, C_GOLD, ow=3)
        sc  = self.font_med.render(
            f"Final Score  P1: {self.p1.rounds_won}   P2: {self.p2.rounds_won}", True, C_GRAY)
        sub = self._outlined(self.font_big, "Press R to play again", C_WHITE)
        surface.blit(msg, (SCREEN_W//2 - msg.get_width()//2, SCREEN_H//2 - 100))
        surface.blit(sc,  (SCREEN_W//2 - sc.get_width()//2,  SCREEN_H//2))
        surface.blit(sub, (SCREEN_W//2 - sub.get_width()//2, SCREEN_H//2 + 70))

    # ── Round-end overlay ─────────────────────────────────────────────────────
    def _draw_round_end(self, surface):
        ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 180)); surface.blit(ov, (0, 0))
        msg = self._outlined(self.font_huge, self.round_msg, C_GOLD, ow=3)
        sub = self._outlined(self.font_big,  "New round in 3 s…  (R to restart)", C_WHITE)
        sc  = self.font_med.render(
            f"Score  P1: {self.p1.rounds_won}   P2: {self.p2.rounds_won}", True, C_GRAY)
        surface.blit(msg, (SCREEN_W//2 - msg.get_width()//2, SCREEN_H//2 - 90))
        surface.blit(sub, (SCREEN_W//2 - sub.get_width()//2, SCREEN_H//2 + 10))
        surface.blit(sc,  (SCREEN_W//2 - sc.get_width()//2,  SCREEN_H//2 + 70))

    # ══════════════════════════════════════════════════════════════════════════
    #  Main loop
    # ══════════════════════════════════════════════════════════════════════════
    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT: self._quit()
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE): self._quit()
                    if event.key == pygame.K_r:
                        if self.phase == "match_end":
                            self.p1.rounds_won = 0; self.p2.rounds_won = 0
                            self.p1.gojo_uses  = 0; self.p2.gojo_uses  = 0
                            self.round_num = 1
                        self._new_round()

            g1, g2 = self.reader[0].gesture, self.reader[1].gesture
            s1, s2 = self.reader[0].score,   self.reader[1].score

            if self.phase == "countdown":
                elapsed = time.time() - self.countdown_start
                if elapsed >= COUNTDOWN_TOTAL:
                    self.phase       = "playing"
                    self.round_start = time.time()
                time_left = ROUND_TIME
            elif self.phase == "playing":
                f1 = self.p1.update(g1, s1, dt)
                f2 = self.p2.update(g2, s2, dt)
                if f1: self._resolve_attack(self.p1, self.p2, f1)
                if f2: self._resolve_attack(self.p2, self.p1, f2)
                time_left = ROUND_TIME - (time.time() - self.round_start)
                if time_left <= 0:
                    if   self.p1.hp > self.p2.hp: self._end_round("PLAYER 1 WINS! (Time)", 0)
                    elif self.p2.hp > self.p1.hp: self._end_round("PLAYER 2 WINS! (Time)", 1)
                    else: self._end_round("DRAW!")
            elif self.phase == "round_end":
                time_left = 0
                if time.time() >= self.round_end_t: self._new_round()
            else:
                time_left = 0

            # ── Render ────────────────────────────────────────────────────────
            cam1, cam2 = self._get_frames()
            self.screen.fill(C_BG)
            self._draw_panel(self.screen, self.p1, cam1, flip=False)
            self._draw_panel(self.screen, self.p2, cam2, flip=True)
            self._draw_top_hud(self.screen, time_left)
            self._draw_particles(self.screen)
            self._draw_atk_effects(self.screen)

            if self.phase == "countdown":
                self._draw_countdown(self.screen, time.time() - self.countdown_start)
            elif self.phase == "round_end":
                self._draw_round_end(self.screen)
            elif self.phase == "match_end":
                self._draw_match_end(self.screen)

            pygame.display.flip()

    def _quit(self):
        self._running = False
        self.cap.release()
        for r in self.reader: r.close()
        pygame.quit(); sys.exit()


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[ERROR] Model not found: {MODEL_PATH}\n"
                 "Place gesture_recognizer-2.task next to this script.")
    StreetFighterGame().run()
