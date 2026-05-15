# Street Fighter — Gesture Edition

A two-player Street Fighter-style game controlled entirely by hand gestures, captured through a single webcam. Each half of the camera feed is assigned to one player.

## Requirements

- Python **3.9 – 3.11** (recommended — mediapipe does not fully support Python 3.12+)
- A webcam
- The model file `gesture_recognizer-2.task` placed in the same directory as `street_fighter_v4.py`

## Setup

### 1. Create a virtual environment

```bash
python3.10 -m venv venv
```

### 2. Activate it

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the game

```bash
python street_fighter_v4.py
```

## Gameplay

The camera is split down the middle: **Player 1** is on the left half, **Player 2** on the right.

| Gesture | Action | Damage | Mana cost |
|---------|--------|--------|-----------|
| Gun (👆) | Shoot | 15 HP | 20 |
| Heart (🤟) | Heart attack | 5 HP | 8 |
| Gojo (open hand) | Domain Expansion | 40 HP | 80 |
| Shield | Block | — | 20 (reduces incoming damage by 60%) |

- **Hold** a gesture for 0.4 s to trigger it — prevents accidental fires.
- **Gojo** can only be used **3 times** per match across all rounds.
- Mana regenerates automatically when your hand is idle.
- Each match is **best of 3 rounds**, 90 seconds per round.

### Controls

| Key | Action |
|-----|--------|
| `R` | Restart round / match |
| `Q` or `Esc` | Quit |

## Project structure

```
street_fighter_v4.py        # main game
gesture_recognizer-2.task   # MediaPipe gesture model (required)
HandTracker.py              # standalone hand tracking utility
ModelMaker.ipynb            # notebook used to train the gesture model
requirements.txt            # Python dependencies
dataset/                    # training images for the gesture model
assets/                     # additional assets
```
