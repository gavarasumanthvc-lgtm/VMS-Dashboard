"""
Central configuration for the VMS Inspection Dashboard.

Pulling these out of Dashboard.py means you can tune thresholds, weights,
and prompts without touching app/logic code — and you can unit test
scoring.py against these constants directly.
"""

# ── HuggingFace router config ────────────────────────────────────────────────
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
# NOTE: verify this model id is what you intend — "gemma-4-31B" doesn't match
# any published Gemma release as of writing. Double check against your
# HF router dashboard before relying on it in production.
HF_MODEL = "google/gemma-4-31B-it:together"
API_TIMEOUT_SECONDS = 45
API_RETRY_ON_503_DELAY_SECONDS = 5

# ── Frame extraction ─────────────────────────────────────────────────────────
FRAME_TARGET_WIDTH = 960
FRAME_JPEG_QUALITY = 92
DEFAULT_FRAMES_PER_STEP = 2
MIN_FRAMES_PER_STEP = 2
MAX_FRAMES_PER_STEP = 5

# Candidates sampled per step when picking the "stillest" frames. Higher =
# better selection quality but more decode work; these candidates are reused
# directly as the extracted frames (no second decode pass).
CANDIDATES_PER_STEP = 6

# ── SOP step segmentation ────────────────────────────────────────────────────
# Percent-of-duration windows per step, keyed by inspection stream type.
# Overlap between adjacent steps is intentionally minimized (vs. the original
# 5-10% overlaps) to avoid extracting near-duplicate frames for two different
# steps. This still assumes a roughly linear label -> unbox -> tags -> product
# flow; if your SOP genuinely varies in order between Return and Forward
# streams, edit the "Forward" entry below to match reality — the code already
# threads `video_type` through to pick the right profile.
SOP_SEGMENTS = {
    "Return": {
        "label": (0.00, 0.22),
        "unboxing": (0.20, 0.48),
        "tags": (0.46, 0.75),
        "product": (0.73, 1.00),
    },
    "Forward": {
        "label": (0.00, 0.22),
        "unboxing": (0.20, 0.48),
        "tags": (0.46, 0.75),
        "product": (0.73, 1.00),
    },
}

STEP_ORDER = ["label", "unboxing", "tags", "product"]

# ── Scoring ───────────────────────────────────────────────────────────────────
STEP_SCORE_LIMITS = {
    "label": 30,
    "unboxing": 20,
    "tags": 25,
    "product": 25,
}

STEP_WEIGHTS = [
    ("Shipping Label & Barcode", "label_check", 30),
    ("Packaging & Unboxing", "unboxing_check", 20),
    ("Brand Tag & Price Tag", "brand_tag_check", 25),
    ("Product Display & Condition", "product_check", 25),
]

VERDICT_ACCEPT_MIN = 85
VERDICT_REVIEW_MIN = 50

# Words that indicate a positive finding when NOT negated.
POSITIVE_PHRASES = [
    "visible", "can see", "shows", "displays", "readable", "present",
    "detected", "found", "held", "opened", "unfolded", "clearly",
    "label", "barcode", "tag", "brand", "tracking", "good", "compliant",
]

# Negated phrases are stripped from the text BEFORE positive matching, so
# "not visible" doesn't also get counted as a hit for "visible". Order
# matters: longer/more specific phrases first.
NEGATIVE_PHRASES = [
    "not visible", "cannot see", "can not see", "no label", "not present",
    "unclear", "cannot read", "can not read", "not shown", "error",
    "missing", "absent", "fail", "no tag", "not found", "obscured",
    "blurred", "off camera", "off-camera",
]

# ── SOP prompts per step ──────────────────────────────────────────────────────
STEP_PROMPTS = {
    "label": """You are auditing a warehouse return inspection video for VMS Logistics SOP compliance.

TASK: Analyze the shipping label and barcode visibility.

Look carefully for:
- Outer polybag or box with a shipping/AWB label
- Ekart, Delhivery, Bluedart, or other carrier labels
- Any barcode, QR code, or tracking number
- Whether the label is held clearly toward the camera
- Whether text is readable and not blurred

READ AND REPORT:
1. Exact tracking/AWB number if visible (e.g. MYER1069106708)
2. Carrier name if visible
3. Whether label is clear and unobstructed

SCORING CRITERIA (30 points):
- Label clearly visible and held toward camera: GOOD
- Barcode/tracking number readable: GOOD
- Label blurred, obscured, or not shown: BAD
- No label visible at all: FAIL

Give your assessment in 3-4 specific sentences. Start with what you actually see.""",

    "unboxing": """You are auditing a warehouse return inspection video for VMS Logistics SOP compliance.

TASK: Analyze the packaging integrity and unboxing procedure.

Look carefully for:
- Is the outer polybag or box seal being inspected before opening?
- Is the operator cutting or unsealing the package on camera?
- Is the entire unboxing happening within the camera frame?
- Any signs of pre-cutting, tampering, or damage to the seal?
- Is the operator rotating the package to show all sides?

SCORING CRITERIA (20 points):
- Package opened fully on camera: GOOD
- Seal inspected before opening: GOOD
- Package rotated to show all sides: GOOD
- Pre-cut or tampered seal: BAD
- Unboxing happening off-camera: FAIL

Give your assessment in 3-4 specific sentences. Start with what you actually see.""",

    "tags": """You are auditing a warehouse return inspection video for VMS Logistics SOP compliance.

TASK: Analyze brand tag and price tag verification.

Look carefully for:
- Brand fabric labels (neck label, waistband label)
- Paper hangtags with price, size, or brand name
- Plastic fasteners/tags still attached
- Tags being held close to the camera lens
- Whether tag text (brand, price, size) is readable

SCORING CRITERIA (25 points):
- Brand tag clearly shown close to camera: GOOD
- Price/size tag visible and readable: GOOD
- Tag text legible: GOOD
- Tags not shown or too far from camera: BAD
- Tags missing or removed: FAIL

READ AND REPORT any brand name, price, or size you can see.

Give your assessment in 3-4 specific sentences. Start with what you actually see.""",

    "product": """You are auditing a warehouse return inspection video for VMS Logistics SOP compliance.

TASK: Analyze product display and condition assessment.

Look carefully for:
- Is the item fully unfolded and displayed?
- What type of item is it (shirt, pants, dress, etc)?
- Is the front side shown clearly?
- Is the back side also shown?
- Any visible defects, stains, tears, or damage?
- Signs of wear or incorrect item return?

SCORING CRITERIA (25 points):
- Item fully unfolded: GOOD
- Both front and back shown: GOOD
- Condition clearly visible: GOOD
- Item only partially shown: BAD
- Item kept folded or off-camera: FAIL

Give your assessment in 3-4 specific sentences describing exactly what you see.""",
}
