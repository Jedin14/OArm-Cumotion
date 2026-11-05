"""The pick loop's window: a status bar with the four steps (find, pick, check, next), the camera
picture (colour or depth) with only outlines drawn over the boxes, and a side panel with two tabs:
PICK (the target close up, its position, the tries, the check's result, what comes next, the last
picks) and ADVANCED (buttons for the views and the less used commands); then the tally and the keys.
Text is drawn with PIL, so it is sharp and never runs off the edge. Tabs, buttons and key chips
can be clicked: render() leaves their places in `hits`."""
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIEW_H = 660             # camera picture height on screen, px
PANEL_W = 440
BAR_H = 64
PAD = 24
BG, PANEL, MUTED, WHITE = (18, 18, 18), (32, 32, 32), (150, 150, 150), (240, 240, 240)   # RGB
GREEN, RED, ORANGE, BLUE, GREY = (40, 160, 70), (210, 50, 60), (230, 140, 20), (40, 105, 190), (85, 85, 85)
STEPS = ["FIND", "PICK", "CHECK", "NEXT"]
TABS = [("pick", "PICK"), ("advanced", "ADVANCED")]
KEYS = {"pick": [("C", "done"), ("T", "next target"), ("S", "snapshot"), ("A", "advanced"), ("Q", "quit")],
        "advanced": [("A", "back"), ("Q", "quit")]}
TAB_H = 46
DEPTH_RANGE = (0.30, 0.65)   # metres shown from warm (near) to cool (far) in the depth picture
FONTS = "/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf"
STYLE = {                # name: (size, bold, colour)
    "label": (16, True, MUTED),
    "title": (32, True, WHITE),
    "value": (26, False, WHITE),
    "text": (20, False, WHITE),
    "small": (16, False, MUTED),
}
THUMB = 58               # size of a past pick's thumbnail, px
_fonts = {}
hits = []                # (x1, y1, x2, y2, key) of what can be clicked in the last render()


def font(size, bold=False):
    if (size, bold) not in _fonts:
        try:
            _fonts[size, bold] = ImageFont.truetype(FONTS.format("-Bold" if bold else ""), size)
        except OSError:                              # no DejaVu on this machine
            _fonts[size, bold] = ImageFont.load_default(size)
    return _fonts[size, bold]


def wrap(draw, text, f, width):
    """Split text into lines no wider than width."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=f) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line] if line else lines


def busy(text):
    """text with dots that count up, for a step that takes a few seconds."""
    return text + "." * (int(time.time() * 3) % 4)


def to_pil(bgr, w=None, h=None):
    img = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
    if w or h:
        s = min((w or 1e9) / img.width, (h or 1e9) / img.height)
        img = img.resize((max(round(img.width * s), 1), max(round(img.height * s), 1)), Image.BILINEAR)
    return img


def render(view, status, colour, panel, tally="", step=None, fps=None, history=(), tab="pick", depth=False):
    """view: the camera picture (BGR); status: the bar's text; colour: the bar's colour (RGB);
    panel: rows of (style, text[, colour]), ("image", bgr), ("dots", used, total, colour),
    ("button", key, text, value[, on]), ("setting", name, text, value, selected) (clicking its
    - / + gives the key "-:name" / "+:name"), or None for a gap; step: which of STEPS is lit; fps:
    camera frames a second (None: not shown); history: the last picks as (thumbnail BGR, colour
    RGB), oldest first; tab: which of TABS is open; depth: the view is a depth picture (adds its
    scale). -> the window (BGR)."""
    hits.clear()
    h, w = view.shape[:2]
    vw = round(w * VIEW_H / h)
    img = Image.new("RGB", (vw + PANEL_W, BAR_H + VIEW_H), BG)
    img.paste(to_pil(cv2.resize(view, (vw, VIEW_H), interpolation=cv2.INTER_LINEAR)), (0, BAR_H))
    d = ImageDraw.Draw(img)

    # Bar: the status on the left, the steps on the right.
    d.rectangle([0, 0, img.width, BAR_H], fill=colour)
    x = img.width - PAD
    f = font(16, True)
    for k in reversed(range(len(STEPS))):
        x -= d.textlength(STEPS[k], font=f)
        lit = k == step
        if lit:
            d.rounded_rectangle([x - 10, BAR_H // 2 - 15, x + d.textlength(STEPS[k], font=f) + 10, BAR_H // 2 + 15],
                                8, fill=WHITE)
        d.text((x, BAR_H // 2), STEPS[k], font=f, anchor="lm",
               fill=colour if lit else tuple(int(c * 0.45 + 255 * 0.55) for c in colour))
        x -= 40 if k else 0
        if k:
            d.text((x + 20, BAR_H // 2), "›", font=f, anchor="mm", fill=tuple(int(c * 0.5 + 255 * 0.5) for c in colour))
    room = x - 2 * PAD
    f = font(28, True)
    while d.textlength(status, font=f) > room and f.size > 16:
        f = font(f.size - 2, True)
    d.text((PAD, BAR_H // 2), status, font=f, fill=WHITE, anchor="lm")

    # Live pill in the picture's corner: is the camera still sending?
    if fps is not None:
        ok = fps >= 1
        text = f"LIVE  {fps:.0f} fps" if ok else "CAMERA SLOW" if fps > 0 else "NO FRAMES"
        f = font(14, True)
        tw = d.textlength(text, font=f)
        x1, y1 = vw - 12, BAR_H + 12
        d.rounded_rectangle([x1 - tw - 36, y1, x1, y1 + 26], 13, fill=(0, 0, 0))
        d.ellipse([x1 - tw - 26, y1 + 8, x1 - tw - 16, y1 + 18], fill=GREEN if ok else RED)
        d.text((x1 - 10, y1 + 13), text, font=f, fill=WHITE, anchor="rm")

    # Depth scale in the picture's corner: near (warm) to far (cool), black: no depth.
    if depth:
        x1, y1, sw = 12, img.height - 40, 180
        d.rounded_rectangle([x1, y1, x1 + sw + 150, y1 + 28], 14, fill=(0, 0, 0))
        ramp = cv2.applyColorMap(np.linspace(255, 0, sw).astype(np.uint8)[None, :], cv2.COLORMAP_TURBO)
        img.paste(to_pil(np.repeat(ramp, 10, axis=0)), (x1 + 50, y1 + 9))
        f = font(13, True)
        d.text((x1 + 44, y1 + 14), f"{DEPTH_RANGE[0] * 100:.0f} cm", font=f, fill=WHITE, anchor="rm")
        d.text((x1 + 56 + sw, y1 + 14), f"{DEPTH_RANGE[1] * 100:.0f} cm   black: none", font=f, fill=WHITE, anchor="lm")

    # Panel.
    x0, width = vw + PAD, PANEL_W - 2 * PAD
    d.rectangle([vw, BAR_H, img.width, img.height], fill=PANEL)
    x = x0
    f = font(15, True)
    for name, title in TABS:                         # tabs: the open one white and underlined
        tw = d.textlength(title, font=f)
        d.text((x, BAR_H + TAB_H // 2 + 2), title, font=f, anchor="lm", fill=WHITE if name == tab else MUTED)
        if name == tab:
            d.rectangle([x, BAR_H + TAB_H - 3, x + tw, BAR_H + TAB_H], fill=WHITE)
        else:
            hits.append((x - 8, BAR_H, x + tw + 8, BAR_H + TAB_H, "a"))
        x += tw + 32
    d.line([vw, BAR_H + TAB_H, img.width, BAR_H + TAB_H], fill=GREY, width=1)
    y = BAR_H + TAB_H + PAD - 4
    for row in panel:
        if row is None:
            y += 18
        elif row[0] == "button":                     # ("button", key, text, value[, on]): a clickable row
            _, key, text, value, *on = row
            d.rounded_rectangle([x0, y, x0 + width, y + 32], 7, fill=(44, 44, 44))
            right = x0 + width - 12
            if on:                                   # a toggle: a switch, knob right when on
                sx = right - 40
                d.rounded_rectangle([sx, y + 6, right, y + 26], 10, fill=BLUE if on[0] else GREY)
                kx = right - 10 if on[0] else sx + 10
                d.ellipse([kx - 8, y + 8, kx + 8, y + 24], fill=WHITE)
                right = sx - 10
                d.text((x0 + 12, y + 16), text, font=font(17), fill=WHITE, anchor="lm")
            else:
                d.rounded_rectangle([x0 + 5, y + 5, x0 + 27, y + 27], 5, fill=GREY)
                d.text((x0 + 16, y + 16), key, font=font(15, True), fill=WHITE, anchor="mm")
                d.text((x0 + 38, y + 16), text, font=font(17), fill=WHITE, anchor="lm")
            d.text((right, y + 16), value, font=font(15, True), anchor="rm",
                   fill=WHITE if on and on[0] else MUTED)
            hits.append((x0, y, x0 + width, y + 32, key.lower()))
            y += 38
        elif row[0] == "setting":                    # ("setting", name, text, value, selected): - value +
            _, name, text, value, sel = row
            d.rounded_rectangle([x0, y, x0 + width, y + 29], 6, fill=(52, 52, 52) if sel else (40, 40, 40),
                                outline=BLUE if sel else None, width=2)
            d.text((x0 + 10, y + 15), text, font=font(15), fill=WHITE, anchor="lm")
            right = x0 + width - 4
            for sign, bx in (("+", right - 22), ("-", right - 22 - 84 - 22)):
                d.rounded_rectangle([bx, y + 4, bx + 22, y + 25], 5, fill=GREY)
                d.text((bx + 11, y + 14), "\u2212" if sign == "-" else "+", font=font(16, True), fill=WHITE, anchor="mm")
                hits.append((bx - 2, y, bx + 24, y + 29, f"{sign}:{name}"))
            d.text((right - 22 - 42, y + 15), value, font=font(15, True), anchor="mm",
                   fill=ORANGE if value.endswith("*") else WHITE)
            y += 33
        elif row[0] == "coords":                     # ("coords", [(name, value), ...]): side by side
            col_w = width // len(row[1])
            for k, (name, value) in enumerate(row[1]):
                d.text((x0 + k * col_w, y), name, font=font(14, True), fill=MUTED)
                d.text((x0 + k * col_w, y + 20), value, font=font(24), fill=WHITE)
            y += 58
        elif row[0] == "image":
            thumb = to_pil(row[1], width, 130)
            img.paste(thumb, (x0, y))
            d.rectangle([x0, y, x0 + thumb.width - 1, y + thumb.height - 1], outline=GREY)
            y += thumb.height + 12
        elif row[0] == "dots":
            _, used, total, col = row
            for k in range(total):
                cx = x0 + 10 + k * 28
                box = [cx - 9, y + 2, cx + 9, y + 20]
                d.ellipse(box, fill=col if k < used else None, outline=col if k < used else GREY, width=2)
            d.text((x0 + total * 28 + 6, y + 11), f"try {used} of {total}", font=font(18), fill=WHITE, anchor="lm")
            y += 32
        else:
            style, text, *col = row
            size, bold, fill = STYLE[style]
            f = font(size, bold)
            for line in wrap(d, text, f, width):
                d.text((x0, y), line, font=f, fill=col[0] if col else fill)
                y += round(size * 1.3)
            y += 4

    # Bottom of the panel: the last picks, the tally, the keys.
    f = font(16)
    rows, x = 1, x0                                  # how many rows the key chips need
    for key, what in KEYS[tab]:
        w = 34 + d.textlength(what, font=f)
        if x + w > x0 + width:
            rows, x = rows + 1, x0
        x += w + 26
    panel_end, y = y, img.height - PAD - rows * 34
    if tally:
        d.text((x0, y - 34), tally, font=font(18, True), fill=WHITE)
    if history and y - 34 - THUMB - 34 > panel_end:         # only when the panel leaves room
        d.text((x0, y - 34 - THUMB - 34), "LAST PICKS", font=font(16, True), fill=MUTED)
        for k, (thumb, col) in enumerate(list(history)[-((width + 8) // (THUMB + 8)):]):
            tx, ty = x0 + k * (THUMB + 8), y - 34 - THUMB - 10
            t = to_pil(thumb, THUMB - 6, THUMB - 6)
            d.rounded_rectangle([tx, ty, tx + THUMB, ty + THUMB], 6, fill=col)
            img.paste(t, (tx + (THUMB - t.width) // 2, ty + (THUMB - t.height) // 2))
    d.line([vw + PAD, y - 8, img.width - PAD, y - 8], fill=GREY, width=1)
    x = x0
    for key, what in KEYS[tab]:
        if x + 34 + d.textlength(what, font=f) > x0 + width:
            x, y = x0, y + 34
        hits.append((x, y, x + 34 + d.textlength(what, font=f), y + 26, key.lower()))
        d.rounded_rectangle([x, y, x + 26, y + 26], 5, fill=GREY)
        d.text((x + 13, y + 13), key, font=font(16, True), fill=WHITE, anchor="mm")
        d.text((x + 34, y + 13), what, font=f, fill=MUTED, anchor="lm")
        x += 34 + d.textlength(what, font=f) + 26
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def clicked(x, y):
    """The key of what is at window pixel (x, y) in the last render(), or ''."""
    return next((k for x1, y1, x2, y2, k in hits if x1 <= x <= x2 and y1 <= y <= y2), "")


def depth_view(depth_m):
    """Depth (metres) as a picture: DEPTH_RANGE from warm (near) to cool (far), black where none."""
    near, far = DEPTH_RANGE
    t = np.clip((far - depth_m) / (far - near), 0, 1)
    out = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[depth_m <= 0] = 0
    return out


def contours(mask):
    return cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


TARGET_RED = (0, 0, 255)   # BGR: the target's colour in the picture, as in the picker's view
SCALE = 2                # the overlay is drawn at this many times the camera's size, so it stays smooth on screen
_decor = {}


def smooth_contours(mask, s=SCALE):
    """The outline of mask at s times its size, with the pixel steps rounded off."""
    m = cv2.GaussianBlur(cv2.resize(mask.astype(np.float32), None, fx=s, fy=s), (0, 0), 1.5 * s)
    return contours(m > 0.5)


def decor(before, shape):
    """The overlay for one target, drawn once: (premultiplied colour, alpha) at SCALE times shape."""
    if _decor.get("key") is before:
        return _decor["value"]
    h, w = shape[0] * SCALE, shape[1] * SCALE
    col, alpha = np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32)

    def layer(m, colour, a=1.0):                     # m: coverage 0..1 (HxW float)
        nonlocal col, alpha
        m = m * a
        col = col * (1 - m[..., None]) + np.float32(colour) * m[..., None]
        alpha = alpha * (1 - m) + m

    def stroke(cnts, width, blur=0):
        canvas = np.zeros((h, w), np.uint8)
        cv2.drawContours(canvas, cnts, -1, 255, width, cv2.LINE_AA)
        canvas = canvas.astype(np.float32) / 255
        return cv2.GaussianBlur(canvas, (0, 0), blur) if blur else canvas

    # The target only, as the picker draws it: its pick face's rectangle and the grip point, in red.
    face = getattr(before, "face_px", None)
    if face is not None:
        rect = [np.round(np.asarray(face) * SCALE).astype(np.int32)]
    else:                                             # no rectangle given: the box's outline
        rect = smooth_contours(before.target)
    layer(stroke(rect, 3 * SCALE), TARGET_RED)
    u, v = round(before.result["u"] * SCALE), round(before.result["v"] * SCALE)
    canvas = np.zeros((h, w), np.uint8)
    cv2.drawMarker(canvas, (u, v), 255, cv2.MARKER_CROSS, 18 * SCALE, 2 * SCALE, cv2.LINE_AA)
    layer(canvas.astype(np.float32) / 255, TARGET_RED)

    _decor.update(key=before, value=(col, alpha[..., None]))
    return _decor["value"]


def target_view(img, before):
    """The live picture with the target's pick face boxed in red and a red cross on its grip point.
    Twice the camera's size."""
    col, alpha = decor(before, img.shape)
    big = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return (big * (1 - alpha) + col).clip(0, 255).astype(np.uint8)


def close_up(img, before, pad=30):
    """The target box cut out of img, with a margin, its pick face boxed in red."""
    ys, xs = np.nonzero(before.target)
    h, w = before.target.shape
    out = img.copy()
    face = getattr(before, "face_px", None)
    rect = [np.round(np.asarray(face)).astype(np.int32)] if face is not None else contours(before.target)
    cv2.drawContours(out, rect, -1, TARGET_RED, 2, cv2.LINE_AA)
    return out[max(ys.min() - pad, 0):min(ys.max() + pad, h), max(xs.min() - pad, 0):min(xs.max() + pad, w)]


def check_view(pc, before, after, verdict):
    """The check's picture without its text bar (the panel shows the numbers): the target red
    where it's gone, green where unchanged, blue where nearer, purple where another box now stands;
    disturbed neighbours in orange. Same colours as pick_check.draw."""
    try:
        depth, colour = pc.median_frames(after)
        gone, same, closer, other = pc._changes(before, depth, colour)[:4]
    except (AttributeError, TypeError, ValueError):  # the check changed: use its own drawing
        return pc.draw(before, after, verdict)
    img = after[-1][0].copy()
    tint = img.copy()
    tint[before.target & gone] = (0, 0, 255)
    tint[before.target & same] = (0, 200, 0)
    tint[before.target & closer] = (255, 0, 0)
    tint[before.target & other] = (255, 0, 200)
    img = cv2.addWeighted(tint, 0.5, img, 0.5, 0)
    cv2.drawContours(img, contours(before.target), -1, (0, 220, 255), 2, cv2.LINE_AA)
    for b in verdict["disturbed"]:
        cv2.drawContours(img, contours(before.others == b["box"] + 1), -1, (0, 140, 255), 2, cv2.LINE_AA)
    return img
