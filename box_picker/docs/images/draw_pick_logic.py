"""Draws docs/images/pick_logic_flow.png, the flowchart in picking_rules.md and robot_pick_logic.md.

Run: python docs/images/draw_pick_logic.py   (needs Pillow and the DejaVu fonts)
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
W, H = 1700, 1250
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
F = lambda s, b=False: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if b else ""), s)
INK, GREY, BLUE, RED, GREEN, ORANGE = (30, 30, 30), (120, 120, 120), (40, 90, 200), (200, 40, 40), (30, 140, 60), (210, 120, 0)

def box(x, y, w, h, text, fill=(245, 245, 245), edge=INK, bold=False, size=20, diamond=False):
    if diamond:
        cx, cy = x + w / 2, y + h / 2
        d.polygon([(cx, y), (x + w, cy), (cx, y + h), (x, cy)], fill=fill, outline=edge, width=2)
    else:
        d.rounded_rectangle([x, y, x + w, y + h], 12, fill=fill, outline=edge, width=2)
    lines = text.split("\n")
    f = F(size, bold)
    th = sum(f.getbbox(l)[3] + 6 for l in lines)
    ty = y + (h - th) / 2
    for l in lines:
        tw = f.getlength(l)
        d.text((x + (w - tw) / 2, ty), l, fill=INK, font=f)
        ty += f.getbbox(l)[3] + 6

def arrow(x1, y1, x2, y2, label=None, col=INK, lx=8, ly=-28):
    d.line([(x1, y1), (x2, y2)], fill=col, width=3)
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    p = [(x2, y2), (x2 - 16 * math.cos(a - 0.4), y2 - 16 * math.sin(a - 0.4)), (x2 - 16 * math.cos(a + 0.4), y2 - 16 * math.sin(a + 0.4))]
    d.polygon(p, fill=col)
    if label:
        d.text(((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly), label, fill=col, font=F(19, True))

d.text((40, 25), "How the next box to pick is chosen", fill=INK, font=F(34, True))

# ---------- Part 1
d.text((40, 85), "STEP 1   Check every box SAM found", fill=BLUE, font=F(26, True))
cx = 300
box(cx - 150, 135, 300, 60, "A box SAM found", fill=(225, 235, 255), bold=True)
arrow(cx, 195, cx, 235)
box(cx - 210, 235, 420, 110, "Does it show a front\nor a top face?", fill=(255, 250, 225), diamond=True)
arrow(cx + 210, 290, 760, 290, "no", RED)
box(760, 255, 520, 70, "SKIP: not a whole box\n(only a side or a sliver)", fill=(255, 230, 230))
arrow(cx, 345, cx, 385, "yes", GREEN, ly=-8)
box(cx - 210, 385, 420, 110, "Is another box\nresting ON TOP of it?", fill=(255, 250, 225), diamond=True)
arrow(cx + 210, 440, 760, 440, "yes", RED)
box(760, 405, 520, 70, "WAIT: the box on top\nmust be picked first", fill=(255, 230, 230))
arrow(cx, 495, cx, 535, "no", GREEN, ly=-8)
box(cx - 210, 535, 420, 110, "Is another box\nIN FRONT of it?", fill=(255, 250, 225), diamond=True)
arrow(cx + 210, 590, 760, 590, "yes", RED)
box(760, 555, 520, 70, "WAIT: the box in front\nmust be picked first", fill=(255, 230, 230))
arrow(cx, 645, cx, 685, "no", GREEN, ly=-8)
box(cx - 150, 685, 300, 60, "FREE box", fill=(220, 245, 225), bold=True, size=22)

# ---------- Part 2
d.text((40, 790), "STEP 2   Choose ONE of the free boxes", fill=BLUE, font=F(26, True))
y = 845
box(40, y, 250, 70, "All free boxes", fill=(220, 245, 225), bold=True)
arrow(290, y + 35, 340, y + 35)
box(340, y, 330, 70, "A. ON A BOX first,\nthe HIGHEST top", fill=(245, 245, 245), size=20)
arrow(670, y + 35, 720, y + 35)
box(720, y, 300, 70, "B. Keep the NEAREST\n(within 3 cm)", fill=(245, 245, 245))
arrow(1020, y + 35, 1070, y + 35)
box(1070, y, 270, 70, "C. Take the LEFTMOST", fill=(245, 245, 245))
arrow(1340, y + 35, 1390, y + 35)
box(1390, y - 5, 270, 80, "TARGET\n(sent to robot)", fill=(255, 215, 215), edge=RED, bold=True, size=22)
d.text((340, y + 92), "tops within 1.5 cm count as equal   nearest = distance to the point the robot grips", fill=GREY, font=F(20))

# ---------- Example
d.text((40, 1060), "Example, frame 5280:", fill=BLUE, font=F(22, True))
rows = [("small box behind (0.53 m away)", "WAIT: box in front", RED),
        ("box in front of it (0.49 m away)", "FREE", GREEN)]
for k, (a, b, c) in enumerate(rows):
    d.text((70, 1095 + k * 32), "- " + a, fill=INK, font=F(20))
    d.text((520, 1095 + k * 32), "-> " + b, fill=c, font=F(20, True))
d.text((820, 1095), "-> TARGET: the box in front", fill=RED, font=F(20, True))
d.text((820, 1127), "its top is seen beside the small box, ending there", fill=GREY, font=F(20))
d.text((70, 1170), "If that top can't be seen, the box above counts as resting on it: the box underneath waits (nothing falls).", fill=GREY, font=F(19))
img.save(Path(__file__).with_name("pick_logic_flow.png"))
print("saved")
