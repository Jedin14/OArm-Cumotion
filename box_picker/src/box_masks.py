"""Find boxes: SAM 3.1 outlines every box from a text prompt ("box"), no points needed.

SAM 3.1 ships as a video tracker; for one frame
only its detector is needed. That is SAM 3's image model with the 3.1 weights and backbone, so it runs
through the official Sam3Processor.

Of SAM's outlines: the ones covering most of the image (carton, background) and the
ones holding two or more others (a whole stack) are dropped; of two that mostly
overlap (the same box twice, or a sticker on a box) the larger is kept, unless SAM is
unsure of it and much surer of the smaller: then it is a box behind, drawn over the
boxes in front of it.
Each box's point is the pixel deepest inside its outline.
"""
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

# Meta's repo is gated (needs a Hugging Face account and approved access); the public
# copy holds the same file under the same licence, and is used only if byte-identical.
SAM_REPOS = ("facebook/sam3.1", "AEmotionStudio/sam3.1")
SAM_CKPT = "sam3.1_multiplex.pt"
SAM_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"   # Meta's file
DEFAULT_PROMPT = "box"
SCORE_THRESHOLD = 0.5      # SAM 3's own default
MAX_MASK_FRACTION = 0.25   # an outline covering more of the image is the carton or background
INSIDE = 0.8               # an outline this much inside another is part of it
SCORE_MARGIN = 0.15        # of two overlapping outlines, the larger wins unless it scores this much lower
WEAK_COVER = 0.65          # ... and SAM is this unsure of it (real boxes score 0.75-0.96), so a box keeps its sticker's place


def sam31_checkpoint():
    """Path to the SAM 3.1 weights: the file in $SAM31_CHECKPOINT if set (a machine without
    internet), else from Meta's repo if this machine has access, else from the public copy.
    The cache stores a download under its SHA-256, so the name is checked."""
    if os.environ.get("SAM31_CHECKPOINT"):
        return os.environ["SAM31_CHECKPOINT"]
    for repo in SAM_REPOS:
        try:
            path = hf_hub_download(repo, SAM_CKPT)
        except (HfHubHTTPError, LocalEntryNotFoundError):
            continue
        if os.path.basename(os.path.realpath(path)) != SAM_SHA256:
            raise RuntimeError(f"{repo}/{SAM_CKPT} is not Meta's SAM 3.1 file (SHA-256 differs)")
        return path
    raise RuntimeError(f"could not download {SAM_CKPT} from {' or '.join(SAM_REPOS)}")


def build_detector(checkpoint_path=None, device="cuda"):
    """The detector half of SAM 3.1, as build_sam3_multiplex_video_predictor builds it,
    without the tracker. FlashAttention 3 is off: it needs a Hopper GPU."""
    import importlib.resources
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.vl_combiner import SAM3VLBackboneTri
    from sam3.model_builder import (_create_dot_product_scoring, _create_geometry_encoder,
                                    _create_multiplex_tri_backbone, _create_sam3_transformer,
                                    _create_segmentation_head, _create_text_encoder)

    bpe = str(importlib.resources.files("sam3") / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    backbone = SAM3VLBackboneTri(scalp=0, visual=_create_multiplex_tri_backbone(), text=_create_text_encoder(bpe))
    model = Sam3MultiplexDetector(
        num_feature_levels=1, backbone=backbone, transformer=_create_sam3_transformer(),
        segmentation_head=_create_segmentation_head(), semantic_segmentation_head=None,
        input_geometry_encoder=_create_geometry_encoder(), use_early_fusion=True,
        use_dot_prod_scoring=True, dot_prod_scoring=_create_dot_product_scoring(),
        supervise_joint_box_scores=True, is_multiplex=True,
    )
    if checkpoint_path != "random":   # "random": no weights, for testing the code without access
        ckpt = torch.load(checkpoint_path or sam31_checkpoint(), map_location="cpu", weights_only=True)
        ckpt = ckpt.get("model", ckpt)
        ckpt = {k[len("detector."):]: v for k, v in ckpt.items() if k.startswith("detector.")}
        if not ckpt:
            raise RuntimeError(f"no detector weights in {checkpoint_path or SAM_CKPT}")
        missing, _ = model.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"warning: SAM 3.1 checkpoint lacks {len(missing)} detector weights, e.g. {missing[:5]}")
    return model.to(device).eval()


def deepest_point(mask):
    """The pixel farthest from the outline's edge, as (x, y)."""
    dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return float(x), float(y)


def component_at(mask, p):
    """The connected part of `mask` containing point p (or its largest part if p is outside it)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    h, w = mask.shape
    lab = labels[min(int(p[1]), h - 1), min(int(p[0]), w - 1)]
    if lab == 0:
        lab = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == lab


def select_masks(masks, scores):
    """masks: bool tensor (N, H, W) on the GPU -> indices of the outlines that are single boxes."""
    flat = masks.flatten(1).float()
    area = flat.sum(1)
    inter = flat @ flat.T
    inside = inter / area.clamp(min=1)[:, None]          # inside[i, j]: share of i that lies in j
    inside.fill_diagonal_(0)
    ok = (area > 0) & (area < MAX_MASK_FRACTION * masks[0].numel())
    ok &= (inside.T > INSIDE).sum(1) < 2                  # not a group: holds fewer than 2 others
    # Not a weak cover: an outline SAM is much less sure of than one that lies mostly inside it
    # is usually a box behind drawn over the boxes in front, so it goes, not them.
    weaker = scores[None, :] - scores[:, None] > SCORE_MARGIN   # weaker[i, j]: j much surer than i
    ok &= ~((inside.T > 0.5) & weaker & ok[None, :]).any(1) | (scores >= WEAK_COVER)
    keep = []
    for i in sorted(torch.nonzero(ok).flatten().tolist(), key=lambda i: -area[i].item()):
        if all(inside[i, j] <= 0.5 and inside[j, i] <= 0.5 for j in keep):
            keep.append(i)
    return sorted(keep, key=lambda i: -scores[i].item())


def remove_overlaps(masks, points):
    """Pixels claimed by more than one outline belong to none (their depth can't be trusted);
    each outline is then cut back to its part around its own point."""
    if not masks:
        return masks
    count = np.sum(masks, axis=0)
    return [component_at(m & (count == 1), p) for m, p in zip(masks, points)]


class BoxFinder:
    def __init__(self, prompt=DEFAULT_PROMPT, score_threshold=SCORE_THRESHOLD, checkpoint_path=None):
        from sam3.model.sam3_image_processor import Sam3Processor
        t0 = time.time()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.processor = Sam3Processor(build_detector(checkpoint_path), confidence_threshold=score_threshold)
        self.prompt = prompt
        print(f"Loaded SAM 3.1 detector in {time.time() - t0:.1f}s "
              f"({torch.cuda.memory_allocated() / 1e9:.1f} GB VRAM)", flush=True)

    @torch.inference_mode()
    def __call__(self, color_bgr):
        """BGR image -> (points [(x, y)], masks [bool (H, W)], timings {step: seconds}, scores [float])."""
        image = Image.fromarray(np.ascontiguousarray(color_bgr[:, :, ::-1]))
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(self.prompt, state)
        masks, scores = state["masks"][:, 0], state["scores"].float()
        keep = select_masks(masks, scores) if len(masks) else []
        masks = [m for m in masks[keep].cpu().numpy()]
        scores = [round(float(s), 3) for s in scores[keep].cpu()]
        points = [deepest_point(m) for m in masks]
        masks = remove_overlaps(masks, points)
        torch.cuda.synchronize()
        return points, masks, {"sam_s": round(time.time() - t0, 3)}, scores
