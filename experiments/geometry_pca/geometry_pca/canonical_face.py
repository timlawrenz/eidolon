"""
Canonical 3D 68-point face template (300W / iBUG landmark layout).

These are biologically-proportioned mean-face coordinates in a right-handed
frame: +X = subject's left (image right), +Y = up, +Z = toward camera (forward).
Units are arbitrary but internally consistent; the orthographic-PnP solver in
pose_normalize.py only needs relative proportions, and we re-center + scale
before use.

This replaces the hand-built synthetic radial "dome" depth prior used in the
Phase 1-R spike. Real orbital regression (eyes set back) and nose-bridge
protrusion are encoded here, which removes the warping the synthetic prior
introduced (see ledger Phase 1-R caveats).

Layout (iBUG 68):
  0-16  jaw line (left ear -> chin -> right ear)
  17-21 right eyebrow      22-26 left eyebrow
  27-30 nose bridge        31-35 lower nose
  36-41 right eye          42-47 left eye
  48-59 outer mouth        60-67 inner mouth

Source: the canonical mean-face 3D model widely used for solvePnP head-pose
estimation (derived from the 3DDFA / 300W-LP mean face), normalized here.
"""
import numpy as np

# (68, 3) canonical landmarks. Z encodes real facial depth: nose tip most
# forward, eye sockets and jaw sides set back.
_CANONICAL_68 = np.array([
    # jaw (0-16)
    [-0.7126, 0.1419, -0.5410], [-0.7088, -0.0387, -0.4916], [-0.6814, -0.2197, -0.4253],
    [-0.6398, -0.3942, -0.3501], [-0.5556, -0.5500, -0.2580], [-0.4419, -0.6857, -0.1530],
    [-0.3058, -0.7951, -0.0497], [-0.1592, -0.8607,  0.0457], [ 0.0000, -0.8810,  0.0810],
    [ 0.1592, -0.8607,  0.0457], [ 0.3058, -0.7951, -0.0497], [ 0.4419, -0.6857, -0.1530],
    [ 0.5556, -0.5500, -0.2580], [ 0.6398, -0.3942, -0.3501], [ 0.6814, -0.2197, -0.4253],
    [ 0.7088, -0.0387, -0.4916], [ 0.7126,  0.1419, -0.5410],
    # right eyebrow (17-21)
    [-0.5524, 0.4636, -0.0780], [-0.4533, 0.5430,  0.0480], [-0.3084, 0.5654,  0.1330],
    [-0.1606, 0.5453,  0.1850], [-0.0204, 0.4986,  0.2130],
    # left eyebrow (22-26)
    [ 0.0204, 0.4986, 0.2130], [ 0.1606, 0.5453, 0.1850], [ 0.3084, 0.5654, 0.1330],
    [ 0.4533, 0.5430, 0.0480], [ 0.5524, 0.4636, -0.0780],
    # nose bridge (27-30)
    [ 0.0000, 0.3819, 0.2640], [ 0.0000, 0.2643, 0.3530], [ 0.0000, 0.1468, 0.4420],
    [ 0.0000, 0.0292, 0.5310],
    # lower nose (31-35)
    [-0.1392, -0.0681, 0.2350], [-0.0710, -0.0921, 0.2880], [ 0.0000, -0.1100, 0.3140],
    [ 0.0710, -0.0921, 0.2880], [ 0.1392, -0.0681, 0.2350],
    # right eye (36-41)
    [-0.4307, 0.3225, 0.0480], [-0.3404, 0.3686, 0.1180], [-0.2410, 0.3658, 0.1230],
    [-0.1560, 0.3175, 0.0930], [-0.2430, 0.3000, 0.1180], [-0.3450, 0.3000, 0.0830],
    # left eye (42-47)
    [ 0.1560, 0.3175, 0.0930], [ 0.2410, 0.3658, 0.1230], [ 0.3404, 0.3686, 0.1180],
    [ 0.4307, 0.3225, 0.0480], [ 0.3450, 0.3000, 0.0830], [ 0.2430, 0.3000, 0.1180],
    # outer mouth (48-59)
    [-0.2592, -0.3596, 0.1080], [-0.1500, -0.3071, 0.1880], [-0.0667, -0.2768, 0.2280],
    [ 0.0000, -0.2950, 0.2380], [ 0.0667, -0.2768, 0.2280], [ 0.1500, -0.3071, 0.1880],
    [ 0.2592, -0.3596, 0.1080], [ 0.1500, -0.4280, 0.1880], [ 0.0667, -0.4610, 0.2180],
    [ 0.0000, -0.4690, 0.2280], [-0.0667, -0.4610, 0.2180], [-0.1500, -0.4280, 0.1880],
    # inner mouth (60-67)
    [-0.2000, -0.3596, 0.1280], [-0.0667, -0.3350, 0.2080], [ 0.0000, -0.3400, 0.2180],
    [ 0.0667, -0.3350, 0.2080], [ 0.2000, -0.3596, 0.1280], [ 0.0667, -0.3950, 0.2080],
    [ 0.0000, -0.4010, 0.2180], [-0.0667, -0.3950, 0.2080],
], dtype=np.float32)


def canonical_template() -> np.ndarray:
    """Return the centered canonical 3D 68-point face template, shape (68, 3)."""
    t = _CANONICAL_68.copy()
    return t - t.mean(axis=0)
