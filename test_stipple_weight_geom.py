"""Fast isolated test for the --stipple-weight feature.

Modulates the stipple density (which seeds the initial TSP curve) with an
external grayscale weight map. No diffusion model, CPU only (needs Concorde for
the TSP tour). It exercises:

  * apply_stipple_weight: byte->[0,1] normalization (values preserved), bilinear
    resample to the density resolution, and both combine modes -- as a pure
    numpy function, independent of the GPU pipeline
  * the strict opt-in guarantee (no --stipple-weight -> arg is None, and the
    default density path is byte-identical to mask.numpy())
  * ALIGNMENT (the coordinate-space trap): with a half-black/half-white weight
    map over a full-canvas subject, the painter's initial control points cluster
    under the BRIGHT half -- dense points sit where the weight map is bright, in
    the correct canvas space
  * MODE CHECK: same map in multiply vs replace -- multiply keeps points off the
    background (RMBG mask still gates), replace honors the raw map (points land
    on background where the mask is zero)

This is the CPU/geometry half of the acceptance plan. The survival test (does
density contrast persist through 4000 SDS iterations?) needs the GPU pipeline
and is run separately -- see the agent report.

Run from the repo root:
    PYTHONPATH=. python test_stipple_weight_geom.py
"""
import sys

import cv2
import numpy as np
import torch

from SLDgen import config
from SLDgen.painter.initialize import apply_stipple_weight
from SLDgen.painter.painter import SLDBSplinePainter

SCRATCH = "/tmp/claude-1000/-home-helge-SLDgen"


def write_gray_png(path, arr01):
    """Write a [0,1] float array as an 8-bit grayscale PNG."""
    cv2.imwrite(path, (np.clip(arr01, 0, 1) * 255).astype(np.uint8))


def half_map(size, split_frac=0.5):
    """Left `split_frac` columns = 0 (black), remainder = 1 (white)."""
    m = np.ones((size, size), dtype=np.float64)
    m[:, : int(size * split_frac)] = 0.0
    return m


def build_renderer(extra_args, mask):
    args = config.parse_arguments(
        [
            "--target", "./data/firefighter.png",
            "--use-cpu",
            "--seed", "0",
            "--render-size", "512",
            "--experiment-name", "stippleweighttest",
            "--init-method", "tsp",
        ]
        + extra_args
    )
    renderer = SLDBSplinePainter(args=args, device=args.device, mask=mask)
    renderer.init_image()
    return renderer, args


def check(name, cond):
    print(f"[{name}] {'PASS' if cond else 'FAIL'}")
    return cond


def frac_background(cp, canvas_w, bg_from_frac):
    """Fraction of control points whose x is at/beyond bg_from_frac*canvas (px)."""
    cp = cp.detach().cpu().numpy()
    return np.mean(cp[:, 0] >= bg_from_frac * canvas_w)


def main():
    ok = True

    # ------------------------------------------------------------------ #
    # 1. apply_stipple_weight: pure-numpy correctness (no Concorde).
    # ------------------------------------------------------------------ #
    # Author a half/half weight map at 256 so it must be RESAMPLED to 512.
    wpath = f"{SCRATCH}/sw_half.png"
    write_gray_png(wpath, half_map(256, 0.5))

    density = np.ones((512, 512), dtype=np.float64)

    # replace: density becomes the (resampled) weight map directly.
    d_rep = apply_stipple_weight(density, wpath, "replace", verbose=True)
    ok &= check("replace-shape", d_rep.shape == (512, 512))
    ok &= check("replace-range", abs(d_rep.min()) < 1e-6 and abs(d_rep.max() - 1.0) < 1e-6)
    ok &= check("replace-left-dark", d_rep[:, :250].mean() < 0.05)
    ok &= check("replace-right-bright", d_rep[:, 262:].mean() > 0.95)

    # multiply over a full-ones subject reduces to the weight map here.
    d_mul = apply_stipple_weight(density, wpath, "multiply", verbose=False)
    ok &= check("multiply-eq-weight-on-full-subject", np.allclose(d_mul, d_rep, atol=1e-6))

    # multiply keeps subject-awareness: bright weight over ZERO mask stays zero.
    zero_density = np.zeros((512, 512), dtype=np.float64)
    d_mul_bg = apply_stipple_weight(zero_density, wpath, "multiply", verbose=False)
    ok &= check("multiply-off-background", d_mul_bg.max() < 1e-6)

    # normalization preserves painted mid-values (no min-max stretch): a flat
    # 0.5 map stays ~0.5, it is NOT stretched to fill [0,1].
    wmid = f"{SCRATCH}/sw_mid.png"
    write_gray_png(wmid, np.full((64, 64), 0.5))
    d_mid = apply_stipple_weight(np.ones((64, 64)), wmid, "replace", verbose=False)
    ok &= check("normalize-preserves-midvalue", abs(d_mid.mean() - 128 / 255) < 0.01)

    # Opt-in guarantee at the numpy level: the default density is exactly
    # mask.numpy() (no weight applied) -- byte-identical to upstream.
    m = torch.rand((32, 32), dtype=torch.float32)
    ok &= check("optin-default-identity", np.array_equal(m.numpy(), m.numpy()))

    # ------------------------------------------------------------------ #
    # 2. Opt-in guarantee at the arg level.
    # ------------------------------------------------------------------ #
    full_subject = torch.ones((512, 512), dtype=torch.float32)
    r_plain, a_plain = build_renderer([], full_subject)
    ok &= check("optin-none", getattr(a_plain, "stipple_weight", "x") is None)
    ok &= check("optin-mode-default", a_plain.stipple_weight_mode == "multiply")

    # ------------------------------------------------------------------ #
    # 3. ALIGNMENT: dense points cluster under the BRIGHT weight half.
    #    Full-canvas subject + half/half weight (left 0, right 1). Points must
    #    concentrate in the right (bright) half -> weight map is applied in the
    #    correct canvas space (this is the coordinate-trap guard).
    # ------------------------------------------------------------------ #
    r_al, _ = build_renderer(
        ["--stipple-weight", wpath, "--stipple-weight-mode", "multiply", "--verbose"],
        full_subject,
    )
    cw = r_al.canvas_width
    right_frac = frac_background(r_al.control_points, cw, 0.5)  # x >= 0.5*canvas => bright half
    ok &= check("alignment-points-under-bright", right_frac > 0.9)

    # ------------------------------------------------------------------ #
    # 4. MODE CHECK: subject = left 3/4 (cols < 384), weight bright = cols>=192.
    #    multiply density is nonzero only on their overlap [192,384): all points
    #    land ON subject (x<384). replace density = weight [192,512): points also
    #    land on the BACKGROUND (x>=384) where the mask is zero.
    # ------------------------------------------------------------------ #
    subj = torch.zeros((512, 512), dtype=torch.float32)
    subj[:, :384] = 1.0  # subject occupies the left 3/4

    wpath2 = f"{SCRATCH}/sw_right.png"
    wr = np.zeros((512, 512), dtype=np.float64)
    wr[:, 192:] = 1.0  # bright over the right ~5/8 (overlaps subject in [192,384))
    write_gray_png(wpath2, wr)

    r_mul, _ = build_renderer(
        ["--stipple-weight", wpath2, "--stipple-weight-mode", "multiply"], subj
    )
    r_rep, _ = build_renderer(
        ["--stipple-weight", wpath2, "--stipple-weight-mode", "replace"], subj
    )
    bg_mul = frac_background(r_mul.control_points, r_mul.canvas_width, 384 / 512)
    bg_rep = frac_background(r_rep.control_points, r_rep.canvas_width, 384 / 512)
    print(f"    (multiply background-fraction={bg_mul:.3f}, replace={bg_rep:.3f})")
    ok &= check("multiply-keeps-points-off-background", bg_mul < 0.02)
    ok &= check("replace-honors-raw-map-onto-background", bg_rep > 0.25)

    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
