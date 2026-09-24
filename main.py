from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_4K = 3840


def fmt(v):
    return f"{float(v):.2f}".rstrip("0").rstrip(".")


def to_hex(bgr):
    b, g, r = [int(max(0, min(255, round(c)))) for c in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def is_background_color(bgr):
    b, g, r = [float(c) for c in bgr]
    return min(r, g, b) > 232 and (max(r, g, b) - min(r, g, b)) < 22


def _is_chroma_center(bgr_center):
    b, g, r = [float(c) for c in bgr_center]
    span = max(r, g, b) - min(r, g, b)
    # Bright steel red (#xx / high R) counts even if B/G are moderate
    if r > 160 and r > g + 30 and r > b + 30 and span > 25:
        return True
    return span > 35 and max(r, g, b) > 70


def cluster_1d(values, k):
    vals = values.reshape(-1, 1).astype(np.float32)
    if len(vals) < k:
        return [float(values.mean())] if len(vals) else [0.0]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    _, _, centers = cv2.kmeans(vals, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    return sorted(float(c[0]) for c in centers)


def background_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mn = np.min(bgr, axis=2)
    # Near-white paper only — never treat vivid red/blue (high sat) as background
    return ((v >= 250) & (s <= 22) & (mn >= 230)) | ((v >= 242) & (s <= 18) & (mn >= 232))


def build_palette(bgr):
    bg = background_mask(bgr)
    ink = ~bg
    centers = [np.array([255.0, 255.0, 255.0])]
    if not np.any(ink):
        return centers

    chroma_m, h, s, v = chroma_mask(bgr)
    chroma = ink & chroma_m
    gray = ink & ~chroma_m

    if np.any(chroma):
        hue = h[chroma].astype(np.int32)
        buckets = {}
        for hv, px in zip(hue, bgr[chroma]):
            key = "red" if hv <= 12 or hv >= 168 else f"h{int(hv) // 15}"
            buckets.setdefault(key, []).append(px)
        for key, pts in buckets.items():
            if len(pts) < 6:
                continue
            mean = np.mean(np.array(pts, dtype=np.float32), axis=0)
            # Force vivid red center so steel fills stay #dc0000-ish
            if key == "red":
                mean = np.array([
                    min(float(mean[0]), 70.0),
                    min(float(mean[1]), 70.0),
                    max(float(mean[2]), 220.0),
                ], dtype=np.float32)
            centers.append(mean)

    if np.any(gray):
        luma = v[gray].astype(np.float32)
        core = luma < 190
        src = luma[core] if np.count_nonzero(core) > 30 else luma
        unique_levels = len(np.unique((src / 28).astype(np.int32)))
        k = 1 if unique_levels <= 1 else min(2, max(2, unique_levels))
        if len(src) < 40:
            k = 1
        for level in cluster_1d(src, k):
            if level > 200:
                continue
            band = gray & (np.abs(v.astype(np.float32) - level) <= 16)
            if np.count_nonzero(band) >= 8:
                centers.append(bgr[band].mean(axis=0).astype(np.float32))

    unique = [centers[0]]
    for c in centers[1:]:
        # Never drop chroma (red/blue) just because mean luma is high
        if (not _is_chroma_center(c)) and float(np.mean(c)) > 175:
            continue
        if min(np.linalg.norm(c - u) for u in unique) > 18:
            unique.append(c)
    return unique


def assign_labels(bgr):
    palette = build_palette(bgr)
    pixels = bgr.reshape(-1, 3).astype(np.float32)
    centers = np.array(palette, dtype=np.float32)
    chroma_m, _, sat2d, val2d = chroma_mask(bgr)
    sat = sat2d.reshape(-1)
    val = val2d.reshape(-1)
    chroma_flat = chroma_m.reshape(-1)

    chroma_idx = [i for i, c in enumerate(palette) if i and _is_chroma_center(c)]
    gray_idx = [i for i, c in enumerate(palette) if i and not _is_chroma_center(c)]

    d_white = np.linalg.norm(pixels - centers[0], axis=1)
    labels = np.zeros(len(pixels), dtype=np.int32)
    ink = (d_white >= 22) & ~((val >= 248) & (sat <= 18))
    chroma_px = ink & chroma_flat
    gray_px = ink & ~chroma_flat

    if chroma_idx and np.any(chroma_px):
        cd = np.linalg.norm(pixels[:, None, :] - centers[chroma_idx], axis=2)
        labels[chroma_px] = np.array(chroma_idx, dtype=np.int32)[cd[chroma_px].argmin(axis=1)]

    if gray_idx and np.any(gray_px):
        gd = np.linalg.norm(pixels[:, None, :] - centers[gray_idx], axis=2)
        labels[gray_px] = np.array(gray_idx, dtype=np.int32)[gd[gray_px].argmin(axis=1)]
        pale = gray_px & (d_white < 28) & (val > 232)
        labels[pale] = 0

    return palette, labels.reshape(bgr.shape[:2])


def enhance_to_4k(bgr, target_long=TARGET_4K):
    """Turn a soft/low-res CAD scan into a crisp 4K line drawing."""
    h, w = bgr.shape[:2]
    work = cv2.bilateralFilter(bgr, 7, 45, 45)
    blur = cv2.GaussianBlur(work, (0, 0), 0.7)
    work = cv2.addWeighted(work, 1.5, blur, -0.5, 0)
    work = np.clip(work, 0, 255).astype(np.uint8)

    palette, labels = assign_labels(work)

    long_edge = max(h, w)
    scale = target_long / float(long_edge)
    if long_edge < 900:
        scale = max(scale, 4.0)
    if scale < 1.0:
        scale = 1.0
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if max(nw, nh) > target_long:
        fit = target_long / max(nw, nh)
        nw = max(1, int(round(nw * fit)))
        nh = max(1, int(round(nh * fit)))

    canvas = np.full((nh, nw, 3), 255, np.uint8)
    kernel = np.ones((2, 2), np.uint8)
    for i, color in enumerate(palette):
        if i == 0 or is_background_color(color):
            continue
        if (not _is_chroma_center(color)) and float(np.mean(color)) > 188:
            continue
        mask = (labels == i).astype(np.uint8) * 255
        if cv2.countNonZero(mask) < 3:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        up = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas[up > 127] = np.array(color, dtype=np.uint8)
    return canvas


def encode_png_data_url(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def contour_to_d(cnt):
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    if peri < 3:
        return ""
    if peri < 48 or area < 36:
        epsilon = 0.12
    else:
        epsilon = min(0.75, max(0.2, peri * 0.00035))
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 2:
        return ""
    d = f"M {fmt(pts[0][0])} {fmt(pts[0][1])}"
    for p in pts[1:]:
        d += f" L {fmt(p[0])} {fmt(p[1])}"
    d += " Z"
    return d


def mask_to_path(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    parts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area < 0.6 and peri < 4:
            continue
        d = contour_to_d(cnt)
        if d:
            parts.append(d)
    return " ".join(parts)


def vectorize_with_vtracer(bgr):
    """Color-faithful Bezier SVG via vtracer (subprocess — avoids native crashes)."""
    import tempfile
    import os
    import re
    import subprocess
    import sys

    h, w = bgr.shape[:2]
    fd_in, in_path = tempfile.mkstemp(suffix=".png")
    fd_out, out_path = tempfile.mkstemp(suffix=".svg")
    os.close(fd_in)
    os.close(fd_out)
    try:
        cv2.imwrite(in_path, bgr)
        script = (
            "import vtracer\n"
            f"vtracer.convert_image_to_svg_py({in_path!r}, {out_path!r}, "
            "colormode='color', hierarchical='stacked', mode='spline', "
            "filter_speckle=2, color_precision=6, layer_difference=8, "
            "corner_threshold=60, length_threshold=3.5, max_iterations=10, "
            "splice_threshold=45, path_precision=2)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=90,
        )
        if proc.returncode != 0 or not os.path.isfile(out_path):
            return None, None
        with open(out_path, "r", encoding="utf-8") as f:
            svg = f.read()
    except Exception:
        return None, None
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass

    if not svg or "<path" not in svg:
        return None, None

    paths = []
    for m in re.finditer(r"<path\b([^>]*)/?>", svg, flags=re.IGNORECASE):
        attrs = m.group(1)
        d_m = re.search(r'\bd="([^"]+)"', attrs)
        if not d_m:
            continue
        fill_m = re.search(r'\bfill="([^"]+)"', attrs)
        stroke_m = re.search(r'\bstroke="([^"]+)"', attrs)
        fr_m = re.search(r'\bfill-rule="([^"]+)"', attrs)
        fill = (fill_m.group(1) if fill_m else "#000000").strip()
        stroke = (stroke_m.group(1) if stroke_m else "none").strip()
        if fill.lower() in ("#fff", "#ffffff", "white", "rgb(255,255,255)"):
            continue
        if re.fullmatch(r"#f{3,6}", fill, flags=re.I):
            continue
        paths.append({
            "d": d_m.group(1),
            "fill": fill if fill.lower() != "none" else "none",
            "stroke": stroke if stroke.lower() != "none" else "none",
            "fillRule": (fr_m.group(1) if fr_m else "evenodd"),
        })
    return paths or None, svg


def vectorize_opencv_color(bgr):
    """Fallback color regions via palette labels (preserves red fills)."""
    palette, labels = assign_labels(bgr)
    layers = []
    for i, center in enumerate(palette):
        if i == 0:
            continue
        mask = (labels == i).astype(np.uint8) * 255
        if cv2.countNonZero(mask) < 4:
            continue
        if (not _is_chroma_center(center)) and float(np.mean(center)) > 188:
            continue
        if _is_chroma_center(center):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        d = mask_to_path(mask)
        if d:
            layers.append((to_hex(center), d))
    layers.sort(key=lambda item: int(item[0][1:3], 16) + int(item[0][3:5], 16) + int(item[0][5:7], 16))
    return [{
        "d": d,
        "fill": color,
        "stroke": "none",
        "fillRule": "evenodd",
    } for color, d in layers]


def vectorize_as_is(bgr):
    vt_paths, _ = vectorize_with_vtracer(bgr)
    if vt_paths:
        return vt_paths
    return vectorize_opencv_color(bgr)


def walk_skeleton(skel):
    pts = cv2.findNonZero(skel)
    if pts is None:
        return []
    unvisited = {tuple(p) for p in pts.reshape(-1, 2)}
    neighbor_offsets = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if dx or dy]

    def neighbors(p):
        found = []
        for dx, dy in neighbor_offsets:
            q = (p[0] + dx, p[1] + dy)
            if q in unvisited:
                found.append(q)
        return found

    paths = []
    while unvisited:
        start = unvisited.pop()
        chain = [start]
        for direction in (1, -1):
            curr = start
            while True:
                nxts = neighbors(curr)
                if not nxts:
                    break
                nxts.sort(key=lambda q: (q[0] - curr[0]) ** 2 + (q[1] - curr[1]) ** 2)
                nxt = nxts[0]
                unvisited.discard(nxt)
                if direction == 1:
                    chain.append(nxt)
                else:
                    chain.insert(0, nxt)
                curr = nxt
        if len(chain) >= 3:
            paths.append(chain)
    return paths


def trace_centerline(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skel = cv2.ximgproc.thinning(thresh)
    parts = []
    for path in walk_skeleton(skel):
        arr = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
        epsilon = max(0.6, 0.0025 * cv2.arcLength(arr, False))
        approx = cv2.approxPolyDP(arr, epsilon, False)
        simplified = [p[0] for p in approx]
        if len(simplified) < 2:
            continue
        d = f"M {fmt(simplified[0][0])} {fmt(simplified[0][1])}"
        for p in simplified[1:]:
            d += f" L {fmt(p[0])} {fmt(p[1])}"
        parts.append(d)
    if not parts:
        return []
    return [{
        "d": " ".join(parts),
        "fill": "none",
        "stroke": "#111111",
        "strokeWidth": 1.15,
        "fillRule": "nonzero",
    }]


def encode_cleaned_png(bgr):
    # Compression 1 = clearer PNG, less blockiness on thin CAD lines
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def isolate_main_drawing(bgr):
    """
    Paint watermark lattice to white ONLY. Keep every drawing pixel —
    cyan glass, pink dims, green labels, red markers must not be stripped.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int32)
    sat = hsv[:, :, 1].astype(np.int32)
    val = hsv[:, :, 2].astype(np.int32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int32)
    b = bgr[:, :, 0].astype(np.int32)
    g = bgr[:, :, 1].astype(np.int32)
    r = bgr[:, :, 2].astype(np.int32)
    span = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    # Real CAD paint (any vivid color) — never treat as watermark
    drawing_chroma = (sat >= 35) & (span >= 30) & (val >= 30)

    # Pale cyan diamond watermark only (desaturated lattice, not glass/seals)
    watermark = (
        (hue >= 70)
        & (hue <= 150)
        & (sat >= 2)
        & (sat <= 45)
        & (gray >= 160)
        & (gray <= 248)
        & (span < 35)
        & ~drawing_chroma
    )
    wash = (val >= 235) & (sat >= 2) & (sat <= 35) & (gray >= 225) & (span < 28) & ~drawing_chroma

    mark = watermark | wash
    if not np.any(mark):
        return bgr.copy(), False

    out = bgr.copy()
    out[mark] = (255, 255, 255)
    return out, True


def is_multicolor_cad(bgr):
    """True when drawing uses several distinct hues (dims, glass, labels)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    chroma = sat >= 40
    if np.count_nonzero(chroma) < 80:
        return False
    # Bucket hues; multi-color if 3+ buckets have meaningful pixels
    buckets = {}
    for hv in hue[chroma]:
        key = int(hv) // 12
        buckets[key] = buckets.get(key, 0) + 1
    strong = sum(1 for c in buckets.values() if c >= 40)
    return strong >= 3


def estimate_sharpness(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


_SR_CACHE = {}


def _espcn_model_path(scale=4):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "models", f"ESPCN_x{scale}.pb")


def ai_upscale_espcn(bgr, scale=4):
    """
    AI super-resolution (Upscale.media-style): ESPCN reconstructs detail
    instead of simple stretch. Falls back to None if model missing.
    """
    path = _espcn_model_path(scale)
    if not os.path.isfile(path):
        return None
    try:
        from cv2 import dnn_superres
        key = (path, scale)
        if key not in _SR_CACHE:
            sr = dnn_superres.DnnSuperResImpl_create()
            sr.readModel(path)
            sr.setModel("espcn", scale)
            _SR_CACHE[key] = sr
        # Large images: tile to avoid OOM
        h, w = bgr.shape[:2]
        if h * w > 900_000:
            fit = 800 / float(max(h, w))
            if fit < 1:
                small = cv2.resize(
                    bgr,
                    (max(1, int(round(w * fit))), max(1, int(round(h * fit)))),
                    interpolation=cv2.INTER_AREA,
                )
                return _SR_CACHE[key].upsample(small)
        return _SR_CACHE[key].upsample(bgr)
    except Exception:
        return None


def upscale_lanczos_steps(bgr, target_long):
    """Fallback stepwise 2× Lanczos when AI model unavailable."""
    out = bgr.copy()
    guard = 0
    while max(out.shape[:2]) * 2 <= target_long and guard < 5:
        guard += 1
        h, w = out.shape[:2]
        out = cv2.resize(out, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    h, w = out.shape[:2]
    long_edge = max(h, w)
    if long_edge < target_long:
        scale = target_long / float(long_edge)
        out = cv2.resize(
            out,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_LANCZOS4,
        )
    elif long_edge > target_long:
        scale = target_long / float(long_edge)
        out = cv2.resize(
            out,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return out


def clarity_boost(bgr):
    """Upscale.media-like clarity: denoise + edge restore + contrast, keep colors."""
    out = cv2.bilateralFilter(bgr, d=5, sigmaColor=30, sigmaSpace=30)

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    # Reconnect micro-gaps in thin dark CAD strokes after upscale
    ink = (gray <= 165) & (sat < 55)
    soft = (gray > 165) & (gray < 220) & (sat < 40)
    near = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    ink_mask = ((ink | (soft & near)).astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, k, iterations=2)
    gap = (ink_mask > 0) & (gray >= 175) & (sat < 45)
    if np.any(gap):
        out[gap] = (55, 55, 55)
    real_ink = ink & (gray <= 140)
    if np.any(real_ink):
        deepened = np.maximum(out[real_ink].astype(np.int16) - 12, 25)
        out[real_ink] = deepened.astype(np.uint8)

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, bch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, bch]), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(out, (0, 0), 0.75)
    out = cv2.addWeighted(out, 1.28, blur, -0.28, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_low_quality_to_hq(bgr, target_long=2800, max_long=4096):
    """
    Upscale.media-style: low-quality / blurry / pixelated → high-quality clear.
    1) AI ESPCN ×4 super-resolution when model present
    2) Fit to target resolution
    3) Clarity boost (sharpen + line heal)
    """
    h0, w0 = bgr.shape[:2]
    long0 = max(h0, w0)
    sharp0 = estimate_sharpness(bgr)
    was_low = long0 < target_long or sharp0 < 150.0

    out = bgr.copy()
    used_ai = False

    if was_low and long0 < target_long:
        ai = ai_upscale_espcn(out, scale=4)
        if ai is not None:
            out = ai
            used_ai = True
        else:
            out = upscale_lanczos_steps(out, target_long)

        if max(out.shape[:2]) < target_long:
            out = upscale_lanczos_steps(out, target_long)
        elif max(out.shape[:2]) > max_long:
            scale = max_long / float(max(out.shape[:2]))
            h, w = out.shape[:2]
            out = cv2.resize(
                out,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
    elif long0 > max_long:
        scale = max_long / float(long0)
        out = cv2.resize(
            out,
            (max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    if was_low:
        out = clarity_boost(out)

    return out, was_low, used_ai


def as_is_clear_display(bgr, min_long=1600, max_long=4096):
    """
    Produce high-quality clear output for SVG (Upscale.media-style enhance).
    Returns (image, was_low, used_ai).
    """
    hq, was_low, used_ai = enhance_low_quality_to_hq(
        bgr, target_long=max(min_long, 2800), max_long=max_long
    )
    return hq, was_low, used_ai


def prepare_for_vectorize(bgr, max_dim=2000):
    """Downscale only for path tracing — never alter the as-is display image."""
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        bgr = cv2.resize(
            bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return bgr


def chroma_mask(bgr):
    """
    Detect solid color fills (red steel, blue inserts, pink dims, cyan glass).
    Bright saturated colors MUST be included.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int32)
    sat = hsv[:, :, 1].astype(np.int32)
    val = hsv[:, :, 2].astype(np.int32)
    b = bgr[:, :, 0].astype(np.int32)
    g = bgr[:, :, 1].astype(np.int32)
    r = bgr[:, :, 2].astype(np.int32)
    chroma_span = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    vivid = (sat >= 35) & (val >= 30) & (chroma_span >= 28)
    is_red = ((hue <= 12) | (hue >= 168)) & (r >= 120) & (r > g + 20) & (r > b + 20) & (sat >= 25)
    is_blue = (hue >= 85) & (hue <= 140) & (sat >= 30) & (val >= 35) & (b > r + 10)
    is_cyan = (hue >= 75) & (hue <= 105) & (sat >= 30) & (b > 100)
    is_pink = (hue >= 140) & (hue <= 175) & (sat >= 40) & (r > 100)
    is_green = (hue >= 35) & (hue <= 85) & (sat >= 40) & (g > r) & (g > b)
    return vivid | is_red | is_blue | is_cyan | is_pink | is_green, hue, sat, val


def preserve_chroma_colors(bgr, chroma, hue):
    """Paint each chroma blob with its true median color (keep CAD palette)."""
    out = np.full_like(bgr, 255)
    if not np.any(chroma):
        return out
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    chroma_u8 = (chroma.astype(np.uint8) * 255)
    # Light close only — heavy close merges thin pink dim lines
    chroma_u8 = cv2.morphologyEx(chroma_u8, cv2.MORPH_CLOSE, close_k, iterations=1)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(chroma_u8, 8)
    min_area = max(3, (bgr.shape[0] * bgr.shape[1]) // 300000)
    for i in range(1, nlab):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        sel = labels == i
        med = np.median(bgr[sel].reshape(-1, 3), axis=0).astype(np.float32)
        hh = float(np.median(hue[sel]))
        mb, mg, mr = float(med[0]), float(med[1]), float(med[2])
        # Keep steel red vivid
        if ((hh <= 12) or (hh >= 168)) and mr > mg + 20 and mr > mb + 20 and mr > 160:
            med = np.array([min(mb, 80), min(mg, 80), max(mr, 220)], dtype=np.float32)
        out[sel] = med.astype(np.uint8)
    return out


def crisp_cad_display(bgr, min_long=1400):
    """
    Mono / two-tone CAD harden: black linework + solid chroma fills.
    Do NOT use for multi-color dimensioned drawings (use as_is_clear_display).
    """
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge < min_long:
        scale = min(2.0, min_long / float(long_edge))
        bgr = cv2.resize(
            bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_LANCZOS4,
        )

    chroma, hue, sat, val = chroma_mask(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int32)

    out = preserve_chroma_colors(bgr, chroma, hue)
    chroma_keep = np.any(out != 255, axis=2)

    cyanish = (hue >= 65) & (hue <= 155) & (sat >= 3) & (gray >= 150) & ~chroma
    hard = (gray <= 165) & ~chroma & ~cyanish
    soft = (gray > 165) & (gray < 225) & (sat < 50) & ~chroma & ~cyanish
    near = cv2.dilate(hard.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    ink = hard | (soft & near)

    ink_u8 = (ink.astype(np.uint8) * 255)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink_u8 = cv2.morphologyEx(ink_u8, cv2.MORPH_CLOSE, close_k, iterations=1)
    ink = (ink_u8 > 0) & ~chroma_keep

    out[ink] = (0, 0, 0)
    return out


def downscale_for_trace(bgr, max_dim=3200):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr, 1.0
    scale = max_dim / max(h, w)
    small = cv2.resize(
        bgr,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return small, 1.0 / scale


def scale_path_d(d, scale):
    if scale == 1:
        return d
    out = []
    for tok in d.replace(",", " ").split():
        try:
            out.append(fmt(float(tok) * scale))
        except ValueError:
            out.append(tok)
    return " ".join(out)


def decode_image(contents):
    """Decode PNG/JPG/GIF/WebP/BMP/TIFF; flatten alpha onto white."""
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        alpha = a.astype(np.float32) / 255.0
        bg = np.full(b.shape, 255.0, dtype=np.float32)
        b = (b.astype(np.float32) * alpha + bg * (1 - alpha)).astype(np.uint8)
        g = (g.astype(np.float32) * alpha + bg * (1 - alpha)).astype(np.uint8)
        r = (r.astype(np.float32) * alpha + bg * (1 - alpha)).astype(np.uint8)
        return cv2.merge([b, g, r])
    return img[:, :, :3]


def paths_to_svg(paths, width, height):
    """W3C SVG 1.1 with real path data (not embedded raster)."""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
    ]
    for p in paths:
        attrs = [f'd="{p["d"]}"']
        fill = p.get("fill", "none")
        stroke = p.get("stroke", "none")
        attrs.append(f'fill="{fill}"')
        attrs.append(f'stroke="{stroke}"')
        if p.get("strokeWidth") is not None and stroke != "none":
            attrs.append(f'stroke-width="{p["strokeWidth"]}"')
        if p.get("fillRule"):
            attrs.append(f'fill-rule="{p["fillRule"]}"')
        parts.append(f'<path {" ".join(attrs)}/>')
    parts.append("</svg>")
    return "".join(parts)


@app.post("/api/trace")
async def trace_image(
    file: UploadFile = File(...),
    mode: str = Form("outline"),
    remove_background: str = Form("true"),
):
    """
    svgai.org-style vectorize prep:
    1) optional background / watermark removal
    2) harden edges for clean color regions
    3) return cleaned raster for Bezier tracer + fallback OpenCV paths
    Real editable SVG paths are produced by the client ImageTracer (Bezier).
    """
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        return {"error": "File too large (max 20MB)"}

    bgr = decode_image(contents)
    if bgr is None:
        return {"error": "Invalid or unsupported image. Use PNG, JPG, GIF, WebP, BMP, or TIFF."}

    do_strip = str(remove_background).lower() not in ("0", "false", "no", "off")
    bg_removed = False
    if do_strip:
        bgr, bg_removed = isolate_main_drawing(bgr)

    # Upscale.media-style: low-quality → AI/HQ clear for SVG
    display, was_low, used_ai = as_is_clear_display(bgr, min_long=1600, max_long=4096)
    quality = "hq" if was_low else "as-is"

    disp_h, disp_w = display.shape[:2]
    prepared = prepare_for_vectorize(display, max_dim=2800)
    prev_h, prev_w = prepared.shape[:2]

    if mode == "centerline":
        paths = trace_centerline(prepared)
        svg_doc = paths_to_svg(paths, prev_w, prev_h)
    else:
        vt_paths, vt_svg = vectorize_with_vtracer(prepared)
        if vt_paths:
            paths = vt_paths
            svg_doc = vt_svg or paths_to_svg(paths, prev_w, prev_h)
        else:
            paths = vectorize_opencv_color(prepared)
            svg_doc = paths_to_svg(paths, prev_w, prev_h)

    return {
        "paths": paths,
        "svg": svg_doc,
        "cleaned_image": encode_cleaned_png(display),
        "trace_image": encode_cleaned_png(prepared),
        "width": disp_w,
        "height": disp_h,
        "trace_width": prev_w,
        "trace_height": prev_h,
        "path_count": len(paths),
        "background_removed": bg_removed,
        "mode": mode,
        "quality": quality,
        "enhanced": was_low,
        "ai_upscale": used_ai,
        "vectorize": True,
    }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
