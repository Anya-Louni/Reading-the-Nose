"""Capture cover images from the built report.

Tall viewport, heading-based clipping: each cover is defined by the section it
starts at and the element it ends at, so the crop follows the content rather than
a hardcoded pixel box that breaks whenever the copy changes.

Serves `docs/` over a local HTTP server rather than opening the file directly,
because `file://` pages are treated as opaque origins by some browser features and
because that is how the page is actually deployed.

Covers produced:
  cover_quadrant     the validation scatter, the headline visual
  cover_showcase     trained vs untrained attention vs IG, the LinkedIn shot
  cover_galaxy       the chemical-space point cloud
  cover_hero         masthead plus the stat strip
  cover_social       1200x630 OG card composited from the quadrant
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socket
import socketserver
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"
OUT = SITE / "covers"

# (name, selector to scroll to, selector whose bottom ends the crop, pad px)
SHOTS = [
    ("cover_hero", "header", ".stat-strip", 28),
    ("cover_quadrant", "section:nth-of-type(1)", ".quad-wrap", 24),
    ("cover_showcase", "section:nth-of-type(2)", ".show-wrap", 24),
    ("cover_galaxy", "section:nth-of-type(4)", "#stage", 24),
    # plot area alone, for compositing into the social card
    ("_plot_only", "#quadbox", "#quadbox", 0),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(directory: Path, port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))

    class Quiet(socketserver.TCPServer):
        allow_reuse_address = True

        def __init__(self, *a, **k):
            super().__init__(*a, **k)

    httpd = Quiet(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def capture(url: str, theme: str, scale: int = 2) -> dict[str, bytes]:
    from playwright.sync_api import sync_playwright

    shots: dict[str, bytes] = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1440, "height": 1800},   # tall: whole sections fit
            device_scale_factor=scale,
            color_scheme=theme,
        )
        page.goto(url, wait_until="networkidle")
        page.emulate_media(reduced_motion="no-preference")
        # let fonts settle, the reveal observers fire, and the canvases paint
        page.wait_for_timeout(2500)
        page.evaluate("""() => {
            document.querySelectorAll('.reveal').forEach(s => {
                s.classList.add('in');
                s.querySelectorAll('[data-w]').forEach(b =>
                    b.style.width = Math.min(100, parseFloat(b.dataset.w)) + '%');
            });
        }""")
        page.wait_for_timeout(2200)

        for name, start_sel, end_sel, pad in SHOTS:
            try:
                page.locator(start_sel).first.scroll_into_view_if_needed()
            except Exception:
                pass
            page.wait_for_timeout(1400)   # canvases redraw on reveal
            box = page.evaluate("""([s, e, pad]) => {
                const a = document.querySelector(s), b = document.querySelector(e);
                if (!a || !b) return null;
                const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
                const top = ra.top + scrollY, bottom = rb.bottom + scrollY;
                return {x: Math.max(0, ra.left - pad), y: Math.max(0, top - pad),
                        width: Math.min(innerWidth, ra.width + pad * 2),
                        height: bottom - top + pad * 2};
            }""", [start_sel, end_sel, pad])
            if not box or box["height"] < 40:
                print(f"   skip {name}: selector not found or empty")
                continue
            # The box is in page coordinates (rect + scrollY), so the screenshot
            # must be full_page; a viewport screenshot treats clip as
            # viewport-relative and throws "clipped area outside the image".
            shots[name] = page.screenshot(clip=box, type="png", full_page=True)
            print(f"   {name}  {box['width']:.0f}x{box['height']:.0f}", flush=True)

        browser.close()
    return shots


def social_card(quadrant_png: bytes, dest: Path) -> None:
    """1200x630 OG card: the quadrant, matted, with the headline number."""
    from PIL import Image, ImageDraw, ImageFont
    import io

    W, H = 1200, 630
    bg, ink, muted, alert = (19, 21, 24), (232, 234, 237), (124, 132, 140), (255, 107, 107)
    card = Image.new("RGB", (W, H), bg)

    # The chart alone, scaled to fill the right half edge to edge. Cover-fit, so
    # it crops rather than letterboxing into dead space.
    src = Image.open(io.BytesIO(quadrant_png)).convert("RGB")
    target_w, target_h = int(W * 0.56), H
    scale = max(target_w / src.width, target_h / src.height)
    src = src.resize((max(1, int(src.width * scale)), max(1, int(src.height * scale))),
                     Image.LANCZOS)
    # Anchor right, not centre: the red "faithful + spurious" cluster sits at the
    # right edge of the plot and is the whole reason the image exists. A centred
    # crop silently removes it.
    left = src.width - target_w
    top = (src.height - target_h) // 2
    src = src.crop((left, top, left + target_w, top + target_h))
    card.paste(src, (W - target_w, 0))

    d = ImageDraw.Draw(card)

    def font(size, bold=False):
        for p in (r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
                  r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
        return ImageFont.load_default()

    x = 56
    d.text((x, 76), "READING THE NOSE", font=font(19, True), fill=muted)
    d.text((x, 128), "36.3%", font=font(104, True), fill=alert)
    for i, line in enumerate([
            "of attention maps were faithful",
            "and reproducible from a model",
            "with its weights thrown away."]):
        d.text((x, 258 + i * 38), line, font=font(29), fill=ink)
    d.text((x, 404), "The faithfulness score flagged none of them.",
           font=font(21), fill=muted)
    d.line([(x, 452), (x + 300, 452)], fill=(46, 52, 58), width=2)
    d.text((x, 474), "750 held-out molecules  \u00b7  integrated gradients: 6.4%",
           font=font(19), fill=muted)

    dest.parent.mkdir(parents=True, exist_ok=True)
    card.save(dest, "PNG", optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="dark", choices=["dark", "light", "both"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if not (SITE / "index.html").exists():
        raise SystemExit("webapp/index.html missing; run scripts/build_webapp.py")

    port = free_port()
    httpd = serve(SITE, port)
    url = f"http://127.0.0.1:{port}/index.html"
    print(f"serving {SITE} at {url}")

    try:
        themes = ["dark", "light"] if args.theme == "both" else [args.theme]
        for theme in themes:
            print(f"\ncapturing {theme} theme...", flush=True)
            shots = capture(url, theme)
            suffix = "" if theme == "dark" else "_light"
            for name, png in shots.items():
                if name.startswith("_"):
                    continue          # internal source, not a deliverable
                (OUT / f"{name}{suffix}.png").write_bytes(png)
            if theme == "dark" and "_plot_only" in shots:
                social_card(shots["_plot_only"], OUT / "cover_social.png")
                print("   cover_social  1200x630")
    finally:
        httpd.shutdown()

    print(f"\nwrote {len(list(OUT.glob('*.png')))} images to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"   {p.name:28s} {p.stat().st_size / 1024:7.0f} KB")


if __name__ == "__main__":
    main()
