# Smell Galaxy

Static, self-contained page. `index.html` is the build output and needs no server,
no build step at view time, and no network beyond the Google Fonts stylesheet.

## Deploy to GitHub Pages

Point Pages at this directory, or copy `index.html` to the site root. Nothing else
is required; the data is inlined.

## Rebuild after changing the model or the analysis

    python scripts/export_galaxy_data.py   # embeddings, 3D layout, confusions, attributions
    python scripts/build_webapp.py         # inline data/galaxy.json into template.html

Edit `template.html`, never `index.html` (it is generated and will be overwritten).

## Framing, which is deliberate

The page is about the **sensor's blind spots**, not the model's confidence. Phase 2
found no generalisation to unseen molecules, so a page built around model confidence
would rest on the one claim that failed. The graph network appears only where it
demonstrably works: laying out the 4,975-molecule odour corpus it was trained on
(validation macro AUC 0.803).

Three things are load-bearing and should not be quietly changed:

- **No attention maps anywhere.** Not because they are empty (an earlier claim
  that attention was uniform was an aggregation artifact, see
  `data_audit/RETRACTION_attention_uniform.md`) but because they fail the Adebayo
  randomisation check: mean rho +0.79 against an untrained model, 60.4% failing
  outright on 750 held-out molecule-seed observations. All atom heatmaps are integrated gradients.
- **Only four molecules get heatmaps** (ethanol, acetaldehyde, acetone, butan-1-ol),
  the ones passing both the weight-randomisation sanity check and the deletion
  faithfulness test. The rest show their exclusion reason instead.
- **Only ammonia/toluene is concentration-clean.** The four wind-tunnel pairs are
  tagged as confounded because UCI 251 ran one concentration per gas.


## Page structure (after the scope change)

The odour task is the primary content. The sensor work is section 05, explicitly
banner-labelled as the small-sample stress test that was run first and failed.

1. **The test** - 1,500 validation points (750 attention + 750 IG) plotted as
   sanity vs faithfulness, with the "faithful and spurious" field marked in red.
   Toggle between methods; the two clouds sit in different places.
2. **The same map from an untrained model** - three panels per molecule: trained
   attention, attention from a weight-randomised model, and integrated gradients.
   The first two are near-identical (rho +0.95 to +0.97 on these four).
3. **Does the architecture earn itself** - the Phase 2 bar chart.
4. **Chemical space** - the rotating galaxy.
5. **The small-sample stress test** - the sensor confusion pairs, labelled.
6. **Limits** - including the retracted uniformity finding.

## Two things that bite when editing

- **`<meta charset="utf-8">` is load-bearing.** The Artifact host injects a
  charset, so a missing one looks fine when published and breaks on GitHub Pages
  or any plain file server. Non-ASCII in the template is written as JS escapes
  (`→`, `ρ`) for the same reason.
- **The quadrant canvas needs its `ResizeObserver`.** The first boot frame can
  land before the aspect-ratio box is laid out, which leaves the canvas at its
  300x150 default and silently draws nothing useful.

## Cover images

`python scripts/make_covers.py --theme dark` (or `both`). It serves `webapp/` on a
local port, drives headless Chromium at a 1440x1800 viewport, and clips each crop
between two selectors so the frame follows the content instead of a pixel box that
rots whenever the copy changes.

Two things it gets right that are easy to get wrong:

- Clip boxes are built from `rect.top + scrollY`, which is page coordinates, so the
  screenshot call must pass `full_page=True`. A viewport screenshot treats `clip`
  as viewport-relative and fails with "clipped area outside the image".
- The social card crops the plot **anchored to the right edge**. The red
  faithful-and-spurious cluster sits against the right-hand axis, and a centred
  cover-fit crop removes exactly the thing the image exists to show.
