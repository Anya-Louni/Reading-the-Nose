"""Inline the data bundle into the template to produce one self-contained page.

The result works two ways with no change: dropped on GitHub Pages as
`webapp/index.html`, and published as an Artifact. Both need the data embedded
rather than fetched, because an Artifact cannot fetch a sibling file and a
`file://` page cannot fetch anything at all.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "webapp"

MARKER = re.compile(r"/\*__DATA__\*/.*?/\*__END__\*/", re.S)
ODOR_MARKER = re.compile(r"/\*__ODOR__\*/.*?/\*__END__ODOR__\*/", re.S)


def main() -> None:
    template = (WEB / "template.html").read_text(encoding="utf-8")
    data = json.loads((WEB / "data" / "galaxy.json").read_text(encoding="utf-8"))

    payload = json.dumps(data, separators=(",", ":"))
    # `</script>` inside a string literal would close the host script tag early.
    payload = payload.replace("</", "<\\/")

    if not MARKER.search(template):
        raise SystemExit("data marker not found in template.html")
    out = MARKER.sub(lambda _: payload, template, count=1)

    # The odour layer is the primary content now, so it ships inlined too.
    odor_path = WEB / "data" / "odor_layer.json"
    if ODOR_MARKER.search(out):
        if not odor_path.exists():
            raise SystemExit(f"{odor_path} missing; run scripts/export_odor_layer.py")
        odor = json.loads(odor_path.read_text(encoding="utf-8"))
        op = json.dumps(odor, separators=(",", ":")).replace("</", "<\\/")
        out = ODOR_MARKER.sub(lambda _: op, out, count=1)

    dest = WEB / "index.html"
    dest.write_text(out, encoding="utf-8")
    kb = dest.stat().st_size / 1024
    print(f"wrote {dest} ({kb:.0f} KB)")
    print(f"  background points : {len(data['galaxy']['background'])}")
    print(f"  analytes          : {len(data['galaxy']['analytes'])}")
    print(f"  confusion pairs   : {len(data['pairs'])}")
    print(f"  IG heatmaps       : "
          f"{sorted(k for k, v in data['molecules'].items() if v['ig_validated'])}")


if __name__ == "__main__":
    main()
