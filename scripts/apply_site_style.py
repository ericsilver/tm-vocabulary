"""Give the published pages under docs/ the look of aporia.institute.

The pages are served at aporia.institute/tm-vocabulary/, beside the site itself, so they can
load the site's stylesheet by root-relative URL. This script makes two insertions in each page
and changes nothing else:

  1. <link rel="stylesheet" href="/css/appendix.css"> directly after the page's own <style>
     block, so the site sheet overrides it (the sheet lives in the ericsilver.github.io repo);
  2. a header with the site's monogram wordmark, linking to https://aporia.institute/, as the
     first element of <body>.

It is idempotent: a page that already carries an insertion is left as it is. The emitters
(online_appendix.py, theme_pages.py, theme_explorer.py, themes_page.py) pass their HTML through
apply() before writing, so a regeneration keeps both insertions.

  python scripts/apply_site_style.py            # every docs/**/*.html
  python scripts/apply_site_style.py --check    # report what would change, write nothing
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

LINK = '<link rel="stylesheet" href="/css/appendix.css">'
HEADER = (
    '<header class="aporia-strip"><a class="aporia-wordmark" href="https://aporia.institute/" '
    'aria-label="aporia.institute, the home page">'
    '<svg viewBox="0 -2.5 28 29" fill="none" stroke="currentColor" stroke-width="2.4" aria-hidden="true">'
    '<svg width="21" height="23" viewBox="0 0 21 23"><path d="M1.6 24 10.5 4 19.4 24"/><path d="M5.2 16h4M11.8 16h4"/>'
    '<g opacity=".24"><path d="M10.5 13.7V24" stroke-width="2.2"/>'
    '<circle cx="10.5" cy="11.1" r="1.2" fill="currentColor" stroke="none"/></g></svg>'
    '<path d="M25.5 -1.5v27"/></svg>aporia.institute</a></header>'
)
_BODY = re.compile(r"<body[^>]*>")


def apply(html: str) -> str:
    """Return html with the stylesheet link and the wordmark header inserted (once each)."""
    if 'href="/css/appendix.css"' not in html:
        head_end = html.find("</head>")
        i = html.rfind("</style>", 0, head_end if head_end >= 0 else len(html))
        if i >= 0:
            i += len("</style>")
        elif head_end >= 0:
            i = head_end
        else:
            raise ValueError("no <style> block or </head> to place the stylesheet link after")
        html = html[:i] + LINK + html[i:]
    if 'class="aporia-strip"' not in html:
        m = _BODY.search(html)
        if not m:
            raise ValueError("no <body> tag to place the header after")
        html = html[:m.end()] + HEADER + html[m.end():]
    return html


def main() -> int:
    check = "--check" in sys.argv[1:]
    changed = same = 0
    for p in sorted(DOCS.rglob("*.html")):
        # newline="" keeps each file's line endings exactly as they are
        with open(p, encoding="utf-8", newline="") as f:
            old = f.read()
        new = apply(old)
        if new == old:
            same += 1
            continue
        changed += 1
        if not check:
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(new)
    print(f"[site-style] {changed} page(s) {'would change' if check else 'changed'}, {same} already styled", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
