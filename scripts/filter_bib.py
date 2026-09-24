"""Write a reference list holding only the entries a document actually cites.

The papers share one hand-written thebibliography (paper/v3_rp/bibliography_rp.tex),
and thebibliography prints every entry whether cited or not. This reads the
citations from a document's .aux and writes the filtered list next to it.

Usage: python scripts/filter_bib.py <document.aux> <output.tex>
Build: pdflatex doc; python scripts/filter_bib.py doc.aux bib_doc.tex; pdflatex doc (x2)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1] / "paper" / "v3_rp" / "bibliography_rp.tex"


def main() -> int:
    aux, out = Path(sys.argv[1]), Path(sys.argv[2])
    cited: set[str] = set()
    if aux.exists():
        for group in re.findall(r"\\citation\{([^}]+)\}", aux.read_text(encoding="utf-8", errors="replace")):
            cited.update(k.strip() for k in group.split(","))
    text = SHARED.read_text(encoding="utf-8")
    head, rest = text.split("\\begin{thebibliography}", 1)
    body, _ = rest.split("\\end{thebibliography}", 1)
    first_line, entries_text = body.split("\n", 1)
    entries = re.split(r"(?=\\bibitem)", entries_text)
    kept = [e for e in entries if (m := re.match(r"\\bibitem\[[^\]]*\]\{([^}]+)\}", e)) and m.group(1) in cited]
    out.write_text("\\begin{thebibliography}" + first_line + "\n\n" + "".join(kept).rstrip()
                   + "\n\n\\end{thebibliography}\n", encoding="utf-8")
    missing = sorted(cited - {re.match(r"\\bibitem\[[^\]]*\]\{([^}]+)\}", e).group(1) for e in kept})
    print(f"{out.name}: {len(kept)} entries kept" + (f"; cited but absent: {missing}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
