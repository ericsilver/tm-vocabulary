#!/usr/bin/env bash
# Build both papers and their supplements, each with a reference list holding
# only what it cites. Output goes to <paper>/build/ so an open PDF viewer never
# blocks the build; copy from there to the submission files.
#   usage:  bash paper/split/build.sh        (from the repo root)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
PY="${PYTHON:-python}"

build_doc () {   # $1 = directory, $2 = document, $3 = bib file name
  local dir="$1" doc="$2" bib="$3"
  ( cd "$dir"
    mkdir -p build
    rm -f "build/$doc.aux" "build/$doc.out" "$doc.aux"
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$doc.tex" > /dev/null
    [ -n "$bib" ] && "$PY" "$ROOT/scripts/filter_bib.py" "build/$doc.aux" "$bib.tex"
    for i in 1 2; do
      pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$doc.tex" > /dev/null
    done
    cp -f "build/$doc.aux" "$doc.aux"      # read by the supplement through xr
    echo "$dir/$doc: $(grep -c 'Warning.*undefined' "build/$doc.log" || true) undefined references"
  )
}

build_doc A_qss main bib_main
build_doc A_qss supp bib_supp
build_doc B_lead main bib_main
build_doc B_lead supp ""
