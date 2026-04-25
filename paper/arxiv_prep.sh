#!/bin/bash
# arxiv_prep.sh — assemble the arXiv upload bundle
#
# arXiv accepts a tex file + any figures it references. We ship
# a self-contained tarball at paper/arxiv_bundle.tar.gz so the
# upload form has exactly one file to accept.
#
# Usage:
#   cd paper
#   ./arxiv_prep.sh
#
# Output:
#   paper/arxiv_bundle.tar.gz
#
# Contents of the tarball:
#   paper.tex
#   figures/*.png
#
# The tarball is NOT committed — run this script immediately before
# uploading. The .gitignore excludes it.

set -euo pipefail

cd "$(dirname "$0")"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "staging area: $STAGE"

# Copy sources (tex expects figures/ relative to cwd)
cp paper.tex "$STAGE/"
mkdir -p "$STAGE/figures"
cp figures/*.png "$STAGE/figures/"

# Sanity: smoke-build in the staged tree to catch any missing deps
if command -v pdflatex >/dev/null 2>&1; then
    echo "smoke-building in staging to verify self-contained..."
    (cd "$STAGE" && pdflatex -interaction=nonstopmode paper.tex > /tmp/arxiv_prep.log 2>&1 && \
                    pdflatex -interaction=nonstopmode paper.tex >> /tmp/arxiv_prep.log 2>&1) || {
        echo "BUILD FAILED — see /tmp/arxiv_prep.log"
        exit 1
    }
    [ -f "$STAGE/paper.pdf" ] && echo "  build OK: $(stat -f %z "$STAGE/paper.pdf") bytes"
else
    echo "pdflatex not installed — skipping smoke build"
fi

# Remove intermediate artifacts so the tarball is clean
rm -f "$STAGE/paper.aux" "$STAGE/paper.log" "$STAGE/paper.out" \
      "$STAGE/paper.toc" "$STAGE/paper.pdf"

# Create tarball
tar -czf arxiv_bundle.tar.gz -C "$STAGE" paper.tex figures
echo "wrote arxiv_bundle.tar.gz ($(stat -f %z arxiv_bundle.tar.gz) bytes)"
echo
echo "Contents:"
tar -tzf arxiv_bundle.tar.gz

echo
echo "Next:"
echo "  1. Go to https://arxiv.org/submit"
echo "  2. Login / register (ORCID recommended)"
echo "  3. New submission → Upload → select paper/arxiv_bundle.tar.gz"
echo "  4. Primary category: cs.SE"
echo "     Secondary:       cs.LG"
echo "  5. MSC-class:        68T05, 68N99"
echo "     ACM-class:        I.2.7, D.2.8"
echo "  6. Title + author + abstract should auto-populate from \\title / \\author /"
echo "     \\begin{abstract} — verify, then Preview."
echo "  7. License choice: CC-BY 4.0 is the standard open choice."
echo "  8. Submit. You'll get an arXiv identifier (e.g. 2604.01234) within 1-2"
echo "     business days after moderator review."
