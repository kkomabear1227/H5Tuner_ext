#!/bin/sh
# Build the tarball handed to a collaborating lab.
#
# Since 2026-09-14 everything here is also on the public h5tuner-ext remote,
# so `git clone` works too.  This script stays useful for handing over a
# self-contained snapshot without git history or the 191 MB virtualenv.
#
# What is deliberately excluded:
#   .venv/                      191 MB of the repo's 194 MB, and machine-specific
#   .git/                       carries the public remote; a collaborator should
#                               start from the tarball, not inherit our history
#   .claude/settings.local.json local tool permissions, names a home directory
#   __pycache__, *.pyc          build noise
#
# Pass --no-research to drop 07-후속-연구-계획.md, the research agenda.
# Everything else is self-contained without it.

set -e

OUT_DIR="${OUT_DIR:-$HOME/Downloads}"
NAME="h5tuner-autotuner-$(date +%Y%m%d)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
DROP_RESEARCH=0

for arg in "$@"; do
    case "$arg" in
        --no-research) DROP_RESEARCH=1 ;;
        *) echo "unknown option: $arg" >&2; exit 1 ;;
    esac
done

trap 'rm -rf "$STAGE"' EXIT

echo "staging from $ROOT"
rsync -a \
    --exclude '.venv/' \
    --exclude '.git/' \
    --exclude '.claude/settings.local.json' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude 'autom4te.cache/' \
    "$ROOT/" "$STAGE/$NAME/"

if [ "$DROP_RESEARCH" -eq 1 ]; then
    rm -f "$STAGE/$NAME/07-후속-연구-계획.md"
    echo "dropped 07-후속-연구-계획.md"
fi

mkdir -p "$OUT_DIR"
tar -czf "$OUT_DIR/$NAME.tar.gz" -C "$STAGE" "$NAME"

echo
echo "wrote $OUT_DIR/$NAME.tar.gz"
du -sh "$OUT_DIR/$NAME.tar.gz"
echo
echo "contents:"
tar -tzf "$OUT_DIR/$NAME.tar.gz" | sed "s|^$NAME/||" \
    | cut -d/ -f1 | grep -v '^$' | sort -u | sed 's/^/  /'
echo
echo "check before sending:"
echo "  tar -xzf $OUT_DIR/$NAME.tar.gz -O $NAME/README.md | head -5"
