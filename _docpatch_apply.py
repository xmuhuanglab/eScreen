# Temporary helper: applies line-number based insertions from a "part" module.
# Usage: python _docpatch_apply.py _docpatch_p1.py
# Reads each target file with its own line ending, inserts whole lines *before*
# the given 1-based line numbers (applied in descending order), and rewrites the
# file with the same line ending / trailing-newline state as the original.
import sys
import importlib.util
from collections import defaultdict

spec = importlib.util.spec_from_file_location('part', sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

by_file = defaultdict(list)
for f, ln, txt in mod.BLOCKS:
    by_file[f].append((ln, txt))

for f, items in by_file.items():
    raw = open(f, 'rb').read()
    if raw.startswith(b'\xef\xbb\xbf'):
        raise SystemExit('BOM found in ' + f)
    data = raw.decode('utf-8')
    eol = '\r\n' if '\r\n' in data else '\n'
    ended = data.endswith(eol)
    lines = data.split(eol)
    if ended:
        lines = lines[:-1]
    n_before = len(lines)
    # Blocks sharing a line number are concatenated in the order they were
    # declared (so a docstring followed by a blank line stays in that order).
    merged = []
    for ln, txt in items:
        if merged and merged[-1][0] == ln:
            merged[-1][1] = merged[-1][1] + '\n' + txt
        else:
            merged.append([ln, txt])
    for ln, txt in sorted(merged, key=lambda t: -t[0]):
        blk = txt.split('\n')
        if blk and blk[-1] == '':
            blk.pop()
        if ln < 1 or ln > n_before:
            raise SystemExit('%s: line %d out of range 1..%d' % (f, ln, n_before))
        lines[ln - 1:ln - 1] = blk
    out = eol.join(lines) + (eol if ended else '')
    open(f, 'wb').write(out.encode('utf-8'))
    print('%s: %d -> %d lines (eol=%r, %d blocks)' % (f, n_before, len(lines), eol, len(items)))
