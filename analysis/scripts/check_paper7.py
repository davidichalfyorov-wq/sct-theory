#!/usr/bin/env python3
"""Pre-push check for Paper 7."""
import re, os

tex_path = r"F:\Black Mesa Research Facility\Main Facility\Physics department\SCT Theory\papers\drafts\sct_cj_bridge.tex"
bib_path = r"F:\Black Mesa Research Facility\Main Facility\Physics department\SCT Theory\papers\references\sct_cj_bridge.bib"
fig_dir = r"F:\Black Mesa Research Facility\Main Facility\Physics department\SCT Theory\papers\drafts\figures"

with open(tex_path, "r", encoding="utf-8") as f:
    tex = f.read()

# 1. References
pat_ref = re.compile(r"\\ref\{([^}]+)\}")
pat_label = re.compile(r"\\label\{([^}]+)\}")
refs = set(pat_ref.findall(tex))
labels = set(pat_label.findall(tex))
missing_refs = refs - labels
print("=== REFERENCE CHECK ===")
if missing_refs:
    print(f"  BROKEN refs: {missing_refs}")
else:
    print(f"  All {len(refs)} refs have labels. OK")

# 2. Figures
pat_fig = re.compile(r"\\includegraphics[^{]*\{([^}]+)\}")
graphics = pat_fig.findall(tex)
missing_figs = []
for g in graphics:
    found = False
    for d in [os.path.dirname(tex_path), fig_dir]:
        for ext in ["", ".pdf", ".png"]:
            if os.path.exists(os.path.join(d, g + ext)):
                found = True
                break
            if os.path.exists(os.path.join(d, os.path.basename(g) + ext)):
                found = True
                break
        if found:
            break
    if not found:
        missing_figs.append(g)
print(f"\n=== FIGURE CHECK ({len(graphics)} referenced) ===")
if missing_figs:
    print(f"  MISSING: {missing_figs}")
else:
    print("  All figures found. OK")

# 3. Bibliography
pat_cite = re.compile(r"\\cite\{([^}]+)\}")
all_keys = set()
for m in pat_cite.finditer(tex):
    for k in m.group(1).split(","):
        all_keys.add(k.strip())
with open(bib_path, "r", encoding="utf-8") as f:
    bib = f.read()
bib_keys = set(re.findall(r"@\w+\{(\w[^,]*)", bib))
missing_bib = all_keys - bib_keys
print(f"\n=== BIBLIOGRAPHY ({len(all_keys)} cited, {len(bib_keys)} in bib) ===")
if missing_bib:
    print(f"  MISSING bib entries: {missing_bib}")
else:
    print("  All citations found. OK")

# 4. Check for old prop:factor4
if "prop:factor4" in tex:
    for i, line in enumerate(tex.split("\n")):
        if "prop:factor4" in line:
            print(f"\n  WARNING line {i+1}: old prop:factor4 reference")

# 5. Equation numbering
pat_eqlabel = re.compile(r"\\label\{eq:([^}]+)\}")
eq_labels = pat_eqlabel.findall(tex)
new_labels = [l for l in eq_labels if "factor4" in l or "gs-ga" in l or "Mss" in l]
print(f"\n=== NEW EQUATION LABELS ===")
for l in new_labels:
    print(f"  eq:{l}")

# 6. Stats
n_lines = len(tex.split("\n"))
n_eq = tex.count("\\begin{equation")
n_tab = tex.count("\\begin{table")
n_fig = tex.count("\\begin{figure")
print(f"\n=== DOCUMENT STATS ===")
print(f"  Lines: {n_lines}, Equations: {n_eq}, Tables: {n_tab}, Figures: {n_fig}")
print(f"  Approx pages: {n_lines // 55}")

# 7. Check no AI markers
ai_markers = ["Claude", "GPT", "LLM", "AI-generated", "automated", "Codex", "Anthropic",
               "language model", "artificial intelligence"]
issues = []
for marker in ai_markers:
    for i, line in enumerate(tex.split("\n")):
        if marker.lower() in line.lower() and not line.strip().startswith("%"):
            issues.append((i+1, marker, line.strip()[:80]))
print(f"\n=== AI MARKER CHECK ===")
if issues:
    for ln, m, txt in issues:
        print(f"  WARNING line {ln}: '{m}' in: {txt}")
else:
    print("  No AI markers found. OK")

print("\n=== ALL CHECKS COMPLETE ===")
