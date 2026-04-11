import re

with open('sct_predictions.tex', 'r', encoding='utf-8') as f:
    tex = f.read()
with open('../references/sct_predictions.bib', 'r', encoding='utf-8') as f:
    bib = f.read()

# Cite keys
cite_keys = set()
for m in re.finditer(r'\\cite\{([A-Za-z0-9:,\s]+)\}', tex):
    for k in m.group(1).split(','):
        cite_keys.add(k.strip())

# Bib keys
bib_keys = set()
for m in re.finditer(r'@\w+\{([A-Za-z0-9:]+),', bib):
    bib_keys.add(m.group(1))

missing = cite_keys - bib_keys
unused = bib_keys - cite_keys
print('MISSING in bib:', sorted(missing) if missing else 'NONE')
print('UNUSED in tex:', sorted(unused) if unused else 'NONE')
print()

# Ref/eqref keys
ref_keys = set()
for m in re.finditer(r'\\ref\{([A-Za-z0-9:_-]+)\}', tex):
    ref_keys.add(m.group(1))
for m in re.finditer(r'\\eqref\{([A-Za-z0-9:_-]+)\}', tex):
    ref_keys.add(m.group(1))

# Label keys
label_keys = set()
for m in re.finditer(r'\\label\{([A-Za-z0-9:_-]+)\}', tex):
    label_keys.add(m.group(1))

missing_labels = ref_keys - label_keys
print('Referenced:', sorted(ref_keys))
print('MISSING labels:', sorted(missing_labels) if missing_labels else 'NONE')
