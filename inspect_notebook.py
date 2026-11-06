import json

with open('perbandingan_3_arsitektur_indobert_cnn_svm.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell.get('cell_type', '')
    src = ''.join(cell.get('source', []))
    lines = src.split('\n')
    preview = lines[0] if lines else ''
    if len(lines) > 1:
        preview += f" ... ({len(lines)} lines)"
    print(f"Cell {i} [{ctype}]: {preview[:80]}")

