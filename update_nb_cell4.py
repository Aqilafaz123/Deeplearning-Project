import nbformat

nb = nbformat.read('perbandingan_3_arsitektur_indobert_cnn_svm.ipynb', as_version=4)
src = nb.cells[4].source

target = '{"file": "dataset_multimedia/RAW/dataset_liputan6_raw.xlsx", "source": "Liputan6", "label": 1},'
replacement = '''{"file": "dataset_multimedia/RAW/dataset_liputan6_raw.xlsx", "source": "Liputan6", "label": 1},
    {"file": "dataset_multimedia/RAW/dataset_merdeka_raw.xlsx", "source": "Merdeka", "label": 1},
    {"file": "dataset_multimedia/RAW/dataset_detik_raw.xlsx", "source": "Detik", "label": 0},'''

if target in src and 'dataset_merdeka_raw.xlsx' not in src:
    nb.cells[4].source = src.replace(target, replacement)
    nbformat.write(nb, 'perbandingan_3_arsitektur_indobert_cnn_svm.ipynb')
    print("Notebook Cell 4 berhasil ditambahkan Merdeka & Detik!")
else:
    print("Sudah terpasang.")

