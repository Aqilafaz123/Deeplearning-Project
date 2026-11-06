import nbformat
from nbclient import NotebookClient

notebook_path = 'perbandingan_3_arsitektur_indobert_cnn_svm.ipynb'
nb = nbformat.read(notebook_path, as_version=4)

# Buat client dan jalankan dalam kernel context
client = NotebookClient(nb, timeout=600, kernel_name='python3')

print("Memulai eksekusi sel notebook (Cell 0 s/d 7)...")
with client.setup_kernel():
    for idx in range(8):
        cell = nb.cells[idx]
        if cell.cell_type == 'code':
            print(f"  Menjalankan Cell {idx}...")
            client.execute_cell(cell, idx)

nbformat.write(nb, notebook_path)
print("Berhasil mengeksekusi dan memperbarui output Cell 0 s/d 7 di notebook!")
