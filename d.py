
import nbformat
from nbmerge import merge_notebooks  # o el método que uses

nb1 = nbformat.read(r"C:\Users\ACER-A315-59\OneDrive\Desktop\Proyecto final IA\primero.ipynb", as_version=4)
nb2 = nbformat.read(r"C:\Users\ACER-A315-59\OneDrive\Desktop\Proyecto final IA\segundo.ipynb", as_version=4)

# Une las celdas
nb1.cells += nb2.cells

nbformat.write(nb1, "resultado.ipynb")