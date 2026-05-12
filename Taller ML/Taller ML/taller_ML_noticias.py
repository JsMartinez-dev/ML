# =============================================================
# TALLER FORMATIVO – UNIDAD ML
# Inteligencia Artificial IA-2026-I
# Dataset: News Category Classification
# https://www.kaggle.com/datasets/bornaetminan/news-category-classification-dataset
# =============================================================

# ── INSTALACIÓN (ejecutar una sola vez si es necesario) ───────
# pip install pandas matplotlib seaborn nltk scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Descargar recursos NLTK necesarios
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# =============================================================
# TAREA 1 – DESCRIPCIÓN DEL PROBLEMA
# =============================================================
"""
El dataset "News Category Classification" contiene titulares y descripciones
cortas de noticias de HuffPost, etiquetadas por categoría (ej: POLITICS,
ENTERTAINMENT, SPORTS, etc.). El objetivo es entrenar un modelo de ML capaz
de predecir la categoría de una noticia a partir de su texto.

Variable objetivo: 'category'
"""

print("=" * 60)
print("TALLER ML – DATASET DE NOTICIAS")
print("=" * 60)


# =============================================================
# TAREA 2 – CARGAR EL DATASET
# =============================================================
# Descarga el archivo desde Kaggle y colócalo en la misma carpeta
# que este script. El archivo se llama algo como 'news.csv' o similar.

# AJUSTA el nombre del archivo según lo que descargaste de Kaggle
ARCHIVO = r"C:\Users\ACER-A315-59\OneDrive\Desktop\Proyecto final IA\Taller ML\Taller ML\archive\news_category_clean.csv"   # <-- cambia si el nombre es diferente

try:
    df = pd.read_csv(ARCHIVO)
    print(f"\n✅ Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
except FileNotFoundError:
    print(f"\n⚠️  Archivo '{ARCHIVO}' no encontrado.")
    print("   Descarga el dataset de Kaggle y colócalo en esta carpeta.")
    print("   Luego cambia la variable ARCHIVO con el nombre correcto.")
    exit()


# =============================================================
# TAREA 3 – EXPLORACIÓN INICIAL CON PANDAS
# =============================================================
print("\n" + "─" * 50)
print("TAREA 3 – EXPLORACIÓN CON PANDAS")
print("─" * 50)

print("\n📌 .head() – primeras 5 filas:")
print(df.head(5))

print("\n📌 .info() – estructura del dataset:")
df.info()

print("\n📌 .describe() – estadísticas generales:")
print(df.describe(include='all'))

print("\n📌 Valores nulos por columna:")
print(df.isnull().sum())


# =============================================================
# TAREA 4 – VARIABLE OBJETIVO Y DISTRIBUCIÓN DE CATEGORÍAS
# =============================================================
print("\n" + "─" * 50)
print("TAREA 4 – DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
print("─" * 50)

# Ajusta 'category' si tu columna tiene otro nombre
COLUMNA_OBJETIVO = 'category'

if COLUMNA_OBJETIVO not in df.columns:
    print(f"⚠️  Columna '{COLUMNA_OBJETIVO}' no encontrada.")
    print(f"   Columnas disponibles: {list(df.columns)}")
    print("   Cambia COLUMNA_OBJETIVO con el nombre correcto.")
else:
    conteo = df[COLUMNA_OBJETIVO].value_counts()
    print(f"\nCategorías encontradas ({len(conteo)}):")
    print(conteo)

    # ── Gráfico de barras ─────────────────────────────────────
    plt.figure(figsize=(14, 6))
    conteo.plot(kind='bar', color='steelblue', edgecolor='white')
    plt.title('Distribución de Categorías de Noticias', fontsize=14, fontweight='bold')
    plt.xlabel('Categoría')
    plt.ylabel('Cantidad de noticias')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig('distribucion_categorias_barras.png', dpi=150)
    plt.show()
    print("✅ Gráfico de barras guardado como 'distribucion_categorias_barras.png'")

    # ── Gráfico de torta (top 10 para que sea legible) ────────
    top10 = conteo.head(10)
    plt.figure(figsize=(9, 9))
    plt.pie(
        top10.values,
        labels=top10.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette('tab10')
    )
    plt.title('Top 10 Categorías (torta)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('distribucion_categorias_torta.png', dpi=150)
    plt.show()
    print("✅ Gráfico de torta guardado como 'distribucion_categorias_torta.png'")


# =============================================================
# TAREA 5 – LIMPIEZA DE TEXTO (con NLTK)
# =============================================================
print("\n" + "─" * 50)
print("TAREA 5 – LIMPIEZA DE TEXTO")
print("─" * 50)

# Ajusta según las columnas de texto de tu dataset
# Normalmente son 'headline' y/o 'short_description'
COLUMNA_TEXTO = 'clean_headline'   # <-- cambia si es necesario

if COLUMNA_TEXTO not in df.columns:
    print(f"⚠️  Columna de texto '{COLUMNA_TEXTO}' no encontrada.")
    print(f"   Columnas disponibles: {list(df.columns)}")
    print("   Cambia COLUMNA_TEXTO con el nombre correcto.")
else:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def limpiar_texto(texto):
        """
        Aplica los 4 pasos de limpieza:
        1. Minúsculas
        2. Eliminar puntuación y caracteres especiales
        3. Eliminar stopwords
        4. Lematizar
        """
        if not isinstance(texto, str):
            return ""

        # 1. Minúsculas
        texto = texto.lower()

        # 2. Eliminar puntuación y caracteres no alfabéticos
        texto = re.sub(r'[^a-z\s]', '', texto)

        # 3. Tokenizar, eliminar stopwords y lematizar
        palabras = texto.split()
        palabras_limpias = [
            lemmatizer.lemmatize(p)
            for p in palabras
            if p not in stop_words and len(p) > 2
        ]

        return ' '.join(palabras_limpias)

    print(f"\nProcesando columna '{COLUMNA_TEXTO}'...")
    df['texto_limpio'] = df[COLUMNA_TEXTO].apply(limpiar_texto)

    print("\n📌 Ejemplo antes y después de la limpieza:")
    for i in range(3):
        print(f"\n  Original : {df[COLUMNA_TEXTO].iloc[i]}")
        print(f"  Limpio   : {df['texto_limpio'].iloc[i]}")

    print("\n✅ Limpieza completada. Nueva columna: 'texto_limpio'")

# =============================================================
# TAREA 5 – PREPROCESAMIENTO DE TEXTO
# =============================================================
print("\n" + "─" * 50)
print("TAREA 5 – PREPROCESAMIENTO DE TEXTO")
print("─" * 50)

# Ajusta el nombre de la columna de texto si es diferente
COLUMNA_TEXTO = 'headline'   # <-- cambia según tu dataset (puede ser 'text', 'title', etc.)

# Verifica que las columnas existen
if COLUMNA_TEXTO not in df.columns:
    print(f"⚠️  Columna '{COLUMNA_TEXTO}' no encontrada.")
    print(f"   Columnas disponibles: {list(df.columns)}")
else:
    lemmatizer = WordNetLemmatizer()
    stop_words  = set(stopwords.words('english'))

    def limpiar_texto(texto):
        """Limpia y lematiza un texto."""
        if not isinstance(texto, str):
            return ""
        texto = texto.lower()                          # minúsculas
        texto = re.sub(r'[^a-z\s]', '', texto)        # eliminar puntuación y números
        tokens = texto.split()
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
        return ' '.join(tokens)

    print("⏳ Limpiando texto... (puede tardar unos segundos)")
    df['texto_limpio'] = df[COLUMNA_TEXTO].apply(limpiar_texto)
    print("✅ Columna 'texto_limpio' creada.")
    print(df[['headline', 'texto_limpio']].head(3))


# =============================================================
# TAREA 6 – MATRICES DE CORRELACIÓN (3 enfoques)
# =============================================================
print("\n" + "─" * 50)
print("TAREA 6 – MATRICES DE CORRELACIÓN")
print("─" * 50)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Parámetros generales ──────────────────────────────────────
TOP_CATEGORIAS = 15   # categorías más frecuentes a mostrar
TOP_PALABRAS   = 20   # palabras más importantes para la matriz de términos

# Filtrar solo las TOP_CATEGORIAS más frecuentes (para legibilidad)
cats_top = df[COLUMNA_OBJETIVO].value_counts().head(TOP_CATEGORIAS).index.tolist()
df_top   = df[df[COLUMNA_OBJETIVO].isin(cats_top)].copy()

print(f"\n📌 Trabajando con las {TOP_CATEGORIAS} categorías más frecuentes.")


# ─────────────────────────────────────────────────────────────
# 6A – MATRIZ DE SIMILITUD COSENO ENTRE CATEGORÍAS
#      (¿qué tan parecidas son dos categorías en vocabulario?)
# ─────────────────────────────────────────────────────────────
print("\n[6A] Calculando matriz de similitud coseno entre categorías...")

# Vectorizador TF-IDF sobre todo el corpus filtrado
tfidf_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf   = tfidf_vec.fit_transform(df_top['texto_limpio'])

# Vector promedio TF-IDF por categoría → representa el "perfil" de cada categoría
cat_matrix = np.zeros((len(cats_top), X_tfidf.shape[1]))
for i, cat in enumerate(cats_top):
    idx = df_top[df_top[COLUMNA_OBJETIVO] == cat].index
    # Reindexar al subconjunto df_top
    mask = df_top[COLUMNA_OBJETIVO] == cat
    cat_matrix[i] = X_tfidf[mask.values].mean(axis=0)

# Similitud coseno entre perfiles de categoría
sim_matrix = cosine_similarity(cat_matrix)
df_sim     = pd.DataFrame(sim_matrix, index=cats_top, columns=cats_top)

# Gráfico 6A
plt.figure(figsize=(13, 10))
mask_diag = np.eye(len(cats_top), dtype=bool)   # ocultar diagonal (siempre = 1)

sns.heatmap(
    df_sim,
    annot=True,
    fmt='.2f',
    cmap='YlOrRd',
    mask=mask_diag,
    linewidths=0.5,
    linecolor='white',
    vmin=0, vmax=0.6,
    annot_kws={'size': 8},
    xticklabels=cats_top,
    yticklabels=cats_top
)
plt.title(
    'Matriz de Similitud Coseno entre Categorías\n(basada en perfiles TF-IDF promedio)',
    fontsize=13, fontweight='bold', pad=15
)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('matriz_similitud_categorias.png', dpi=150)
plt.show()
print("✅ Guardada como 'matriz_similitud_categorias.png'")


# ─────────────────────────────────────────────────────────────
# 6B – MATRIZ DE CORRELACIÓN DE TÉRMINOS CLAVE POR CATEGORÍA
#      (¿qué tan correlacionados están los top términos entre sí?)
# ─────────────────────────────────────────────────────────────
print("\n[6B] Calculando matriz de correlación de términos clave...")

# Seleccionar los TOP_PALABRAS términos con mayor peso promedio global
mean_tfidf     = np.asarray(X_tfidf.mean(axis=0)).flatten()
top_idx        = mean_tfidf.argsort()[::-1][:TOP_PALABRAS]
feature_names  = np.array(tfidf_vec.get_feature_names_out())
top_words      = feature_names[top_idx]

# Submatriz de solo esas columnas
X_top_words    = pd.DataFrame(
    X_tfidf[:, top_idx].toarray(),
    columns=top_words
)

# Correlación de Pearson entre los vectores de los términos
corr_words     = X_top_words.corr()

# Gráfico 6B
plt.figure(figsize=(13, 11))
sns.heatmap(
    corr_words,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    linewidths=0.4,
    linecolor='white',
    annot_kws={'size': 8},
    square=True,
    vmin=-0.4, vmax=0.4
)
plt.title(
    f'Matriz de Correlación – Top {TOP_PALABRAS} Términos TF-IDF\n(correlación de Pearson entre frecuencias de términos)',
    fontsize=12, fontweight='bold', pad=15
)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('matriz_correlacion_terminos.png', dpi=150)
plt.show()
print("✅ Guardada como 'matriz_correlacion_terminos.png'")


# ─────────────────────────────────────────────────────────────
# 6C – HEATMAP DE PRESENCIA DE TÉRMINOS POR CATEGORÍA
#      (¿qué palabras clave dominan cada categoría?)
# ─────────────────────────────────────────────────────────────
print("\n[6C] Calculando heatmap de términos por categoría...")

# Peso TF-IDF promedio de cada top-palabra dentro de cada categoría
heat_data = pd.DataFrame(index=cats_top, columns=top_words, dtype=float)

for cat in cats_top:
    mask  = df_top[COLUMNA_OBJETIVO] == cat
    media = np.asarray(X_tfidf[mask.values][:, top_idx].mean(axis=0)).flatten()
    heat_data.loc[cat] = media

# Normalizar por fila para comparar proporciones dentro de cada categoría
heat_norm = heat_data.div(heat_data.max(axis=1), axis=0)

# Gráfico 6C
plt.figure(figsize=(18, 8))
sns.heatmap(
    heat_norm.astype(float),
    cmap='Blues',
    linewidths=0.3,
    linecolor='white',
    annot=True,
    fmt='.2f',
    annot_kws={'size': 7},
    xticklabels=top_words,
    yticklabels=cats_top
)
plt.title(
    f'Presencia Relativa de Top {TOP_PALABRAS} Términos por Categoría\n(TF-IDF promedio normalizado por categoría)',
    fontsize=12, fontweight='bold', pad=15
)
plt.xlabel('Términos', fontsize=10)
plt.ylabel('Categoría', fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('heatmap_terminos_por_categoria.png', dpi=150)
plt.show()
print("✅ Guardada como 'heatmap_terminos_por_categoria.png'")


# ─────────────────────────────────────────────────────────────
# RESUMEN DE INTERPRETACIÓN
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN – INTERPRETACIÓN DE LAS MATRICES")
print("=" * 60)
print("""
6A · Similitud coseno entre categorías
   → Valores cercanos a 1 indican categorías con vocabulario
     muy similar (posible confusión en el modelo).
   → Valores cercanos a 0 indican categorías bien separadas.

6B · Correlación de términos clave
   → Rojo (+) : dos términos aparecen juntos frecuentemente.
   → Azul (−) : dos términos raramente co-ocurren.
   → Útil para detectar redundancia entre features.

6C · Presencia de términos por categoría
   → Muestra qué palabras son "firma" de cada categoría.
   → Un término con valor alto en una sola fila es un 
     discriminador excelente para el clasificador.
""")

# =============================================================
# TAREA 6 – VECTORIZACIÓN
# =============================================================
print("\n" + "─" * 50)
print("TAREA 6 – VECTORIZACIÓN")
print("─" * 50)

if 'texto_limpio' in df.columns:

    # Eliminar filas donde el texto limpio quedó vacío
    df_limpio = df[df['texto_limpio'].str.strip() != ''].copy()
    print(f"Registros después de limpiar vacíos: {len(df_limpio)}")

    # ── Opción A: CountVectorizer ─────────────────────────────
    print("\n📌 CountVectorizer (bolsa de palabras):")
    count_vec = CountVectorizer(max_features=5000)
    X_count = count_vec.fit_transform(df_limpio['texto_limpio'])
    print(f"  Forma de la matriz: {X_count.shape}")
    print(f"  (filas = noticias, columnas = palabras del vocabulario)")
    print(f"  Ejemplo de primeras 10 palabras del vocabulario:")
    print(f"  {count_vec.get_feature_names_out()[:10]}")

    # ── Opción B: TfidfVectorizer ─────────────────────────────
    print("\n📌 TfidfVectorizer (frecuencia inversa de documento):")
    tfidf_vec = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vec.fit_transform(df_limpio['texto_limpio'])
    print(f"  Forma de la matriz: {X_tfidf.shape}")
    print(f"  Ejemplo de primeras 10 palabras del vocabulario:")
    print(f"  {tfidf_vec.get_feature_names_out()[:10]}")

    print("\n✅ Vectorización lista.")
    print("   'X_count' y 'X_tfidf' son las matrices de características.")
    print("   La variable objetivo (y) sería:")
    print(f"   y = df_limpio['{COLUMNA_OBJETIVO}']")

    # ── Guardar dataset limpio para uso futuro ─────────────────
    df_limpio[['texto_limpio', COLUMNA_OBJETIVO]].to_csv(
        'dataset_limpio.csv', index=False
    )
    print("\n✅ Dataset limpio guardado como 'dataset_limpio.csv'")


# =============================================================
print("\n" + "=" * 60)
print("TALLER COMPLETADO")
print("Archivos generados:")
print("  - distribucion_categorias_barras.png")
print("  - distribucion_categorias_torta.png")
print("  - dataset_limpio.csv")
print("=" * 60)
print(df.columns.tolist())
print(df.head())
