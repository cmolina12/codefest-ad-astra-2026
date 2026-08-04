import json
import re 
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable

from chunking import DocumentoSinTexto, contar_palabras, detectar_idioma

# Umbrales
MIN_PALABRAS_DOC = 15

MIN_PALABRAS_HTML = 8

MIN_PALABRAS_PAGINA = 8

DPI_OCR = 150

MAX_FILAS_CSV = 5000

PACK_TESSERACT = {"es": "spa", "pt": "por", "en": "eng"}
 
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_LIGADURAS = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
                            "\u00ad": "", "\u200b": "", "\ufeff": ""})

_NUMERO_PAGINA = re.compile(
    r"^\s*(?:p[áa]g(?:ina)?\.?\s*)?\d{1,4}(?:\s*(?:de|of|/)\s*\d{1,4})?\s*$",
    re.IGNORECASE,
)


def normalizar(texto: str) -> str:
    """NFC, ligaduras, guiones de corte y espacios. No reescribe contenido.
 
    Se limita a lo que no puede equivocarse. La detección estadística de
    encabezados y pies se probó y se descartó: fallaba contra PDFs reales
    borrando párrafos completos, y con chunks de 140 palabras una cabecera
    repetida es ruido menor comparado con perder evidencia.
    """
    texto = unicodedata.normalize("NFC", texto.translate(_LIGADURAS))
    texto = _CONTROL.sub(" ", texto)
    # 'infraes-\ntructura' -> 'infraestructura': el corte de línea del PDF no
    # es un guion real y parte la palabra para el tokenizador.
    texto = re.sub(r"(\w)[-\u2010\u2011]\s*\n\s*(\w)", r"\1\2", texto)
    texto = re.sub(r"[ \t\u00a0]+", " ", texto)
    texto = re.sub(r" *\n *", "\n", texto)
    lineas = [l for l in texto.split("\n") if not _NUMERO_PAGINA.match(l)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lineas)).strip()


# PDF

def _ocr_imagen(datos: bytes, idioma: str | None) -> str:
    """OCR de una imagen en memoria con Tesseract.
 
    Escala de grises: ~10% más rápido, mismo resultado en texto impreso.
    El pack de idioma importa: con el equivocado Tesseract corrompe los
    diacríticos en silencio ('Relatório' -> 'Relatério'), lo que degrada el
    embedding sin lanzar ningún error.
    """
    import io
 
    import pytesseract
    from PIL import Image
 
    img = Image.open(io.BytesIO(datos)).convert("L")
    return pytesseract.image_to_string(img, lang=PACK_TESSERACT.get(idioma or "", "spa+por+eng"))


def extraer_pdf(path: Path, ocr: bool = True) -> str:
    """Texto de un PDF, con OCR solo en las páginas que no tienen capa de texto.
 
    `get_text("blocks", sort=True)` ordena por posición y respeta el doble
    columnado; la extracción lineal mezcla las dos columnas línea a línea y
    produce frases sin sentido.
 
    El OCR es por página, no por documento: un PDF mixto (texto + páginas
    escaneadas) se recupera entero y solo se paga OCR donde hace falta.
    """
    import fitz
 
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise DocumentoSinTexto(f"{path.name}: no se pudo abrir como PDF ({exc})") from exc
 
    paginas: list[str] = []
    sin_capa: list[int] = []
    for n, pagina in enumerate(doc):
        bloques = pagina.get_text("blocks", sort=True)
        plano = "\n".join(b[4] for b in bloques if isinstance(b[4], str))
        if contar_palabras(plano) < MIN_PALABRAS_PAGINA:
            sin_capa.append(n)
            paginas.append("")
        else:
            paginas.append(plano)
 
    if sin_capa and ocr:
        # El idioma se detecta sobre las páginas que sí tienen texto. Si no hay
        # ninguna (PDF escaneado entero), se OCRea con el pack combinado, se
        # detecta sobre ese texto sucio y se vuelve a OCRear con el correcto.
        limpio = " ".join(p for p in paginas if p)
        idioma = detectar_idioma(limpio) if contar_palabras(limpio) >= 15 else None
 
        if idioma is None:
            sucio = _ocr_imagen(doc[sin_capa[0]].get_pixmap(dpi=DPI_OCR).tobytes("png"), None)
            if contar_palabras(sucio) >= 15:
                idioma = detectar_idioma(sucio)
 
        for n in sin_capa:
            paginas[n] = _ocr_imagen(doc[n].get_pixmap(dpi=DPI_OCR).tobytes("png"), idioma)
 
    n_paginas = doc.page_count
    doc.close()
 
    texto = normalizar("\n\n".join(p for p in paginas if p.strip()))
    if contar_palabras(texto) < MIN_PALABRAS_DOC:
        raise DocumentoSinTexto(
            f"{path.name}: {n_paginas} páginas, {contar_palabras(texto)} palabras extraídas. "
            f"{len(sin_capa)} páginas sin capa de texto. ¿Escaneado con OCR fallido o protegido?"
        )
    return texto

def titulo_pdf(path: Path) -> str | None:
    """Título de los metadatos, si existe y no es basura del generador."""
    import fitz
 
    try:
        doc = fitz.open(path)
        titulo = (doc.metadata or {}).get("title", "")
        doc.close()
    except Exception:
        return None
    titulo = normalizar(titulo or "")
    # Los generadores de PDF meten el nombre del archivo o rutas como título.
    if not titulo or titulo.lower().endswith((".pdf", ".doc", ".docx")) or "\\" in titulo:
        return None
    return titulo if len(titulo.split()) >= 2 else None


# HTML

_ETIQUETAS_FUERA = ("script", "style", "nav", "footer", "header", "aside",
                    "form", "noscript", "iframe", "svg", "button")

def _sopa(path: Path):
    from bs4 import BeautifulSoup
 
    crudo = path.read_bytes().decode("utf-8", errors="replace")
    try:
        return BeautifulSoup(crudo, "lxml")
    except Exception:
        return BeautifulSoup(crudo, "html.parser")


def extraer_html(path: Path) -> str:
    """Contenido principal de un HTML.
 
    El corpus actual no tiene HTML; esta rama existe por si ADL amplía el
    índice. Si llega a haber muchos, conviene cambiar a `trafilatura`, que
    detecta el cuerpo por densidad de texto en vez de por lista de etiquetas.
    """
    sopa = _sopa(path)
    for etiqueta in sopa(_ETIQUETAS_FUERA):
        etiqueta.decompose()
 
    # Se prefiere el contenedor semántico si existe; si no, el body entero.
    cuerpo = sopa.find("main") or sopa.find("article") or sopa.body or sopa
    texto = normalizar(cuerpo.get_text("\n"))
 
    if contar_palabras(texto) < MIN_PALABRAS_HTML:
        raise DocumentoSinTexto(
            f"{path.name}: {contar_palabras(texto)} palabras tras quitar boilerplate. "
            "¿Página renderizada por JavaScript?"
        )
    return texto


def titulo_html(path: Path) -> str | None:
    sopa = _sopa(path)
    if sopa.title and sopa.title.string:
        return normalizar(sopa.title.string) or None
    h1 = sopa.find("h1")
    return normalizar(h1.get_text()) or None if h1 else None


# CSV

def _fila_a_texto(encabezados, valores) -> str:
    """Serializa una fila REPITIENDO los encabezados.
 
    Sin ellos, un chunk de tabla es una lista de valores sueltos sin nada con
    qué emparejar contra una consulta en lenguaje natural. Con ellos, cada
    fila es autocontenida: 'País: Colombia | Año: 2024 | Índice: 0.42'.
    """
    partes = []
    for h, v in zip(encabezados, valores):
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none", "nat", "<na>"):
            partes.append(f"{str(h).strip()}: {s}")
    return " | ".join(partes)



def extraer_csv(path: Path) -> str:
    """Cabecera + filas, cada fila como una oración autocontenida.
 
    El separador se detecta probando: los datasets de observatorios distintos
    usan coma, punto y coma o tabulador indistintamente, y acertar importa
    porque con el separador equivocado pandas devuelve una única columna con
    toda la fila dentro.
    """
    import pandas as pd
 
    # El sniffer de pandas (`sep=None`) inventa separadores dentro del texto y
    # parte palabras por la mitad, así que se prueban separadores explícitos y
    # se elige el que produzca más columnas. Empatar en 1 columna es válido:
    # un CSV de una sola columna de texto es legítimo.
    mejor, mejor_cols = None, 0
    for sep in (",", ";", "\t", "|"):
        try:
            candidato = pd.read_csv(
                path, sep=sep, engine="python", nrows=MAX_FILAS_CSV,
                encoding="utf-8", on_bad_lines="skip", dtype=str,
            )
        except Exception:
            continue
        if candidato.shape[1] > mejor_cols:
            mejor, mejor_cols = candidato, candidato.shape[1]
    df = mejor
 
    if df is None or df.empty:
        raise DocumentoSinTexto(f"{path.name}: CSV vacío o ilegible")
 
    lineas = [f"{path.stem}. Columnas: {', '.join(str(c) for c in df.columns)}."]
    for fila in df.itertuples(index=False, name=None):
        linea = _fila_a_texto(df.columns, fila)
        if linea:
            lineas.append(linea + ".")
 
    texto = normalizar("\n".join(lineas))
    if contar_palabras(texto) < MIN_PALABRAS_DOC:
        raise DocumentoSinTexto(f"{path.name}: CSV sin filas de datos aprovechables")
    return texto


# Dispatch

EXTRACTORES: dict[str, Callable[[Path], str]] = {
    "pdf": extraer_pdf,
    "html": extraer_html,
    "csv": extraer_csv,
}
 
TITULADORES: dict[str, Callable[[Path], "str | None"]] = {
    "pdf": titulo_pdf,
    "html": titulo_html,
}
 
 
def extraer(path: Path | str, formato: str) -> str:
    """Punto de entrada único. `formato` viene del manifiesto."""
    path = Path(path)
    extractor = EXTRACTORES.get(formato)
    if extractor is None:
        raise DocumentoSinTexto(
            f"{path.name}: formato {formato!r} sin extractor. Disponibles: {sorted(EXTRACTORES)}"
        )
    return extractor(path)
 
 
def titulo(path: Path | str, formato: str) -> str | None:
    """Título del documento si el formato lo expone. Alimenta `texto_embed`."""
    titulador = TITULADORES.get(formato)
    return titulador(Path(path)) if titulador else None
 
