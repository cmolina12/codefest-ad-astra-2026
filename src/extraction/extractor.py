import json
import re 
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable

from chunking import DocumentoSinTexto, contar_palabras, detectar_idioma

# Umbrales
MIN_PALABRAS_DOC = 15

MIN_PALABRAS_HTML = 8

MIN_PALABRAS_IMAGEN = 6

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


# XLSX

def extraer_xlsx(path: Path) -> str:
    """Una pasada por hoja: cabecera primero, luego fila a fila.
 
    `read_only=True` es obligatorio, no una optimización: openpyxl en modo
    normal carga la hoja entera en memoria y los datasets del AI Index tienen
    decenas de miles de filas.
 
    `data_only=True` devuelve el valor calculado de las fórmulas en vez de
    '=SUM(A1:A10)', que no es indexable.
    """
    import openpyxl
 
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise DocumentoSinTexto(f"{path.name}: no se pudo abrir como XLSX ({exc})") from exc
 
    bloques: list[str] = []
    for hoja in wb.worksheets:
        filas = hoja.iter_rows(values_only=True)
        encabezados = _primera_fila_util(filas)
        if encabezados is None:
            continue
        # El nombre de la hoja es contexto real: 'Migración interna' dice más
        # que los propios encabezados en muchos datasets.
        lineas = [f"{path.stem} — {hoja.title}. "
                  f"Columnas: {', '.join(str(h) for h in encabezados)}."]
        for _, fila in zip(range(MAX_FILAS_CSV), filas):
            linea = _fila_a_texto(encabezados, fila)
            if linea:
                lineas.append(linea + ".")
        if len(lineas) > 1:
            bloques.append("\n".join(lineas))
    wb.close()
 
    texto = normalizar("\n\n".join(bloques))
    if contar_palabras(texto) < MIN_PALABRAS_DOC:
        raise DocumentoSinTexto(f"{path.name}: XLSX sin filas de datos aprovechables")
    return texto
 
 
def _primera_fila_util(filas):
    """Salta las filas de título y decorativas que preceden a la cabecera real.
 
    Es habitual que un XLSX empiece con 'Informe anual 2024' en A1 y varias
    filas vacías antes de los encabezados; tomar A1 como cabecera produciría
    una única columna y todo el dataset se serializaría mal.
    """
    for _ in range(20):
        fila = next(filas, None)
        if fila is None:
            return None
        no_vacias = [c for c in fila if c is not None and str(c).strip()]
        if len(no_vacias) >= 2:
            return [h if h is not None and str(h).strip() else f"col{i}"
                    for i, h in enumerate(fila)]
    return None


# JSON
_CLAVES_CUERPO = ("content", "contenido", "text", "texto", "body", "cuerpo",
                  "articlebody", "article_body", "full_text", "fulltext",
                  "description", "descripcion", "descrição", "summary",
                  "resumen", "resumo", "abstract", "extract")


_CLAVES_RUIDO = ("id", "_id", "uuid", "guid", "hash", "checksum", "etag",
                 "url", "link", "href", "src", "image", "imagen", "thumbnail",
                 "slug", "path", "filename", "encoding", "mimetype", "base64")

_RE_RUIDO_VALOR = re.compile(r"^(?:https?://|data:|[0-9a-f]{32,}$)", re.IGNORECASE)


def _es_ruido(clave: str, valor: str) -> bool:
    c = clave.lower()
    return (any(c == r or c.endswith("_" + r) for r in _CLAVES_RUIDO)
            or bool(_RE_RUIDO_VALOR.match(valor.strip())))


def extraer_json(path: Path) -> str:
    """Aplana un JSON a texto, separando cuerpo de metadatos.
 
    El corpus tiene 954 JSON de 16 observatorios distintos, cada uno con su
    propio esquema. En vez de un adaptador por fuente, se aprovecha lo que
    casi todos comparten: una o varias claves de contenido con la prosa
    (`content`, `body`, `description`...) y el resto metadatos.
 
    El cuerpo se emite tal cual —es texto natural y el encoder lo aprovecha
    mejor sin prefijos— y los metadatos como 'clave: valor', que es lo que les
    da sentido ('2024' no dice nada; 'año: 2024' sí).
    """
    try:
        datos = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise DocumentoSinTexto(f"{path.name}: JSON mal formado ({exc.msg} línea {exc.lineno})") from exc
 
    cuerpo: list[str] = []
    metadatos: list[str] = []
 
    def recorrer(nodo, ruta: str = "") -> None:
        clave = ruta.split(".")[-1].split("[")[0]
        if isinstance(nodo, dict):
            for k, v in nodo.items():
                recorrer(v, f"{ruta}.{k}" if ruta else str(k))
        elif isinstance(nodo, list):
            # Lista de escalares: una sola línea, para no fragmentar
            # ['residuos', 'órbita'] en tres entradas sin contexto.
            if nodo and all(not isinstance(x, (dict, list)) for x in nodo):
                vals = ", ".join(str(x).strip() for x in nodo if str(x).strip())
                if vals and not _es_ruido(clave, vals):
                    metadatos.append(f"{clave.replace('_', ' ')}: {vals}")
            else:
                for i, x in enumerate(nodo):
                    recorrer(x, f"{ruta}[{i}]")
        elif nodo is not None and isinstance(nodo, (str, int, float, bool)):
            valor = str(nodo).strip()
            if not valor or _es_ruido(clave, valor):
                return
            if clave.lower() in _CLAVES_CUERPO and len(valor.split()) >= 5:
                cuerpo.append(valor)
            else:
                metadatos.append(f"{clave.replace('_', ' ')}: {valor}")
 
    recorrer(datos)
 
    # Metadatos primero (son el título, la fecha, el autor) y luego la prosa.
    partes = [". ".join(m.rstrip(".") for m in metadatos)] if metadatos else []
    partes.extend(cuerpo)
    texto = normalizar("\n\n".join(p for p in partes if p.strip()))
 
    if contar_palabras(texto) < MIN_PALABRAS_DOC:
        raise DocumentoSinTexto(
            f"{path.name}: {contar_palabras(texto)} palabras tras aplanar. "
            "¿Esquema no previsto? Revisar las claves de este observatorio."
        )
    return texto
 
 
def titulo_json(path: Path) -> str | None:
    """Primer título encontrado en cualquier nivel.
 
    Se recorre en anchura porque los esquemas anidan de formas distintas: unos
    lo ponen en la raíz y otros bajo `data.articulo.titulo`.
    """
    try:
        datos = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
 
    claves = ("title", "titulo", "título", "headline", "name", "nombre", "titre")
    cola = [datos]
    while cola:
        nodo = cola.pop(0)
        if isinstance(nodo, dict):
            for c in claves:
                valor = nodo.get(c)
                if isinstance(valor, str) and len(valor.split()) >= 2:
                    return normalizar(valor)
            cola.extend(v for v in nodo.values() if isinstance(v, (dict, list)))
        elif isinstance(nodo, list):
            cola.extend(v for v in nodo if isinstance(v, (dict, list)))
    return None


# Imagenes

_lector_easyocr = None


def _ocr_easyocr(path: Path) -> str:
    """OCR con easyocr: una sola pasada para es+pt+en.
 
    Medido contra tesseract sobre una figura con texto disperso (ejes,
    etiquetas de barras, pie): easyocr CER 0.000, tesseract 0.010. En texto
    corrido empatan. easyocr es ~8x más lento y descarga 95 MB de modelos,
    pero el corpus solo tiene 8 imágenes, así que el tiempo es irrelevante y
    la calidad en layout disperso decide.
 
    NO se usa para las páginas de PDF escaneadas: ahí son miles de páginas y
    8 s/página frente a 1 s/página cambia días por horas.
    """
    global _lector_easyocr
    if _lector_easyocr is None:
        import easyocr
 
        _lector_easyocr = easyocr.Reader(["es", "pt", "en"], gpu=False, verbose=False)
    return " ".join(_lector_easyocr.readtext(str(path), detail=0))
 
 
def extraer_imagen(path: Path, motor: str = "tesseract") -> str:
    """OCR más el nombre del archivo como contexto.
 
    El nombre suele ser la única señal fiable en figuras y portadas
    ('grafico_residuos_orbitales_2024.png' aporta cuatro términos indexables),
    así que encabeza el texto aunque el OCR falle.
 
    `motor`: 'tesseract' (por defecto, sin dependencias extra) o 'easyocr'
    (mejor en figuras con texto disperso; ver `_ocr_easyocr`).
 
    Umbral más bajo que el resto: una figura con título y fuente es contenido
    legítimo y exigirle 15 palabras la dejaría fuera del índice.
    """
    contexto = re.sub(r"[_\-]+", " ", path.stem).strip()
    try:
        if motor == "easyocr":
            texto = _ocr_easyocr(path)
        elif motor == "tesseract":
            texto = _ocr_imagen(path.read_bytes(), None)
        else:
            raise ValueError(f"motor OCR desconocido: {motor!r}. Usa 'tesseract' o 'easyocr'.")
    except ValueError:
        raise
    except Exception as exc:
        raise DocumentoSinTexto(f"{path.name}: OCR falló ({exc})") from exc
 
    completo = normalizar(f"{contexto}. {texto}" if texto.strip() else contexto)
    if contar_palabras(completo) < MIN_PALABRAS_IMAGEN:
        raise DocumentoSinTexto(
            f"{path.name}: {contar_palabras(completo)} palabras entre OCR y nombre. "
            "¿Imagen decorativa sin texto?"
        )
    return completo


# PBF

_TAGS_TIPO = ("place", "boundary", "natural", "waterway", "landuse",
              "amenity", "tourism", "highway")
 
 
def extraer_pbf(path: Path, max_nombres: int = 3000) -> str:
    """Topónimos de un PBF de OpenStreetMap.
 
    Los nombres de lugares son lo único de un mapa que puede emparejar con una
    consulta en lenguaje natural: una consulta sobre dinámicas territoriales en
    la Amazonía puede coincidir con los municipios y ríos que contiene.
 
    osmium y no pyrosm/quackosm: aquellos construyen GeoDataFrames y arrastran
    geopandas, shapely y pyarrow (156 MB) para dar geometrías que no se usan;
    quackosm además descarga una extensión de DuckDB en tiempo de ejecución.
    """
    import osmium
 
    nombres: dict[str, str] = {}
    try:
        for obj in osmium.FileProcessor(str(path)):
            nombre = obj.tags.get("name")
            if not nombre or nombre in nombres:
                continue
            tipo = next((obj.tags.get(t) for t in _TAGS_TIPO if obj.tags.get(t)), "lugar")
            nombres[nombre] = tipo
            if len(nombres) >= max_nombres:
                break
    except Exception as exc:
        raise DocumentoSinTexto(f"{path.name}: no se pudo leer el PBF ({exc})") from exc
 
    if not nombres:
        raise DocumentoSinTexto(f"{path.name}: PBF sin elementos con nombre")
 
    por_tipo: dict[str, list[str]] = {}
    for nombre, tipo in nombres.items():
        por_tipo.setdefault(tipo, []).append(nombre)
 
    contexto = re.sub(r"[_\-]+", " ", path.stem).strip()
    lineas = [f"Datos geográficos de {contexto}. {len(nombres)} elementos con nombre."]
    for tipo, lista in sorted(por_tipo.items(), key=lambda kv: -len(kv[1])):
        lineas.append(f"{tipo}: {', '.join(sorted(lista))}.")
    return normalizar("\n".join(lineas))


# Texto plano

def extraer_texto_plano(path: Path) -> str:
    """`.md` y `.txt` se leen tal cual.
 
    Nada de pymupdf4llm aquí: esa librería convierte PDF *a* markdown, y al
    pasarle un .md lo renderiza y lo devuelve alterado (añade negritas, borra
    las URL de los enlaces). El campo `texto` debe ir sin modificaciones.
    """
    texto = normalizar(path.read_text(encoding="utf-8", errors="replace"))
    if contar_palabras(texto) < MIN_PALABRAS_IMAGEN:
        raise DocumentoSinTexto(f"{path.name}: archivo de texto casi vacío")
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
 
