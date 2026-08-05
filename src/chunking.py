import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

OBJETIVO_PALABRAS = 140
PRESUPUESTO_SALIDA = 235
LIMITE_DURO = 250

_RE_FALLBACK = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÑÜ¿¡«\"(\d])")

_ABREVIATURAS = {
    "p.", "ej.", "etc.", "vs.", "art.", "núm.", "no.", "fig.", "cap.", "pág.",
    "ss.", "aprox.", "ed.", "eds.", "vol.", "op.", "cit.", "cf.", "i.e.", "e.g.",
    "ca.", "km.", "ha.", "hab.", "sr.", "sra.", "dr.", "dra.", "prof.", "ing.",
}


def contar_palabras(texto: str) -> int:
    return len(texto.split())

#: pysbd NO soporta portugués: `Segmenter(language='pt')` lanza ValueError.
IDIOMAS_PYSBD = {"es": "es", "pt": "es", "en": "en"}


_identificador = None
_segmentadores: dict = {}
 

def detectar_idioma(texto: str) -> str:
    """Devuelve 'es', 'pt' o 'en'.
 
    Llamar UNA VEZ por documento sobre el texto completo. Por debajo de ~8
    palabras la detección deja de ser fiable en cualquier herramienta: sobre
    'Relatório anual 2024' no hay señal.
 
    Se restringe a los tres idiomas del corpus a propósito: sin restringir,
    py3langid clasifica texto gallego o catalán como tal, y ese código no
    existe en IDIOMAS_PYSBD.
    """
    global _identificador
    if not texto.strip():
        return "es"
    if _identificador is None:
        try:
            from py3langid.langid import MODEL_FILE, LanguageIdentifier
        except ImportError as exc:
            raise ImportError(
                "py3langid es obligatorio: el idioma decide las reglas de segmentación y, "
                "por tanto, las fronteras de los chunks. Instala la versión pinneada."
            ) from exc
        _identificador = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
        _identificador.set_languages(list(IDIOMAS_PYSBD))
    return _identificador.classify(texto)[0]
 
 

def _segmentador(idioma: str):
    """Cachea el Segmenter: compilar las reglas de pysbd en cada llamada es caro."""
    codigo = IDIOMAS_PYSBD.get(idioma)
    if codigo is None:
        raise ValueError(
            f"idioma '{idioma}' no previsto. El corpus es es/pt/en; "
            f"añádelo a IDIOMAS_PYSBD mapeándolo a uno de {sorted(set(IDIOMAS_PYSBD.values()))}"
        )
    if codigo not in _segmentadores:
        try:
            import pysbd
        except ImportError as exc:
            raise ImportError(
                "pysbd es obligatorio: sin él las fronteras de oración cambian, y con ellas "
                "`posicion` y `chunk_id`, que dejarían de corresponder a los vectores del índice "
                "FAISS de forma silenciosa. Instala la versión pinneada de requirements.txt. "
                "Para pruebas aisladas: segmentar(..., permitir_regex=True)."
            ) from exc
        _segmentadores[codigo] = pysbd.Segmenter(language=codigo, clean=False)
    return _segmentadores[codigo]
 


def segmentar(texto: str, idioma: Optional[str] = None, permitir_regex: bool = False) -> List[str]:
    """Divide en oraciones. Si `idioma` es None se detecta sobre este texto.
 
    Pasar siempre el idioma del DOCUMENTO cuando se conozca: detectarlo por
    fragmento es poco fiable en títulos, tablas y listas.
    """
    texto = texto.strip()
    if not texto:
        return []
 
    if permitir_regex:
        crudas = _RE_FALLBACK.split(texto)
    else:
        crudas = _segmentador(idioma or detectar_idioma(texto)).segment(texto)
 
    # Reunir lo que pysbd partió por abreviatura, y las líneas sueltas de
    # tablas y viñetas que quedan sin puntuación final.
    oraciones: List[str] = []
    for cruda in crudas:
        s = cruda.strip()
        if not s:
            continue
        if oraciones and oraciones[-1].split()[-1].lower() in _ABREVIATURAS:
            oraciones[-1] = f"{oraciones[-1]} {s}"
        else:
            oraciones.append(s)
    return oraciones
 
 
def trocear_oracion_larga(oracion: str, maximo: int = PRESUPUESTO_SALIDA) -> List[str]:
    """Parte una 'oración' que por sí sola excede el presupuesto.
 
    Pasa con tablas mal extraídas de PDF y con filas de CSV serializadas.
    Corta por punto y coma, luego por coma, y como último recurso por ventana
    de palabras: preferimos un corte feo a un fragmento descalificado.
    """
    if contar_palabras(oracion) <= maximo:
        return [oracion]
 
    for separador in (";", ","):
        # El separador se queda pegado a la izquierda: `texto` no se altera.
        crudas = oracion.split(separador)
        partes = [p + separador for p in crudas[:-1]] + [crudas[-1]]
        partes = [p.strip() for p in partes if p.strip()]
        if len(partes) > 1:
            resultado: List[str] = []
            for p in partes:
                resultado.extend(trocear_oracion_larga(p, maximo))
            return resultado
 
    palabras = oracion.split()
    return [" ".join(palabras[i : i + maximo]) for i in range(0, len(palabras), maximo)]


def agrupar(
    oraciones: Sequence[str],
    objetivo: int = OBJETIVO_PALABRAS,
    minimo: int = 25,
) -> List[str]:
    """Agrupa oraciones en chunks de ~`objetivo` palabras, sin solape.
 
    Cierra el chunk en cuanto alcanza el objetivo; nunca corta una oración por
    dentro. Los restos por debajo de `minimo` se absorben en el chunk vecino:
    un chunk de una palabra ocupa un vector del índice y no puede emparejar
    con ninguna consulta.
    """
    chunks: List[str] = []
    actual: List[str] = []
    n = 0
 
    for oracion in oraciones:
        for pieza in trocear_oracion_larga(oracion):
            p = contar_palabras(pieza)
            if actual and n + p > objetivo:
                chunks.append(" ".join(actual))
                actual, n = [], 0
            actual.append(pieza)
            n += p
 
    if actual:
        chunks.append(" ".join(actual))
    if not chunks:
        return []
 
    # Absorber huérfanos en ambos extremos y en el medio. El primer chunk solo
    # puede fusionarse hacia delante, así que un `agrupar` que solo mire la
    # cola deja títulos sueltos como 'ancho.' convertidos en chunk propio.
    fusionados: List[str] = [chunks[0]]
    for pieza in chunks[1:]:
        if contar_palabras(fusionados[-1]) < minimo or contar_palabras(pieza) < minimo:
            candidato = f"{fusionados[-1]} {pieza}"
            if contar_palabras(candidato) <= LIMITE_DURO:
                fusionados[-1] = candidato
                continue
        fusionados.append(pieza)
 
    if len(fusionados) > 1 and contar_palabras(fusionados[-1]) < minimo:
        cola = fusionados.pop()
        if contar_palabras(f"{fusionados[-1]} {cola}") <= LIMITE_DURO:
            fusionados[-1] = f"{fusionados[-1]} {cola}"
        else:
            fusionados.append(cola)
 
    return fusionados



def expandir(
    texto: str,
    antes: str = "",
    despues: str = "",
    presupuesto: int = PRESUPUESTO_SALIDA,
    idioma: Optional[str] = None,
) -> str:
    """Crece el fragmento con texto de los chunks vecinos hasta el presupuesto.
 
    `antes` y `despues` son el campo `texto` de los chunks en `posicion` ± 1
    del mismo documento. Se añaden ORACIÓN a oración —no chunk a chunk— porque
    un vecino entero casi nunca cabe: 140 + 140 > 235. Alterna hacia atrás y
    hacia delante para centrar el contexto y se detiene cuando la siguiente
    oración no cabe completa, que es lo que preserva la completitud (§3.3).
    """
    izq = segmentar(antes, idioma)[::-1]   # de la más cercana a la más lejana
    der = segmentar(despues, idioma)
    total = contar_palabras(texto)
    tomadas_izq: List[str] = []
    tomadas_der: List[str] = []
    i = j = 0
 
    while i < len(izq) or j < len(der):
        creció = False
        if j < len(der) and total + contar_palabras(der[j]) <= presupuesto:
            tomadas_der.append(der[j])
            total += contar_palabras(der[j])
            j += 1
            creció = True
        if i < len(izq) and total + contar_palabras(izq[i]) <= presupuesto:
            tomadas_izq.insert(0, izq[i])
            total += contar_palabras(izq[i])
            i += 1
            creció = True
        if not creció:
            break
 
    return " ".join([*tomadas_izq, texto, *tomadas_der])
 
 
def expandir_desde(chunks: Sequence["Chunk"], i: int,
                   presupuesto: int = PRESUPUESTO_SALIDA) -> str:
    """Texto de salida para `chunks[i]`, expandido con sus vecinos.
 
    Lo que consume `generador.py`. `chunks` deben ser los del MISMO documento
    ordenados por `posicion`: la spec solo permite concatenar con el fragmento
    inmediatamente anterior o posterior del mismo documento (§9.2.1).
    """
    antes = chunks[i - 1].texto if i > 0 else ""
    despues = chunks[i + 1].texto if i + 1 < len(chunks) else ""
    return expandir(chunks[i].texto, antes, despues, presupuesto, chunks[i].idioma)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    fuente: str        # nombre EXACTO del archivo de ADL: clave del ground truth
    formato: str
    fenomeno: int
    posicion: int      # empieza en 0
    num_tokens: int
    texto: str         # sin modificaciones
    idioma: Optional[str] = None
    titulo_doc: Optional[str] = None


    def to_dict(self) -> dict:
        d = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "fuente": self.fuente,
            "formato": self.formato,
            "fenomeno": self.fenomeno,
            "posicion": self.posicion,
            "num_tokens": self.num_tokens,
            "texto": self.texto,
        }
        if self.idioma:
            d["idioma"] = self.idioma
        if self.titulo_doc:
            d["titulo_doc"] = self.titulo_doc
        return d


    @property
    def texto_embed(self) -> str:
        """Lo que se le pasa al encoder; `texto` se persiste sin tocar."""
        return f"{self.titulo_doc} | {self.texto}" if self.titulo_doc else self.texto


class DocumentoSinTexto(Exception):
    """El documento no produjo texto indexable.
 
    Casi siempre: PDF escaneado sin capa de texto, o imagen sin OCR. Es un
    error y no un aviso porque un documento con 0 chunks no está en el índice
    y es IRRECUPERABLE: si el ground truth lo marca relevante para alguna
    consulta, ese F1@3 se pierde y nada en el pipeline lo delataría.
    """


def documento_a_chunks(
    texto: str,
    *,
    doc_id: str,
    fuente: str,
    formato: str,
    fenomeno: int,
    idioma: Optional[str] = None,
    titulo_doc: Optional[str] = None,
    objetivo: int = OBJETIVO_PALABRAS,
    contar_tokens: Optional[Callable[[str], int]] = None,
) -> List[Chunk]:
    """Texto limpio de un documento -> lista de Chunk lista para indexar.
 
    Lanza DocumentoSinTexto si no hay texto suficiente: hay que resolverlo en
    la ingesta (OCR, otro extractor), no dejar el documento fuera del índice.
    """
    if not texto.strip():
        raise DocumentoSinTexto(
            f"{fuente!r}: texto vacío. Cada extractor decide su propio umbral mínimo "
            "según el formato; aquí solo se rechaza la cadena vacía."
        )
    tok = contar_tokens or contar_palabras
    idioma = idioma or detectar_idioma(texto)   # una sola vez, sobre el documento entero
    piezas = agrupar(segmentar(texto, idioma), objetivo=objetivo)
 
    # Garantía dura: ningún chunk puede superar el límite de la spec. Si esto
    # salta, el troceo de oraciones largas dejó pasar algo y hay que arreglarlo
    # ahí, no truncar aquí (truncar cortaría una oración por la mitad).
    desbordados = [(i, contar_palabras(p)) for i, p in enumerate(piezas)
                   if contar_palabras(p) > LIMITE_DURO]
    if desbordados:
        raise AssertionError(
            f"{fuente!r}: chunks por encima de {LIMITE_DURO} palabras en las posiciones "
            f"{desbordados[:5]}. Revisar trocear_oracion_larga."
        )
 
    return [
        Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}-chunk-{i:04d}",
            fuente=fuente,
            formato=formato,
            fenomeno=fenomeno,
            posicion=i,
            num_tokens=tok(p),
            texto=p,
            idioma=idioma,
            titulo_doc=titulo_doc,
        )
        for i, p in enumerate(piezas)
    ]
