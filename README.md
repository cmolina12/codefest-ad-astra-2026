# CODEFEST Ad Astra 2026 — Etapa 1

**Reto:** Construcción de la Base de Conocimiento | **Organizan:** Universidad de los Andes · Fuerza Aeroespacial Colombiana | **EQUIPO:** Talon Systems

## Equipo

| Nombre | Código |
|---|---|
| Daniel Benavides | d.benavidess@uniandes.edu.co |
| Samuel Rozen | s.rozen@uniandes.edu.co |
| Camilo Molina | c.molinap@uniandes.edu.co |
| Andrés Felipe Alfonso | a.alfonsog@uniandes.edu.co |

## Descripción

Sistema de recuperación de información sobre un corpus de fuentes abiertas provisto por la organización. Recibe una pregunta en lenguaje natural y devuelve los fragmentos de texto y los documentos que mejor responden a esa pregunta: no genera texto ni redacta respuestas, únicamente busca y ordena por relevancia.

El corpus cubre tres fenómenos—inteligencia artificial en el sector defensa, seguridad espacial y congestión de la órbita baja terrestre, y dinámicas territoriales en América Latina—en archivos PDF, HTML, JSON, CSV, XLSX, imágenes y PBF, en español, inglés y portugués.

## Fases

**Fase 1 — Extracción de texto.** Se abre cada archivo del corpus según su formato y se recupera su contenido textual donde hay un extractor distinto por tipo de archivo, con sus propias particularidades (OCR para imágenes, recorrido de capas para PBF, eliminación de etiquetas para HTML).

**Fase 2 — Limpieza y normalización.** Se depura el texto eliminando lo que no aporta información—encabezados, pies de página, *boilerplate* de sitios web—y se detecta el idioma predominante de cada documento.

**Fase 3 — Fragmentación (*chunking*).** Se parte cada documento en fragmentos que puedan buscarse por separado.

**Fase 4 — Codificación semántica.** Cada fragmento se convierte en un vector numérico mediante un modelo *encoder*, de modo que fragmentos con significados parecidos queden representados por vectores cercanos.

**Fase 5 — Indexación y almacenamiento.** Los vectores se guardan en un índice FAISS que permite encontrar rápidamente los más parecidos a una consulta, junto con un archivo de metadata que conserva la información de cada fragmento y su documento de origen.

**Fase 6 — Recuperación.** Se convierte la consulta en vector con el mismo modelo de la Fase 4, se buscan los fragmentos más similares y se arma la respuesta: los 10 fragmentos más relevantes y los 3 documentos más relevantes.

**Fase 7 — Evaluación.** Se mide la calidad de la búsqueda con NDCG@10 sobre fragmentos y F1@3 sobre documentos; como la organización no publica las respuestas correctas durante el reto, incluye construir un conjunto propio de consultas de prueba.

**Fase bonus — Grafo de conocimiento.** Componente opcional con puntaje adicional: extraer entidades (países, organizaciones, tecnologías) y las relaciones entre ellas para complementar la búsqueda vectorial.

## Estructura del repositorio

├── README.md
├── requirements.txt
├── src/
│   ├── extraction/        ← Fase 1
│   ├── limpieza.py        ← Fase 2
│   ├── chunking.py        ← Fase 3
│   ├── encoders.py        ← Fase 4
│   ├── indexado.py        ← Fase 5
│   ├── recuperacion.py    ← Fase 6
│   └── evaluacion.py      ← Fase 7
├── scripts/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
└── tests/
