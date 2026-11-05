# Harness de Razonamiento sobre Bases de Datos SQL

IIC-3694 Inteligencia Artificial Generativa — Desafío de Investigación 2026.

Entorno de entrenamiento (harness) + dataset para que un LLM pequeño (~1-2B, familia Qwen)
aprenda a responder preguntas complejas consultando una base de datos: el modelo genera una
consulta SQL, el harness la ejecuta y el modelo usa el resultado para responder. La recompensa
sigue la interfaz común del curso: `reward(prompt, completion) -> float`.

## Estructura del repositorio

```
desafio/
├── harness/            # entorno, prompt, parser, ejecutor SQL, verificador y reward()
├── generator/          # esquema, datos sintéticos, SQL canónico y preguntas en lenguaje natural
├── dataset/            # train.jsonl / eval.jsonl / test.jsonl (generados por código)
├── evaluation/         # línea base pass@1 / pass@k, métricas y robustez del verificador
├── training_data/      # trazas SFT y pares DPO derivados con el verificador
├── tests/              # pruebas unitarias (incluye casos adversariales)
├── db/                 # schema.sql (DDL generado) y hospital.sqlite (datos, versionado)
├── outputs/            # resultados de experimentos (ignorado por git)
├── docs/               # enunciados del desafío y del proyecto
└── requirements.txt
```

## Instalación

Sin dependencias externas: la base de datos es SQLite (incluido en Python).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  |  source .venv/bin/activate en Linux/macOS
pip install -r requirements.txt
pytest -q
```

El archivo `.env` es opcional (ver `.env.example`); solo hace falta para apuntar a modelos
distintos en Ollama o cambiar límites de la herramienta SQL.

## Dominio: hospital

Siete tablas: `departments`, `doctors` (con `supervisor_id` recursivo), `patients`, `appointments`,
`admissions`, `medications` y `prescriptions` (N:M entre consultas y medicamentos). El esquema se
declara en `generator/schema_generator.py` y desde ahí se emite el DDL:

```bash
python -m generator.schema_generator      # escribe db/schema.sql
python -m generator.data_generator        # escribe db/hospital.sqlite (determinista, seed 2026)
```

Los datos son sintéticos y deterministas (~8.800 filas: 10 departamentos, 60 médicos, 400 pacientes,
3.000 consultas, 500 hospitalizaciones, 42 medicamentos, ~2.800 recetas). `db/hospital.sqlite` se
versiona en git para que el ground truth sea reproducible; `tests/test_generator.py` comprueba que
coincide con lo que produce el generador. Incluye correlaciones deliberadas (especialidad ↔
diagnóstico ↔ medicamento, Pediatría atiende solo menores, jerarquía de supervisión con anomalías
salariales, pacientes y medicamentos sin registros asociados) para que las preguntas con varias
condiciones, self-joins y anti-joins tengan respuestas no triviales.

## Dataset

```bash
python -m generator.question_generator    # escribe dataset/{train,eval,test}.jsonl (2.500 tareas)
python -m generator.query_generator       # diagnóstico: capacidad de instancias por plantilla
```

Las tareas se generan a partir de 62 plantillas SQL composicionales con slots muestreados desde la
base (valores categóricos, nombres, umbrales, períodos) y 3 paráfrasis en español por plantilla.
Cada instancia se ejecuta sobre `db/hospital.sqlite` y se acepta solo si está bien formada: resultado
no vacío y sin NULL, números distintos de 0, listas/tablas de 1 a 8 filas, sin empates en la
frontera de rankings y booleanos balanceados. `expected_result` siempre se obtiene ejecutando
`gold_sql`; nunca se escribe a mano.

| Nivel  | Construcciones                                                                          | Plantillas |
| ------ | --------------------------------------------------------------------------------------- | ---------- |
| easy   | 1 tabla, filtros múltiples, agregación simple, EXISTS                                   | 12         |
| medium | 1 JOIN y/o GROUP BY, HAVING simple, IN, ORDER BY + LIMIT 1                              | 13         |
| hard   | 2+ JOINs, HAVING con varios agregados, subconsultas, INTERSECT, top-k, self-join simple | 16         |
| expert | anti-join, subconsulta correlacionada, CTE, comparación contra agregados, CASE, fechas  | 21         |

Estructura de una instancia:

```json
{
  "id": "hosp_1730",
  "database": "hospital.sqlite",
  "difficulty": "expert",
  "template_id": "top_k_patients_total_stay_city",
  "features": ["join", "group_by", "order_limit", "date_arith"],
  "question": "¿Cuáles son los 4 pacientes de Temuco con más días acumulados de hospitalización (solo altas registradas)? Indica nombre y total de días, de mayor a menor.",
  "gold_sql": "SELECT p.full_name, CAST(SUM(julianday(ad.discharge_date) - julianday(ad.admission_date)) AS INTEGER) AS days FROM admissions ad JOIN patients p ON ... ORDER BY days DESC LIMIT 4",
  "answer_type": "table",
  "ordered": true,
  "result_columns": ["full_name", "total_days"],
  "expected_result": [
    ["Constanza Reyes Martínez", 90],
    ["Andrea Soto Fernández", 29],
    ["Magdalena Flores Orellana", 25],
    ["Andrés Vergara Martínez", 24]
  ],
  "template_heldout": false
}
```

`answer_type` ∈ {`number`, `text`, `boolean`, `list`, `table`}; `ordered` indica si el orden importa
(rankings). Particiones: train 1.565 / eval 333 / test 602. El test incluye 333 instancias de
plantillas vistas en train y 269 de 9 plantillas reservadas (`template_heldout: true`) para medir
generalización a estructuras SQL no vistas.

## Uso de la interfaz común

```python
from harness import reward
score = reward(prompt, completion)   # 1.0 si la respuesta final es correcta, 0.0 en otro caso
```

El `prompt` contiene una etiqueta `[TASK_ID: ...]` que permite al harness recuperar el ground truth
de `dataset/*.jsonl`. La `completion` debe incluir la consulta en `<sql>...</sql>` y la respuesta
final en `<answer>...</answer>`.

## Estado

Esquema, datos y dataset listos. Pendiente: implementación del harness (parser, ejecutor,
verificador, reward), línea base y datos derivados.
