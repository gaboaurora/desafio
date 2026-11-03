"""
Definición y emisión del esquema relacional del entorno (dominio: hospital).

El esquema se declara una vez como estructura Python (tablas, columnas, tipos, FKs, dominios de
valores) y desde ahí se derivan:
  - El DDL para SQLite (db/schema.sql).
  - La descripción compacta que se inserta en el prompt (prompt_builder).
  - La metadata que usan data_generator (dominios de valores) y query_generator (joins válidos).

Diseño del dominio (7 tablas):

    departments ──< doctors >── doctors (supervisor_id, relación recursiva)
         │             │
         │             └──< appointments >── patients
         │                      │
         │                      └──< prescriptions >── medications   (N:M)
         │
         └──< admissions >── patients, doctors

Propiedades buscadas para las tareas:
  - Relación recursiva (doctors.supervisor_id) => self-joins.
  - N:M (appointments <-> medications vía prescriptions) => joins de 3+ tablas.
  - Dos caminos distintos entre patients y departments (appointments/doctors vs admissions)
    => el modelo debe elegir el join correcto según la pregunta.
  - Columnas numéricas con unidades naturales (CLP, minutos, días, mg) => agregaciones.
  - Fechas ISO (TEXT 'YYYY-MM-DD') => filtros temporales y aritmética con julianday().
  - Varias categóricas (specialty, status, diagnosis, insurance, blood_type) => GROUP BY / HAVING.

Uso:
    python -m generator.schema_generator          # escribe db/schema.sql
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

ValueDomain = Union[Sequence[str], Tuple[float, float]]


@dataclass
class Column:
    name: str
    sql_type: str                              # INTEGER | REAL | TEXT
    nullable: bool = False
    primary_key: bool = False
    references: Optional[str] = None           # "tabla(columna)"
    description: str = ""
    values: Optional[Sequence[str]] = None     # dominio categórico
    value_range: Optional[Tuple[float, float]] = None   # dominio numérico (min, max)
    unique: bool = False
    check: Optional[str] = None                # expresión CHECK adicional


@dataclass
class Table:
    name: str
    columns: List[Column]
    description: str = ""
    constraints: List[str] = field(default_factory=list)   # constraints a nivel de tabla

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"{self.name}.{name}")

    @property
    def primary_key(self) -> Column:
        return next(c for c in self.columns if c.primary_key)


@dataclass
class Schema:
    schema_id: str
    tables: List[Table]
    description: str = ""

    def table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(name)

    def foreign_keys(self) -> List[Tuple[str, str, str, str]]:
        """Lista (tabla_origen, col_origen, tabla_destino, col_destino)."""
        fks = []
        for t in self.tables:
            for c in t.columns:
                if c.references:
                    target_table, target_col = c.references.rstrip(")").split("(")
                    fks.append((t.name, c.name, target_table, target_col))
        return fks


# ----------------------------------------------------------------------------------------------
# Dominio: Hospital
# ----------------------------------------------------------------------------------------------

SPECIALTIES = [
    "Cardiología", "Pediatría", "Neurología", "Oncología", "Traumatología",
    "Dermatología", "Ginecología", "Medicina Interna", "Psiquiatría", "Urgencias",
]

DEPARTMENT_NAMES = SPECIALTIES  # un departamento por especialidad

CITIES = [
    "Santiago", "Valparaíso", "Concepción", "La Serena", "Antofagasta",
    "Temuco", "Rancagua", "Puerto Montt", "Iquique", "Talca",
]

INSURANCE_TYPES = ["Fonasa", "Isapre", "Particular"]
BLOOD_TYPES = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
SEXES = ["F", "M"]

APPOINTMENT_STATUSES = ["completed", "cancelled", "no_show"]

DIAGNOSES = [
    "Hipertensión", "Diabetes tipo 2", "Migraña", "Asma", "Fractura", "Dermatitis",
    "Arritmia", "Depresión", "Gastritis", "Lumbalgia", "Anemia", "Infección respiratoria",
    "Control sano", "Artrosis", "Epilepsia", "Tumor benigno", "Infección urinaria",
    "Trastorno de ansiedad",
]

ROOM_TYPES = ["general", "intermedio", "UCI"]

MEDICATION_CATEGORIES = [
    "analgésico", "antibiótico", "antihipertensivo", "antidiabético",
    "antiinflamatorio", "antidepresivo", "anticoagulante", "antihistamínico",
    "anticonvulsivo", "suplemento",
]


def build_schema() -> Schema:
    """Devuelve el esquema del dominio hospitalario."""

    departments = Table(
        name="departments",
        description="Departamentos clínicos del hospital.",
        columns=[
            Column("department_id", "INTEGER", primary_key=True, description="Identificador del departamento."),
            Column("name", "TEXT", unique=True, values=DEPARTMENT_NAMES, description="Nombre del departamento (coincide con una especialidad)."),
            Column("floor", "INTEGER", value_range=(1, 8), description="Piso del edificio donde está el departamento."),
            Column("annual_budget", "INTEGER", value_range=(200_000_000, 2_000_000_000), description="Presupuesto anual en pesos chilenos (CLP)."),
        ],
    )

    doctors = Table(
        name="doctors",
        description="Médicos del hospital. Cada médico pertenece a un departamento y puede tener un supervisor (otro médico).",
        columns=[
            Column("doctor_id", "INTEGER", primary_key=True, description="Identificador del médico."),
            Column("full_name", "TEXT", description="Nombre completo."),
            Column("specialty", "TEXT", values=SPECIALTIES, description="Especialidad médica."),
            Column("department_id", "INTEGER", references="departments(department_id)", description="Departamento al que pertenece."),
            Column("supervisor_id", "INTEGER", nullable=True, references="doctors(doctor_id)", description="Médico supervisor (NULL si es jefe)."),
            Column("hire_date", "TEXT", description="Fecha de contratación, formato 'YYYY-MM-DD'."),
            Column("salary", "INTEGER", value_range=(1_800_000, 9_000_000), description="Salario mensual en CLP."),
        ],
    )

    patients = Table(
        name="patients",
        description="Pacientes registrados.",
        columns=[
            Column("patient_id", "INTEGER", primary_key=True, description="Identificador del paciente."),
            Column("full_name", "TEXT", description="Nombre completo."),
            Column("age", "INTEGER", value_range=(0, 95), description="Edad en años cumplidos."),
            Column("sex", "TEXT", values=SEXES, description="Sexo del paciente."),
            Column("blood_type", "TEXT", values=BLOOD_TYPES, description="Grupo sanguíneo."),
            Column("city", "TEXT", values=CITIES, description="Ciudad de residencia."),
            Column("insurance", "TEXT", values=INSURANCE_TYPES, description="Previsión de salud."),
        ],
    )

    appointments = Table(
        name="appointments",
        description="Consultas ambulatorias de un paciente con un médico.",
        columns=[
            Column("appointment_id", "INTEGER", primary_key=True, description="Identificador de la consulta."),
            Column("patient_id", "INTEGER", references="patients(patient_id)", description="Paciente atendido."),
            Column("doctor_id", "INTEGER", references="doctors(doctor_id)", description="Médico que atiende."),
            Column("appointment_date", "TEXT", description="Fecha de la consulta, formato 'YYYY-MM-DD'."),
            Column("duration_minutes", "INTEGER", value_range=(10, 90), description="Duración en minutos."),
            Column("status", "TEXT", values=APPOINTMENT_STATUSES, description="Estado de la consulta."),
            Column("diagnosis", "TEXT", nullable=True, values=DIAGNOSES, description="Diagnóstico registrado (NULL si la consulta no se realizó)."),
            Column("cost", "INTEGER", value_range=(15_000, 120_000), description="Costo de la consulta en CLP."),
        ],
    )

    admissions = Table(
        name="admissions",
        description="Hospitalizaciones (ingresos con cama) de pacientes en un departamento.",
        columns=[
            Column("admission_id", "INTEGER", primary_key=True, description="Identificador de la hospitalización."),
            Column("patient_id", "INTEGER", references="patients(patient_id)", description="Paciente hospitalizado."),
            Column("department_id", "INTEGER", references="departments(department_id)", description="Departamento donde fue hospitalizado."),
            Column("attending_doctor_id", "INTEGER", references="doctors(doctor_id)", description="Médico tratante."),
            Column("admission_date", "TEXT", description="Fecha de ingreso, formato 'YYYY-MM-DD'."),
            Column("discharge_date", "TEXT", nullable=True, description="Fecha de alta, formato 'YYYY-MM-DD' (NULL si sigue hospitalizado)."),
            Column("room_type", "TEXT", values=ROOM_TYPES, description="Tipo de cama."),
            Column("daily_cost", "INTEGER", value_range=(80_000, 900_000), description="Costo diario de la cama en CLP."),
        ],
        constraints=["CHECK (discharge_date IS NULL OR discharge_date >= admission_date)"],
    )

    medications = Table(
        name="medications",
        description="Catálogo de medicamentos.",
        columns=[
            Column("medication_id", "INTEGER", primary_key=True, description="Identificador del medicamento."),
            Column("name", "TEXT", unique=True, description="Nombre comercial o genérico."),
            Column("category", "TEXT", values=MEDICATION_CATEGORIES, description="Categoría terapéutica."),
            Column("unit_price", "INTEGER", value_range=(500, 60_000), description="Precio unitario en CLP."),
            Column("requires_prescription", "INTEGER", values=["0", "1"], description="1 si requiere receta médica, 0 si es de venta libre."),
        ],
    )

    prescriptions = Table(
        name="prescriptions",
        description="Medicamentos recetados en una consulta (relación N:M entre consultas y medicamentos).",
        columns=[
            Column("prescription_id", "INTEGER", primary_key=True, description="Identificador de la receta."),
            Column("appointment_id", "INTEGER", references="appointments(appointment_id)", description="Consulta en la que se recetó."),
            Column("medication_id", "INTEGER", references="medications(medication_id)", description="Medicamento recetado."),
            Column("quantity", "INTEGER", value_range=(1, 60), description="Cantidad de unidades recetadas."),
            Column("duration_days", "INTEGER", value_range=(1, 90), description="Duración del tratamiento en días."),
        ],
        constraints=["UNIQUE (appointment_id, medication_id)"],
    )

    return Schema(
        schema_id="hospital_v1",
        description="Base de datos de gestión de un hospital: departamentos, médicos, pacientes, consultas, hospitalizaciones y recetas.",
        tables=[departments, doctors, patients, appointments, admissions, medications, prescriptions],
    )


# ----------------------------------------------------------------------------------------------
# Emisión
# ----------------------------------------------------------------------------------------------

def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _column_ddl(col: Column) -> str:
    parts = [col.name, col.sql_type]
    if col.primary_key:
        parts.append("PRIMARY KEY")
    elif not col.nullable:
        parts.append("NOT NULL")
    if col.unique:
        parts.append("UNIQUE")
    if col.references:
        parts.append(f"REFERENCES {col.references}")
    if col.values is not None and not col.primary_key:
        literals = ", ".join(_sql_literal(v) if col.sql_type == "TEXT" else v for v in col.values)
        parts.append(f"CHECK ({col.name} IN ({literals}))")
    elif col.value_range is not None:
        lo, hi = col.value_range
        parts.append(f"CHECK ({col.name} BETWEEN {_num(lo)} AND {_num(hi)})")
    if col.check:
        parts.append(f"CHECK ({col.check})")
    return " ".join(parts)


def to_ddl(schema: Schema) -> str:
    """Genera el DDL SQLite a partir del esquema."""
    lines = [
        f"-- Esquema: {schema.schema_id}",
        f"-- {schema.description}",
        "-- Generado por generator/schema_generator.py. No editar a mano.",
        "PRAGMA foreign_keys = ON;",
        "",
    ]
    for t in schema.tables:
        lines.append(f"-- {t.description}")
        lines.append(f"CREATE TABLE {t.name} (")
        body = [f"    {_column_ddl(c)}" for c in t.columns]
        body += [f"    {c}" for c in t.constraints]
        lines.append(",\n".join(body))
        lines.append(");")
        lines.append("")
    # Índices sobre claves foráneas para acelerar los JOINs.
    for src_table, src_col, _, _ in schema.foreign_keys():
        lines.append(f"CREATE INDEX idx_{src_table}_{src_col} ON {src_table}({src_col});")
    lines.append("")
    return "\n".join(lines)


def to_prompt_description(schema: Schema) -> str:
    """
    Descripción compacta del esquema para el prompt del modelo.

    Formato por tabla:
        tabla(col1 TIPO PK, col2 TIPO -> otra(col), ...)
          - col: descripción [valores: a, b, c]
    """
    out = [f"Base de datos SQLite: {schema.description}", ""]
    for t in schema.tables:
        sig = []
        for c in t.columns:
            s = f"{c.name} {c.sql_type}"
            if c.primary_key:
                s += " PK"
            if c.references:
                s += f" -> {c.references}"
            if c.nullable:
                s += " NULL"
            sig.append(s)
        out.append(f"{t.name}({', '.join(sig)})")
        out.append(f"  {t.description}")
        for c in t.columns:
            if c.primary_key or not c.description:
                continue
            line = f"  - {c.name}: {c.description}"
            if c.values is not None and c.sql_type == "TEXT" and len(c.values) <= 20:
                line += f" Valores: {', '.join(c.values)}."
            out.append(line)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_ddl(path: str | Path = "db/schema.sql") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_ddl(build_schema()), encoding="utf-8")
    return path


if __name__ == "__main__":
    out = write_ddl()
    print(f"DDL escrito en {out}")
