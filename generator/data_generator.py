"""
Poblamiento sintético y determinista de la base de datos SQLite (dominio hospital).

Principios:
  - Semilla fija (`RANDOM_SEED`): misma semilla => mismos bytes en db/hospital.sqlite, por lo que el
    ground truth del dataset es reproducible por cualquiera que clone el repo.
  - Volumen pequeño (~8.000 filas en total) para que cada consulta tarde milisegundos durante RL,
    pero con suficiente variedad para producir miles de preguntas distintas por nivel.
  - Correlaciones realistas que vuelven no triviales las preguntas con varias condiciones:
      * especialidad del médico <-> diagnóstico <-> categoría del medicamento recetado,
      * Pediatría atiende solo menores de 16; Ginecología solo mujeres,
      * jerarquía de supervisión de 2-3 niveles con salarios mayormente (no siempre) crecientes,
      * distribución de visitas por paciente de cola pesada (pocos pacientes con muchas consultas),
      * costo diario de cama según tipo de habitación; estadías más largas en UCI.
  - Valores únicos donde la unicidad evita respuestas ambiguas (nombres, presupuestos, precios).
  - Validación final contra los dominios declarados en schema_generator (fuente única de verdad).

Uso:
    python -m generator.data_generator [--out db/hospital.sqlite] [--seed 2026]
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

from generator.schema_generator import (
    BLOOD_TYPES,
    CITIES,
    DEPARTMENT_NAMES,
    INSURANCE_TYPES,
    MEDICATION_CATEGORIES,
    ROOM_TYPES,
    Schema,
    build_schema,
    to_ddl,
)

load_dotenv()

Row = Dict[str, Any]
Tables = Dict[str, List[Row]]


# ----------------------------------------------------------------------------------------------
# Tamaños y constantes del dominio
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Sizes:
    doctors_per_department: tuple[int, int] = (5, 7)     # => ~60 médicos
    patients: int = 400
    appointments: int = 3000
    admissions: int = 500
    current_admissions: int = 20                         # hospitalizados sin alta
    new_doctors: int = 2                                 # contratados hace poco, sin atenciones
    unprescribed_medications: int = 3                    # en catálogo pero nunca recetados


DATE_START = date(2024, 1, 1)
DATE_END = date(2025, 12, 31)       # "hoy" para efectos de la base de datos

FIRST_NAMES_F = [
    "María", "Camila", "Valentina", "Javiera", "Francisca", "Catalina", "Antonia", "Isidora",
    "Fernanda", "Constanza", "Daniela", "Carolina", "Josefa", "Martina", "Sofía", "Florencia",
    "Paula", "Andrea", "Macarena", "Rocío", "Trinidad", "Emilia", "Amanda", "Victoria",
    "Ignacia", "Magdalena", "Renata", "Agustina", "Pilar", "Claudia",
]
FIRST_NAMES_M = [
    "José", "Juan", "Matías", "Benjamín", "Vicente", "Sebastián", "Diego", "Nicolás", "Felipe",
    "Tomás", "Cristóbal", "Joaquín", "Maximiliano", "Agustín", "Gabriel", "Martín", "Lucas",
    "Ignacio", "Francisco", "Pablo", "Rodrigo", "Andrés", "Gonzalo", "Álvaro", "Patricio",
    "Mauricio", "Esteban", "Renato", "Bastián", "Emilio",
]
LAST_NAMES = [
    "González", "Muñoz", "Rojas", "Díaz", "Pérez", "Soto", "Contreras", "Silva", "Martínez",
    "Sepúlveda", "Morales", "Rodríguez", "López", "Fuentes", "Hernández", "Torres", "Araya",
    "Flores", "Espinoza", "Valenzuela", "Castillo", "Tapia", "Reyes", "Gutiérrez", "Castro",
    "Pizarro", "Álvarez", "Vásquez", "Sánchez", "Fernández", "Ramírez", "Carrasco", "Gómez",
    "Cortés", "Herrera", "Núñez", "Jara", "Vergara", "Rivera", "Figueroa", "Riquelme", "Garcia",
    "Miranda", "Bravo", "Vera", "Molina", "Vega", "Campos", "Sandoval", "Orellana",
]

# Diagnósticos plausibles por especialidad (la primera lista pesa más).
DIAGNOSES_BY_SPECIALTY: Dict[str, List[str]] = {
    "Cardiología": ["Hipertensión", "Arritmia"],
    "Pediatría": ["Asma", "Infección respiratoria", "Control sano"],
    "Neurología": ["Migraña", "Epilepsia"],
    "Oncología": ["Tumor benigno", "Anemia", "Control sano"],
    "Traumatología": ["Fractura", "Lumbalgia", "Artrosis"],
    "Dermatología": ["Dermatitis", "Control sano"],
    "Ginecología": ["Control sano", "Infección urinaria", "Anemia"],
    "Medicina Interna": ["Diabetes tipo 2", "Hipertensión", "Gastritis", "Anemia"],
    "Psiquiatría": ["Depresión", "Trastorno de ansiedad"],
    "Urgencias": ["Fractura", "Infección respiratoria", "Gastritis", "Arritmia"],
}

# Categorías de medicamento plausibles por diagnóstico.
CATEGORIES_BY_DIAGNOSIS: Dict[str, List[str]] = {
    "Hipertensión": ["antihipertensivo"],
    "Arritmia": ["anticoagulante", "antihipertensivo"],
    "Diabetes tipo 2": ["antidiabético"],
    "Migraña": ["analgésico", "antiinflamatorio"],
    "Asma": ["antihistamínico", "antiinflamatorio"],
    "Fractura": ["analgésico", "antiinflamatorio"],
    "Dermatitis": ["antihistamínico", "antiinflamatorio"],
    "Depresión": ["antidepresivo"],
    "Trastorno de ansiedad": ["antidepresivo"],
    "Gastritis": ["antibiótico", "analgésico"],
    "Lumbalgia": ["analgésico", "antiinflamatorio"],
    "Anemia": ["suplemento"],
    "Infección respiratoria": ["antibiótico", "antihistamínico"],
    "Infección urinaria": ["antibiótico"],
    "Control sano": ["suplemento"],
    "Artrosis": ["antiinflamatorio", "analgésico"],
    "Epilepsia": ["anticonvulsivo"],
    "Tumor benigno": ["analgésico"],
}
# Probabilidad de recetar 0/1/2/3 medicamentos según diagnóstico (Control sano casi no receta).
N_MEDS_WEIGHTS_DEFAULT = [0.15, 0.45, 0.30, 0.10]
N_MEDS_WEIGHTS_CHECKUP = [0.70, 0.25, 0.05, 0.00]

MEDICATIONS_BY_CATEGORY: Dict[str, List[str]] = {
    "analgésico": ["Paracetamol", "Tramadol", "Metamizol", "Ketorolaco"],
    "antibiótico": ["Amoxicilina", "Azitromicina", "Ciprofloxacino", "Cefadroxilo", "Nitrofurantoína", "Claritromicina"],
    "antihipertensivo": ["Losartán", "Enalapril", "Amlodipino", "Atenolol", "Hidroclorotiazida"],
    "antidiabético": ["Metformina", "Glibenclamida", "Insulina glargina", "Sitagliptina"],
    "antiinflamatorio": ["Ibuprofeno", "Diclofenaco", "Naproxeno", "Celecoxib", "Prednisona"],
    "antidepresivo": ["Sertralina", "Fluoxetina", "Escitalopram", "Venlafaxina"],
    "anticoagulante": ["Warfarina", "Rivaroxabán", "Enoxaparina", "Ácido acetilsalicílico"],
    "antihistamínico": ["Loratadina", "Cetirizina", "Desloratadina", "Clorfenamina"],
    "anticonvulsivo": ["Levetiracetam", "Ácido valproico", "Carbamazepina"],
    "suplemento": ["Sulfato ferroso", "Vitamina D", "Ácido fólico"],
}
OTC_MEDICATIONS = {  # venta libre (requires_prescription = 0)
    "Paracetamol", "Metamizol", "Ibuprofeno", "Naproxeno", "Loratadina", "Cetirizina",
    "Clorfenamina", "Ácido acetilsalicílico", "Sulfato ferroso", "Vitamina D", "Ácido fólico",
}
PRICE_RANGE_BY_CATEGORY: Dict[str, tuple[int, int]] = {
    "analgésico": (500, 6_000), "antibiótico": (3_000, 25_000), "antihipertensivo": (2_000, 15_000),
    "antidiabético": (4_000, 60_000), "antiinflamatorio": (800, 12_000), "antidepresivo": (5_000, 30_000),
    "anticoagulante": (1_000, 45_000), "antihistamínico": (1_000, 8_000), "anticonvulsivo": (8_000, 40_000),
    "suplemento": (1_500, 9_000),
}

# Costo base y duración media (minutos) de una consulta por especialidad.
APPOINTMENT_PROFILE: Dict[str, tuple[int, int]] = {
    "Cardiología": (60_000, 30), "Pediatría": (30_000, 20), "Neurología": (65_000, 40),
    "Oncología": (90_000, 45), "Traumatología": (45_000, 25), "Dermatología": (40_000, 20),
    "Ginecología": (45_000, 30), "Medicina Interna": (35_000, 30), "Psiquiatría": (70_000, 50),
    "Urgencias": (50_000, 35),
}
# Peso relativo de cada departamento en hospitalizaciones.
ADMISSION_WEIGHTS: Dict[str, float] = {
    "Cardiología": 1.5, "Pediatría": 1.0, "Neurología": 0.8, "Oncología": 1.2, "Traumatología": 1.8,
    "Dermatología": 0.2, "Ginecología": 0.9, "Medicina Interna": 2.0, "Psiquiatría": 0.6, "Urgencias": 2.5,
}
STAY_RANGE_BY_ROOM: Dict[str, tuple[int, int]] = {"general": (1, 10), "intermedio": (3, 20), "UCI": (5, 30)}
DAILY_COST_BY_ROOM: Dict[str, tuple[int, int]] = {"general": (80_000, 200_000), "intermedio": (200_000, 450_000), "UCI": (450_000, 900_000)}


# ----------------------------------------------------------------------------------------------
# Generador
# ----------------------------------------------------------------------------------------------

class DataGenerator:
    def __init__(self, schema: Optional[Schema] = None, seed: Optional[int] = None, sizes: Sizes = Sizes()) -> None:
        self.schema = schema or build_schema()
        self.seed = seed if seed is not None else int(os.getenv("RANDOM_SEED", "2026"))
        self.sizes = sizes
        self.rng = random.Random(self.seed)

    # -- utilidades ----------------------------------------------------------------------------

    def _date_between(self, start: date, end: date) -> date:
        return start + timedelta(days=self.rng.randint(0, (end - start).days))

    def _unique_names(self, n: int, female_ratio: float) -> List[tuple[str, str]]:
        """Devuelve n (nombre_completo, sexo) únicos."""
        seen, out = set(), []
        while len(out) < n:
            sex = "F" if self.rng.random() < female_ratio else "M"
            first = self.rng.choice(FIRST_NAMES_F if sex == "F" else FIRST_NAMES_M)
            name = f"{first} {self.rng.choice(LAST_NAMES)} {self.rng.choice(LAST_NAMES)}"
            if name not in seen:
                seen.add(name)
                out.append((name, sex))
        return out

    def _unique_ints(self, n: int, lo: int, hi: int, step: int) -> List[int]:
        pool = list(range(lo, hi + 1, step))
        return self.rng.sample(pool, n)

    # -- tablas --------------------------------------------------------------------------------

    def _gen_departments(self) -> List[Row]:
        budgets = self._unique_ints(len(DEPARTMENT_NAMES), 200_000_000, 2_000_000_000, 1_000_000)
        return [
            {"department_id": i + 1, "name": name, "floor": self.rng.randint(1, 8), "annual_budget": budgets[i]}
            for i, name in enumerate(DEPARTMENT_NAMES)
        ]

    def _gen_doctors(self, departments: List[Row]) -> List[Row]:
        counts = [self.rng.randint(*self.sizes.doctors_per_department) for _ in departments]
        names = iter(self._unique_names(sum(counts), female_ratio=0.5))
        doctors: List[Row] = []
        for dept, n in zip(departments, counts):
            chief_id = len(doctors) + 1
            senior_ids: List[int] = []
            for k in range(n):
                level = "chief" if k == 0 else ("senior" if k <= 2 else "staff")
                name = next(names)[0]
                doctor_id = len(doctors) + 1
                if level == "chief":
                    supervisor, salary = None, self.rng.randrange(6_500_000, 9_000_001, 1_000)
                    hire = self._date_between(date(2005, 1, 1), date(2012, 12, 31))
                elif level == "senior":
                    supervisor, salary = chief_id, self.rng.randrange(3_800_000, 7_000_001, 1_000)
                    hire = self._date_between(date(2010, 1, 1), date(2018, 12, 31))
                    senior_ids.append(doctor_id)
                else:
                    supervisor = self.rng.choice(senior_ids or [chief_id])
                    if self.rng.random() < 0.12:   # anomalía: gana más que su supervisor
                        salary = min(9_000_000, doctors[supervisor - 1]["salary"] + self.rng.randrange(50_000, 600_001, 1_000))
                    else:
                        salary = self.rng.randrange(2_000_000, 5_500_001, 1_000)
                    hire = self._date_between(date(2012, 1, 1), date(2025, 6, 30))
                doctors.append({
                    "doctor_id": doctor_id, "full_name": name, "specialty": dept["name"],
                    "department_id": dept["department_id"], "supervisor_id": supervisor,
                    "hire_date": hire.isoformat(), "salary": salary,
                })
        # Médicos recién contratados (sin consultas ni hospitalizaciones todavía): habilitan
        # preguntas de anti-join ("médicos que no han atendido") y de fechas.
        staff = [d for d in doctors if d["supervisor_id"] is not None and d["doctor_id"] not in
                 {s for s in (x["supervisor_id"] for x in doctors) if s}]
        for d in self.rng.sample(staff, self.sizes.new_doctors):
            d["hire_date"] = self._date_between(DATE_END - timedelta(days=45), DATE_END).isoformat()
        return doctors

    @staticmethod
    def _is_active(doctor: Row) -> bool:
        return date.fromisoformat(doctor["hire_date"]) <= DATE_END - timedelta(days=45)

    def _gen_patients(self) -> List[Row]:
        n = self.sizes.patients
        names = self._unique_names(n, female_ratio=0.52)
        blood_w = [0.38, 0.07, 0.30, 0.06, 0.09, 0.02, 0.06, 0.02]
        city_w = [0.35, 0.10, 0.10, 0.06, 0.06, 0.08, 0.06, 0.06, 0.05, 0.08]
        ins_w = [0.55, 0.35, 0.10]
        patients = []
        for i, (name, sex) in enumerate(names):
            r = self.rng.random()
            if r < 0.15:
                age = self.rng.randint(0, 15)
            elif r < 0.70:
                age = self.rng.randint(16, 59)
            else:
                age = self.rng.randint(60, 95)
            patients.append({
                "patient_id": i + 1, "full_name": name, "age": age, "sex": sex,
                "blood_type": self.rng.choices(BLOOD_TYPES, blood_w)[0],
                "city": self.rng.choices(CITIES, city_w)[0],
                "insurance": self.rng.choices(INSURANCE_TYPES, ins_w)[0],
            })
        return patients

    def _patient_pools(self, patients: List[Row]) -> Dict[str, tuple[List[Row], List[float]]]:
        """Pools de pacientes elegibles por especialidad, con pesos de cola pesada (visitas)."""
        visit_weight = {p["patient_id"]: self.rng.paretovariate(1.6) for p in patients}
        pools: Dict[str, tuple[List[Row], List[float]]] = {}
        for spec in DEPARTMENT_NAMES:
            if spec == "Pediatría":
                elig = [p for p in patients if p["age"] <= 15]
            elif spec == "Ginecología":
                elig = [p for p in patients if p["sex"] == "F" and p["age"] >= 14]
            else:
                elig = [p for p in patients if p["age"] >= 14] if spec != "Urgencias" else patients
            pools[spec] = (elig, [visit_weight[p["patient_id"]] for p in elig])
        return pools

    def _gen_appointments(self, doctors: List[Row], patients: List[Row]) -> List[Row]:
        pools = self._patient_pools(patients)
        active = [d for d in doctors if self._is_active(d)]
        rows = []
        for i in range(self.sizes.appointments):
            doc = self.rng.choice(active)
            spec = doc["specialty"]
            elig, weights = pools[spec]
            pat = self.rng.choices(elig, weights)[0]
            status = self.rng.choices(["completed", "cancelled", "no_show"], [0.80, 0.12, 0.08])[0]
            diag_options = DIAGNOSES_BY_SPECIALTY[spec]
            diag_w = [3.0] + [1.0] * (len(diag_options) - 1)
            diagnosis = self.rng.choices(diag_options, diag_w)[0] if status == "completed" else None
            base_cost, base_min = APPOINTMENT_PROFILE[spec]
            cost = int(round(base_cost * self.rng.uniform(0.7, 1.4) / 500) * 500)
            duration = int(round(self.rng.gauss(base_min, 10) / 5) * 5)
            first_day = max(DATE_START, date.fromisoformat(doc["hire_date"]))
            rows.append({
                "appointment_id": i + 1, "patient_id": pat["patient_id"], "doctor_id": doc["doctor_id"],
                "appointment_date": self._date_between(first_day, DATE_END).isoformat(),
                "duration_minutes": max(10, min(90, duration)), "status": status,
                "diagnosis": diagnosis, "cost": max(15_000, min(120_000, cost)),
            })
        return rows

    def _gen_admissions(self, departments: List[Row], doctors: List[Row], patients: List[Row]) -> List[Row]:
        by_dept: Dict[int, List[Row]] = {}
        for d in doctors:
            if self._is_active(d):
                by_dept.setdefault(d["department_id"], []).append(d)
        dept_w = [ADMISSION_WEIGHTS[d["name"]] for d in departments]
        pools = self._patient_pools(patients)
        rows = []
        n_total = self.sizes.admissions
        for i in range(n_total):
            dept = self.rng.choices(departments, dept_w)[0]
            elig, weights = pools[dept["name"]]
            pat = self.rng.choices(elig, weights)[0]
            doc = self.rng.choice(by_dept[dept["department_id"]])
            room = self.rng.choices(ROOM_TYPES, [0.6, 0.28, 0.12])[0]
            lo, hi = STAY_RANGE_BY_ROOM[room]
            stay = self.rng.randint(lo, hi)
            is_current = i >= n_total - self.sizes.current_admissions
            first_day = max(DATE_START, date.fromisoformat(doc["hire_date"]))
            if is_current:
                admission = self._date_between(DATE_END - timedelta(days=29), DATE_END)
                discharge: Optional[date] = None
            else:
                admission = self._date_between(first_day, DATE_END - timedelta(days=stay))
                discharge = admission + timedelta(days=stay)
            clo, chi = DAILY_COST_BY_ROOM[room]
            rows.append({
                "admission_id": i + 1, "patient_id": pat["patient_id"], "department_id": dept["department_id"],
                "attending_doctor_id": doc["doctor_id"], "admission_date": admission.isoformat(),
                "discharge_date": discharge.isoformat() if discharge else None, "room_type": room,
                "daily_cost": self.rng.randrange(clo, chi + 1, 1_000),
            })
        return rows

    def _gen_medications(self) -> List[Row]:
        rows = []
        used_prices: set[int] = set()
        for category in MEDICATION_CATEGORIES:
            lo, hi = PRICE_RANGE_BY_CATEGORY[category]
            for name in MEDICATIONS_BY_CATEGORY[category]:
                price = self.rng.randrange(lo, hi + 1, 100)
                while price in used_prices:
                    price = self.rng.randrange(lo, hi + 1, 100)
                used_prices.add(price)
                rows.append({
                    "medication_id": len(rows) + 1, "name": name, "category": category,
                    "unit_price": price, "requires_prescription": 0 if name in OTC_MEDICATIONS else 1,
                })
        return rows

    def _gen_prescriptions(self, appointments: List[Row], medications: List[Row]) -> List[Row]:
        # Algunos medicamentos del catálogo nunca se recetan (anti-join no vacío). Se eligen solo
        # entre categorías con 4+ opciones para no vaciar ninguna categoría.
        candidates = [m for m in medications if len(MEDICATIONS_BY_CATEGORY[m["category"]]) >= 4]
        unprescribed = {m["medication_id"] for m in self.rng.sample(candidates, self.sizes.unprescribed_medications)}
        by_cat: Dict[str, List[Row]] = {}
        for m in medications:
            if m["medication_id"] not in unprescribed:
                by_cat.setdefault(m["category"], []).append(m)
        rows = []
        for appt in appointments:
            if appt["status"] != "completed":
                continue
            diag = appt["diagnosis"]
            weights = N_MEDS_WEIGHTS_CHECKUP if diag == "Control sano" else N_MEDS_WEIGHTS_DEFAULT
            n_meds = self.rng.choices([0, 1, 2, 3], weights)[0]
            chosen: set[int] = set()
            for _ in range(n_meds):
                if self.rng.random() < 0.8:
                    category = self.rng.choice(CATEGORIES_BY_DIAGNOSIS[diag])
                else:
                    category = self.rng.choice(MEDICATION_CATEGORIES)
                med = self.rng.choice(by_cat[category])
                if med["medication_id"] in chosen:
                    continue
                chosen.add(med["medication_id"])
                duration = self.rng.choice([3, 5, 7, 10, 14, 21, 30, 30, 60, 90])
                dose_per_day = self.rng.choice([1, 1, 2, 3])
                rows.append({
                    "prescription_id": len(rows) + 1, "appointment_id": appt["appointment_id"],
                    "medication_id": med["medication_id"],
                    "quantity": max(1, min(60, duration * dose_per_day)), "duration_days": duration,
                })
        return rows

    # -- API pública ---------------------------------------------------------------------------

    def generate(self) -> Tables:
        """Genera todas las tablas en memoria, en orden de dependencias."""
        self.rng = random.Random(self.seed)
        departments = self._gen_departments()
        doctors = self._gen_doctors(departments)
        patients = self._gen_patients()
        appointments = self._gen_appointments(doctors, patients)
        admissions = self._gen_admissions(departments, doctors, patients)
        medications = self._gen_medications()
        prescriptions = self._gen_prescriptions(appointments, medications)
        data: Tables = {
            "departments": departments, "doctors": doctors, "patients": patients,
            "appointments": appointments, "admissions": admissions,
            "medications": medications, "prescriptions": prescriptions,
        }
        self.validate(data)
        return data

    def validate(self, data: Tables) -> None:
        """Comprueba que cada fila respete los dominios declarados en el esquema."""
        for table in self.schema.tables:
            rows = data[table.name]
            assert rows, f"tabla vacía: {table.name}"
            for row in rows:
                assert set(row) == {c.name for c in table.columns}, f"columnas inconsistentes en {table.name}"
                for col in table.columns:
                    v = row[col.name]
                    if v is None:
                        assert col.nullable, f"NULL no permitido en {table.name}.{col.name}"
                        continue
                    if col.values is not None:
                        assert str(v) in col.values, f"valor fuera de dominio: {table.name}.{col.name}={v!r}"
                    if col.value_range is not None:
                        lo, hi = col.value_range
                        assert lo <= v <= hi, f"fuera de rango: {table.name}.{col.name}={v!r}"

    def build_database(self, data: Tables, db_path: str | Path) -> sqlite3.Connection:
        """Crea (o reemplaza) el archivo SQLite: aplica el DDL, inserta los datos y verifica FKs."""
        if str(db_path) != ":memory:":
            db_path = Path(db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            if db_path.exists():
                db_path.unlink()
        conn = sqlite3.connect(str(db_path))
        conn.executescript(to_ddl(self.schema))
        for table in self.schema.tables:
            cols = [c.name for c in table.columns]
            placeholders = ", ".join("?" for _ in cols)
            conn.executemany(
                f"INSERT INTO {table.name} ({', '.join(cols)}) VALUES ({placeholders})",
                [tuple(row[c] for c in cols) for row in data[table.name]],
            )
        conn.commit()
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert not violations, f"violaciones de clave foránea: {violations[:5]}"
        return conn

    @staticmethod
    def summary(conn: sqlite3.Connection, schema: Schema) -> Dict[str, int]:
        return {t.name: conn.execute(f"SELECT COUNT(*) FROM {t.name}").fetchone()[0] for t in schema.tables}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Genera db/hospital.sqlite de forma determinista.")
    parser.add_argument("--out", default=os.getenv("DB_PATH", "db/hospital.sqlite"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    gen = DataGenerator(seed=args.seed)
    data = gen.generate()
    conn = gen.build_database(data, args.out)
    counts = DataGenerator.summary(conn, gen.schema)
    conn.execute("VACUUM")
    conn.close()
    print(f"Base de datos escrita en {args.out} (seed={gen.seed})")
    for name, n in counts.items():
        print(f"  {name:<14} {n:>6}")


if __name__ == "__main__":
    main()
