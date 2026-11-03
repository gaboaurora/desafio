"""
Generación de consultas SQL canónicas por nivel de dificultad y cálculo del ground truth.

Enfoque: plantillas composicionales con *slots* que se muestrean desde la propia base de datos
(valores categóricos, nombres de entidades, umbrales numéricos, períodos). Cada instancia se ejecuta
sobre db/hospital.sqlite y se acepta solo si está "bien formada":

  - resultado no vacío y sin NULLs;
  - `number`: distinto de 0 (evita que "0" sea una respuesta frecuente y adivinable);
  - `list` / `table`: entre 1 y MAX_ROWS filas, sin duplicados;
  - `text` y rankings (`ordered=True`): sin empate en la frontera (p.ej. dos médicos con el mismo
    máximo de consultas harían ambigua la respuesta);
  - `boolean`: se balancean verdaderos y falsos por plantilla.

Niveles (definidos por construcciones, no por longitud):
  - easy   : 1 tabla, filtros múltiples, agregación simple, EXISTS simple.
  - medium : 1 JOIN y/o GROUP BY, HAVING simple, subconsulta IN simple, ORDER BY + LIMIT 1.
  - hard   : 2+ JOINs, HAVING con varios agregados, subconsultas, INTERSECT, aritmética, top-k.
  - expert : self-join, anti-join (NOT EXISTS), subconsulta correlacionada, CTE, comparación
             contra agregados globales, aritmética de fechas, CASE en agregaciones.

Uso:
    python -m generator.query_generator            # imprime capacidad por plantilla (diagnóstico)
"""

from __future__ import annotations

import math
import os
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

MAX_ROWS = 8
DATABASE_FILE = "hospital.sqlite"

MONTHS_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "septiembre", "octubre", "noviembre", "diciembre",
]
STATUS_TEXT = {
    "completed": "completadas",
    "cancelled": "canceladas",
    "no_show": "a las que el paciente no asistió",
}
SEX_TEXT = {"F": "femenino", "M": "masculino"}


def fmt_clp(v: int) -> str:
    return f"{int(v):,}".replace(",", ".")


# ----------------------------------------------------------------------------------------------
# Slots y samplers
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Slot:
    value: Any          # valor crudo (clave de deduplicación)
    sql: str            # literal/expresión para interpolar en SQL
    text: str           # realización en lenguaje natural
    fmt: str = ""       # uso interno (formato strftime de un período)


class Context:
    """Conexión de solo lectura + caché de valores distintos por columna."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._cache: Dict[str, List[Any]] = {}

    def distinct(self, table: str, column: str) -> List[Any]:
        key = f"{table}.{column}"
        if key not in self._cache:
            rows = self.conn.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY 1").fetchall()
            self._cache[key] = [r[0] for r in rows]
        return self._cache[key]


Sampler = Callable[[random.Random, Context], Slot]


def _q(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def col_value(table: str, column: str, text_map: Optional[Dict[str, str]] = None) -> Sampler:
    def sample(rng: random.Random, ctx: Context) -> Slot:
        v = rng.choice(ctx.distinct(table, column))
        return Slot(v, _q(v), (text_map or {}).get(v, str(v)))
    return sample


def entity_name(table: str, column: str = "full_name") -> Sampler:
    return col_value(table, column)


def number(values: Sequence[int], money: bool = False) -> Sampler:
    def sample(rng: random.Random, ctx: Context) -> Slot:
        v = rng.choice(list(values))
        return Slot(v, str(v), fmt_clp(v) if money else str(v))
    return sample


def choice(options: Sequence[Slot]) -> Sampler:
    def sample(rng: random.Random, ctx: Context) -> Slot:
        return rng.choice(list(options))
    return sample


def period(kind: str = "any", years: Sequence[int] = (2024, 2025)) -> Sampler:
    """Período temporal: un año ('%Y') o un mes ('%Y-%m')."""
    def sample(rng: random.Random, ctx: Context) -> Slot:
        k = kind if kind != "any" else rng.choice(["year", "month"])
        y = rng.choice(list(years))
        if k == "year":
            return Slot(str(y), _q(str(y)), str(y), fmt="%Y")
        m = rng.randint(1, 12)
        v = f"{y}-{m:02d}"
        return Slot(v, _q(v), f"{MONTHS_ES[m - 1]} de {y}", fmt="%Y-%m")
    return sample


def hire_year(years: Sequence[int] = range(2008, 2025)) -> Sampler:
    def sample(rng: random.Random, ctx: Context) -> Slot:
        y = rng.choice(list(years))
        return Slot(y, _q(f"{y}-01-01"), str(y))
    return sample


# ----------------------------------------------------------------------------------------------
# Plantillas
# ----------------------------------------------------------------------------------------------

@dataclass
class Template:
    id: str
    difficulty: str
    features: List[str]
    answer_type: str                       # number | text | boolean | list | table
    result_columns: List[str]
    sql: str                               # con placeholders {slot.sql}, {slot.fmt}
    slots: Dict[str, Sampler]
    ordered: bool = False
    tie_sql: Optional[str] = None          # devuelve la clave de orden; detecta empates en la frontera
    constraint: Optional[Callable[[Dict[str, Slot]], bool]] = None


@dataclass
class GeneratedQuery:
    template: Template
    slots: Dict[str, Slot]
    sql: str
    expected_result: Any

    @property
    def difficulty(self) -> str:
        return self.template.difficulty

    def dedupe_key(self) -> tuple:
        return (self.template.id, tuple(sorted((k, repr(s.value)) for k, s in self.slots.items())))


AGE = number([18, 30, 40, 50, 60, 65, 70, 80])
COST = number([30_000, 40_000, 50_000, 60_000, 70_000, 80_000], money=True)
SALARY = number([3_000_000, 3_500_000, 4_000_000, 4_500_000, 5_000_000, 6_000_000], money=True)
PRICE = number([2_000, 5_000, 10_000, 20_000, 30_000], money=True)
BUDGET = number([600_000_000, 800_000_000, 1_000_000_000, 1_200_000_000, 1_500_000_000], money=True)
MINUTES = number([25, 30, 35, 40, 45])
K = number([2, 3, 4, 5])

CITY = col_value("patients", "city")
INSURANCE = col_value("patients", "insurance")
BLOOD = col_value("patients", "blood_type")
SEX = col_value("patients", "sex", SEX_TEXT)
SPECIALTY = col_value("doctors", "specialty")
DEPARTMENT = col_value("departments", "name")
FLOOR = col_value("departments", "floor")
DIAGNOSIS = col_value("appointments", "diagnosis")
STATUS = col_value("appointments", "status", STATUS_TEXT)
ROOM = col_value("admissions", "room_type")
CATEGORY = col_value("medications", "category")
PATIENT = entity_name("patients")
DOCTOR = entity_name("doctors")

SALARY_AGG = choice([
    Slot("max", "MAX(salary)", "más alto"),
    Slot("min", "MIN(salary)", "más bajo"),
    Slot("avg", "ROUND(AVG(salary), 2)", "promedio"),
])
AGE_AGG = choice([Slot("max", "MAX(age)", "mayor"), Slot("min", "MIN(age)", "menor")])
CMP = choice([Slot("gt", ">", "más de"), Slot("lt", "<", "menos de")])
DIRECTION = choice([Slot("desc", "DESC", "mayor"), Slot("asc", "ASC", "menor")])
FACTOR = choice([Slot(2, "2", "el doble"), Slot(3, "3", "el triple")])

STAY_DAYS = "(julianday(ad.discharge_date) - julianday(ad.admission_date))"

TEMPLATES: List[Template] = [
    # ---------------------------------------------------------------- EASY
    Template("count_patients_city_insurance", "easy", ["multi_condition", "aggregation"], "number", ["count"],
             "SELECT COUNT(*) FROM patients WHERE city = {city.sql} AND insurance = {insurance.sql}",
             {"city": CITY, "insurance": INSURANCE}),
    Template("count_patients_city_age", "easy", ["multi_condition", "aggregation"], "number", ["count"],
             "SELECT COUNT(*) FROM patients WHERE city = {city.sql} AND age > {age.sql}",
             {"city": CITY, "age": AGE}),
    Template("avg_age_city_insurance", "easy", ["multi_condition", "aggregation"], "number", ["avg_age"],
             "SELECT ROUND(AVG(age), 2) FROM patients WHERE city = {city.sql} AND insurance = {insurance.sql}",
             {"city": CITY, "insurance": INSURANCE}),
    Template("count_appointments_status_period", "easy", ["multi_condition", "date_filter", "aggregation"], "number", ["count"],
             "SELECT COUNT(*) FROM appointments WHERE status = {status.sql} AND strftime('{period.fmt}', appointment_date) = {period.sql}",
             {"status": STATUS, "period": period()}),
    Template("salary_agg_specialty", "easy", ["aggregation"], "number", ["salary"],
             "SELECT {agg.sql} FROM doctors WHERE specialty = {specialty.sql}",
             {"agg": SALARY_AGG, "specialty": SPECIALTY}),
    Template("list_medications_category_price", "easy", ["multi_condition"], "list", ["name"],
             "SELECT name FROM medications WHERE category = {category.sql} AND unit_price < {price.sql} ORDER BY name",
             {"category": CATEGORY, "price": PRICE}),
    Template("count_admissions_room_period", "easy", ["multi_condition", "date_filter", "aggregation"], "number", ["count"],
             "SELECT COUNT(*) FROM admissions WHERE room_type = {room.sql} AND strftime('{period.fmt}', admission_date) = {period.sql}",
             {"room": ROOM, "period": period()}),
    Template("sum_cost_diagnosis_period", "easy", ["multi_condition", "date_filter", "aggregation"], "number", ["total_cost"],
             "SELECT SUM(cost) FROM appointments WHERE diagnosis = {diagnosis.sql} AND strftime('{period.fmt}', appointment_date) = {period.sql}",
             {"diagnosis": DIAGNOSIS, "period": period()}),
    Template("count_doctors_specialty_hired_after", "easy", ["multi_condition", "date_filter", "aggregation"], "number", ["count"],
             "SELECT COUNT(*) FROM doctors WHERE specialty = {specialty.sql} AND hire_date >= {year.sql}",
             {"specialty": SPECIALTY, "year": hire_year()}),
    Template("list_patients_city_blood_age", "easy", ["multi_condition"], "list", ["full_name"],
             "SELECT full_name FROM patients WHERE city = {city.sql} AND blood_type = {blood.sql} AND age > {age.sql} ORDER BY full_name",
             {"city": CITY, "blood": BLOOD, "age": AGE}),
    Template("exists_patient_city_age_insurance", "easy", ["multi_condition", "exists"], "boolean", ["exists"],
             "SELECT EXISTS(SELECT 1 FROM patients WHERE city = {city.sql} AND age > {age.sql} AND insurance = {insurance.sql})",
             {"city": CITY, "age": AGE, "insurance": INSURANCE}),
    Template("age_extreme_city_sex", "easy", ["multi_condition", "aggregation"], "number", ["age"],
             "SELECT {agg.sql} FROM patients WHERE city = {city.sql} AND sex = {sex.sql}",
             {"agg": AGE_AGG, "city": CITY, "sex": SEX}),

    # ---------------------------------------------------------------- MEDIUM
    Template("count_appointments_specialty_status_period", "medium", ["join", "multi_condition", "date_filter", "aggregation"], "number", ["count"],
             "SELECT COUNT(*) FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "WHERE d.specialty = {specialty.sql} AND a.status = {status.sql} AND strftime('{period.fmt}', a.appointment_date) = {period.sql}",
             {"specialty": SPECIALTY, "status": STATUS, "period": period()}),
    Template("doctors_min_appointments_specialty_year", "medium", ["join", "group_by", "having", "date_filter"], "list", ["full_name"],
             "SELECT d.full_name FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "WHERE d.specialty = {specialty.sql} AND a.status = 'completed' AND strftime('{period.fmt}', a.appointment_date) = {period.sql} "
             "GROUP BY d.doctor_id, d.full_name HAVING COUNT(*) > {n.sql} ORDER BY d.full_name",
             {"specialty": SPECIALTY, "period": period("year"), "n": number([12, 15, 18, 20, 25])}),
    Template("doctors_min_appointments_specialty_month", "medium", ["join", "group_by", "having", "date_filter"], "list", ["full_name"],
             "SELECT d.full_name FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "WHERE d.specialty = {specialty.sql} AND strftime('{period.fmt}', a.appointment_date) = {period.sql} "
             "GROUP BY d.doctor_id, d.full_name HAVING COUNT(*) >= {n.sql} ORDER BY d.full_name",
             {"specialty": SPECIALTY, "period": period("month"), "n": number([2, 3, 4])}),
    Template("top_doctor_appointments_specialty_period", "medium", ["join", "group_by", "order_limit", "date_filter"], "text", ["full_name"],
             "SELECT d.full_name FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "WHERE d.specialty = {specialty.sql} AND a.status = 'completed' AND strftime('{period.fmt}', a.appointment_date) = {period.sql} "
             "GROUP BY d.doctor_id, d.full_name ORDER BY COUNT(*) DESC, d.full_name LIMIT 1",
             {"specialty": SPECIALTY, "period": period("year")},
             tie_sql="SELECT COUNT(*) FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
                     "WHERE d.specialty = {specialty.sql} AND a.status = 'completed' AND strftime('{period.fmt}', a.appointment_date) = {period.sql} "
                     "GROUP BY d.doctor_id ORDER BY COUNT(*) DESC"),
    Template("avg_age_distinct_patients_diagnosis_sex", "medium", ["subquery", "multi_condition", "aggregation"], "number", ["avg_age"],
             "SELECT ROUND(AVG(age), 2) FROM patients WHERE sex = {sex.sql} "
             "AND patient_id IN (SELECT patient_id FROM appointments WHERE diagnosis = {diagnosis.sql})",
             {"sex": SEX, "diagnosis": DIAGNOSIS}),
    Template("count_admissions_department_room_year", "medium", ["join", "multi_condition", "date_filter", "aggregation"], "number", ["count"],
             "SELECT COUNT(*) FROM admissions ad JOIN departments dp ON ad.department_id = dp.department_id "
             "WHERE dp.name = {department.sql} AND ad.room_type = {room.sql} AND strftime('{period.fmt}', ad.admission_date) = {period.sql}",
             {"department": DEPARTMENT, "room": ROOM, "period": period("year")}),
    Template("doctors_per_department_budget_hired_after", "medium", ["join", "group_by", "multi_condition", "date_filter", "aggregation"], "table", ["department", "n_doctors"],
             "SELECT dp.name, COUNT(*) FROM doctors d JOIN departments dp ON d.department_id = dp.department_id "
             "WHERE dp.annual_budget > {budget.sql} AND d.hire_date >= {year.sql} GROUP BY dp.name ORDER BY dp.name",
             {"budget": BUDGET, "year": hire_year()}),
    Template("sum_quantity_category_price", "medium", ["join", "multi_condition", "aggregation"], "number", ["total_units"],
             "SELECT SUM(pr.quantity) FROM prescriptions pr JOIN medications m ON pr.medication_id = m.medication_id "
             "WHERE m.category = {category.sql} AND m.unit_price {cmp.sql} {price.sql}",
             {"category": CATEGORY, "cmp": CMP, "price": PRICE}),
    Template("patients_city_admitted_room", "medium", ["join", "multi_condition", "distinct"], "list", ["full_name"],
             "SELECT DISTINCT p.full_name FROM admissions ad JOIN patients p ON ad.patient_id = p.patient_id "
             "WHERE p.city = {city.sql} AND ad.room_type = {room.sql} ORDER BY p.full_name",
             {"city": CITY, "room": ROOM}),
    Template("exists_doctor_department_salary_hired_after", "medium", ["join", "exists", "multi_condition", "date_filter"], "boolean", ["exists"],
             "SELECT EXISTS(SELECT 1 FROM doctors d JOIN departments dp ON d.department_id = dp.department_id "
             "WHERE dp.name = {department.sql} AND d.salary > {salary.sql} AND d.hire_date >= {year.sql})",
             {"department": DEPARTMENT, "salary": SALARY, "year": hire_year()}),
    Template("avg_duration_by_specialty_having", "medium", ["join", "group_by", "having", "aggregation"], "table", ["specialty", "avg_duration"],
             "SELECT d.specialty, ROUND(AVG(a.duration_minutes), 2) FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "WHERE a.status = 'completed' AND strftime('{period.fmt}', a.appointment_date) = {period.sql} "
             "GROUP BY d.specialty HAVING AVG(a.duration_minutes) > {minutes.sql} ORDER BY d.specialty",
             {"minutes": MINUTES, "period": period("year")}),
    Template("count_distinct_patients_doctor_period", "medium", ["join", "distinct", "date_filter", "aggregation"], "number", ["n_patients"],
             "SELECT COUNT(DISTINCT a.patient_id) FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "WHERE d.full_name = {doctor.sql} AND strftime('{period.fmt}', a.appointment_date) = {period.sql}",
             {"doctor": DOCTOR, "period": period("year")}),
    Template("sum_cost_specialty_period", "medium", ["join", "multi_condition", "date_filter", "aggregation"], "number", ["total_cost"],
             "SELECT SUM(a.cost) FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "WHERE d.specialty = {specialty.sql} AND a.status = 'completed' AND strftime('{period.fmt}', a.appointment_date) = {period.sql}",
             {"specialty": SPECIALTY, "period": period()}),

    # ---------------------------------------------------------------- HARD
    Template("patient_prescription_spend", "hard", ["multi_join", "arithmetic", "aggregation"], "number", ["total_spend"],
             "SELECT SUM(pr.quantity * m.unit_price) FROM prescriptions pr "
             "JOIN appointments a ON pr.appointment_id = a.appointment_id "
             "JOIN patients p ON a.patient_id = p.patient_id "
             "JOIN medications m ON pr.medication_id = m.medication_id WHERE p.full_name = {patient.sql}",
             {"patient": PATIENT}),
    Template("top_k_departments_admission_spend_room", "hard", ["join", "group_by", "order_limit", "date_arith", "arithmetic"], "table", ["department", "total_spend"],
             "SELECT dp.name, CAST(SUM(ad.daily_cost * " + STAY_DAYS + ") AS INTEGER) AS total FROM admissions ad "
             "JOIN departments dp ON ad.department_id = dp.department_id "
             "WHERE ad.discharge_date IS NOT NULL AND ad.room_type = {room.sql} GROUP BY dp.name ORDER BY total DESC LIMIT {k.sql}",
             {"room": ROOM, "k": K}, ordered=True,
             tie_sql="SELECT SUM(ad.daily_cost * " + STAY_DAYS + ") FROM admissions ad "
                     "WHERE ad.discharge_date IS NOT NULL AND ad.room_type = {room.sql} GROUP BY ad.department_id ORDER BY 1 DESC"),
    Template("avg_cost_by_department_having_year", "hard", ["multi_join", "group_by", "having", "date_filter", "aggregation"], "table", ["department", "avg_cost"],
             "SELECT dp.name, ROUND(AVG(a.cost), 2) FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "JOIN departments dp ON d.department_id = dp.department_id "
             "WHERE a.status = 'completed' AND strftime('{period.fmt}', a.appointment_date) = {period.sql} "
             "GROUP BY dp.name HAVING AVG(a.cost) > {cost.sql} ORDER BY dp.name",
             {"period": period("year"), "cost": COST}),
    Template("medications_prescribed_diagnosis_min_times", "hard", ["multi_join", "group_by", "having"], "list", ["name"],
             "SELECT m.name FROM prescriptions pr JOIN appointments a ON pr.appointment_id = a.appointment_id "
             "JOIN medications m ON pr.medication_id = m.medication_id WHERE a.diagnosis = {diagnosis.sql} "
             "GROUP BY m.medication_id, m.name HAVING COUNT(*) > {n.sql} ORDER BY m.name",
             {"diagnosis": DIAGNOSIS, "n": number([5, 8, 10, 15, 20])}),
    Template("patients_above_avg_appointments_city_insurance", "hard", ["join", "group_by", "having", "subquery", "aggregation"], "list", ["full_name"],
             "SELECT p.full_name FROM patients p JOIN appointments a ON a.patient_id = p.patient_id "
             "WHERE p.city = {city.sql} AND p.insurance = {insurance.sql} GROUP BY p.patient_id, p.full_name "
             "HAVING COUNT(*) > (SELECT AVG(c) FROM (SELECT COUNT(*) AS c FROM appointments GROUP BY patient_id)) ORDER BY p.full_name",
             {"city": CITY, "insurance": INSURANCE}),
    Template("count_patients_both_specialties", "hard", ["join", "set_operation", "subquery", "aggregation"], "number", ["n_patients"],
             "SELECT COUNT(*) FROM (SELECT a.patient_id FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id WHERE d.specialty = {s1.sql} "
             "INTERSECT SELECT a.patient_id FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id WHERE d.specialty = {s2.sql})",
             {"s1": SPECIALTY, "s2": SPECIALTY}, constraint=lambda s: s["s1"].value < s["s2"].value),
    Template("departments_min_doctors_avg_salary", "hard", ["join", "group_by", "having", "aggregation"], "table", ["department", "n_doctors", "avg_salary"],
             "SELECT dp.name, COUNT(*), ROUND(AVG(d.salary), 2) FROM doctors d JOIN departments dp ON d.department_id = dp.department_id "
             "GROUP BY dp.name HAVING COUNT(*) >= {n.sql} AND AVG(d.salary) > {salary.sql} ORDER BY dp.name",
             {"n": number([5, 6, 7]), "salary": SALARY}),
    Template("doctor_most_stay_days_department_room", "hard", ["multi_join", "group_by", "order_limit", "date_arith"], "text", ["full_name"],
             "SELECT d.full_name FROM admissions ad JOIN doctors d ON ad.attending_doctor_id = d.doctor_id "
             "JOIN departments dp ON ad.department_id = dp.department_id "
             "WHERE dp.name = {department.sql} AND ad.room_type = {room.sql} AND ad.discharge_date IS NOT NULL "
             "GROUP BY d.doctor_id, d.full_name ORDER BY SUM(" + STAY_DAYS + ") DESC, d.full_name LIMIT 1",
             {"department": DEPARTMENT, "room": ROOM},
             tie_sql="SELECT SUM(" + STAY_DAYS + ") FROM admissions ad JOIN departments dp ON ad.department_id = dp.department_id "
                     "WHERE dp.name = {department.sql} AND ad.room_type = {room.sql} AND ad.discharge_date IS NOT NULL "
                     "GROUP BY ad.attending_doctor_id ORDER BY 1 DESC"),
    Template("revenue_category_period", "hard", ["multi_join", "arithmetic", "date_filter", "aggregation"], "number", ["total_revenue"],
             "SELECT SUM(pr.quantity * m.unit_price) FROM prescriptions pr JOIN medications m ON pr.medication_id = m.medication_id "
             "JOIN appointments a ON pr.appointment_id = a.appointment_id "
             "WHERE m.category = {category.sql} AND strftime('{period.fmt}', a.appointment_date) = {period.sql}",
             {"category": CATEGORY, "period": period()}),
    Template("most_common_diagnosis_city_sex", "hard", ["join", "group_by", "order_limit", "multi_condition"], "text", ["diagnosis"],
             "SELECT a.diagnosis FROM appointments a JOIN patients p ON a.patient_id = p.patient_id "
             "WHERE p.city = {city.sql} AND p.sex = {sex.sql} AND a.diagnosis IS NOT NULL "
             "GROUP BY a.diagnosis ORDER BY COUNT(*) DESC, a.diagnosis LIMIT 1",
             {"city": CITY, "sex": SEX},
             tie_sql="SELECT COUNT(*) FROM appointments a JOIN patients p ON a.patient_id = p.patient_id "
                     "WHERE p.city = {city.sql} AND p.sex = {sex.sql} AND a.diagnosis IS NOT NULL GROUP BY a.diagnosis ORDER BY 1 DESC"),
    Template("count_admitted_patients_with_diagnosis", "hard", ["join", "subquery", "distinct", "aggregation"], "number", ["n_patients"],
             "SELECT COUNT(DISTINCT ad.patient_id) FROM admissions ad JOIN departments dp ON ad.department_id = dp.department_id "
             "WHERE dp.name = {department.sql} AND ad.patient_id IN (SELECT patient_id FROM appointments WHERE diagnosis = {diagnosis.sql})",
             {"department": DEPARTMENT, "diagnosis": DIAGNOSIS}),
    Template("exists_department_floor_avg_cost", "hard", ["multi_join", "exists", "group_by", "having"], "boolean", ["exists"],
             "SELECT EXISTS(SELECT 1 FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "JOIN departments dp ON d.department_id = dp.department_id WHERE a.status = 'completed' AND dp.floor = {floor.sql} "
             "GROUP BY dp.department_id HAVING AVG(a.cost) > {cost.sql})",
             {"floor": FLOOR, "cost": COST}),
    Template("top_k_medications_quantity_category", "hard", ["join", "group_by", "order_limit", "aggregation"], "table", ["name", "total_units"],
             "SELECT m.name, SUM(pr.quantity) AS total FROM prescriptions pr JOIN medications m ON pr.medication_id = m.medication_id "
             "WHERE m.category = {category.sql} GROUP BY m.medication_id, m.name ORDER BY total DESC LIMIT {k.sql}",
             {"category": CATEGORY, "k": number([2, 3])}, ordered=True,
             tie_sql="SELECT SUM(pr.quantity) FROM prescriptions pr JOIN medications m ON pr.medication_id = m.medication_id "
                     "WHERE m.category = {category.sql} GROUP BY m.medication_id ORDER BY 1 DESC"),
    Template("supervisor_of_doctor", "hard", ["self_join"], "text", ["supervisor_name"],
             "SELECT s.full_name FROM doctors d JOIN doctors s ON d.supervisor_id = s.doctor_id WHERE d.full_name = {doctor.sql}",
             {"doctor": DOCTOR}),
    Template("subordinates_of_doctor", "hard", ["self_join"], "list", ["full_name"],
             "SELECT d.full_name FROM doctors d JOIN doctors s ON d.supervisor_id = s.doctor_id WHERE s.full_name = {doctor.sql} ORDER BY d.full_name",
             {"doctor": DOCTOR}),
    Template("doctor_hired_before_supervisor_bool", "hard", ["self_join", "date_filter"], "boolean", ["hired_before"],
             "SELECT d.hire_date < s.hire_date FROM doctors d JOIN doctors s ON d.supervisor_id = s.doctor_id WHERE d.full_name = {doctor.sql}",
             {"doctor": DOCTOR}),

    # ---------------------------------------------------------------- EXPERT
    Template("doctors_earning_more_than_supervisor_hired_after", "expert", ["self_join", "multi_condition", "date_filter"], "list", ["full_name"],
             "SELECT d.full_name FROM doctors d JOIN doctors s ON d.supervisor_id = s.doctor_id "
             "WHERE d.salary > s.salary AND d.hire_date >= {year.sql} ORDER BY d.full_name",
             {"year": hire_year(range(2012, 2024))}),
    Template("count_doctors_earning_more_than_supervisor_before", "expert", ["self_join", "multi_condition", "date_filter", "aggregation"], "number", ["count"],
             "SELECT COUNT(*) FROM doctors d JOIN doctors s ON d.supervisor_id = s.doctor_id "
             "WHERE d.salary > s.salary AND d.hire_date < {year.sql}",
             {"year": hire_year(range(2014, 2026))}),
    Template("patient_spend_above_city_avg_bool", "expert", ["cte", "multi_join", "subquery", "arithmetic", "aggregation"], "boolean", ["above_avg"],
             "WITH spend AS (SELECT p.patient_id, p.full_name, p.city, SUM(pr.quantity * m.unit_price) AS total FROM prescriptions pr "
             "JOIN appointments a ON pr.appointment_id = a.appointment_id JOIN patients p ON a.patient_id = p.patient_id "
             "JOIN medications m ON pr.medication_id = m.medication_id GROUP BY p.patient_id, p.full_name, p.city) "
             "SELECT (SELECT total FROM spend WHERE full_name = {patient.sql}) > "
             "(SELECT AVG(total) FROM spend WHERE city = (SELECT city FROM patients WHERE full_name = {patient.sql}))",
             {"patient": PATIENT}),
    Template("doctor_more_appointments_than_supervisor_bool", "expert", ["self_join", "correlated_subquery", "date_filter", "aggregation"], "boolean", ["more_than_supervisor"],
             "SELECT (SELECT COUNT(*) FROM appointments a WHERE a.doctor_id = d.doctor_id AND strftime('{period.fmt}', a.appointment_date) = {period.sql}) > "
             "(SELECT COUNT(*) FROM appointments a WHERE a.doctor_id = s.doctor_id AND strftime('{period.fmt}', a.appointment_date) = {period.sql}) "
             "FROM doctors d JOIN doctors s ON d.supervisor_id = s.doctor_id WHERE d.full_name = {doctor.sql}",
             {"doctor": DOCTOR, "period": period("year")}),
    Template("patients_admitted_dept_without_appointment_same_dept", "expert", ["multi_join", "anti_join", "correlated_subquery", "distinct"], "list", ["full_name"],
             "SELECT DISTINCT p.full_name FROM admissions ad JOIN patients p ON ad.patient_id = p.patient_id "
             "JOIN departments dp ON ad.department_id = dp.department_id WHERE dp.name = {department.sql} AND p.city = {city.sql} "
             "AND NOT EXISTS (SELECT 1 FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "WHERE a.patient_id = p.patient_id AND d.department_id = ad.department_id) ORDER BY p.full_name",
             {"department": DEPARTMENT, "city": CITY}),
    Template("days_between_first_last_admission_patient", "expert", ["join", "date_arith", "aggregation"], "number", ["days"],
             "SELECT CAST(julianday(MAX(ad.admission_date)) - julianday(MIN(ad.admission_date)) AS INTEGER) "
             "FROM admissions ad JOIN patients p ON ad.patient_id = p.patient_id WHERE p.full_name = {patient.sql}",
             {"patient": PATIENT}),
    Template("patient_most_expensive_admission_city_room", "expert", ["join", "order_limit", "date_arith", "arithmetic"], "text", ["full_name"],
             "SELECT p.full_name FROM admissions ad JOIN patients p ON ad.patient_id = p.patient_id "
             "WHERE p.city = {city.sql} AND ad.room_type = {room.sql} AND ad.discharge_date IS NOT NULL "
             "ORDER BY ad.daily_cost * " + STAY_DAYS + " DESC, p.full_name LIMIT 1",
             {"city": CITY, "room": ROOM},
             tie_sql="SELECT ad.daily_cost * " + STAY_DAYS + " FROM admissions ad JOIN patients p ON ad.patient_id = p.patient_id "
                     "WHERE p.city = {city.sql} AND ad.room_type = {room.sql} AND ad.discharge_date IS NOT NULL ORDER BY 1 DESC"),
    Template("count_admitted_and_consulted_same_dept_year", "expert", ["subquery", "correlated_subquery", "multi_join", "distinct", "date_filter", "aggregation"], "number", ["n_patients"],
             "SELECT COUNT(DISTINCT ad.patient_id) FROM admissions ad "
             "WHERE ad.department_id = (SELECT department_id FROM departments WHERE name = {department.sql}) "
             "AND strftime('{period.fmt}', ad.admission_date) = {period.sql} "
             "AND EXISTS (SELECT 1 FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "WHERE a.patient_id = ad.patient_id AND d.department_id = ad.department_id AND a.status = 'completed')",
             {"department": DEPARTMENT, "period": period("year")}),
    Template("medications_never_prescribed_price", "expert", ["anti_join", "multi_condition"], "list", ["name"],
             "SELECT m.name FROM medications m WHERE NOT EXISTS (SELECT 1 FROM prescriptions pr WHERE pr.medication_id = m.medication_id) "
             "AND m.unit_price {cmp.sql} {price.sql} ORDER BY m.name",
             {"cmp": CMP, "price": PRICE}),
    Template("doctors_no_appointments_specialty_month", "expert", ["anti_join", "date_filter"], "list", ["full_name"],
             "SELECT d.full_name FROM doctors d WHERE d.specialty = {specialty.sql} AND NOT EXISTS "
             "(SELECT 1 FROM appointments a WHERE a.doctor_id = d.doctor_id AND strftime('{period.fmt}', a.appointment_date) = {period.sql}) "
             "ORDER BY d.full_name",
             {"specialty": SPECIALTY, "period": period("month")}),
    Template("patients_no_appointments_city_insurance_year", "expert", ["anti_join", "multi_condition", "date_filter"], "list", ["full_name"],
             "SELECT p.full_name FROM patients p WHERE p.city = {city.sql} AND p.insurance = {insurance.sql} AND NOT EXISTS "
             "(SELECT 1 FROM appointments a WHERE a.patient_id = p.patient_id AND strftime('{period.fmt}', a.appointment_date) = {period.sql}) "
             "ORDER BY p.full_name",
             {"city": CITY, "insurance": INSURANCE, "period": period("year")}),
    Template("departments_avg_stay_above_overall_room", "expert", ["join", "group_by", "having", "subquery", "date_arith", "aggregation"], "list", ["department"],
             "SELECT dp.name FROM admissions ad JOIN departments dp ON ad.department_id = dp.department_id "
             "WHERE ad.room_type = {room.sql} AND ad.discharge_date IS NOT NULL AND strftime('{period.fmt}', ad.admission_date) = {period.sql} "
             "GROUP BY dp.name HAVING AVG(" + STAY_DAYS + ") > "
             "(SELECT AVG(julianday(discharge_date) - julianday(admission_date)) FROM admissions "
             "WHERE room_type = {room.sql} AND discharge_date IS NOT NULL AND strftime('{period.fmt}', admission_date) = {period.sql}) ORDER BY dp.name",
             {"room": ROOM, "period": period("year")}),
    Template("top_k_patients_total_stay_city", "expert", ["join", "group_by", "order_limit", "date_arith"], "table", ["full_name", "total_days"],
             "SELECT p.full_name, CAST(SUM(" + STAY_DAYS + ") AS INTEGER) AS days FROM admissions ad JOIN patients p ON ad.patient_id = p.patient_id "
             "WHERE ad.discharge_date IS NOT NULL AND p.city = {city.sql} GROUP BY p.patient_id, p.full_name ORDER BY days DESC LIMIT {k.sql}",
             {"city": CITY, "k": K}, ordered=True,
             tie_sql="SELECT SUM(" + STAY_DAYS + ") FROM admissions ad JOIN patients p ON ad.patient_id = p.patient_id "
                     "WHERE ad.discharge_date IS NOT NULL AND p.city = {city.sql} GROUP BY p.patient_id ORDER BY 1 DESC"),
    Template("doctors_more_appointments_than_supervisor_year", "expert", ["self_join", "correlated_subquery", "date_filter", "aggregation"], "list", ["full_name"],
             "SELECT d.full_name FROM doctors d JOIN doctors s ON d.supervisor_id = s.doctor_id WHERE d.specialty = {specialty.sql} AND "
             "(SELECT COUNT(*) FROM appointments a WHERE a.doctor_id = d.doctor_id AND strftime('{period.fmt}', a.appointment_date) = {period.sql}) > "
             "(SELECT COUNT(*) FROM appointments a WHERE a.doctor_id = s.doctor_id AND strftime('{period.fmt}', a.appointment_date) = {period.sql}) "
             "ORDER BY d.full_name",
             {"specialty": SPECIALTY, "period": period("year")}),
    Template("patients_spend_above_factor_city_avg", "expert", ["cte", "multi_join", "subquery", "arithmetic", "aggregation"], "list", ["full_name"],
             "WITH spend AS (SELECT p.patient_id, p.full_name, p.city, SUM(pr.quantity * m.unit_price) AS total FROM prescriptions pr "
             "JOIN appointments a ON pr.appointment_id = a.appointment_id JOIN patients p ON a.patient_id = p.patient_id "
             "JOIN medications m ON pr.medication_id = m.medication_id GROUP BY p.patient_id, p.full_name, p.city) "
             "SELECT full_name FROM spend WHERE city = {city.sql} AND total > {factor.sql} * (SELECT AVG(total) FROM spend WHERE city = {city.sql}) ORDER BY full_name",
             {"city": CITY, "factor": FACTOR}),
    Template("month_most_appointments_specialty_year", "expert", ["join", "group_by", "order_limit", "date_arith"], "text", ["month"],
             "SELECT strftime('%Y-%m', a.appointment_date) AS month FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "WHERE d.specialty = {specialty.sql} AND strftime('{period.fmt}', a.appointment_date) = {period.sql} "
             "GROUP BY month ORDER BY COUNT(*) DESC, month LIMIT 1",
             {"specialty": SPECIALTY, "period": period("year")},
             tie_sql="SELECT COUNT(*) FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
                     "WHERE d.specialty = {specialty.sql} AND strftime('{period.fmt}', a.appointment_date) = {period.sql} "
                     "GROUP BY strftime('%Y-%m', a.appointment_date) ORDER BY 1 DESC"),
    Template("doctors_hired_before_supervisor_specialty", "expert", ["self_join", "date_filter"], "list", ["full_name"],
             "SELECT d.full_name FROM doctors d JOIN doctors s ON d.supervisor_id = s.doctor_id "
             "WHERE d.specialty = {specialty.sql} AND d.hire_date < s.hire_date ORDER BY d.full_name",
             {"specialty": SPECIALTY}),
    Template("department_extreme_status_ratio_year", "expert", ["multi_join", "group_by", "order_limit", "case", "date_filter"], "text", ["department"],
             "SELECT dp.name FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
             "JOIN departments dp ON d.department_id = dp.department_id WHERE strftime('{period.fmt}', a.appointment_date) = {period.sql} "
             "GROUP BY dp.name ORDER BY AVG(CASE WHEN a.status = {status.sql} THEN 1.0 ELSE 0.0 END) {direction.sql}, dp.name LIMIT 1",
             {"period": period("year"), "status": choice([Slot("cancelled", "'cancelled'", "canceladas"), Slot("no_show", "'no_show'", "con inasistencia del paciente")]),
              "direction": DIRECTION},
             tie_sql="SELECT AVG(CASE WHEN a.status = {status.sql} THEN 1.0 ELSE 0.0 END) FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id "
                     "WHERE strftime('{period.fmt}', a.appointment_date) = {period.sql} GROUP BY d.department_id ORDER BY 1 {direction.sql}"),
    Template("count_admitted_room_patients_prescribed_category", "expert", ["subquery", "multi_join", "distinct", "aggregation"], "number", ["n_patients"],
             "SELECT COUNT(DISTINCT ad.patient_id) FROM admissions ad WHERE ad.room_type = {room.sql} AND ad.patient_id IN "
             "(SELECT a.patient_id FROM appointments a JOIN prescriptions pr ON pr.appointment_id = a.appointment_id "
             "JOIN medications m ON pr.medication_id = m.medication_id WHERE m.category = {category.sql})",
             {"room": ROOM, "category": CATEGORY}),
    Template("doctors_above_department_avg_salary_specialty", "expert", ["correlated_subquery", "aggregation"], "list", ["full_name"],
             "SELECT d.full_name FROM doctors d WHERE d.specialty = {specialty.sql} AND d.salary > "
             "(SELECT AVG(d2.salary) FROM doctors d2 WHERE d2.department_id = d.department_id) ORDER BY d.full_name",
             {"specialty": SPECIALTY}),
    Template("count_doctors_above_department_avg_salary_hired_after", "expert", ["correlated_subquery", "date_filter", "aggregation"], "number", ["count"],
             "SELECT COUNT(*) FROM doctors d WHERE d.hire_date >= {year.sql} AND d.salary > "
             "(SELECT AVG(d2.salary) FROM doctors d2 WHERE d2.department_id = d.department_id)",
             {"year": hire_year(range(2010, 2025))}),
]

TEMPLATES_BY_ID: Dict[str, Template] = {t.id: t for t in TEMPLATES}
assert len(TEMPLATES_BY_ID) == len(TEMPLATES), "ids de plantilla duplicados"


# ----------------------------------------------------------------------------------------------
# Generador
# ----------------------------------------------------------------------------------------------

def _coerce(v: Any) -> Any:
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 2)
    return v


class QueryGenerator:
    def __init__(self, db_path: str | Path | None = None, seed: int = 2026, max_rows: int = MAX_ROWS) -> None:
        self.db_path = Path(db_path or os.getenv("DB_PATH", "db/hospital.sqlite"))
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_rows = max_rows
        self.conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)
        self.ctx = Context(self.conn)
        self._seen: set = set()
        self._bool_counts: Dict[str, Dict[bool, int]] = {}

    # -- instanciación ---------------------------------------------------------------------------

    def _sample_slots(self, t: Template) -> Dict[str, Slot]:
        return {name: sampler(self.rng, self.ctx) for name, sampler in t.slots.items()}

    def execute(self, sql: str) -> List[tuple]:
        return self.conn.execute(sql).fetchall()

    def _well_posed(self, t: Template, slots: Dict[str, Slot], rows: List[tuple]) -> Optional[Any]:
        """Devuelve el expected_result tipado si la instancia es válida; None en caso contrario."""
        if not rows:
            return None
        if any(v is None for r in rows for v in r):
            return None
        if t.answer_type in ("number", "text", "boolean"):
            if len(rows) != 1 or len(rows[0]) != 1:
                return None
            v = rows[0][0]
            if t.answer_type == "number":
                v = _coerce(v)
                return None if v == 0 else v
            if t.answer_type == "boolean":
                return bool(v)
            if t.tie_sql and self._has_boundary_tie(t, slots, 1):
                return None
            return str(v)
        if len(rows) > self.max_rows:
            return None
        if t.answer_type == "list":
            values = [_coerce(r[0]) for r in rows]
            return values if len(set(map(repr, values))) == len(values) else None
        if t.answer_type == "table":
            if t.ordered and t.tie_sql and self._has_boundary_tie(t, slots, len(rows)):
                return None
            table = [[_coerce(v) for v in r] for r in rows]
            return table if len(set(map(repr, table))) == len(table) else None
        raise ValueError(t.answer_type)

    def _has_boundary_tie(self, t: Template, slots: Dict[str, Slot], n: int) -> bool:
        keys = [r[0] for r in self.execute(t.tie_sql.format(**slots))]
        return len(keys) > n and keys[n - 1] == keys[n]

    def try_instantiate(self, t: Template, attempts: int = 40) -> Optional[GeneratedQuery]:
        for _ in range(attempts):
            slots = self._sample_slots(t)
            if t.constraint and not t.constraint(slots):
                continue
            key = (t.id, tuple(sorted((k, repr(s.value)) for k, s in slots.items())))
            if key in self._seen:
                continue
            sql = t.sql.format(**slots)
            result = self._well_posed(t, slots, self.execute(sql))
            if result is None:
                continue
            if t.answer_type == "boolean":
                counts = self._bool_counts.setdefault(t.id, {True: 0, False: 0})
                if counts[result] > counts[not result] + 1:
                    continue
                counts[result] += 1
            self._seen.add(key)
            return GeneratedQuery(t, slots, sql, result)
        return None

    # -- generación por nivel ---------------------------------------------------------------------

    def generate(self, targets: Dict[str, int], patience: int = 3) -> List[GeneratedQuery]:
        """
        Genera hasta `targets[nivel]` instancias por nivel, repartidas round-robin entre plantillas
        para balancearlas. Una plantilla se retira cuando falla `patience` rondas consecutivas.
        """
        out: List[GeneratedQuery] = []
        for difficulty, target in targets.items():
            active = [t for t in TEMPLATES if t.difficulty == difficulty]
            failures = {t.id: 0 for t in active}
            produced = 0
            while produced < target and active:
                for t in list(active):
                    if produced >= target:
                        break
                    q = self.try_instantiate(t)
                    if q is None:
                        failures[t.id] += 1
                        if failures[t.id] >= patience:
                            active.remove(t)
                        continue
                    failures[t.id] = 0
                    out.append(q)
                    produced += 1
        return out

    def capacity_report(self, per_template: int = 400) -> Dict[str, int]:
        """Diagnóstico: cuántas instancias distintas y bien formadas admite cada plantilla."""
        report = {}
        for t in TEMPLATES:
            n = 0
            while n < per_template and self.try_instantiate(t, attempts=60):
                n += 1
            report[t.id] = n
        return report


if __name__ == "__main__":
    gen = QueryGenerator()
    rep = gen.capacity_report()
    for t in TEMPLATES:
        print(f"{t.difficulty:<7} {t.id:<55} {rep[t.id]:>4}")
    by_level: Dict[str, int] = {}
    for t in TEMPLATES:
        by_level[t.difficulty] = by_level.get(t.difficulty, 0) + rep[t.id]
    print(by_level, "total:", sum(by_level.values()))
