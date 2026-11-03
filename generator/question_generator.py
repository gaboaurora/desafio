"""
Generación de la pregunta en lenguaje natural, construcción de las tareas y particiones.

Cada plantilla de query_generator tiene varias paráfrasis en español que se instancian con los
mismos slots que el SQL (`{slot.text}`). Las preguntas se redactan para que determinen una única
consulta: cuando hay filtro de estado se dice explícitamente ("completadas"), cuando se cuentan
entidades se aclara "distintos", y en respuestas tipo tabla se indican las columnas pedidas.

Particiones (train / eval / test):
  - ~20 % de las plantillas de cada nivel se reservan íntegramente para test (`template_heldout`),
    lo que permite medir generalización a estructuras SQL no vistas en entrenamiento.
  - El resto se estratifica por plantilla para que las proporciones sean ~70 / 15 / 15 en total.
  - Deduplicación por (plantilla, valores de slots): una misma instancia nunca cae en dos particiones.

Uso:
    python -m generator.question_generator --total 2500 --seed 2026
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from generator.query_generator import DATABASE_FILE, TEMPLATES, GeneratedQuery, QueryGenerator
from harness.environment import Task

DEFAULT_TARGETS = {"easy": 650, "medium": 700, "hard": 700, "expert": 450}

# --------------------------------------------------------------------------------------------------
# Paráfrasis por plantilla
# --------------------------------------------------------------------------------------------------

QUESTIONS: Dict[str, List[str]] = {
    # ---------------------------------------------------------------- EASY
    "count_patients_city_insurance": [
        "¿Cuántos pacientes de {city.text} tienen previsión {insurance.text}?",
        "¿Cuántos pacientes registrados viven en {city.text} y están afiliados a {insurance.text}?",
        "Indica el número de pacientes con previsión {insurance.text} que residen en {city.text}.",
    ],
    "count_patients_city_age": [
        "¿Cuántos pacientes de {city.text} tienen más de {age.text} años?",
        "¿Cuántos pacientes mayores de {age.text} años viven en {city.text}?",
        "Indica cuántos pacientes residentes en {city.text} superan los {age.text} años de edad.",
    ],
    "avg_age_city_insurance": [
        "¿Cuál es la edad promedio de los pacientes de {city.text} con previsión {insurance.text}?",
        "¿Qué edad promedio tienen los pacientes afiliados a {insurance.text} que viven en {city.text}?",
        "Calcula la edad promedio de los pacientes de {city.text} cuya previsión es {insurance.text}.",
    ],
    "count_appointments_status_period": [
        "¿Cuántas consultas {status.text} hubo en {period.text}?",
        "¿Cuál fue el número de consultas {status.text} registradas en {period.text}?",
        "Indica cuántas consultas {status.text} se registraron en {period.text}.",
    ],
    "salary_agg_specialty": [
        "¿Cuál es el salario {agg.text} de los médicos de {specialty.text}?",
        "Entre los médicos de {specialty.text}, ¿cuál es el salario mensual {agg.text}?",
        "Indica el salario {agg.text} dentro de la especialidad {specialty.text}.",
    ],
    "list_medications_category_price": [
        "¿Qué medicamentos de la categoría {category.text} tienen un precio unitario menor a {price.text} pesos?",
        "Lista los nombres de los medicamentos {category.text} que cuestan menos de {price.text} pesos por unidad.",
        "¿Cuáles son los medicamentos de categoría {category.text} con precio unitario inferior a {price.text} pesos?",
    ],
    "count_admissions_room_period": [
        "¿Cuántas hospitalizaciones en cama {room.text} comenzaron en {period.text}?",
        "¿Cuántos ingresos hospitalarios con tipo de cama {room.text} tuvieron fecha de ingreso en {period.text}?",
        "Indica el número de hospitalizaciones en cama {room.text} cuya fecha de ingreso fue en {period.text}.",
    ],
    "sum_cost_diagnosis_period": [
        "¿Cuánto suman los costos de las consultas con diagnóstico {diagnosis.text} realizadas en {period.text}?",
        "¿Cuál es el costo total de las consultas diagnosticadas como {diagnosis.text} en {period.text}?",
        "Calcula la suma de los costos de consulta con diagnóstico {diagnosis.text} durante {period.text}.",
    ],
    "count_doctors_specialty_hired_after": [
        "¿Cuántos médicos de {specialty.text} fueron contratados a partir del año {year.text}?",
        "¿Cuántos médicos con especialidad {specialty.text} tienen fecha de contratación desde el 1 de enero de {year.text} en adelante?",
        "Indica el número de médicos de {specialty.text} contratados desde {year.text} (inclusive).",
    ],
    "list_patients_city_blood_age": [
        "¿Cómo se llaman los pacientes de {city.text} con grupo sanguíneo {blood.text} y más de {age.text} años?",
        "Lista los nombres de los pacientes mayores de {age.text} años que viven en {city.text} y tienen grupo sanguíneo {blood.text}.",
        "¿Qué pacientes de {city.text} tienen grupo sanguíneo {blood.text} y superan los {age.text} años?",
    ],
    "exists_patient_city_age_insurance": [
        "¿Existe algún paciente de {city.text} mayor de {age.text} años con previsión {insurance.text}?",
        "¿Hay al menos un paciente que viva en {city.text}, tenga más de {age.text} años y esté afiliado a {insurance.text}?",
        "Responde sí o no: ¿algún paciente de {city.text} con previsión {insurance.text} tiene más de {age.text} años?",
    ],
    "age_extreme_city_sex": [
        "¿Cuál es la {agg.text} edad entre los pacientes de sexo {sex.text} de {city.text}?",
        "Entre los pacientes de sexo {sex.text} que viven en {city.text}, ¿qué edad tiene el de {agg.text} edad?",
        "Indica la {agg.text} edad registrada para pacientes de sexo {sex.text} residentes en {city.text}.",
    ],

    # ---------------------------------------------------------------- MEDIUM
    "count_appointments_specialty_status_period": [
        "¿Cuántas consultas {status.text} realizaron los médicos de {specialty.text} en {period.text}?",
        "¿Cuál fue el número de consultas {status.text} atendidas por médicos con especialidad {specialty.text} durante {period.text}?",
        "Indica cuántas consultas {status.text} tuvieron los médicos de {specialty.text} en {period.text}.",
    ],
    "doctors_min_appointments_specialty_year": [
        "¿Qué médicos de {specialty.text} completaron más de {n.text} consultas en {period.text}?",
        "Lista los nombres de los médicos de {specialty.text} con más de {n.text} consultas completadas durante {period.text}.",
        "¿Cuáles médicos con especialidad {specialty.text} superaron las {n.text} consultas completadas en {period.text}?",
    ],
    "doctors_min_appointments_specialty_month": [
        "¿Qué médicos de {specialty.text} tuvieron al menos {n.text} consultas registradas (de cualquier estado) en {period.text}?",
        "Lista los médicos de {specialty.text} con {n.text} o más consultas registradas, sin importar su estado, en {period.text}.",
        "¿Cuáles médicos de {specialty.text} registraron {n.text} o más consultas en {period.text}, contando todos los estados?",
    ],
    "top_doctor_appointments_specialty_period": [
        "¿Qué médico de {specialty.text} completó más consultas en {period.text}?",
        "¿Cuál es el nombre del médico de {specialty.text} con mayor número de consultas completadas en {period.text}?",
        "Indica el médico de {specialty.text} que acumuló más consultas completadas durante {period.text}.",
    ],
    "avg_age_distinct_patients_diagnosis_sex": [
        "¿Cuál es la edad promedio de los pacientes distintos de sexo {sex.text} que han tenido al menos una consulta con diagnóstico {diagnosis.text}?",
        "Considerando cada paciente una sola vez, ¿qué edad promedio tienen los pacientes de sexo {sex.text} diagnosticados alguna vez con {diagnosis.text}?",
        "Calcula la edad promedio (sin repetir pacientes) de las personas de sexo {sex.text} con algún diagnóstico de {diagnosis.text}.",
    ],
    "count_admissions_department_room_year": [
        "¿Cuántas hospitalizaciones en cama {room.text} registró el departamento de {department.text} con ingreso en {period.text}?",
        "¿Cuántos ingresos hospitalarios con cama {room.text} tuvo {department.text} durante {period.text}?",
        "Indica el número de hospitalizaciones de tipo {room.text} en {department.text} cuya fecha de ingreso fue en {period.text}.",
    ],
    "doctors_per_department_budget_hired_after": [
        "Para cada departamento con presupuesto anual superior a {budget.text} pesos, ¿cuántos médicos fueron contratados a partir de {year.text}? Indica departamento y cantidad (solo departamentos con al menos uno).",
        "¿Cuántos médicos contratados desde {year.text} (inclusive) tiene cada departamento cuyo presupuesto anual supera {budget.text} pesos? Entrega nombre del departamento y cantidad, omitiendo los que no tienen ninguno.",
        "Lista los departamentos con presupuesto anual mayor a {budget.text} pesos junto con el número de médicos contratados a partir de {year.text} (excluye departamentos sin médicos en ese período).",
    ],
    "sum_quantity_category_price": [
        "¿Cuántas unidades en total se han recetado de medicamentos {category.text} con precio unitario {cmp.text} {price.text} pesos?",
        "Suma la cantidad recetada de todos los medicamentos de categoría {category.text} cuyo precio unitario es {cmp.text} {price.text} pesos.",
        "¿Cuál es el total de unidades recetadas de medicamentos {category.text} que cuestan {cmp.text} {price.text} pesos por unidad?",
    ],
    "patients_city_admitted_room": [
        "¿Qué pacientes de {city.text} han estado hospitalizados en cama {room.text}?",
        "Lista los nombres (sin repetir) de los pacientes residentes en {city.text} con alguna hospitalización en cama {room.text}.",
        "¿Cuáles pacientes de {city.text} tienen al menos un ingreso hospitalario con tipo de cama {room.text}?",
    ],
    "exists_doctor_department_salary_hired_after": [
        "¿Hay algún médico del departamento de {department.text} que gane más de {salary.text} pesos y haya sido contratado a partir de {year.text}?",
        "¿Existe al menos un médico en {department.text} con salario superior a {salary.text} pesos y fecha de contratación desde {year.text} en adelante?",
        "Responde sí o no: ¿algún médico de {department.text} contratado desde {year.text} (inclusive) tiene un salario mayor a {salary.text} pesos?",
    ],
    "avg_duration_by_specialty_having": [
        "¿Qué especialidades tuvieron una duración promedio de consulta completada superior a {minutes.text} minutos en {period.text}? Indica especialidad y promedio.",
        "Para {period.text}, lista las especialidades cuya duración promedio de consultas completadas supera los {minutes.text} minutos, junto con ese promedio.",
        "¿Cuáles especialidades promedian más de {minutes.text} minutos por consulta completada en {period.text}? Entrega especialidad y duración promedio.",
    ],
    "count_distinct_patients_doctor_period": [
        "¿A cuántos pacientes distintos atendió el médico {doctor.text} en {period.text}, considerando consultas de cualquier estado?",
        "¿Cuántos pacientes diferentes tuvieron consultas registradas (de cualquier estado) con {doctor.text} durante {period.text}?",
        "Indica el número de pacientes distintos con alguna consulta registrada con el médico {doctor.text} en {period.text}.",
    ],
    "sum_cost_specialty_period": [
        "¿Cuánto recaudaron en total las consultas completadas de {specialty.text} en {period.text}?",
        "¿Cuál es la suma de los costos de las consultas completadas por médicos de {specialty.text} durante {period.text}?",
        "Calcula el costo total de las consultas completadas en la especialidad {specialty.text} en {period.text}.",
    ],

    # ---------------------------------------------------------------- HARD
    "patient_prescription_spend": [
        "¿Cuánto gastó en total el paciente {patient.text} en medicamentos recetados (cantidad por precio unitario)?",
        "Calcula el gasto total en medicamentos del paciente {patient.text}, sumando cantidad recetada por precio unitario de cada receta.",
        "¿Cuál es el monto total (cantidad × precio unitario) de las recetas del paciente {patient.text}?",
    ],
    "top_k_departments_admission_spend_room": [
        "¿Cuáles son los {k.text} departamentos con mayor gasto total en hospitalizaciones con alta registrada en cama {room.text}? Indica nombre y gasto (costo diario × días de estadía), de mayor a menor.",
        "Ordena de mayor a menor los {k.text} departamentos que más gastaron en hospitalizaciones finalizadas en cama {room.text}, donde el gasto es el costo diario multiplicado por los días entre ingreso y alta. Entrega nombre y gasto.",
        "Entrega el ranking (descendente) de los {k.text} departamentos con mayor costo acumulado en hospitalizaciones con alta en cama {room.text}, calculado como costo diario × días de estadía, con nombre y monto.",
    ],
    "avg_cost_by_department_having_year": [
        "¿Qué departamentos tuvieron en {period.text} un costo promedio de consulta completada superior a {cost.text} pesos? Indica nombre del departamento y promedio.",
        "Para las consultas completadas de {period.text}, lista los departamentos (del médico que atiende) cuyo costo promedio supera {cost.text} pesos, con su promedio.",
        "¿Cuáles departamentos promediaron más de {cost.text} pesos por consulta completada durante {period.text}? Entrega departamento y costo promedio.",
    ],
    "medications_prescribed_diagnosis_min_times": [
        "¿Qué medicamentos fueron recetados más de {n.text} veces en consultas con diagnóstico {diagnosis.text}?",
        "Lista los medicamentos que aparecen en más de {n.text} recetas asociadas a consultas diagnosticadas como {diagnosis.text}.",
        "¿Cuáles medicamentos superan las {n.text} recetas en consultas cuyo diagnóstico es {diagnosis.text}?",
    ],
    "patients_above_avg_appointments_city_insurance": [
        "¿Qué pacientes de {city.text} con previsión {insurance.text} tienen más consultas registradas que el promedio de consultas por paciente (calculado sobre todos los pacientes con al menos una consulta)?",
        "Lista los pacientes de {city.text} afiliados a {insurance.text} cuyo número de consultas supera el promedio de consultas por paciente entre quienes tienen al menos una.",
        "¿Cuáles pacientes residentes en {city.text} y con previsión {insurance.text} superan el promedio general de consultas por paciente (considerando solo pacientes con alguna consulta)?",
    ],
    "count_patients_both_specialties": [
        "¿Cuántos pacientes distintos han tenido consultas tanto con médicos de {s1.text} como con médicos de {s2.text}?",
        "¿Cuántos pacientes diferentes registran al menos una consulta con {s1.text} y al menos una con {s2.text}?",
        "Indica el número de pacientes distintos atendidos (en cualquier estado de consulta) por ambas especialidades: {s1.text} y {s2.text}.",
    ],
    "departments_min_doctors_avg_salary": [
        "¿Qué departamentos tienen al menos {n.text} médicos y un salario promedio superior a {salary.text} pesos? Indica nombre, cantidad de médicos y salario promedio.",
        "Lista los departamentos con {n.text} o más médicos cuyo salario promedio supera {salary.text} pesos, entregando nombre, número de médicos y promedio.",
        "¿Cuáles departamentos cumplen tener mínimo {n.text} médicos y promedio salarial mayor a {salary.text} pesos? Entrega departamento, cantidad y promedio.",
    ],
    "doctor_most_stay_days_department_room": [
        "¿Qué médico tratante de {department.text} acumuló más días de hospitalización (solo altas registradas) en cama {room.text}?",
        "Entre las hospitalizaciones con alta en cama {room.text} del departamento de {department.text}, ¿qué médico tratante suma más días de estadía?",
        "Indica el médico tratante con mayor total de días de hospitalización finalizada en cama {room.text} dentro de {department.text}.",
    ],
    "revenue_category_period": [
        "¿Cuál fue el monto total (cantidad × precio unitario) de los medicamentos {category.text} recetados en consultas de {period.text}?",
        "Calcula el ingreso total por recetas de medicamentos de categoría {category.text} en consultas realizadas en {period.text}, multiplicando cantidad por precio unitario.",
        "¿Cuánto suman, en pesos, las recetas de medicamentos {category.text} (cantidad por precio unitario) asociadas a consultas de {period.text}?",
    ],
    "most_common_diagnosis_city_sex": [
        "¿Cuál es el diagnóstico más frecuente en las consultas de pacientes de sexo {sex.text} de {city.text}?",
        "Entre las consultas con diagnóstico de pacientes de sexo {sex.text} residentes en {city.text}, ¿qué diagnóstico aparece más veces?",
        "Indica el diagnóstico que más se repite en consultas de pacientes de {city.text} de sexo {sex.text}.",
    ],
    "count_admitted_patients_with_diagnosis": [
        "¿Cuántos pacientes distintos hospitalizados en {department.text} tienen al menos una consulta con diagnóstico {diagnosis.text}?",
        "¿Cuántos pacientes diferentes con alguna hospitalización en el departamento de {department.text} han sido diagnosticados alguna vez con {diagnosis.text}?",
        "Indica el número de pacientes distintos que estuvieron hospitalizados en {department.text} y registran alguna consulta diagnosticada como {diagnosis.text}.",
    ],
    "exists_department_floor_avg_cost": [
        "¿Hay algún departamento en el piso {floor.text} cuyo costo promedio de consulta completada supere {cost.text} pesos?",
        "¿Existe al menos un departamento ubicado en el piso {floor.text} con un promedio de costo de consultas completadas mayor a {cost.text} pesos?",
        "Responde sí o no: ¿algún departamento del piso {floor.text} promedia más de {cost.text} pesos por consulta completada?",
    ],
    "top_k_medications_quantity_category": [
        "¿Cuáles son los {k.text} medicamentos de la categoría {category.text} con más unidades recetadas en total? Indica nombre y unidades, de mayor a menor.",
        "Ordena de mayor a menor los {k.text} medicamentos {category.text} con mayor cantidad total recetada, entregando nombre y total de unidades.",
        "Entrega el ranking descendente de los {k.text} medicamentos de categoría {category.text} más recetados en unidades, con nombre y cantidad total.",
    ],
    "supervisor_of_doctor": [
        "¿Quién es el supervisor del médico {doctor.text}?",
        "¿Cómo se llama el médico que supervisa a {doctor.text}?",
        "Indica el nombre del supervisor de {doctor.text}.",
    ],
    "subordinates_of_doctor": [
        "¿Qué médicos son supervisados por {doctor.text}?",
        "Lista los nombres de los médicos cuyo supervisor es {doctor.text}.",
        "¿Cuáles médicos tienen a {doctor.text} como supervisor?",
    ],
    "doctor_hired_before_supervisor_bool": [
        "¿El médico {doctor.text} fue contratado antes que su supervisor?",
        "¿Es la fecha de contratación de {doctor.text} anterior a la de su supervisor?",
        "Responde sí o no: ¿{doctor.text} lleva más tiempo contratado que su supervisor?",
    ],

    # ---------------------------------------------------------------- EXPERT
    "doctors_earning_more_than_supervisor_hired_after": [
        "¿Qué médicos contratados a partir de {year.text} ganan más que su supervisor?",
        "Lista los médicos con fecha de contratación desde {year.text} (inclusive) cuyo salario supera el de su supervisor.",
        "¿Cuáles médicos contratados desde el año {year.text} en adelante tienen un salario mayor que el de la persona que los supervisa?",
    ],
    "count_doctors_earning_more_than_supervisor_before": [
        "¿Cuántos médicos contratados antes de {year.text} ganan más que su supervisor?",
        "¿Cuántos médicos con fecha de contratación anterior al 1 de enero de {year.text} tienen un salario superior al de su supervisor?",
        "Indica el número de médicos contratados antes del año {year.text} cuyo salario supera el de quien los supervisa.",
    ],
    "patient_spend_above_city_avg_bool": [
        "¿El gasto total en medicamentos recetados (cantidad × precio unitario) del paciente {patient.text} es mayor que el gasto promedio por paciente de su misma ciudad (considerando solo pacientes con recetas)?",
        "Comparado con el promedio de gasto en recetas de los pacientes de su ciudad que tienen alguna receta, ¿gastó más el paciente {patient.text}?",
        "Responde sí o no: ¿el paciente {patient.text} gastó en medicamentos recetados más que el promedio de los pacientes con recetas de su ciudad?",
    ],
    "doctor_more_appointments_than_supervisor_bool": [
        "¿El médico {doctor.text} tuvo más consultas registradas (de cualquier estado) que su supervisor en {period.text}?",
        "En {period.text}, ¿registró {doctor.text} un número de consultas mayor que el de su supervisor, contando todos los estados?",
        "Responde sí o no: ¿{doctor.text} superó a su supervisor en cantidad de consultas registradas durante {period.text}?",
    ],
    "patients_admitted_dept_without_appointment_same_dept": [
        "¿Qué pacientes de {city.text} han estado hospitalizados en {department.text} sin haber tenido nunca una consulta con un médico de ese mismo departamento?",
        "Lista los pacientes residentes en {city.text} con alguna hospitalización en el departamento de {department.text} que no registran ninguna consulta con médicos de {department.text}.",
        "¿Cuáles pacientes de {city.text} fueron hospitalizados en {department.text} pero nunca tuvieron una consulta (de ningún estado) con un médico de ese departamento?",
    ],
    "days_between_first_last_admission_patient": [
        "¿Cuántos días transcurrieron entre la primera y la última fecha de ingreso hospitalario del paciente {patient.text}?",
        "Calcula la diferencia en días entre el ingreso hospitalario más antiguo y el más reciente del paciente {patient.text}.",
        "¿Cuántos días separan la primera hospitalización de la última (por fecha de ingreso) del paciente {patient.text}?",
    ],
    "patient_most_expensive_admission_city_room": [
        "¿Qué paciente de {city.text} tuvo la hospitalización más costosa en cama {room.text}, considerando costo diario × días entre ingreso y alta?",
        "Entre las hospitalizaciones con alta en cama {room.text} de pacientes de {city.text}, ¿quién tuvo la de mayor costo total (costo diario por días de estadía)?",
        "Indica el paciente residente en {city.text} cuya hospitalización finalizada en cama {room.text} tuvo el mayor costo (costo diario multiplicado por días de estadía).",
    ],
    "count_admitted_and_consulted_same_dept_year": [
        "¿Cuántos pacientes distintos hospitalizados en {department.text} con ingreso en {period.text} han tenido además alguna consulta completada con un médico de ese mismo departamento (en cualquier fecha)?",
        "De los pacientes con ingreso hospitalario en {department.text} durante {period.text}, ¿cuántos (sin repetir) registran al menos una consulta completada con médicos de {department.text}?",
        "Indica el número de pacientes distintos que ingresaron a {department.text} en {period.text} y que también tienen alguna consulta completada con un médico del mismo departamento.",
    ],
    "medications_never_prescribed_price": [
        "¿Qué medicamentos que nunca han sido recetados tienen un precio unitario de {cmp.text} {price.text} pesos?",
        "Lista los medicamentos sin ninguna receta asociada cuyo precio unitario es {cmp.text} {price.text} pesos.",
        "¿Cuáles medicamentos del catálogo no aparecen en ninguna receta y cuestan {cmp.text} {price.text} pesos por unidad?",
    ],
    "doctors_no_appointments_specialty_month": [
        "¿Qué médicos de {specialty.text} no tuvieron ninguna consulta registrada en {period.text}?",
        "Lista los médicos con especialidad {specialty.text} sin consultas (de ningún estado) durante {period.text}.",
        "¿Cuáles médicos de {specialty.text} no registran consultas en {period.text}?",
    ],
    "patients_no_appointments_city_insurance_year": [
        "¿Qué pacientes de {city.text} con previsión {insurance.text} no tuvieron ninguna consulta registrada en {period.text}?",
        "Lista los pacientes residentes en {city.text} afiliados a {insurance.text} sin consultas (de ningún estado) durante {period.text}.",
        "¿Cuáles pacientes de {city.text} con previsión {insurance.text} no registran consultas en {period.text}?",
    ],
    "departments_avg_stay_above_overall_room": [
        "¿Qué departamentos tuvieron en {period.text} una estadía promedio en cama {room.text} (hospitalizaciones con alta) superior a la estadía promedio general en cama {room.text} de ese mismo período?",
        "Para hospitalizaciones con alta en cama {room.text} con ingreso en {period.text}, lista los departamentos cuyo promedio de días de estadía supera el promedio de todas las hospitalizaciones de ese tipo en {period.text}.",
        "¿Cuáles departamentos superan el promedio general de días de estadía en cama {room.text} durante {period.text}, considerando solo hospitalizaciones con alta registrada?",
    ],
    "top_k_patients_total_stay_city": [
        "¿Cuáles son los {k.text} pacientes de {city.text} con más días acumulados de hospitalización (solo altas registradas)? Indica nombre y total de días, de mayor a menor.",
        "Ordena de mayor a menor los {k.text} pacientes residentes en {city.text} que suman más días de estadía hospitalaria finalizada, con nombre y días.",
        "Entrega el ranking descendente de los {k.text} pacientes de {city.text} con mayor total de días entre ingreso y alta, indicando nombre y días acumulados.",
    ],
    "doctors_more_appointments_than_supervisor_year": [
        "¿Qué médicos de {specialty.text} registraron más consultas (de cualquier estado) que su supervisor en {period.text}?",
        "Lista los médicos con especialidad {specialty.text} cuyo número de consultas registradas en {period.text} supera al de su supervisor.",
        "¿Cuáles médicos de {specialty.text} tuvieron en {period.text} más consultas que la persona que los supervisa, contando todos los estados?",
    ],
    "patients_spend_above_factor_city_avg": [
        "¿Qué pacientes de {city.text} gastaron en medicamentos recetados (cantidad × precio unitario) más de {factor.text} del gasto promedio por paciente de {city.text} (considerando solo pacientes con recetas)?",
        "Lista los pacientes residentes en {city.text} cuyo gasto total en recetas supera {factor.text} del promedio de gasto de los pacientes de {city.text} con alguna receta.",
        "¿Cuáles pacientes de {city.text} tienen un gasto en medicamentos recetados mayor a {factor.text} del gasto promedio de su ciudad entre pacientes con recetas?",
    ],
    "month_most_appointments_specialty_year": [
        "¿En qué mes de {period.text} se registraron más consultas de {specialty.text}? Responde en formato AAAA-MM.",
        "Indica el mes (formato AAAA-MM) de {period.text} con mayor número de consultas registradas para la especialidad {specialty.text}.",
        "¿Cuál fue el mes de {period.text} con más consultas (de cualquier estado) de médicos de {specialty.text}? Usa el formato AAAA-MM.",
    ],
    "doctors_hired_before_supervisor_specialty": [
        "¿Qué médicos de {specialty.text} fueron contratados antes que su supervisor?",
        "Lista los médicos con especialidad {specialty.text} cuya fecha de contratación es anterior a la de su supervisor.",
        "¿Cuáles médicos de {specialty.text} llevan más tiempo contratados que la persona que los supervisa?",
    ],
    "department_extreme_status_ratio_year": [
        "¿Qué departamento tuvo la {direction.text} proporción de consultas {status.text} sobre el total de sus consultas en {period.text}?",
        "En {period.text}, ¿cuál departamento (según el médico que atiende) presenta la {direction.text} fracción de consultas {status.text} respecto de todas sus consultas?",
        "Indica el departamento con la {direction.text} tasa de consultas {status.text} durante {period.text}.",
    ],
    "count_admitted_room_patients_prescribed_category": [
        "¿Cuántos pacientes distintos hospitalizados alguna vez en cama {room.text} han recibido recetas de medicamentos {category.text}?",
        "¿Cuántos pacientes diferentes con al menos una hospitalización en cama {room.text} tienen alguna receta de un medicamento de categoría {category.text}?",
        "Indica el número de pacientes distintos que estuvieron en cama {room.text} y a quienes se les recetó algún medicamento {category.text}.",
    ],
    "doctors_above_department_avg_salary_specialty": [
        "¿Qué médicos de {specialty.text} ganan más que el salario promedio de su propio departamento?",
        "Lista los médicos con especialidad {specialty.text} cuyo salario supera el promedio salarial del departamento al que pertenecen.",
        "¿Cuáles médicos de {specialty.text} tienen un salario por encima del promedio de su departamento?",
    ],
    "count_doctors_above_department_avg_salary_hired_after": [
        "¿Cuántos médicos contratados a partir de {year.text} ganan más que el salario promedio de su propio departamento?",
        "¿Cuántos médicos con fecha de contratación desde {year.text} (inclusive) superan el promedio salarial de su departamento?",
        "Indica el número de médicos contratados desde el año {year.text} en adelante cuyo salario es mayor al promedio del departamento al que pertenecen.",
    ],
}


# --------------------------------------------------------------------------------------------------
# Generador de preguntas, tareas y particiones
# --------------------------------------------------------------------------------------------------

class QuestionGenerator:
    def __init__(self, seed: int = 2026, heldout_fraction: float = 0.15) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.heldout_fraction = heldout_fraction

    def to_question(self, query: GeneratedQuery) -> str:
        variants = QUESTIONS[query.template.id]
        return self.rng.choice(variants).format(**query.slots)

    def build_tasks(self, queries: List[GeneratedQuery]) -> List[Task]:
        queries = list(queries)
        self.rng.shuffle(queries)
        width = max(4, len(str(len(queries))))
        tasks = []
        for i, q in enumerate(queries, start=1):
            t = q.template
            tasks.append(Task(
                id=f"hosp_{i:0{width}d}",
                database=DATABASE_FILE,
                difficulty=t.difficulty,
                template_id=t.id,
                features=list(t.features),
                question=self.to_question(q),
                gold_sql=q.sql,
                answer_type=t.answer_type,
                ordered=t.ordered,
                result_columns=list(t.result_columns),
                expected_result=q.expected_result,
            ))
        return tasks

    def choose_heldout_templates(self, tasks: List[Task]) -> List[str]:
        """~20 % de las plantillas por nivel (mínimo 1), evitando las de mayor volumen."""
        by_level: Dict[str, Counter] = defaultdict(Counter)
        for t in tasks:
            by_level[t.difficulty][t.template_id] += 1
        heldout: List[str] = []
        for level, counts in sorted(by_level.items()):
            ids = sorted(counts)
            n = max(1, round(len(ids) * self.heldout_fraction))
            median = sorted(counts.values())[len(counts) // 2]
            candidates = [i for i in ids if counts[i] <= median] or ids
            heldout += self.rng.sample(candidates, min(n, len(candidates)))
        return heldout

    def split(self, tasks: List[Task], ratios: Sequence[float] = (0.70, 0.15, 0.15)) -> Dict[str, List[Task]]:
        """
        Las plantillas reservadas van íntegramente a test (marcadas `template_heldout=True`).
        El resto se reparte por plantilla según `ratios`, de modo que test contiene dos bloques:
        instancias de plantillas vistas en train (misma distribución) y de plantillas no vistas.
        """
        _, eval_r, test_r = ratios
        heldout = set(self.choose_heldout_templates(tasks))
        splits: Dict[str, List[Task]] = {"train": [], "eval": [], "test": []}

        regular = [t for t in tasks if t.template_id not in heldout]
        for t in tasks:
            if t.template_id in heldout:
                t.template_heldout = True
                splits["test"].append(t)

        by_template: Dict[str, List[Task]] = defaultdict(list)
        for t in regular:
            by_template[t.template_id].append(t)
        for template_id in sorted(by_template):
            group = by_template[template_id]
            self.rng.shuffle(group)
            n_test = round(len(group) * test_r)
            n_eval = round(len(group) * eval_r)
            splits["test"] += group[:n_test]
            splits["eval"] += group[n_test:n_test + n_eval]
            splits["train"] += group[n_test + n_eval:]
        for s in splits.values():
            s.sort(key=lambda t: t.id)
        return splits

    @staticmethod
    def export_jsonl(splits: Dict[str, List[Task]], out_dir: str | Path = "dataset") -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, tasks in splits.items():
            with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for t in tasks:
                    f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def summarize(splits: Dict[str, List[Task]]) -> str:
    lines = []
    levels = ["easy", "medium", "hard", "expert"]
    lines.append(f"{'split':<8}" + "".join(f"{l:>8}" for l in levels) + f"{'total':>8}{'heldout':>9}")
    for name, tasks in splits.items():
        c = Counter(t.difficulty for t in tasks)
        h = sum(t.template_heldout for t in tasks)
        lines.append(f"{name:<8}" + "".join(f"{c.get(l, 0):>8}" for l in levels) + f"{len(tasks):>8}{h:>9}")
    all_tasks = [t for s in splits.values() for t in s]
    lines.append("")
    lines.append("answer_type: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(t.answer_type for t in all_tasks).items())))
    lines.append(f"templates: {len({t.template_id for t in all_tasks})} / {len(TEMPLATES)}   "
                 f"heldout: {sorted({t.template_id for t in all_tasks if t.template_heldout})}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Genera dataset/{train,eval,test}.jsonl")
    parser.add_argument("--total", type=int, default=sum(DEFAULT_TARGETS.values()))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="dataset")
    args = parser.parse_args(argv)

    scale = args.total / sum(DEFAULT_TARGETS.values())
    targets = {k: round(v * scale) for k, v in DEFAULT_TARGETS.items()}

    qg = QueryGenerator(seed=args.seed)
    queries = qg.generate(targets)
    gen = QuestionGenerator(seed=args.seed)
    tasks = gen.build_tasks(queries)
    splits = gen.split(tasks)
    gen.export_jsonl(splits, args.out)
    print(summarize(splits))


if __name__ == "__main__":
    main()
