"""
Función de recompensa con la interfaz común del curso:

    reward(prompt, completion) -> float

Flujo:
  1. `extract_task_id(prompt)` recupera la tarea y su ground truth desde el registro
     (dataset/*.jsonl cargado en memoria).
  2. `parse_completion(completion)` extrae SQL y respuesta final; completions inválidas => 0.0.
  3. Si la completion incluye la(s) consulta(s) <sql>, el harness las RE-EJECUTA y verifica que la
     respuesta final sea consistente con el resultado real (evita que el modelo invente <result>).
  4. `verify(...)` compara la respuesta final con el ground truth.

Recompensa binaria por defecto (1.0 / 0.0). Opcionalmente, `shaped_reward` entrega crédito parcial
(p.ej. 0.2 por SQL ejecutable, 0.5 por resultado correcto sin respuesta final bien formateada) para
experimentos del proyecto de curso; la versión final reportada debe ser la binaria.
"""

from __future__ import annotations
import json
import re

from typing import Dict, Optional

from environment import Task
from parser import parse_completion
from verifier import verify
from sql_executor import SQLExecutor

_TASK_REGISTRY: Dict[str, Task] = {}


def load_task_registry(*jsonl_paths: str) -> int:
    """Carga tareas desde archivos JSONL al registro global. Devuelve cuántas tareas hay."""
    for path in jsonl_paths:
        for line in open(path, 'r', encoding='utf-8'):
            task = Task.from_dict(json.loads(line))
            if task.id in _TASK_REGISTRY:
                print(f"CUIDADO: La tarea {task.id} fue reescrita.")
            _TASK_REGISTRY[task.id] = task
    return len(_TASK_REGISTRY)

def get_task(task_id: str) -> Optional[Task]:
    return _TASK_REGISTRY.get(task_id)


def reward(prompt: str, completion: str) -> float:
    """Recompensa binaria: 1.0 si la respuesta final es correcta y la completion es válida."""
    # Se busca la tarea en la base de datos
    task_id = prompt.TASK_ID
    task = get_task(task_id)
    if task is None: 
        raise AttributeError("La tarea pedida no se encuentra en la base de datos.")
    parsed_completion = parse_completion(completion)
    if not parsed_completion.is_valid: return 0.0 # Completion no es válida
    # Se compara el resultado de las queries de sql con el valor entregado por el modelo
    executor = SQLExecutor()
    queries = parsed_completion.sql_queries
    if len(queries) == 0: return 0.0 # No hay query
    real_value = executor.execute(queries[-1])
    # Si hay errores en la ejecución del SQL
    if real_value.error is not None: return 0.0
    clean_value = process_type_sql(task.answer_type, real_value)
    coherence = verify(parsed_completion.final_answer, clean_value, task.answer_type)
    # Si el resultado de la query no es correcto, no hay puntaje
    if not coherence.is_correct: return 0.0
    # Se ejecuta el gold_sql para encontrar el target    
    target = process_type_sql(task.answer_type, executor.execute(task.gold_sql))
    # Se verifica que la respuesta sea correcta
    result = verify(parsed_completion.final_answer, target, task.answer_type)
    if result.is_correct:
        return 1.0
    return 0.0

"""Procesa información extraída de la consulta sql"""
def process_type_sql(answer_type, sql_result):
    rows = sql_result.rows
    if answer_type == "number" or answer_type=="text":
        return rows[0][0]
    elif answer_type == "list":
        return [i[0] for i in rows]
    elif answer_type=="table":
        return [list(i) for i in rows]


def process_type_completion(answer_type, sql_result):
    if answer_type=="table":
        rows = re.findall(r'\[(.*?)\]', sql_result[1:-1])
        result = [row.split(",") for row in rows]
        new_result = []
        for row in result:
            new_row = []
            for value in row:
                try: 
                    value = int(value)
                except ValueError:
                    try: 
                        value = float(value)
                    except ValueError:
                        pass
                new_row.append(value)
            new_result.append(new_row)
        return new_result
    return sql_result

def shaped_reward(prompt: str, completion: str) -> float:
    """Variante con crédito parcial (solo para experimentos; no es la métrica oficial)."""
    raise NotImplementedError

class check:
    def __init__(self, task_id):
        self.TASK_ID = task_id

print(load_task_registry("dataset/train.jsonl"))
print(reward(check('hosp_0007'), "<sql>SELECT m.name FROM prescriptions pr JOIN appointments a ON pr.appointment_id = a.appointment_id JOIN medications m ON pr.medication_id = m.medication_id WHERE a.diagnosis = 'Tumor benigno' GROUP BY m.medication_id, m.name HAVING COUNT(*) > 8 ORDER BY m.name</sql><answer>[Ketorolaco, Metamizol, Paracetamol]</answer>"))
#print(reward(check('hosp_0034'), "<sql>SELECT m.name, SUM(pr.quantity) AS total FROM prescriptions pr JOIN medications m ON pr.medication_id = m.medication_id WHERE m.category = 'anticoagulante' GROUP BY m.medication_id, m.name ORDER BY total DESC LIMIT 3</sql><answer>[[Ácido acetilsalicílico, 1110], [Warfarina, 729], [Rivaroxabán, 704]]</answer>"))
print(process_type_completion("table",'[[Ácido acetilsalicílico, 1110], [Warfarina, 729], [Rivaroxabán, 704]]'))
print(reward(check('hosp_0001'), "<sql>SELECT SUM(a.cost) FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id WHERE d.specialty = 'Medicina Interna' AND a.status = 'completed' AND strftime('%Y-%m', a.appointment_date) = '2024-11'</sql><answer>282000</answer>"))
