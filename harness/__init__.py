"""
Harness de razonamiento sobre bases de datos SQL (IIC-3694, Desafío de Investigación 2026).

Interfaz común del curso:

    from harness import reward
    score = reward(prompt, completion)   # -> float en [0.0, 1.0]

Módulos:
  - environment.py    : ciclo de interacción (prompt -> SQL -> observación -> respuesta final).
  - prompt_builder.py : construcción del prompt (esquema, instrucciones de formato, pregunta).
  - parser.py         : extracción de la consulta SQL y la respuesta final desde la completion.
  - sql_executor.py   : ejecución segura (solo lectura, timeout, límite de filas).
  - verifier.py       : comparación robusta entre respuesta predicha y ground truth.
  - reward.py         : función `reward(prompt, completion) -> float`.
"""

from harness.reward import reward
from harness.environment import SQLEnvironment

__all__ = ["reward", "SQLEnvironment"]
