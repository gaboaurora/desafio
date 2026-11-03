"""
Módulo 3: Datos Derivados para SFT y DPO
Responsabilidad:
  - Generar trazas con un modelo abierto de mayor tamaño e interactuar con el `SQLHarness`.
  - Filtrar trazas exitosas (`reward == 1.0`) para Ajuste Supervisado (SFT).
  - Construir pares de preferencia (`chosen` con `reward == 1.0` vs `rejected` con `reward == 0.0`) para DPO.
"""

from typing import Any, Dict, List


class DerivedDataBuilder:
    """Esqueleto para generar datasets de entrenamiento SFT y DPO."""

    def generate_sft_traces(self, tasks: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
        """Genera interacciones y filtra aquellas con `reward(prompt, completion) == 1.0`."""
        raise NotImplementedError

    def generate_dpo_pairs(self, tasks: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
        """Construye pares `(prompt, chosen, rejected)` para un mismo problema."""
        raise NotImplementedError
