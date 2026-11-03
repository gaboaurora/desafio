"""
Módulo 4: Evaluación de Línea Base y Robustez (Sección I.4 y Sección II del Desafío)
Responsabilidad:
  - Medir `pass@1` y `pass@k` por nivel de dificultad en varios modelos abiertos (incluido el modelo objetivo).
  - Evaluar casos de prueba diseñados para intentar engañar al verificador (reward hacking).
"""

from typing import Any, Dict, List


class BaselineEvaluator:
    """Esqueleto para medir la línea base (`pass@1`, `pass@k`) y auditar el verificador."""

    def evaluate_pass_at_k(
        self, model_name: str, test_tasks: List[Dict[str, Any]], k: int = 5
    ) -> Dict[str, Any]:
        """Calcula `pass@1` y `pass@k` desglosado por nivel de dificultad."""
        raise NotImplementedError

    def test_verifier_robustness(self) -> List[Dict[str, Any]]:
        """Ejecuta casos de prueba de equivalencia de respuestas e intentos de reward hacking."""
        raise NotImplementedError
