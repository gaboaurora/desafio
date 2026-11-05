"""
Pipeline de generación de tareas verificables por construcción.

    schema_generator  -> esquema del hospital: DDL SQLite (db/schema.sql) + descripción para el prompt
    data_generator    -> poblamiento sintético determinista (semilla fija)
    query_generator   -> SQL canónico por nivel de dificultad + ground truth ejecutado en la BD
    question_generator-> pregunta en lenguaje natural asociada a cada SQL (con paráfrasis)
"""
