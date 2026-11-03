-- ============================================================================
-- IIC-3694 Desafío de Investigación: Permisos y Seguridad de Base de Datos
-- Archivo: db/init/02_permissions.sql
-- Descripción: Configura el usuario de solo lectura (`LLM_DB_USER`) que
--              utilizará el harness cuando ejecute consultas generadas por el LLM.
--              Garantiza aislamiento contra DROP/DELETE/UPDATE/INSERT y cuelgues.
-- ============================================================================

\getenv llm_user LLM_DB_USER
\getenv llm_password LLM_DB_PASSWORD
\getenv db_name POSTGRES_DB

-- 1. Crear el rol restringido para el LLM con contraseña desde variables de entorno
CREATE ROLE :"llm_user" WITH
    LOGIN
    PASSWORD :'llm_password'
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOINHERIT
    NOREPLICATION
    CONNECTION LIMIT 20;

-- 2. Revocar permisos públicos peligrosos y otorgar únicamente lectura (SELECT)
REVOKE ALL ON SCHEMA public FROM :"llm_user";
GRANT CONNECT ON DATABASE :"db_name" TO :"llm_user";
GRANT USAGE ON SCHEMA public TO :"llm_user";
GRANT SELECT ON ALL TABLES IN SCHEMA public TO :"llm_user";

-- 3. Asegurar que cualquier tabla futura creada por el admin sea solo de lectura para el LLM
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO :"llm_user";

-- 4. Blindaje a nivel de motor PostgreSQL:
--    - Transacciones forzadas en modo solo-lectura
--    - Timeout estricto de 3000ms para evitar que un CROSS JOIN infinto bloquee el entrenamiento RL
ALTER ROLE :"llm_user" SET default_transaction_read_only = on;
ALTER ROLE :"llm_user" SET statement_timeout = '3000ms';
ALTER ROLE :"llm_user" SET lock_timeout = '1000ms';
ALTER ROLE :"llm_user" SET search_path TO public;
