-- ============================================================================
-- IIC-3694 Desafío de Investigación: Entorno de Razonamiento sobre SQL
-- Archivo: db/init/01_schema.sql
-- Descripción: Define el esquema relacional universitario (Students, Courses,
--              Enrollments) para consultas de razonamiento de múltiples pasos.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS public;

-- 1. Tabla: students (Estudiantes)
CREATE TABLE IF NOT EXISTS public.students (
    student_id      SERIAL PRIMARY KEY,
    full_name       VARCHAR(120) NOT NULL,
    age             INTEGER NOT NULL CHECK (age >= 15 AND age <= 100),
    major           VARCHAR(80) NOT NULL,
    admission_year  INTEGER NOT NULL CHECK (admission_year >= 2000 AND admission_year <= 2030),
    scholarship     BOOLEAN NOT NULL DEFAULT FALSE,
    city            VARCHAR(80)
);

COMMENT ON TABLE public.students IS 'Registro de estudiantes matriculados en la universidad.';
COMMENT ON COLUMN public.students.age IS 'Edad del estudiante en años cumplidos.';
COMMENT ON COLUMN public.students.major IS 'Carrera o especialidad principal del estudiante.';

-- 2. Tabla: courses (Cursos)
CREATE TABLE IF NOT EXISTS public.courses (
    course_id       SERIAL PRIMARY KEY,
    course_code     VARCHAR(20) NOT NULL UNIQUE,
    course_name     VARCHAR(120) NOT NULL,
    department      VARCHAR(80) NOT NULL,
    credits         INTEGER NOT NULL CHECK (credits > 0 AND credits <= 30),
    difficulty_tier INTEGER NOT NULL DEFAULT 1 CHECK (difficulty_tier BETWEEN 1 AND 3)
);

COMMENT ON TABLE public.courses IS 'Catálogo de cursos dictados por departamento.';
COMMENT ON COLUMN public.courses.course_name IS 'Nombre oficial del curso (ej. Matemáticas, Física, Bases de Datos).';

-- 3. Tabla: enrollments (Inscripciones y Calificaciones)
CREATE TABLE IF NOT EXISTS public.enrollments (
    enrollment_id   SERIAL PRIMARY KEY,
    student_id      INTEGER NOT NULL REFERENCES public.students(student_id) ON DELETE CASCADE,
    course_id       INTEGER NOT NULL REFERENCES public.courses(course_id) ON DELETE CASCADE,
    semester        VARCHAR(10) NOT NULL,   -- Ejemplo: '2026-1', '2026-2'
    grade           NUMERIC(4, 2) CHECK (grade >= 1.00 AND grade <= 7.00),
    attendance_pct  NUMERIC(5, 2) CHECK (attendance_pct >= 0.00 AND attendance_pct <= 100.00),
    status          VARCHAR(20) NOT NULL DEFAULT 'completed'
                    CHECK (status IN ('enrolled', 'completed', 'withdrawn')),
    CONSTRAINT uq_student_course_semester UNIQUE (student_id, course_id, semester)
);

COMMENT ON TABLE public.enrollments IS 'Inscripciones de estudiantes en cursos por semestre con su nota final (escala 1.0 a 7.0).';

-- Índices para acelerar los JOINs y filtros frecuentes durante RLVR / GRPO
CREATE INDEX IF NOT EXISTS idx_students_age ON public.students(age);
CREATE INDEX IF NOT EXISTS idx_students_major ON public.students(major);
CREATE INDEX IF NOT EXISTS idx_courses_name ON public.courses(course_name);
CREATE INDEX IF NOT EXISTS idx_courses_dept ON public.courses(department);
CREATE INDEX IF NOT EXISTS idx_enrollments_student ON public.enrollments(student_id);
CREATE INDEX IF NOT EXISTS idx_enrollments_course ON public.enrollments(course_id);
CREATE INDEX IF NOT EXISTS idx_enrollments_semester ON public.enrollments(semester);
