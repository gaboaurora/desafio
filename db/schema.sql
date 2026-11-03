-- Esquema: hospital_v1
-- Base de datos de gestión de un hospital: departamentos, médicos, pacientes, consultas, hospitalizaciones y recetas.
-- Generado por generator/schema_generator.py. No editar a mano.
PRAGMA foreign_keys = ON;

-- Departamentos clínicos del hospital.
CREATE TABLE departments (
    department_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE CHECK (name IN ('Cardiología', 'Pediatría', 'Neurología', 'Oncología', 'Traumatología', 'Dermatología', 'Ginecología', 'Medicina Interna', 'Psiquiatría', 'Urgencias')),
    floor INTEGER NOT NULL CHECK (floor BETWEEN 1 AND 8),
    annual_budget INTEGER NOT NULL CHECK (annual_budget BETWEEN 200000000 AND 2000000000)
);

-- Médicos del hospital. Cada médico pertenece a un departamento y puede tener un supervisor (otro médico).
CREATE TABLE doctors (
    doctor_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    specialty TEXT NOT NULL CHECK (specialty IN ('Cardiología', 'Pediatría', 'Neurología', 'Oncología', 'Traumatología', 'Dermatología', 'Ginecología', 'Medicina Interna', 'Psiquiatría', 'Urgencias')),
    department_id INTEGER NOT NULL REFERENCES departments(department_id),
    supervisor_id INTEGER REFERENCES doctors(doctor_id),
    hire_date TEXT NOT NULL,
    salary INTEGER NOT NULL CHECK (salary BETWEEN 1800000 AND 9000000)
);

-- Pacientes registrados.
CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    age INTEGER NOT NULL CHECK (age BETWEEN 0 AND 95),
    sex TEXT NOT NULL CHECK (sex IN ('F', 'M')),
    blood_type TEXT NOT NULL CHECK (blood_type IN ('O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-')),
    city TEXT NOT NULL CHECK (city IN ('Santiago', 'Valparaíso', 'Concepción', 'La Serena', 'Antofagasta', 'Temuco', 'Rancagua', 'Puerto Montt', 'Iquique', 'Talca')),
    insurance TEXT NOT NULL CHECK (insurance IN ('Fonasa', 'Isapre', 'Particular'))
);

-- Consultas ambulatorias de un paciente con un médico.
CREATE TABLE appointments (
    appointment_id INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(patient_id),
    doctor_id INTEGER NOT NULL REFERENCES doctors(doctor_id),
    appointment_date TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL CHECK (duration_minutes BETWEEN 10 AND 90),
    status TEXT NOT NULL CHECK (status IN ('completed', 'cancelled', 'no_show')),
    diagnosis TEXT CHECK (diagnosis IN ('Hipertensión', 'Diabetes tipo 2', 'Migraña', 'Asma', 'Fractura', 'Dermatitis', 'Arritmia', 'Depresión', 'Gastritis', 'Lumbalgia', 'Anemia', 'Infección respiratoria', 'Control sano', 'Artrosis', 'Epilepsia', 'Tumor benigno', 'Infección urinaria', 'Trastorno de ansiedad')),
    cost INTEGER NOT NULL CHECK (cost BETWEEN 15000 AND 120000)
);

-- Hospitalizaciones (ingresos con cama) de pacientes en un departamento.
CREATE TABLE admissions (
    admission_id INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(patient_id),
    department_id INTEGER NOT NULL REFERENCES departments(department_id),
    attending_doctor_id INTEGER NOT NULL REFERENCES doctors(doctor_id),
    admission_date TEXT NOT NULL,
    discharge_date TEXT,
    room_type TEXT NOT NULL CHECK (room_type IN ('general', 'intermedio', 'UCI')),
    daily_cost INTEGER NOT NULL CHECK (daily_cost BETWEEN 80000 AND 900000),
    CHECK (discharge_date IS NULL OR discharge_date >= admission_date)
);

-- Catálogo de medicamentos.
CREATE TABLE medications (
    medication_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL CHECK (category IN ('analgésico', 'antibiótico', 'antihipertensivo', 'antidiabético', 'antiinflamatorio', 'antidepresivo', 'anticoagulante', 'antihistamínico', 'anticonvulsivo', 'suplemento')),
    unit_price INTEGER NOT NULL CHECK (unit_price BETWEEN 500 AND 60000),
    requires_prescription INTEGER NOT NULL CHECK (requires_prescription IN (0, 1))
);

-- Medicamentos recetados en una consulta (relación N:M entre consultas y medicamentos).
CREATE TABLE prescriptions (
    prescription_id INTEGER PRIMARY KEY,
    appointment_id INTEGER NOT NULL REFERENCES appointments(appointment_id),
    medication_id INTEGER NOT NULL REFERENCES medications(medication_id),
    quantity INTEGER NOT NULL CHECK (quantity BETWEEN 1 AND 60),
    duration_days INTEGER NOT NULL CHECK (duration_days BETWEEN 1 AND 90),
    UNIQUE (appointment_id, medication_id)
);

CREATE INDEX idx_doctors_department_id ON doctors(department_id);
CREATE INDEX idx_doctors_supervisor_id ON doctors(supervisor_id);
CREATE INDEX idx_appointments_patient_id ON appointments(patient_id);
CREATE INDEX idx_appointments_doctor_id ON appointments(doctor_id);
CREATE INDEX idx_admissions_patient_id ON admissions(patient_id);
CREATE INDEX idx_admissions_department_id ON admissions(department_id);
CREATE INDEX idx_admissions_attending_doctor_id ON admissions(attending_doctor_id);
CREATE INDEX idx_prescriptions_appointment_id ON prescriptions(appointment_id);
CREATE INDEX idx_prescriptions_medication_id ON prescriptions(medication_id);
