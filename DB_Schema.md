# Database Schema Documentation

## Overview
This database schema supports a medical EEG (Electroencephalogram) analysis system that manages hospitals, users (patients, doctors, admins), EEG sessions, analysis results, and reporting functionality.

## Core Entities

### Hospitals
**Table:** `hospitals`
- **Purpose:** Stores hospital information and serves as the organizational unit for all users
- **Key Fields:**
  - `id` (UUID, PK): Unique hospital identifier
  - `hospital_code` (VARCHAR, UNIQUE): Human-readable hospital code
  - `name` (VARCHAR): Hospital name
  - `address` (TEXT): Physical address
  - `phone`, `email`: Contact information
  - `license_number` (VARCHAR): Medical license number
  - `status` (VARCHAR): active | inactive | suspended

### User Management

#### User Profiles
**Table:** `user_profiles`
- **Purpose:** Central user table for all system users
- **Key Fields:**
  - `id` (UUID, PK): Links to Supabase auth.users
  - `hospital_id` (UUID, FK): Associated hospital
  - `unique_identifier` (VARCHAR, UNIQUE): System-generated unique ID
  - `full_name`, `email`, `phone`: Personal information
  - `role` (VARCHAR): patient | doctor | admin
  - `account_status` (VARCHAR): pending | active | suspended | inactive
  - `phone_verified` (BOOLEAN): Phone verification status
  - `created_by_admin` (UUID, FK): Admin who created the account

#### Authentication & Security
**Table:** `custom_auth_credentials`
- **Purpose:** Stores authentication credentials and login tracking
- **Key Fields:**
  - `user_profile_id` (UUID, FK): Links to user_profiles
  - `password_hash` (TEXT): Encrypted password
  - `temp_password` (BOOLEAN): Indicates temporary password
  - `failed_login_attempts` (INTEGER): Security tracking
  - `account_locked_until` (TIMESTAMP): Account lockout timestamp

**Table:** `custom_auth_audit_log`
- **Purpose:** Audit trail for authentication events
- **Key Fields:**
  - `user_profile_id` (UUID, FK): User performing action
  - `action` (VARCHAR): Type of action performed
  - `details` (JSONB): Additional action details
  - `ip_address`, `user_agent`: Security tracking information

### Role-Specific Profiles

#### Patient Profiles
**Table:** `patient_profiles`
- **Purpose:** Extended information for patients
- **Key Fields:**
  - `user_id` (UUID, PK, FK): Links to user_profiles
  - `patient_id` (VARCHAR): Hospital-specific patient ID
  - `blood_group_id` (INTEGER, FK): References blood_groups
  - `emergency_contact_name`, `emergency_contact_phone`: Emergency contacts
  - `medical_history`, `current_medications`, `allergies`: Medical information
  - `assigned_doctor_id` (UUID, FK): Primary care doctor
  - `verification_status` (VARCHAR): pending | verified | rejected
  - `prescription_url` (TEXT): Link to prescription document

#### Doctor Profiles
**Table:** `doctor_profiles`
- **Purpose:** Extended information for medical doctors
- **Key Fields:**
  - `user_id` (UUID, PK, FK): Links to user_profiles
  - `medical_license` (VARCHAR): Medical license number
  - `qualification_id` (INTEGER, FK): References qualifications
  - `specialization` (VARCHAR): Medical specialization
  - `experience_years` (INTEGER): Years of practice
  - `consultation_fee` (NUMERIC): Consultation charges
  - `verification_status` (VARCHAR): pending | verified | rejected
  - `verified_by` (UUID, FK): Admin who verified the doctor

#### Admin Profiles
**Table:** `admin_profiles`
- **Purpose:** Administrative users with system permissions
- **Key Fields:**
  - `user_id` (UUID, PK, FK): Links to user_profiles
  - `employee_id` (VARCHAR): Hospital employee ID
  - `department` (VARCHAR): Administrative department
  - `permissions` (JSONB): Role-based permissions object

#### Radiologist Profiles
**Table:** `radiologist_profiles`
- **Purpose:** Specialized medical imaging professionals
- **Key Fields:**
  - `user_id` (UUID, FK): Links to auth.users (legacy table)
  - `radiologist_license` (VARCHAR): Radiologist license
  - `imaging_expertise` (TEXT): Areas of imaging expertise
  - `experience_years` (INTEGER): Years of radiology experience

## EEG Analysis System

### EEG Sessions
**Table:** `eeg_sessions`
- **Purpose:** Stores EEG recording session information
- **Key Fields:**
  - `id` (UUID, PK): Unique session identifier
  - `session_code` (VARCHAR, UNIQUE): Human-readable session code
  - `patient_id` (UUID, FK): Patient being analyzed
  - `doctor_id` (UUID, FK): Doctor conducting session
  - `hospital_id` (UUID, FK): Hospital where session occurred
  - `filename` (VARCHAR): Original EEG data filename
  - `eeg_data_url` (TEXT): Cloud storage URL for EEG data
  - `session_duration` (INTEGER): Duration in minutes/seconds
  - `electrodes_used` (JSONB): Array of electrode positions used
  - `sampling_rate` (INTEGER): Data sampling frequency
  - `analysis_type` (VARCHAR): binary | multiclass | regression
  - `status` (VARCHAR): uploaded | processing | completed | failed

### Analysis Results
**Table:** `eeg_analysis_results`
- **Purpose:** Stores ML analysis results for EEG sessions
- **Key Fields:**
  - `id` (UUID, PK): Unique result identifier
  - `session_id` (UUID, FK): Associated EEG session
  - `prediction` (VARCHAR): Primary analysis result
  - `confidence_score` (NUMERIC): Confidence level (0-1)
  - `probabilities` (JSONB): Class probabilities object
  - `stats_data` (JSONB): Statistical analysis data
  - `similarity_results` (JSONB): Pattern similarity metrics
  - `consistency_metrics` (JSONB): Analysis consistency data
  - `trial_predictions` (JSONB): Individual trial results
  - Plot URLs: `timeseries_plot_url`, `psd_plot_url`, `similarity_plot_url`

### Reports
**Table:** `reports`
- **Purpose:** Generated reports from analysis results
- **Key Fields:**
  - `id` (UUID, PK): Unique report identifier
  - `session_id` (UUID, FK): Source EEG session
  - `analysis_result_id` (UUID, FK): Source analysis results
  - `report_type` (VARCHAR): patient | doctor | technical
  - `report_url` (TEXT): Cloud storage URL for report PDF
  - `generated_for_user_id` (UUID, FK): Target user for report
  - `generated_by_doctor_id` (UUID, FK): Doctor who generated report
  - `is_accessible` (BOOLEAN): Report access status
  - `access_expires_at` (TIMESTAMP): Report expiration date

## Relationships & Assignments

### Doctor-Patient Relationships
**Table:** `doctor_patient_relationships`
- **Purpose:** Manages many-to-many relationships between doctors and patients
- **Key Fields:**
  - `doctor_id` (UUID, FK): Doctor in relationship
  - `patient_id` (UUID, FK): Patient in relationship
  - `hospital_id` (UUID, FK): Hospital context
  - `relationship_status` (VARCHAR): active | inactive | terminated
  - `assigned_by` (UUID, FK): Admin who created assignment
  - `notes` (TEXT): Assignment notes

## Support Tables

### Reference Data
**Table:** `blood_groups`
- **Purpose:** Blood type reference data
- **Fields:** `id` (PK), `blood_type` (UNIQUE)

**Table:** `qualifications`
- **Purpose:** Medical qualification reference data
- **Fields:** `id` (PK), `qualification_name` (UNIQUE), `specialization`

### System Administration
**Table:** `hospital_id_sequences`
- **Purpose:** Manages auto-incrementing ID sequences per hospital and role
- **Key Fields:**
  - `hospital_id` (UUID, FK): Target hospital
  - `role` (VARCHAR): patient | doctor | admin
  - `current_sequence` (INTEGER): Current sequence number

**Table:** `notifications`
- **Purpose:** In-app notification system
- **Key Fields:**
  - `user_id` (UUID, FK): Notification recipient
  - `title`, `message`: Notification content
  - `type` (VARCHAR): report_ready | verification_update | system_alert | assignment
  - `related_resource_type`, `related_resource_id`: Associated system entity

### Security & Auditing
**Table:** `user_access_logs`
- **Purpose:** Tracks user actions for security and compliance
- **Key Fields:**
  - `user_id` (UUID, FK): User performing action
  - `action` (VARCHAR): Action type
  - `resource_type`, `resource_id`: Target resource
  - `ip_address`, `user_agent`: Request metadata
  - `success` (BOOLEAN), `error_message`: Result tracking

**Table:** `password_reset_tokens`
- **Purpose:** Manages password reset functionality
- **Key Fields:**
  - `user_profile_id` (UUID, FK): User requesting reset
  - `token` (TEXT, UNIQUE): Reset token
  - `expires_at` (TIMESTAMP): Token expiration
  - `used` (BOOLEAN): Token usage status

## Legacy Tables

### Profiles (Legacy)
**Table:** `profiles`
- **Purpose:** Legacy user profile table (being phased out)
- **Note:** Links to auth.users, contains role selection functionality

**Table:** `profile_details`
- **Purpose:** Extended details for legacy profiles
- **Note:** Contains clinic and certification information

**Table:** `predictions`
- **Purpose:** Legacy prediction storage
- **Note:** Direct link to auth.users, contains analysis results and report URLs

## Key Relationships

1. **Hospital-Centric Design:** All users belong to a hospital
2. **Role-Based Access:** Users have specific roles with corresponding profile tables
3. **EEG Workflow:** Sessions → Analysis → Reports
4. **Doctor-Patient Assignment:** Flexible many-to-many relationships
5. **Audit Trail:** Comprehensive logging of authentication and access

## Data Flow

1. **User Registration:** Hospital admin creates user profiles
2. **Role Assignment:** Users get role-specific profiles (patient/doctor/admin)
3. **EEG Session Creation:** Doctors create sessions for patients
4. **Analysis Processing:** ML system processes EEG data and stores results
5. **Report Generation:** System generates role-appropriate reports
6. **Access Management:** Reports distributed based on permissions

## Security Features

- Phone and email verification
- Account lockout after failed attempts
- Audit logging for all authentication events
- User access logging for compliance
- Password reset token management
- Role-based permissions system