# Database Schema

**WARNING**: This schema is for context only and is not meant to be run. Table order and constraints may not be valid for execution.

## Tables

### admin_profiles
```sql
CREATE TABLE public.admin_profiles (
  user_id uuid NOT NULL,
  employee_id character varying,
  department character varying,
  permissions jsonb DEFAULT '{"manage_doctors": true, "manage_patients": true, "view_all_reports": true}'::jsonb,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT admin_profiles_pkey PRIMARY KEY (user_id),
  CONSTRAINT admin_profiles_user_fkey FOREIGN KEY (user_id) REFERENCES public.user_profiles(id)
);
```

### blood_groups
```sql
CREATE TABLE public.blood_groups (
  id integer NOT NULL DEFAULT nextval('blood_groups_id_seq'::regclass),
  blood_type character varying NOT NULL UNIQUE,
  CONSTRAINT blood_groups_pkey PRIMARY KEY (id)
);
```

### doctor_patient_relationships
```sql
CREATE TABLE public.doctor_patient_relationships (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  doctor_id uuid NOT NULL,
  patient_id uuid NOT NULL,
  hospital_id uuid NOT NULL,
  relationship_status character varying DEFAULT 'active'::character varying CHECK (relationship_status::text = ANY (ARRAY['active'::character varying, 'inactive'::character varying, 'terminated'::character varying]::text[])),
  assigned_by uuid,
  assigned_at timestamp with time zone DEFAULT now(),
  notes text,
  CONSTRAINT doctor_patient_relationships_pkey PRIMARY KEY (id),
  CONSTRAINT doctor_patient_relationships_assigned_by_fkey FOREIGN KEY (assigned_by) REFERENCES public.user_profiles(id),
  CONSTRAINT doctor_patient_relationships_doctor_fkey FOREIGN KEY (doctor_id) REFERENCES public.doctor_profiles(user_id),
  CONSTRAINT doctor_patient_relationships_patient_fkey FOREIGN KEY (patient_id) REFERENCES public.patient_profiles(user_id),
  CONSTRAINT doctor_patient_relationships_hospital_fkey FOREIGN KEY (hospital_id) REFERENCES public.hospitals(id)
);
```

### doctor_profiles
```sql
CREATE TABLE public.doctor_profiles (
  user_id uuid NOT NULL,
  medical_license character varying NOT NULL,
  qualification_id integer,
  specialization character varying,
  experience_years integer,
  consultation_fee numeric,
  verification_status character varying DEFAULT 'pending'::character varying CHECK (verification_status::text = ANY (ARRAY['pending'::character varying, 'verified'::character varying, 'rejected'::character varying]::text[])),
  verified_by uuid,
  verified_at timestamp with time zone,
  rejection_reason text,
  is_active boolean DEFAULT true,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT doctor_profiles_pkey PRIMARY KEY (user_id),
  CONSTRAINT doctor_profiles_user_fkey FOREIGN KEY (user_id) REFERENCES public.user_profiles(id),
  CONSTRAINT doctor_profiles_qualification_fkey FOREIGN KEY (qualification_id) REFERENCES public.qualifications(id),
  CONSTRAINT doctor_profiles_verified_by_fkey FOREIGN KEY (verified_by) REFERENCES public.user_profiles(id)
);
```

### eeg_analysis_results
```sql
CREATE TABLE public.eeg_analysis_results (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  session_id uuid NOT NULL,
  prediction character varying NOT NULL,
  confidence_score numeric,
  probabilities jsonb,
  stats_data jsonb,
  similarity_results jsonb,
  consistency_metrics jsonb,
  trial_predictions jsonb,
  timeseries_plot_url text,
  psd_plot_url text,
  similarity_plot_url text,
  analysis_completed_at timestamp with time zone DEFAULT now(),
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT eeg_analysis_results_pkey PRIMARY KEY (id),
  CONSTRAINT eeg_analysis_results_session_fkey FOREIGN KEY (session_id) REFERENCES public.eeg_sessions(id)
);
```

### eeg_sessions
```sql
CREATE TABLE public.eeg_sessions (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  session_code character varying NOT NULL UNIQUE,
  patient_id uuid NOT NULL,
  doctor_id uuid NOT NULL,
  hospital_id uuid NOT NULL,
  filename character varying NOT NULL,
  eeg_data_url text NOT NULL,
  session_date timestamp with time zone DEFAULT now(),
  session_duration integer,
  electrodes_used jsonb,
  sampling_rate integer,
  session_notes text,
  analysis_type character varying DEFAULT 'binary'::character varying CHECK (analysis_type::text = ANY (ARRAY['binary'::character varying, 'multiclass'::character varying, 'regression'::character varying]::text[])),
  status character varying DEFAULT 'uploaded'::character varying CHECK (status::text = ANY (ARRAY['uploaded'::character varying, 'processing'::character varying, 'completed'::character varying, 'failed'::character varying]::text[])),
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT eeg_sessions_pkey PRIMARY KEY (id),
  CONSTRAINT eeg_sessions_doctor_fkey FOREIGN KEY (doctor_id) REFERENCES public.doctor_profiles(user_id),
  CONSTRAINT eeg_sessions_patient_fkey FOREIGN KEY (patient_id) REFERENCES public.patient_profiles(user_id),
  CONSTRAINT eeg_sessions_hospital_fkey FOREIGN KEY (hospital_id) REFERENCES public.hospitals(id)
);
```

### hospitals
```sql
CREATE TABLE public.hospitals (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  hospital_code character varying NOT NULL UNIQUE,
  name character varying NOT NULL,
  address text NOT NULL,
  phone character varying,
  email character varying,
  license_number character varying,
  established_date date,
  status character varying DEFAULT 'active'::character varying CHECK (status::text = ANY (ARRAY['active'::character varying, 'inactive'::character varying, 'suspended'::character varying]::text[])),
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT hospitals_pkey PRIMARY KEY (id)
);
```

### notifications
```sql
CREATE TABLE public.notifications (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL,
  title character varying NOT NULL,
  message text NOT NULL,
  type character varying NOT NULL CHECK (type::text = ANY (ARRAY['report_ready'::character varying, 'verification_update'::character varying, 'system_alert'::character varying, 'assignment'::character varying]::text[])),
  is_read boolean DEFAULT false,
  related_resource_type character varying,
  related_resource_id uuid,
  created_at timestamp with time zone DEFAULT now(),
  read_at timestamp with time zone,
  CONSTRAINT notifications_pkey PRIMARY KEY (id),
  CONSTRAINT notifications_user_fkey FOREIGN KEY (user_id) REFERENCES public.user_profiles(id)
);
```

### patient_profiles
```sql
CREATE TABLE public.patient_profiles (
  user_id uuid NOT NULL,
  patient_id character varying,
  blood_group_id integer,
  emergency_contact_name character varying,
  emergency_contact_phone character varying,
  medical_history text,
  current_medications text,
  allergies text,
  assigned_doctor_id uuid,
  verification_status character varying DEFAULT 'pending'::character varying CHECK (verification_status::text = ANY (ARRAY['pending'::character varying, 'verified'::character varying, 'rejected'::character varying]::text[])),
  verified_by uuid,
  verified_at timestamp with time zone,
  prescription_url text,
  prescription_uploaded_at timestamp with time zone,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT patient_profiles_pkey PRIMARY KEY (user_id),
  CONSTRAINT patient_profiles_user_fkey FOREIGN KEY (user_id) REFERENCES public.user_profiles(id),
  CONSTRAINT patient_profiles_blood_group_fkey FOREIGN KEY (blood_group_id) REFERENCES public.blood_groups(id),
  CONSTRAINT patient_profiles_doctor_fkey FOREIGN KEY (assigned_doctor_id) REFERENCES public.doctor_profiles(user_id),
  CONSTRAINT patient_profiles_verified_by_fkey FOREIGN KEY (verified_by) REFERENCES public.user_profiles(id)
);
```

### predictions
```sql
CREATE TABLE public.predictions (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  user_id uuid,
  filename text NOT NULL,
  prediction text NOT NULL,
  created_at timestamp with time zone DEFAULT now(),
  eeg_data_url text,
  probabilities jsonb,
  stats_data jsonb,
  timeseries_plot_url text,
  psd_plot_url text,
  report_generated_at timestamp with time zone,
  status text,
  similarity_results jsonb,
  similarity_plot_url text,
  consistency_metrics jsonb,
  trial_predictions jsonb,
  patient_pdf_url text,
  technical_pdf_url text,
  clinician_pdf_url text,
  analysis_type text DEFAULT 'binary'::text,
  CONSTRAINT predictions_pkey PRIMARY KEY (id),
  CONSTRAINT predictions_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);
```

### profile_details
```sql
CREATE TABLE public.profile_details (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  profile_id uuid NOT NULL UNIQUE,
  clinic_name text,
  specialization text,
  license_number text,
  hospital_affiliation text,
  certification_id text,
  date_of_birth date,
  emergency_contact_name text,
  emergency_contact_phone text,
  CONSTRAINT profile_details_pkey PRIMARY KEY (id),
  CONSTRAINT profile_details_profile_id_fkey FOREIGN KEY (profile_id) REFERENCES public.profiles(id)
);
```

### profiles
```sql
CREATE TABLE public.profiles (
  id uuid NOT NULL,
  full_name text,
  email text,
  created_at timestamp with time zone DEFAULT timezone('utc'::text, now()),
  role text CHECK (role = ANY (ARRAY['patient'::text, 'technician'::text, 'clinician'::text, 'pending_selection'::text])),
  role_confirmed boolean NOT NULL DEFAULT false,
  CONSTRAINT profiles_pkey PRIMARY KEY (id),
  CONSTRAINT profiles_id_fkey FOREIGN KEY (id) REFERENCES auth.users(id)
);
```

### qualifications
```sql
CREATE TABLE public.qualifications (
  id integer NOT NULL DEFAULT nextval('qualifications_id_seq'::regclass),
  qualification_name character varying NOT NULL UNIQUE,
  specialization character varying,
  CONSTRAINT qualifications_pkey PRIMARY KEY (id)
);
```

### reports
```sql
CREATE TABLE public.reports (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  session_id uuid NOT NULL,
  analysis_result_id uuid NOT NULL,
  report_type character varying NOT NULL CHECK (report_type::text = ANY (ARRAY['patient'::character varying, 'doctor'::character varying, 'technical'::character varying]::text[])),
  report_url text NOT NULL,
  generated_for_user_id uuid NOT NULL,
  generated_by_doctor_id uuid NOT NULL,
  is_accessible boolean DEFAULT true,
  access_expires_at timestamp with time zone,
  generated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT reports_pkey PRIMARY KEY (id),
  CONSTRAINT reports_analysis_fkey FOREIGN KEY (analysis_result_id) REFERENCES public.eeg_analysis_results(id),
  CONSTRAINT reports_session_fkey FOREIGN KEY (session_id) REFERENCES public.eeg_sessions(id),
  CONSTRAINT reports_generated_by_fkey FOREIGN KEY (generated_by_doctor_id) REFERENCES public.doctor_profiles(user_id),
  CONSTRAINT reports_generated_for_fkey FOREIGN KEY (generated_for_user_id) REFERENCES public.user_profiles(id)
);
```

### user_access_logs
```sql
CREATE TABLE public.user_access_logs (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL,
  action character varying NOT NULL,
  resource_type character varying,
  resource_id uuid,
  ip_address inet,
  user_agent text,
  success boolean DEFAULT true,
  error_message text,
  timestamp timestamp with time zone DEFAULT now(),
  CONSTRAINT user_access_logs_pkey PRIMARY KEY (id),
  CONSTRAINT user_access_logs_user_fkey FOREIGN KEY (user_id) REFERENCES public.user_profiles(id)
);
```

### user_profiles
```sql
CREATE TABLE public.user_profiles (
  id uuid NOT NULL,
  hospital_id uuid NOT NULL,
  unique_identifier character varying NOT NULL UNIQUE,
  full_name character varying NOT NULL,
  email character varying NOT NULL UNIQUE,
  phone character varying NOT NULL,
  date_of_birth date,
  address text,
  role character varying NOT NULL CHECK (role::text = ANY (ARRAY['patient'::character varying, 'doctor'::character varying, 'admin'::character varying]::text[])),
  account_status character varying DEFAULT 'pending'::character varying CHECK (account_status::text = ANY (ARRAY['pending'::character varying, 'active'::character varying, 'suspended'::character varying, 'inactive'::character varying]::text[])),
  phone_verified boolean DEFAULT false,
  phone_otp character varying,
  phone_otp_expires_at timestamp with time zone,
  phone_otp_attempts integer DEFAULT 0,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT user_profiles_pkey PRIMARY KEY (id),
  CONSTRAINT user_profiles_hospital_fkey FOREIGN KEY (hospital_id) REFERENCES public.hospitals(id),
  CONSTRAINT user_profiles_id_fkey FOREIGN KEY (id) REFERENCES auth.users(id)
);
```