-- Sample data insertion for testing the email/password authentication system
-- Run this in your Supabase SQL editor to add sample hospitals and related data

-- Insert sample hospitals
INSERT INTO hospitals (id, hospital_code, name, address, phone, email, license_number, status) VALUES
(gen_random_uuid(), 'HSP001', 'General Medical Center', '123 Healthcare Ave, Medical City, MC 12345', '+1-555-0101', 'contact@generalmedical.com', 'LIC001', 'active'),
(gen_random_uuid(), 'HSP002', 'St. Mary''s Hospital', '456 Saint Mary Street, Healing Town, HT 67890', '+1-555-0102', 'info@stmarys.org', 'LIC002', 'active'),
(gen_random_uuid(), 'HSP003', 'City Central Hospital', '789 Central Plaza, Downtown, DT 11111', '+1-555-0103', 'admin@citycentral.health', 'LIC003', 'active'),
(gen_random_uuid(), 'HSP004', 'Regional Neurological Institute', '321 Brain Research Blvd, Neuroville, NV 22222', '+1-555-0104', 'research@neurology.edu', 'LIC004', 'active'),
(gen_random_uuid(), 'HSP005', 'Community Health Center', '654 Community Drive, Hometown, HT 33333', '+1-555-0105', 'contact@communityhealth.org', 'LIC005', 'active'),
(gen_random_uuid(), 'HSP006', 'University Medical Hospital', '987 University Campus, College Town, CT 44444', '+1-555-0106', 'hospital@university.edu', 'LIC006', 'active'),
(gen_random_uuid(), 'HSP007', 'Advanced Brain Clinic', '147 Neural Network St, Cognitive City, CC 55555', '+1-555-0107', 'clinic@advancedbrain.com', 'LIC007', 'active'),
(gen_random_uuid(), 'HSP008', 'Memorial Medical Center', '258 Memorial Way, Tribute Town, TT 66666', '+1-555-0108', 'info@memorialmed.org', 'LIC008', 'active')
ON CONFLICT (hospital_code) DO NOTHING;

-- Insert sample blood groups (if table exists and is empty)
INSERT INTO blood_groups (blood_type) VALUES
('A+'), ('A-'), ('B+'), ('B-'), ('AB+'), ('AB-'), ('O+'), ('O-')
ON CONFLICT (blood_type) DO NOTHING;

-- Insert sample qualifications (if table exists and is empty)
INSERT INTO qualifications (qualification_name, specialization) VALUES
('Doctor of Medicine (MD)', 'General Medicine'),
('Doctor of Medicine (MD)', 'Neurology'),
('Doctor of Medicine (MD)', 'Psychiatry'),
('Doctor of Medicine (MD)', 'Geriatric Medicine'),
('Doctor of Osteopathic Medicine (DO)', 'General Medicine'),
('Master of Science in Nursing (MSN)', 'Nursing'),
('Bachelor of Science in Nursing (BSN)', 'Nursing'),
('Certified Medical Technologist', 'Medical Technology'),
('Registered Nurse (RN)', 'Nursing'),
('Licensed Practical Nurse (LPN)', 'Nursing')
ON CONFLICT (qualification_name) DO NOTHING;

-- Verify the data was inserted
SELECT 'Hospitals' as table_name, COUNT(*) as record_count FROM hospitals
UNION ALL
SELECT 'Blood Groups' as table_name, COUNT(*) as record_count FROM blood_groups
UNION ALL  
SELECT 'Qualifications' as table_name, COUNT(*) as record_count FROM qualifications;

-- Display the hospitals that were added
SELECT hospital_code, name, address, status 
FROM hospitals 
ORDER BY hospital_code;