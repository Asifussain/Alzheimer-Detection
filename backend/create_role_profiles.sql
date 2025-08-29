-- Create role-specific profiles for email auth users
-- This ensures dashboard components work properly by creating the expected profile records

-- Create doctor profiles for email auth doctors
INSERT INTO doctor_profiles (user_id, medical_license, specialization, verification_status)
SELECT 
    id,
    'TEMP-' || SUBSTRING(unique_identifier, 1, 10) as medical_license,
    'General Medicine' as specialization,
    'verified' as verification_status
FROM user_profiles 
WHERE auth_provider = 'email' 
AND role = 'doctor'
AND id NOT IN (SELECT user_id FROM doctor_profiles)
ON CONFLICT (user_id) DO UPDATE SET verification_status = 'verified';

-- Create patient profiles for email auth patients
INSERT INTO patient_profiles (user_id, patient_id, verification_status)
SELECT 
    id,
    unique_identifier as patient_id,
    'verified' as verification_status
FROM user_profiles 
WHERE auth_provider = 'email' 
AND role = 'patient'
AND id NOT IN (SELECT user_id FROM patient_profiles)
ON CONFLICT (user_id) DO UPDATE SET verification_status = 'verified';

-- Create admin profiles for email auth admins
INSERT INTO admin_profiles (user_id, employee_id, department)
SELECT 
    id,
    unique_identifier as employee_id,
    'Administration' as department
FROM user_profiles 
WHERE auth_provider = 'email' 
AND role = 'admin'
AND id NOT IN (SELECT user_id FROM admin_profiles)
ON CONFLICT (user_id) DO NOTHING;

-- Verify the profiles were created
SELECT 
    up.email,
    up.role,
    up.auth_provider,
    up.account_status,
    CASE 
        WHEN up.role = 'doctor' THEN dp.verification_status
        WHEN up.role = 'patient' THEN pp.verification_status
        ELSE 'N/A'
    END as profile_verification_status
FROM user_profiles up
LEFT JOIN doctor_profiles dp ON up.id = dp.user_id AND up.role = 'doctor'
LEFT JOIN patient_profiles pp ON up.id = pp.user_id AND up.role = 'patient'
WHERE up.auth_provider = 'email'
ORDER BY up.email;