-- Create patient profile for your email auth user
-- Replace 'hmsiitindore@gmail.com' with your actual email

-- First, let's see your current user profile
SELECT id, email, role, account_status, phone_verified, unique_identifier
FROM user_profiles 
WHERE email = 'hmsiitindore@gmail.com' AND auth_provider = 'email';

-- Create the patient profile record
INSERT INTO patient_profiles (
    user_id, 
    patient_id, 
    verification_status,
    created_at
)
SELECT 
    id as user_id,
    unique_identifier as patient_id,
    'verified' as verification_status,
    NOW() as created_at
FROM user_profiles 
WHERE email = 'hmsiitindore@gmail.com' 
AND auth_provider = 'email'
AND role = 'patient'
AND id NOT IN (SELECT user_id FROM patient_profiles WHERE user_id IS NOT NULL)
ON CONFLICT (user_id) DO UPDATE SET verification_status = 'verified';

-- Verify the patient profile was created
SELECT 
    up.email,
    up.role,
    up.unique_identifier,
    pp.patient_id,
    pp.verification_status,
    pp.created_at
FROM user_profiles up
LEFT JOIN patient_profiles pp ON up.id = pp.user_id
WHERE up.email = 'hmsiitindore@gmail.com' 
AND up.auth_provider = 'email';

-- If you're a doctor instead, run this:
-- INSERT INTO doctor_profiles (user_id, medical_license, specialization, verification_status)
-- SELECT id, 'TEMP-' || SUBSTRING(unique_identifier, 1, 10), 'General Medicine', 'verified'
-- FROM user_profiles WHERE email = 'hmsiitindore@gmail.com' AND role = 'doctor'
-- ON CONFLICT (user_id) DO UPDATE SET verification_status = 'verified';