-- Database schema extensions for email/password authentication
-- WARNING: This is for reference only. Please run these commands carefully in your Supabase SQL editor.

-- Add new columns to user_profiles table for email authentication
ALTER TABLE user_profiles 
ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255),
ADD COLUMN IF NOT EXISTS auth_provider VARCHAR(50) DEFAULT 'google',
ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS email_verification_token VARCHAR(255),
ADD COLUMN IF NOT EXISTS password_reset_token VARCHAR(255),
ADD COLUMN IF NOT EXISTS password_reset_expires TIMESTAMP WITH TIME ZONE;

-- Add index for better performance on email lookups
CREATE INDEX IF NOT EXISTS idx_user_profiles_email_provider ON user_profiles(email, auth_provider);
CREATE INDEX IF NOT EXISTS idx_user_profiles_email_verification ON user_profiles(email_verification_token);
CREATE INDEX IF NOT EXISTS idx_user_profiles_password_reset ON user_profiles(password_reset_token);

-- Update existing users to have 'google' as auth_provider
UPDATE user_profiles SET auth_provider = 'google' WHERE auth_provider IS NULL;

-- Add constraint to ensure password_hash is required for email auth
ALTER TABLE user_profiles 
ADD CONSTRAINT check_password_for_email_auth 
CHECK (
    (auth_provider = 'google' AND password_hash IS NULL) OR
    (auth_provider = 'email' AND password_hash IS NOT NULL)
);

-- Create function to generate unique identifiers for new users
CREATE OR REPLACE FUNCTION generate_unique_identifier(
    p_hospital_id UUID,
    p_role VARCHAR(50)
)
RETURNS VARCHAR(50) AS $$
DECLARE
    hospital_code VARCHAR(10);
    role_prefix VARCHAR(3);
    sequence_num INTEGER;
    unique_id VARCHAR(50);
BEGIN
    -- Get hospital code
    SELECT hospital_code INTO hospital_code 
    FROM hospitals 
    WHERE id = p_hospital_id;
    
    IF hospital_code IS NULL THEN
        RAISE EXCEPTION 'Hospital not found';
    END IF;
    
    -- Set role prefix
    role_prefix := CASE 
        WHEN p_role = 'patient' THEN 'PAT'
        WHEN p_role = 'doctor' THEN 'DOC'
        WHEN p_role = 'admin' THEN 'ADM'
        ELSE 'USR'
    END;
    
    -- Get next sequence number for this hospital and role
    SELECT COUNT(*) + 1 INTO sequence_num
    FROM user_profiles 
    WHERE hospital_id = p_hospital_id AND role = p_role;
    
    -- Generate unique identifier
    unique_id := hospital_code || '-' || role_prefix || '-' || LPAD(sequence_num::text, 4, '0');
    
    -- Ensure uniqueness (in case of concurrent inserts)
    WHILE EXISTS (SELECT 1 FROM user_profiles WHERE unique_identifier = unique_id) LOOP
        sequence_num := sequence_num + 1;
        unique_id := hospital_code || '-' || role_prefix || '-' || LPAD(sequence_num::text, 4, '0');
    END LOOP;
    
    RETURN unique_id;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically generate unique identifiers
CREATE OR REPLACE FUNCTION set_unique_identifier()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.unique_identifier IS NULL THEN
        NEW.unique_identifier := generate_unique_identifier(NEW.hospital_id, NEW.role);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Only create trigger if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
        WHERE tgname = 'trigger_set_unique_identifier' 
        AND tgrelid = 'user_profiles'::regclass
    ) THEN
        CREATE TRIGGER trigger_set_unique_identifier
            BEFORE INSERT ON user_profiles
            FOR EACH ROW
            EXECUTE FUNCTION set_unique_identifier();
    END IF;
END
$$;

-- Add some sample data for testing (optional)
-- INSERT INTO hospitals (id, hospital_code, name, address, phone, email) VALUES 
-- (gen_random_uuid(), 'HSP001', 'Test Hospital', '123 Test St', '+1234567890', 'test@hospital.com')
-- ON CONFLICT DO NOTHING;

-- Ensure JWT secret environment variable is documented
COMMENT ON COLUMN user_profiles.password_hash IS 'bcrypt hashed password for email authentication';
COMMENT ON COLUMN user_profiles.auth_provider IS 'Authentication provider: google or email';
COMMENT ON COLUMN user_profiles.email_verified IS 'Whether email address has been verified (for email auth)';
COMMENT ON COLUMN user_profiles.email_verification_token IS 'Token used for email verification';
COMMENT ON COLUMN user_profiles.password_reset_token IS 'Token used for password reset';
COMMENT ON COLUMN user_profiles.password_reset_expires IS 'Expiration time for password reset token';