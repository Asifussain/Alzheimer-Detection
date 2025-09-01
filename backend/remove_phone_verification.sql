-- Remove phone verification requirement for all email auth users
-- This makes all email users active and phone verified

-- Update all email authentication users
UPDATE user_profiles 
SET 
    account_status = 'active',
    phone_verified = true
WHERE auth_provider = 'email';

-- Verify the update
SELECT email, auth_provider, account_status, phone_verified, role
FROM user_profiles 
WHERE auth_provider = 'email'
ORDER BY email;

-- Show summary
SELECT 
    auth_provider,
    account_status,
    phone_verified,
    COUNT(*) as user_count
FROM user_profiles 
GROUP BY auth_provider, account_status, phone_verified
ORDER BY auth_provider, account_status;