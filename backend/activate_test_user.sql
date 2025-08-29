-- Script to activate your test user for email authentication
-- Replace 'your-test-email@example.com' with your actual email

-- First, check if the user exists
SELECT id, email, account_status, phone_verified, auth_provider, role
FROM user_profiles 
WHERE email = 'your-test-email@example.com' AND auth_provider = 'email';

-- Activate the user (change the email to match your test account)
UPDATE user_profiles 
SET 
    account_status = 'active',
    phone_verified = true,
    email_verified = true
WHERE email = 'your-test-email@example.com' AND auth_provider = 'email';

-- Verify the update
SELECT id, email, account_status, phone_verified, email_verified, auth_provider, role
FROM user_profiles 
WHERE email = 'your-test-email@example.com' AND auth_provider = 'email';