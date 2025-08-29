-- Quick bypass for phone verification during testing
-- Replace 'your-test-email@example.com' with your actual email

-- Option 1: Completely bypass phone verification for your test account
UPDATE user_profiles 
SET phone_verified = true
WHERE email = 'your-test-email@example.com' AND auth_provider = 'email';

-- Option 2: Check current OTP for your account (if you want to use the actual flow)
SELECT email, phone, phone_otp, phone_otp_expires_at, phone_otp_attempts
FROM user_profiles 
WHERE email = 'your-test-email@example.com' AND auth_provider = 'email';

-- Option 3: Generate a test OTP you can use
UPDATE user_profiles 
SET 
    phone_otp = '123456',
    phone_otp_expires_at = (NOW() + INTERVAL '10 minutes'),
    phone_otp_attempts = 0
WHERE email = 'your-test-email@example.com' AND auth_provider = 'email';

-- Verify the update
SELECT email, phone, phone_verified, phone_otp, account_status
FROM user_profiles 
WHERE email = 'your-test-email@example.com' AND auth_provider = 'email';