# Custom Email/Password Authentication Setup Guide

This guide walks you through setting up the custom email/password authentication system alongside your existing Google OAuth.

## 🔧 Setup Instructions

### 1. Database Migration

First, run the database migration to add the necessary columns:

```sql
-- Open your Supabase SQL editor and run the following:
-- (Copy from: backend/database_migration.sql)

ALTER TABLE user_profiles 
ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255),
ADD COLUMN IF NOT EXISTS auth_provider VARCHAR(50) DEFAULT 'google',
ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS email_verification_token VARCHAR(255),
ADD COLUMN IF NOT EXISTS password_reset_token VARCHAR(255),
ADD COLUMN IF NOT EXISTS password_reset_expires TIMESTAMP WITH TIME ZONE;

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_profiles_email_provider ON user_profiles(email, auth_provider);
CREATE INDEX IF NOT EXISTS idx_user_profiles_email_verification ON user_profiles(email_verification_token);
CREATE INDEX IF NOT EXISTS idx_user_profiles_password_reset ON user_profiles(password_reset_token);

-- Update existing users
UPDATE user_profiles SET auth_provider = 'google' WHERE auth_provider IS NULL;
```

### 2. Environment Configuration

#### Backend Environment Variables

Create or update `backend/.env` with:

```bash
# Copy from backend/.env.example
SUPABASE_URL=your_supabase_url_here
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key_here

# CRITICAL: Generate a strong JWT secret
JWT_SECRET=your-super-secret-jwt-key-change-in-production-min-32-characters

# Other required variables
REDIS_URL=redis://localhost:6379/0
FRONTEND_URL=http://localhost:3000
FLASK_ENV=development
PORT=5000
```

**Generate JWT Secret:**
```bash
# Generate a secure JWT secret
openssl rand -base64 32
```

#### Frontend Environment Variables

Your existing `.env.local` should work, but verify:

```bash
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
NEXT_PUBLIC_API_URL=http://127.0.0.1:5000
```

### 3. Install Dependencies

#### Backend Dependencies

```bash
cd backend
pip install PyJWT==2.8.0 bcrypt==4.1.2
```

#### Frontend Dependencies

No new dependencies needed - already using existing libraries.

### 4. Start the Services

#### Start Backend
```bash
cd backend
python app.py
```

#### Start Frontend
```bash
cd frontend
npm run dev
```

## 🧪 Testing the Authentication System

### Test Scenarios

#### 1. **User Registration Test**

**Test Data:**
```json
{
  "email": "test@example.com",
  "password": "TestPassword123",
  "full_name": "Test User",
  "phone": "+1234567890",
  "hospital_id": "your-hospital-uuid",
  "role": "patient"
}
```

**Steps:**
1. Go to `http://localhost:3000/login`
2. Click "Create Account" 
3. Fill in the form with test data
4. Submit and verify:
   - User is created in database
   - JWT token is stored in localStorage
   - User is redirected to account-pending page

**Expected Result:** User account created with `auth_provider = 'email'` and hashed password.

#### 2. **User Login Test**

**Steps:**
1. Go to `http://localhost:3000/login`
2. Enter email and password from registration
3. Submit and verify:
   - JWT token received and stored
   - User redirected to appropriate dashboard

**Expected Result:** Successful login with JWT authentication.

#### 3. **Google OAuth Test**

**Steps:**
1. Go to `http://localhost:3000/login`
2. Click "Sign In with Google"
3. Complete Google OAuth flow

**Expected Result:** Both auth systems work simultaneously without interference.

#### 4. **Security Feature Tests**

**Rate Limiting:**
```bash
# Test rate limiting (should get 429 after 10 attempts)
for i in {1..12}; do
  curl -X POST http://localhost:5000/api/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"test@test.com","password":"wrong"}'
done
```

**Account Lockout:**
- Try logging in with wrong password 5 times
- Should get account locked message

**Password Validation:**
- Try weak passwords like "password123"
- Should get rejection with strength requirements

### 5. **API Endpoint Tests**

Use curl or Postman to test the endpoints:

```bash
# Registration
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "newuser@example.com",
    "password": "StrongPassword123",
    "full_name": "New User",
    "phone": "+1234567890",
    "hospital_id": "your-hospital-uuid",
    "role": "patient"
  }'

# Login
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "newuser@example.com",
    "password": "StrongPassword123"
  }'

# Verify Token (use token from login response)
curl -X POST http://localhost:5000/api/auth/verify-token \
  -H "Authorization: Bearer YOUR_JWT_TOKEN_HERE"
```

## 🔍 Verification Checklist

### Database Verification

Check that the new columns exist:
```sql
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'user_profiles' 
AND column_name IN ('password_hash', 'auth_provider', 'email_verified');
```

### Frontend Verification

1. **Login Page:** Should show both email form and Google OAuth button
2. **Registration:** All fields should validate properly
3. **Authentication State:** Should work with both auth methods
4. **Routing:** Should redirect correctly based on auth status

### Backend Verification

1. **API Endpoints:** All auth endpoints should be accessible at `/api/auth/*`
2. **Security Headers:** Check that security headers are applied
3. **JWT Tokens:** Should generate and validate properly
4. **Password Hashing:** Passwords should be bcrypt hashed in database

## 🚨 Troubleshooting

### Common Issues

**1. JWT Secret Not Set**
- Error: "JWT secret not configured"
- Fix: Set `JWT_SECRET` in backend environment variables

**2. Database Column Missing**
- Error: "column does not exist"
- Fix: Run the database migration script

**3. CORS Issues**
- Error: "CORS policy blocked"
- Fix: Verify `FRONTEND_URL` in backend environment

**4. Rate Limiting Issues in Development**
- Error: "Rate limit exceeded"
- Fix: Set `FLASK_ENV=development` to allow more requests

### Debug Commands

```bash
# Check backend logs
tail -f backend/logs/app.log

# Check if JWT secret is loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('JWT_SECRET loaded:', bool(os.getenv('JWT_SECRET')))"

# Test database connection
python -c "from backend.supabase_client_setup import get_supabase_client; print('DB connected:', get_supabase_client().table('user_profiles').select('id').limit(1).execute())"
```

## 🔒 Security Considerations

### Production Checklist

- [ ] Set strong `JWT_SECRET` (32+ characters, random)
- [ ] Enable HTTPS enforcement (`FLASK_ENV=production`)
- [ ] Configure proper CORS origins
- [ ] Set up proper logging and monitoring
- [ ] Configure rate limiting with Redis
- [ ] Set up email service for password resets
- [ ] Review IP whitelisting for admin endpoints
- [ ] Enable database encryption at rest
- [ ] Set up proper backup procedures
- [ ] Configure session timeout policies

### Security Headers Implemented

- Content Security Policy (CSP)
- X-Content-Type-Options
- X-Frame-Options  
- X-XSS-Protection
- Rate limiting
- Input validation and sanitization
- Password complexity requirements
- Account lockout protection

## 🎉 Success Criteria

Your authentication system is working correctly when:

1. ✅ Users can register with email/password
2. ✅ Users can login with email/password  
3. ✅ Google OAuth continues to work
4. ✅ Both auth types integrate seamlessly
5. ✅ Security features are active (rate limiting, validation, etc.)
6. ✅ JWTs are generated and validated
7. ✅ User sessions persist across page reloads
8. ✅ Proper redirects based on account status
9. ✅ Password reset flow works (when email is configured)
10. ✅ All existing features continue to work

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the application logs
3. Verify all environment variables are set
4. Ensure database migration was successful
5. Test with curl/Postman to isolate frontend vs backend issues

The system is designed to be backwards compatible - all existing Google OAuth users should continue to work without any changes needed.