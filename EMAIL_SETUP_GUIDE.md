# 📧 Email Configuration Setup Guide

## Required Environment Variables

Add these to your `.env.local` file:

```env
# Gmail Configuration for Sending Credentials
EMAIL_USER=your-email@gmail.com
EMAIL_APP_PASSWORD=your-gmail-app-password

# Supabase Service Role Key (for admin operations)
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

## 🔧 Gmail Setup Steps

### 1. Enable 2-Factor Authentication
1. Go to [Google Account Settings](https://myaccount.google.com/)
2. Click "Security" in the left sidebar
3. Enable "2-Step Verification" if not already enabled

### 2. Generate App Password
1. In Google Account Settings → Security
2. Click "2-Step Verification"
3. Scroll down and click "App passwords"
4. Select "Mail" as the app
5. Select "Other (custom name)" as device
6. Enter "AI4NEURO" as the name
7. Copy the 16-character password (no spaces)

### 3. Update .env.local
```env
EMAIL_USER=youremail@gmail.com
EMAIL_APP_PASSWORD=abcd efgh ijkl mnop  # (remove spaces when copying)
```

## 🔑 Supabase Service Role Key Setup

### 1. Get Service Role Key
1. Go to your [Supabase Dashboard](https://app.supabase.com/)
2. Select your project
3. Go to Settings → API
4. Copy the "service_role" key (NOT the anon key)

### 2. Add to .env.local
```env
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 🏥 Database Tables Required

Ensure these tables exist in your Supabase database:

### user_profiles
- id (uuid, primary key)
- email (text)
- full_name (text)
- phone (text)
- role (text)
- hospital_id (text)
- account_status (text)
- unique_identifier (text)
- date_of_birth (date)
- address (text)

### hospitals
- id (uuid, primary key)
- name (text)
- hospital_code (text)

## 🧪 Testing Email Setup

1. Start development server: `npm run dev`
2. Go to Admin Dashboard → Add User tab
3. Try creating a test patient/doctor
4. Check console logs for email sending status
5. Check recipient's email inbox

## 🚨 Common Issues & Solutions

### "Authentication failed" error:
- Double-check Gmail app password (no spaces)
- Ensure 2FA is enabled on Gmail account

### "Invalid token" error:
- Verify SUPABASE_SERVICE_ROLE_KEY is correct
- Check if service role key has proper permissions

### "Admin access required" error:
- Ensure your user has role='admin' in user_profiles table
- Check account_status='active'

### Database errors:
- Verify all required tables exist
- Check RLS (Row Level Security) policies allow admin access

## ✅ Verification Checklist

- [ ] Gmail 2FA enabled
- [ ] App password generated and added to .env.local
- [ ] Service role key added to .env.local  
- [ ] Database tables exist
- [ ] Admin user has proper role and status
- [ ] Test email sending works

## 📋 Example .env.local

```env
# Next.js
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ0eXAiOiJKV1QiLCJhbGc...

# Admin operations
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5c...

# Email configuration
EMAIL_USER=hospital-admin@gmail.com
EMAIL_APP_PASSWORD=abcdefghijklmnop
```

🎉 **Once configured, the system will automatically send welcome emails with login credentials to new users!**