# Quick Database Setup for Sample Hospitals

You need to add sample hospitals to your database so the registration form works properly.

## Step 1: Run Database Migration (if not done already)

1. Go to your **Supabase Dashboard**
2. Click on **SQL Editor** 
3. Copy and paste the contents from `backend/database_migration.sql`
4. Click **RUN** to execute

## Step 2: Add Sample Hospitals

1. In the same **SQL Editor**
2. Copy and paste the contents from `backend/sample_data_insert.sql`
3. Click **RUN** to execute

This will add:
- ✅ **8 Sample Hospitals** (General Medical Center, St. Mary's Hospital, etc.)
- ✅ **Blood Groups** (A+, A-, B+, B-, AB+, AB-, O+, O-)
- ✅ **Qualifications** (MD, DO, MSN, BSN, etc.)

## Step 3: Test Registration

1. Go to `http://localhost:3000/login`
2. Click **"Create Account"**
3. You should now see hospitals in the dropdown:
   - General Medical Center
   - St. Mary's Hospital
   - City Central Hospital
   - Regional Neurological Institute
   - Community Health Center
   - University Medical Hospital
   - Advanced Brain Clinic
   - Memorial Medical Center

## Alternative: Manual Addition via Supabase Dashboard

If you prefer to add hospitals manually:

1. Go to **Supabase Dashboard** → **Table Editor**
2. Select **hospitals** table
3. Click **Insert** → **Insert row**
4. Fill in:
   - `hospital_code`: HSP001
   - `name`: Test Hospital
   - `address`: 123 Test Street
   - `phone`: +1234567890
   - `email`: test@hospital.com
   - `status`: active

## Verify It's Working

After adding hospitals, refresh your registration page and the hospital dropdown should be populated with options.

## Sample Hospital Codes for Testing

- **HSP001** - General Medical Center
- **HSP002** - St. Mary's Hospital  
- **HSP003** - City Central Hospital
- **HSP004** - Regional Neurological Institute

Choose any of these when registering your test account!