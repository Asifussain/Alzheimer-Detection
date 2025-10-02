# Supabase Row Level Security (RLS) Policies

## Table of Contents
- [Admin Profiles](#admin-profiles)
- [Blood Groups](#blood-groups)
- [Custom Auth Audit Log](#custom-auth-audit-log)
- [Custom Auth Credentials](#custom-auth-credentials)
- [Doctor Patient Relationships](#doctor-patient-relationships)
- [Doctor Profiles](#doctor-profiles)
- [EEG Analysis Results](#eeg-analysis-results)
- [EEG Sessions](#eeg-sessions)
- [Hospital ID Sequences](#hospital-id-sequences)
- [Hospitals](#hospitals)
- [Notifications](#notifications)
- [Password Reset Tokens](#password-reset-tokens)
- [Patient Profiles](#patient-profiles)
- [Predictions](#predictions)
- [Profile Details](#profile-details)
- [Profiles](#profiles)
- [Qualifications](#qualifications)
- [Radiologist Profiles](#radiologist-profiles)
- [Reports](#reports)
- [User Access Logs](#user-access-logs)
- [User Profiles](#user-profiles)

---

## Admin Profiles

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Users can insert own admin profile | INSERT | public | Allows users to create their own admin profile |
| Users can update own admin profile | UPDATE | public | Allows users to update their own admin profile |
| Users can view own admin profile | SELECT | public | Allows users to view their own admin profile |

---

## Blood Groups

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Authenticated users can view blood groups | SELECT | authenticated | Allows authenticated users to view blood group data |

---

## Custom Auth Audit Log

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Users can view own audit log | SELECT | public | Allows users to view their own authentication audit logs |

---

## Custom Auth Credentials

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Users can view own credentials | SELECT | public | Allows users to view their own authentication credentials |

---

## Doctor Patient Relationships

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Admins can manage relationships | ALL | public | Allows admins to perform all operations on doctor-patient relationships |
| Users can view their relationships | SELECT | public | Allows users to view their doctor-patient relationships |

---

## Doctor Profiles

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Users can insert own doctor profile | INSERT | public | Allows users to create their own doctor profile |
| Users can update own doctor profile | UPDATE | public | Allows users to update their own doctor profile |
| Users can view own doctor profile | SELECT | public | Allows users to view their own doctor profile |

---

## EEG Analysis Results

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Users can view analysis for their sessions | SELECT | public | Allows users to view EEG analysis results for their own sessions |

---

## EEG Sessions

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Doctors can create sessions | INSERT | public | Allows doctors to create new EEG sessions |
| Doctors can view sessions for their patients | SELECT | public | Allows doctors to view EEG sessions for their assigned patients |
| Patients can view own sessions | SELECT | public | Allows patients to view their own EEG sessions |

---

## Hospital ID Sequences

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Admins can manage sequences | ALL | public | Allows admins to perform all operations on hospital ID sequences |

---

## Hospitals

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Authenticated users can view hospitals | SELECT | authenticated | Allows authenticated users to view hospital data |

---

## Notifications

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| System can insert notifications | INSERT | public | Allows the system to create notifications |
| Users can view own notifications | SELECT | public | Allows users to view their own notifications |

---

## Password Reset Tokens

**RLS Status:** Enabled

**⚠️ Warning:** No policies have been created yet. No data will be selectable via Supabase APIs.

---

## Patient Profiles

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Doctors can view assigned patients | SELECT | public | Allows doctors to view profiles of their assigned patients |
| Users can insert own patient profile | INSERT | public | Allows users to create their own patient profile |
| Users can update own patient profile | UPDATE | public | Allows users to update their own patient profile |
| Users can view own patient profile | SELECT | public | Allows users to view their own patient profile |

---

## Predictions

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Users can access their own predictions | ALL | public | Allows users to perform all operations on their own predictions |

---

## Profile Details

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Allow individual update access | UPDATE | public | Allows users to update profile details |
| Enable read access for all users | SELECT | public | Allows all users to read profile details |
| Users can insert their own profile details | INSERT | public | Allows users to insert their own profile details |

---

## Profiles

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Enable delete for own user | DELETE | authenticated | Allows authenticated users to delete their own profile |
| Enable insert for own profile | INSERT | authenticated | Allows authenticated users to create their own profile |
| Enable read access for own user | SELECT | authenticated | Allows authenticated users to read their own profile |
| Enable update for own user | UPDATE | authenticated | Allows authenticated users to update their own profile |

---

## Qualifications

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Authenticated users can view qualifications | SELECT | authenticated | Allows authenticated users to view qualification data |

---

## Radiologist Profiles

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Admins can create radiologist profiles | INSERT | public | Allows admins to create radiologist profiles |
| Admins can read all radiologist profiles | SELECT | public | Allows admins to view all radiologist profiles |
| Admins can update radiologist profiles | UPDATE | public | Allows admins to update radiologist profiles |
| Radiologists can read own profile | SELECT | public | Allows radiologists to view their own profile |
| Radiologists can update own profile | UPDATE | public | Allows radiologists to update their own profile |

---

## Reports

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Users can view reports generated for them | SELECT | public | Allows users to view reports that were generated for them |

---

## User Access Logs

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| System can insert access logs | INSERT | public | Allows the system to create access log entries |
| Users can view own access logs | SELECT | public | Allows users to view their own access logs |

---

## User Profiles

**RLS Status:** Enabled

| Policy Name | Command | Applied To | Description |
|------------|---------|------------|-------------|
| Allow admin to view users in their own hospital | SELECT | public | Allows admins to view user profiles within their hospital |
| Enable insert for authenticated users | INSERT | public | Allows authenticated users to insert user profiles |
| Enable select for authenticated users | SELECT | public | Allows authenticated users to select user profiles |
| Enable update for authenticated users | UPDATE | public | Allows authenticated users to update user profiles |
| Users can insert own profile | INSERT | public | Allows users to insert their own profile |
| Users can update own profile | UPDATE | public | Allows users to update their own profile |
| Users can view own profile | SELECT | public | Allows users to view their own profile |

---

## Summary

- **Total Tables:** 21
- **Tables with RLS Enabled:** 21
- **Tables without Policies:** 1 (password_reset_tokens)

### Security Notes

1. All tables have RLS enabled, which is a security best practice
2. The `password_reset_tokens` table needs policies to be functional
3. Most policies follow a pattern of allowing users to manage their own data
4. Admin and system-level operations are granted specific permissions
5. Role-based access is implemented for doctors, patients, radiologists, and admins