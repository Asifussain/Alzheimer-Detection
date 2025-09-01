# 🔍 Quick Debug Guide for Port 3000

## Step 1: Browser Console Debugging

1. **Open:** http://localhost:3000/admin/dashboard
2. **Press F12** to open Developer Tools
3. **Go to Network tab**
4. **Try creating a patient**
5. **Look for the `/api/admin/create-account` request**
6. **Click on it and check:**
   - **Request Headers** (Authorization token present?)
   - **Request Payload** (what data was sent?)
   - **Response** (what error came back?)

## Step 2: Test Your Database Connection

**Try this URL in browser:** http://localhost:3000/api/admin/test-connection

This will show if your database and email config is working.

## Step 3: Check Console Logs

**Open a new terminal window** and run:
```bash
cd "C:\Users\sathw\OneDrive\Desktop\alzheimer-detection-app\frontend"
npm run dev
```

Then try creating a patient and watch for any error logs.

## Step 4: Common Issues

### **"Admin access required"**
- Your user might not have `role: 'admin'` in the database
- Check if you're logged in correctly

### **"Missing required fields"** 
- Make sure you fill all required patient fields
- Blood group, emergency contact, etc.

### **"Database error"**
- Tables might not exist or have wrong permissions
- Service role key might be incorrect

## Step 5: Network Tab Details

When you see the API call in Network tab, check:

1. **Status Code:**
   - 401 = Authentication issue
   - 403 = Permission denied (not admin)
   - 500 = Server error (check response body)

2. **Response Body:**
   - Should show the actual error message
   - Will be more detailed than "Failed to create account"

---

**Please check the Network tab in browser first** - that will show us the real error message! 🔧