import { useEffect, useState, createContext, useContext, useCallback, useRef, useMemo } from 'react';
import { useRouter } from 'next/router';
import supabase from '../lib/supabaseClient';
import LoadingSpinner from './LoadingSpinner';

// Session persistence utilities
const SESSION_STORAGE_KEY = 'ai4neuro_session_cache';
const PROFILE_STORAGE_KEY = 'ai4neuro_profile_cache';

const saveToStorage = (key, data) => {
  try {
    if (typeof window !== 'undefined') {
      localStorage.setItem(key, JSON.stringify(data));
    }
  } catch (error) {
    console.warn('Failed to save to localStorage:', error);
  }
};

const getFromStorage = (key) => {
  try {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : null;
    }
  } catch (error) {
    console.warn('Failed to get from localStorage:', error);
  }
  return null;
};

const clearFromStorage = (key) => {
  try {
    if (typeof window !== 'undefined') {
      localStorage.removeItem(key);
    }
  } catch (error) {
    console.warn('Failed to clear from localStorage:', error);
  }
};

// Debounce utility for performance optimization
const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

const AuthContext = createContext({
  session: undefined,
  user: undefined,
  userProfile: undefined,
  hospitalData: undefined,
  isLoading: true,
  isInitialLoad: true,
  signOut: async () => {},
  refreshProfile: async () => {},
  getUserId: () => null,
});

export const PENDING_ROLE_SELECTION = 'pending_selection';

// Hospital-based unique ID generator
const generateHospitalBasedId = (hospitalCode, role, sequence) => {
  const rolePrefix = {
    'patient': 'PAT',
    'doctor': 'DOC', 
    'admin': 'ADM'
  };
  
  const prefix = rolePrefix[role] || 'USR';
  const paddedSequence = sequence.toString().padStart(4, '0');
  return `${hospitalCode}-${prefix}-${paddedSequence}`;
};

export const AuthProvider = ({ children }) => {
  const [session, setSession] = useState(undefined);
  const [user, setUser] = useState(undefined);
  const [userProfile, setUserProfile] = useState(undefined);
  const [hospitalData, setHospitalData] = useState(undefined);
  const [isLoading, setIsLoading] = useState(true);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const router = useRouter();
  const isMountedRef = useRef(false);
  const profileCacheRef = useRef(null);
  const sessionCacheRef = useRef(null);
  const visibilityTimeoutRef = useRef(null);
  const lastSessionCheckRef = useRef(null);

  // Track mount status
  useEffect(() => {
    isMountedRef.current = true;
    return () => { 
      isMountedRef.current = false; 
      if (visibilityTimeoutRef.current) {
        clearTimeout(visibilityTimeoutRef.current);
      }
    };
  }, []);

  const fetchAndSetProfile = useCallback(async (currentUser, currentSession, useCache = false) => {
    if (!isMountedRef.current) return;
    
    if (!currentUser) {
      setUserProfile(null);
      setHospitalData(null);
      clearFromStorage(PROFILE_STORAGE_KEY);
      profileCacheRef.current = null;
      return;
    }

    // Check if we can use cached profile data (for better performance on tab switches)
    if (useCache && profileCacheRef.current && profileCacheRef.current.userId === currentUser.id) {
      const cacheAge = Date.now() - profileCacheRef.current.timestamp;
      if (cacheAge < 1800000) { // Cache valid for 30 minutes
        setUserProfile(profileCacheRef.current.profile);
        setHospitalData(profileCacheRef.current.hospital);
        return;
      }
    }

    // Try to get from localStorage for faster initial loading
    const cachedProfile = getFromStorage(PROFILE_STORAGE_KEY);
    if (cachedProfile && cachedProfile.userId === currentUser.id && useCache) {
      const cacheAge = Date.now() - cachedProfile.timestamp;
      if (cacheAge < 1800000) { // Cache valid for 30 minutes
        setUserProfile(cachedProfile.profile);
        setHospitalData(cachedProfile.hospital);
        profileCacheRef.current = cachedProfile;
        return;
      }
    }

    try {
      console.log('Fetching profile for user:', currentUser.id);
      
      // Try to fetch user profile from new user_profiles table
      let profileData, profileError;
      
      try {
        const result = await supabase
          .from('user_profiles')
          .select(`
            *,
            hospitals(
              id,
              name,
              hospital_code,
              address,
              phone,
              email
            ),
            patient_profiles!patient_profiles_user_fkey(
              patient_id,
              blood_group_id,
              emergency_contact_name,
              emergency_contact_phone,
              medical_history,
              current_medications,
              allergies,
              verification_status,
              prescription_url,
              blood_groups(blood_type)
            ),
            doctor_profiles!doctor_profiles_user_fkey(
              medical_license,
              qualification_id,
              specialization,
              experience_years,
              consultation_fee,
              verification_status,
              qualifications(qualification_name)
            ),
            admin_profiles!admin_profiles_user_fkey(
              employee_id,
              department,
              permissions
            )
          `)
          .eq('id', currentUser.id)
          .maybeSingle();
          
        profileData = result.data;
        profileError = result.error;
        console.log('Profile fetch result:', { profileData, profileError });
      } catch (fetchError) {
        console.warn('Profile fetch failed:', fetchError);
        profileError = fetchError;
      }

      if (!isMountedRef.current) return;

      if (profileError && profileError.code !== 'PGRST116') {
        console.warn('Complex profile query failed, trying simple query:', profileError);
        
        // Try a simpler query without joins
        try {
          const { data: simpleProfileData, error: simpleError } = await supabase
            .from('user_profiles')
            .select('*')
            .eq('id', currentUser.id)
            .maybeSingle();
          
          console.log('Simple profile fetch result:', { simpleProfileData, simpleError });
          
          if (simpleProfileData && !simpleError) {
            const cacheData = {
              userId: currentUser.id,
              profile: simpleProfileData,
              hospital: null,
              timestamp: Date.now()
            };
            
            setUserProfile(simpleProfileData);
            setHospitalData(null);
            
            profileCacheRef.current = cacheData;
            saveToStorage(PROFILE_STORAGE_KEY, cacheData);
            return; // Early return on success
          }
        } catch (fallbackError) {
          console.warn('Simple profile query also failed:', fallbackError);
        }
        
        // If both queries fail, try legacy profiles table
        try {
          console.log('Trying legacy profiles table...');
          const { data: legacyProfileData, error: legacyError } = await supabase
            .from('profiles')
            .select('*')
            .eq('id', currentUser.id)
            .maybeSingle();
            
          console.log('Legacy profile fetch result:', { legacyProfileData, legacyError });
          
          if (legacyProfileData && !legacyError) {
            const cacheData = {
              userId: currentUser.id,
              profile: {
                ...legacyProfileData,
                account_status: 'active', // Default for legacy
                phone_verified: true // Default for legacy
              },
              hospital: null,
              timestamp: Date.now()
            };
            
            setUserProfile(cacheData.profile);
            setHospitalData(null);
            
            profileCacheRef.current = cacheData;
            saveToStorage(PROFILE_STORAGE_KEY, cacheData);
            return; // Early return on success
          }
        } catch (legacyError) {
          console.warn('Legacy profile query failed:', legacyError);
        }
        
        // If all database queries fail, create a minimal profile from user data
        console.warn('All profile queries failed, creating minimal profile from auth user');
        const minimalProfile = {
          id: currentUser.id,
          email: currentUser.email,
          full_name: currentUser.user_metadata?.full_name || currentUser.email?.split('@')[0] || 'User',
          role: currentUser.user_metadata?.role || 'admin', // Default to admin for demo
          account_status: 'active',
          phone_verified: true,
          unique_identifier: `DEMO-${currentUser.id.slice(0, 8)}`,
          created_at: currentUser.created_at,
          isMinimal: true // Flag to indicate this is a minimal profile
        };
        
        setUserProfile(minimalProfile);
        setHospitalData({
          id: 'demo-hospital',
          name: 'Demo Hospital',
          hospital_code: 'DEMO'
        });
      } else if (profileData) {
        const cacheData = {
          userId: currentUser.id,
          profile: profileData,
          hospital: profileData.hospitals,
          timestamp: Date.now()
        };
        
        setUserProfile(profileData);
        setHospitalData(profileData.hospitals);
        
        // Cache the profile data
        profileCacheRef.current = cacheData;
        saveToStorage(PROFILE_STORAGE_KEY, cacheData);
      } else {
        // No profile found - user needs to complete setup
        setUserProfile({ needsSetup: true });
        setHospitalData(null);
      }
    } catch (error) {
      if (isMountedRef.current) {
        console.warn('All profile fetch attempts failed, using minimal profile');
        
        // Create a minimal profile from the authenticated user data
        const minimalProfile = {
          id: currentUser.id,
          email: currentUser.email,
          full_name: currentUser.user_metadata?.full_name || currentUser.email?.split('@')[0] || 'User',
          role: currentUser.user_metadata?.role || 'admin', // Default to admin for demo
          account_status: 'active',
          phone_verified: true,
          unique_identifier: `DEMO-${currentUser.id.slice(0, 8)}`,
          created_at: currentUser.created_at,
          isMinimal: true // Flag to indicate this is a minimal profile
        };
        
        setUserProfile(minimalProfile);
        setHospitalData({
          id: 'demo-hospital',
          name: 'Demo Hospital',
          hospital_code: 'DEMO'
        });
        
        console.log('Using minimal profile:', minimalProfile);
      }
    }
  }, []);

  useEffect(() => {
    setIsLoading(true);
    let currentSession = null;

    const processSession = async (sessionToProcess, useCache = false) => {
      if (!isMountedRef.current) return;
      
      setSession(sessionToProcess);
      const currentUser = sessionToProcess?.user || null;
      setUser(currentUser);
      
      // Cache session data for better performance
      if (sessionToProcess) {
        const sessionCache = {
          session: sessionToProcess,
          timestamp: Date.now()
        };
        sessionCacheRef.current = sessionCache;
        saveToStorage(SESSION_STORAGE_KEY, sessionCache);
        lastSessionCheckRef.current = sessionToProcess;
      } else {
        sessionCacheRef.current = null;
        clearFromStorage(SESSION_STORAGE_KEY);
        lastSessionCheckRef.current = null;
      }
      
      await fetchAndSetProfile(currentUser, sessionToProcess, useCache);
      if (isMountedRef.current) {
        setIsLoading(false);
        setIsInitialLoad(false);
      }
    };

    // Try to load cached session first for faster initial load
    const cachedSession = getFromStorage(SESSION_STORAGE_KEY);
    if (cachedSession && isInitialLoad && Date.now() - cachedSession.timestamp < 300000) { // 5 minutes
      sessionCacheRef.current = cachedSession;
      processSession(cachedSession.session, true);
    }

    supabase.auth.getSession().then(({ data }) => {
      if (!isMountedRef.current) return;
      currentSession = data.session;
      if (window.location.hash.includes('access_token') || window.location.hash.includes('error')) {
        router.replace(router.pathname, undefined, { shallow: true });
      }
      processSession(currentSession, false);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, sessionFromListener) => {
      if (!isMountedRef.current) return;
      if (window.location.hash.includes('access_token') || window.location.hash.includes('error')) {
        if (["SIGNED_IN", "TOKEN_REFRESHED", "USER_UPDATED", "PASSWORD_RECOVERY"].includes(event)) {
          router.replace(router.pathname, undefined, { shallow: true });
        }
      }
      if (event === "SIGNED_OUT" || event === "SIGNED_IN" || JSON.stringify(sessionFromListener) !== JSON.stringify(currentSession)) {
        currentSession = sessionFromListener;
        setIsLoading(true);
        processSession(sessionFromListener, false);
      }
    });

    // DISABLED: Visibility change handler was causing excessive re-fetching
    // Only check session on actual auth state changes, not visibility changes
    const handleVisibilityChange = debounce(() => {
      // Completely disabled to prevent tab-switching refetches
      return;
    }, 1000);
    
    // Don't add the event listener
    // document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      subscription?.unsubscribe();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      if (visibilityTimeoutRef.current) {
        clearTimeout(visibilityTimeoutRef.current);
      }
    };
  }, [fetchAndSetProfile, router, isInitialLoad]);

  useEffect(() => {
    if (isLoading) return;
    const currentPath = router.pathname;

    if (user && userProfile) {
      // Check if user needs to complete profile setup
      if (userProfile.needsSetup || !userProfile.role) {
        if (currentPath !== '/complete-profile' && currentPath !== '/login') {
          router.replace('/complete-profile');
        }
      } 
      // Check if account is pending activation
      else if (userProfile.account_status === 'pending') {
        // Allow user to stay on complete-profile if they're still filling it out
        if (currentPath !== '/account-pending' && currentPath !== '/complete-profile') {
          router.replace('/account-pending');
        }
      }
      // Check if phone verification is needed
      else if (!userProfile.phone_verified && userProfile.account_status === 'active') {
        if (currentPath !== '/VerifyPhone') {
          router.replace('/VerifyPhone');
        }
      }
      // User is fully set up and verified
      else if (userProfile.account_status === 'active' && userProfile.phone_verified) {
        if (currentPath === '/login' || currentPath === '/complete-profile' || currentPath === '/VerifyPhone' || currentPath === '/account-pending') {
          router.replace(`/${userProfile.role}/dashboard`);
        }
      }
    } else if (!user) {
      const publicPaths = ['/', '/home', '/login', '/landing', '/service', '/about', '/contact'];
      if (!publicPaths.includes(currentPath) && !currentPath.startsWith('/_next/')) {
        router.replace('/');
      }
    }
  }, [isLoading, session, user, userProfile, router]);

  const signOut = useCallback(async () => {
    if (!isMountedRef.current) return;
    setIsLoading(true);
    
    // Clear all cached data
    profileCacheRef.current = null;
    sessionCacheRef.current = null;
    lastSessionCheckRef.current = null;
    clearFromStorage(SESSION_STORAGE_KEY);
    clearFromStorage(PROFILE_STORAGE_KEY);
    
    await supabase.auth.signOut();
  }, []);

  const refreshProfile = useCallback(async () => {
    if (user && session && isMountedRef.current) {
      // Clear cache to force fresh data
      profileCacheRef.current = null;
      clearFromStorage(PROFILE_STORAGE_KEY);
      
      setIsLoading(true);
      await fetchAndSetProfile(user, session, false);
      if (isMountedRef.current) setIsLoading(false);
    }
  }, [user, session, fetchAndSetProfile, userProfile?.account_status]);

  const getUserId = useCallback(() => {
    return userProfile?.unique_identifier || user?.id || null;
  }, [userProfile, user]);

  // Memoize the context value to prevent unnecessary re-renders
  const contextValue = useMemo(() => ({
    session,
    user,
    userProfile,
    hospitalData,
    isLoading,
    isInitialLoad,
    signOut,
    refreshProfile,
    getUserId
  }), [session, user, userProfile, hospitalData, isLoading, isInitialLoad, signOut, refreshProfile, getUserId]);

  // Only show full loading screen on initial load to prevent flash
  if (isInitialLoad && isLoading && session === undefined) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh', 
        backgroundColor: 'var(--background-start)',
        flexDirection: 'column',
        gap: '1rem'
      }}>
        <LoadingSpinner />
        <p style={{ color: 'var(--text-secondary)' }}>Initializing AI4NEURO...</p>
      </div>
    );
  }

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const useUser = () => {
  const { user, userProfile } = useContext(AuthContext);
  return { user, userProfile };
};

// Additional hooks for specific role data
export const usePatientData = () => {
  const { userProfile } = useAuth();
  // Handle both array and single object cases
  const patientProfiles = userProfile?.patient_profiles;
  return Array.isArray(patientProfiles) ? patientProfiles?.[0] : patientProfiles || null;
};

export const useDoctorData = () => {
  const { userProfile } = useAuth();
  // Handle both array and single object cases
  const doctorProfiles = userProfile?.doctor_profiles;
  return Array.isArray(doctorProfiles) ? doctorProfiles?.[0] : doctorProfiles || null;
};

export const useAdminData = () => {
  const { userProfile } = useAuth();
  // Handle both array and single object cases
  const adminProfiles = userProfile?.admin_profiles;
  return Array.isArray(adminProfiles) ? adminProfiles?.[0] : adminProfiles || null;
};

export const useHospital = () => {
  const { hospitalData } = useAuth();
  return hospitalData;
};