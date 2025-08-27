import { useEffect, useState, createContext, useContext, useCallback, useRef, useMemo } from 'react';
import { useRouter } from 'next/router';
import supabase from '../lib/supabaseClient';
import LoadingSpinner from './LoadingSpinner';

const AuthContext = createContext({
  session: undefined,
  user: undefined,
  userProfile: undefined,
  hospitalData: undefined,
  isLoading: true,
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
  const router = useRouter();
  const isMountedRef = useRef(false);

  // Track mount status
  useEffect(() => {
    isMountedRef.current = true;
    return () => { isMountedRef.current = false; };
  }, []);

  const fetchAndSetProfile = useCallback(async (currentUser, currentSession) => {
    if (!isMountedRef.current) return;
    if (!currentUser) {
      setUserProfile(null);
      setHospitalData(null);
      return;
    }

    try {
      // Fetch user profile from new user_profiles table
      const { data: profileData, error: profileError } = await supabase
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

      if (!isMountedRef.current) return;

      if (profileError && profileError.code !== 'PGRST116') {
        // Try a simpler query without joins
        try {
          const { data: simpleProfileData, error: simpleError } = await supabase
            .from('user_profiles')
            .select('*')
            .eq('id', currentUser.id)
            .maybeSingle();
          
          if (simpleProfileData && !simpleError) {
            setUserProfile(simpleProfileData);
            setHospitalData(null); // Will fetch separately if needed
          } else {
            setUserProfile(null);
            setHospitalData(null);
          }
        } catch (fallbackError) {
          setUserProfile(null);
          setHospitalData(null);
        }
      } else if (profileData) {
        
        setUserProfile(profileData);
        setHospitalData(profileData.hospitals);
      } else {
        // No profile found - user needs to complete setup
        setUserProfile({ needsSetup: true });
        setHospitalData(null);
      }
    } catch (error) {
      if (isMountedRef.current) {
        // Try simple fetch as fallback
        try {
          const { data: simpleProfileData, error: simpleError } = await supabase
            .from('user_profiles')
            .select('*')
            .eq('id', currentUser.id)
            .maybeSingle();
          
          if (simpleProfileData && !simpleError) {
            setUserProfile(simpleProfileData);
            setHospitalData(null);
          } else {
            setUserProfile(null);
            setHospitalData(null);
          }
        } catch (fallbackError) {
          setUserProfile(null);
          setHospitalData(null);
        }
      }
    }
  }, []);

  useEffect(() => {
    setIsLoading(true);
    let currentSession = null;

    const processSession = async (sessionToProcess) => {
      if (!isMountedRef.current) return;
      setSession(sessionToProcess);
      const currentUser = sessionToProcess?.user || null;
      setUser(currentUser);
      await fetchAndSetProfile(currentUser, sessionToProcess);
      if (isMountedRef.current) setIsLoading(false);
    };

    supabase.auth.getSession().then(({ data }) => {
      if (!isMountedRef.current) return;
      currentSession = data.session;
      if (window.location.hash.includes('access_token') || window.location.hash.includes('error')) {
        router.replace(router.pathname, undefined, { shallow: true });
      }
      processSession(currentSession);
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
        processSession(sessionFromListener);
      }
    });

    const handleVisibilityChange = () => {
      if (!isMountedRef.current || document.visibilityState !== 'visible') return;
      setIsLoading(true);
      supabase.auth.getSession().then(({ data: { session: sessionFromVisibility } }) => {
        if (!isMountedRef.current) return;
        if (JSON.stringify(sessionFromVisibility) !== JSON.stringify(currentSession)) {
          currentSession = sessionFromVisibility;
          processSession(sessionFromVisibility);
        } else {
          if (isMountedRef.current) setIsLoading(false);
        }
      });
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      subscription?.unsubscribe();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [fetchAndSetProfile, router]);

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
    await supabase.auth.signOut();
  }, []);

  const refreshProfile = useCallback(async () => {
    
    if (user && session && isMountedRef.current) {
      setIsLoading(true);
      await fetchAndSetProfile(user, session);
      if (isMountedRef.current) setIsLoading(false);
    } else {
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
    signOut,
    refreshProfile,
    getUserId
  }), [session, user, userProfile, hospitalData, isLoading, signOut, refreshProfile, getUserId]);

  if (isLoading && session === undefined) {
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