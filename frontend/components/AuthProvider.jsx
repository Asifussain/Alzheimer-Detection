import { useEffect, useState, createContext, useContext, useCallback, useRef, useMemo } from 'react';
import { useRouter } from 'next/router';
import supabase from '../lib/supabaseClient';
import LoadingSpinner from './LoadingSpinner';

const AuthContext = createContext({
  session: undefined,
  user: undefined,
  profile: undefined,
  isLoading: true,
  signOut: async () => {},
  refreshProfile: async () => {},
});

export const PENDING_ROLE_SELECTION = 'pending_selection';

export const AuthProvider = ({ children }) => {
  const [session, setSession] = useState(undefined);
  const [user, setUser] = useState(undefined);
  const [profile, setProfile] = useState(undefined);
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
      setProfile(null);
      return;
    }
    try {
      const { data: profileData, error: profileError } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', currentUser.id)
        .maybeSingle();

      if (!isMountedRef.current) return;

      if (profileError && profileError.code !== 'PGRST116') {
        setProfile(null);
      } else if (profileData) {
        // Check if user needs to select a role
        const needsSetup = !profileData.role || profileData.role === '' || typeof profileData.role_confirmed === 'undefined' || !profileData.role_confirmed;
        if (needsSetup && profileData.role !== PENDING_ROLE_SELECTION) {
          setProfile({ ...profileData, role: PENDING_ROLE_SELECTION, role_confirmed: false });
        } else {
          setProfile(profileData);
        }
      } else {
        // Create default profile if not found
        const { data: newProfile, error: insertError } = await supabase
          .from('profiles')
          .insert({ id: currentUser.id, full_name: currentUser.user_metadata?.full_name || currentUser.email, email: currentUser.email, role: PENDING_ROLE_SELECTION, role_confirmed: false })
          .select().single();
        if (!isMountedRef.current) return;
        if (insertError) {
          setProfile(null);
        } else {
          setProfile(newProfile);
        }
      }
    } catch (error) {
      if (isMountedRef.current) setProfile(null);
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

    if (user && profile) {
      const needsRoleSelection = !profile.role || profile.role === PENDING_ROLE_SELECTION || !profile.role_confirmed;
      if (needsRoleSelection) {
        if (currentPath !== '/select-role' && currentPath !== '/login') {
          router.replace('/select-role');
        }
      } else {
        if (currentPath === '/login' || currentPath === '/select-role') {
          router.replace(`/${profile.role}/dashboard`);
        }
      }
    } else if (!user) {
      const publicPaths = ['/', '/home', '/login', '/landing', '/service', '/about', '/contact'];
      if (!publicPaths.includes(currentPath) && !currentPath.startsWith('/_next/')) {
        router.replace('/');
      }
    }
  }, [isLoading, session, user, profile, router]);


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
    }
  }, [user, session, fetchAndSetProfile]);

  // --- FIX IS HERE ---
  // Memoize the context value to prevent unnecessary re-renders in consumers
  const contextValue = useMemo(() => ({
    session,
    user,
    profile,
    isLoading,
    signOut,
    refreshProfile
  }), [session, user, profile, isLoading, signOut, refreshProfile]);
  // --------------------


  if (isLoading && session === undefined) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', backgroundColor: 'var(--background-start)' }}>
        <LoadingSpinner />
        <p style={{ color: 'var(--text-secondary)', marginLeft: '10px' }}>Initializing AI4NEURO...</p>
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
  const { user } = useContext(AuthContext);
  return { user };
};