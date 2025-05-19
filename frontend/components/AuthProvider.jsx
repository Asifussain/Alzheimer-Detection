// frontend/components/AuthProvider.jsx
import { useEffect, useState, createContext, useContext, useCallback, useRef } from 'react';
import { useRouter } from 'next/router';
import supabase from '../lib/supabaseClient';
import LoadingSpinner from './LoadingSpinner';

const AuthContext = createContext({
  session: undefined, // Use undefined to clearly distinguish from null (no session) vs not yet checked
  user: undefined,
  profile: undefined,
  isLoading: true, // True by default
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
  const isMountedRef = useRef(false); // To track mount status

  useEffect(() => {
    isMountedRef.current = true;
    console.log('[AuthProvider] Component Mounted.');
    return () => {
      isMountedRef.current = false;
      console.log('[AuthProvider] Component Will Unmount.'); // Should ideally not see during nav
    };
  }, []);

  const fetchAndSetProfile = useCallback(async (currentUser, currentSession) => {
    if (!isMountedRef.current) return;
    if (!currentUser) {
      console.log('[AuthProvider:fetchProfile] No user, setting profile to null.');
      setProfile(null);
      return;
    }

    console.log(`[AuthProvider:fetchProfile] Fetching for user: ${currentUser.id}`);
    try {
      const { data: profileData, error: profileError } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', currentUser.id)
        .maybeSingle();

      if (!isMountedRef.current) return;

      if (profileError && profileError.code !== 'PGRST116') {
        console.error("[AuthProvider:fetchProfile] Error fetching profile:", profileError);
        setProfile(null);
      } else if (profileData) {
        const needsSetup = !profileData.role || profileData.role === '' || typeof profileData.role_confirmed === 'undefined' || !profileData.role_confirmed;
        if (needsSetup && profileData.role !== PENDING_ROLE_SELECTION) {
          console.log(`[AuthProvider:fetchProfile] Existing user needs role setup. Apparent Role: '${profileData.role}'. Setting to PENDING.`);
          setProfile({ ...profileData, role: PENDING_ROLE_SELECTION, role_confirmed: false });
        } else {
          setProfile(profileData);
        }
        console.log(`[AuthProvider:fetchProfile] Profile set:`, profileData?.role, profileData?.role_confirmed);
      } else {
        console.log('[AuthProvider:fetchProfile] Profile not found, creating default.');
        const { data: newProfile, error: insertError } = await supabase
          .from('profiles')
          .insert({ id: currentUser.id, full_name: currentUser.user_metadata?.full_name || currentUser.email, email: currentUser.email, role: PENDING_ROLE_SELECTION, role_confirmed: false })
          .select().single();
        
        if (!isMountedRef.current) return;
        if (insertError) {
          console.error('[AuthProvider:fetchProfile] Error creating default profile:', insertError);
          setProfile(null);
        } else {
          console.log('[AuthProvider:fetchProfile] Default profile created:', newProfile);
          setProfile(newProfile);
        }
      }
    } catch (error) {
      console.error('[AuthProvider:fetchProfile] Unexpected error:', error);
      if (isMountedRef.current) setProfile(null);
    }
  }, []); // Stable callback

  // Effect for session management (initial + listener + visibility)
  useEffect(() => {
    console.log('[AuthProvider:SessionEffect] Setting up session management.');
    setIsLoading(true); // Always start as loading when this effect runs

    let currentSession = null; // Variable to hold session from getSession

    const processSession = async (sessionToProcess, eventSource = "UNKNOWN") => {
        if (!isMountedRef.current) return;
        console.log(`[AuthProvider:processSession ${eventSource}] Processing:`, !!sessionToProcess);
        setSession(sessionToProcess);
        const currentUser = sessionToProcess?.user || null;
        setUser(currentUser);
        await fetchAndSetProfile(currentUser, sessionToProcess);
        if (isMountedRef.current) setIsLoading(false); // Finish loading
    };

    // Initial session check
    supabase.auth.getSession().then(({ data }) => {
      if (!isMountedRef.current) return;
      console.log('[AuthProvider:SessionEffect] Initial getSession complete.');
      currentSession = data.session; // Store it
      if (window.location.hash.includes('access_token') || window.location.hash.includes('error')) {
        router.replace(router.pathname, undefined, { shallow: true });
      }
      processSession(currentSession, "INITIAL_GET_SESSION");
    });

    // Auth state change listener
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, sessionFromListener) => {
      if (!isMountedRef.current) return;
      console.log(`[AuthProvider:SessionEffect] onAuthStateChange event: ${event}`);
      if (window.location.hash.includes('access_token') || window.location.hash.includes('error')) {
          if (["SIGNED_IN", "TOKEN_REFRESHED", "USER_UPDATED", "PASSWORD_RECOVERY"].includes(event)) {
              router.replace(router.pathname, undefined, { shallow: true });
          }
      }
      // Only update if the session object is different, or if the event is a sign out/in.
      // This helps prevent re-processing if getSession on visibility returns the same session object.
      if (event === "SIGNED_OUT" || event === "SIGNED_IN" || JSON.stringify(sessionFromListener) !== JSON.stringify(currentSession)) {
          currentSession = sessionFromListener; // Update our tracked session
          setIsLoading(true); // Set loading for this change
          processSession(sessionFromListener, `AUTH_EVENT_${event}`);
      } else {
          console.log(`[AuthProvider:SessionEffect] onAuthStateChange event: ${event} - session data appears unchanged, skipping full process.`);
      }
    });

    const handleVisibilityChange = () => {
      if (!isMountedRef.current || document.visibilityState !== 'visible') return;
      console.log('[AuthProvider:SessionEffect] Tab became visible, re-validating session.');
      setIsLoading(true);
      supabase.auth.getSession().then(({ data: { session: sessionFromVisibility } }) => {
        if (!isMountedRef.current) return;
        // Compare with the 'currentSession' tracked by this effect scope
        if (JSON.stringify(sessionFromVisibility) !== JSON.stringify(currentSession)) {
          console.log('[AuthProvider:SessionEffect:Visibility] Session changed.');
          currentSession = sessionFromVisibility; // Update tracked session
          processSession(sessionFromVisibility, "VISIBILITY_CHANGE");
        } else {
          console.log('[AuthProvider:SessionEffect:Visibility] Session unchanged.');
          if (isMountedRef.current) setIsLoading(false); // No change, just ensure loading is false
        }
      });
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      console.log('[AuthProvider:SessionEffect] Cleaning up listeners.');
      subscription?.unsubscribe();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [fetchAndSetProfile, router]); // router is for router.replace

  // Redirect logic
  useEffect(() => {
    if (isLoading) return; // Wait for auth state to be determined
    const currentPath = router.pathname;

    if (user && profile) {
      const needsRoleSelection = !profile.role || profile.role === PENDING_ROLE_SELECTION || !profile.role_confirmed;
      if (needsRoleSelection) {
        if (currentPath !== '/select-role' && currentPath !== '/login') {
          console.log("[AuthProvider:RedirectLogic] Needs role selection. Redirecting to /select-role.");
          router.replace('/select-role');
        }
      } else {
        if (currentPath === '/' || currentPath === '/login' || currentPath === '/select-role') {
          console.log(`[AuthProvider:RedirectLogic] Role '${profile.role}' confirmed. Redirecting to /${profile.role}/dashboard.`);
          router.replace(`/${profile.role}/dashboard`);
        }
      }
    } else if (!user) { // No user (implies no session either)
      const publicPaths = ['/', '/login', '/service', '/about', '/contact'];
      if (!publicPaths.includes(currentPath) && !currentPath.startsWith('/_next/')) {
         console.log(`[AuthProvider:RedirectLogic] No user & not public. Path: ${currentPath}. Redirecting to /.`);
         router.replace('/');
      }
    }
    // If user exists but profile is null, it means profile fetch failed or is in progress.
    // isLoading should cover the "in progress" part. If fetch failed, user might be stuck if not redirected.
    // However, the primary logic should ensure profile is either loaded or set to pending.
  }, [isLoading, session, user, profile, router]);

  const signOut = useCallback(async () => {
    if (!isMountedRef.current) return;
    setIsLoading(true);
    await supabase.auth.signOut();
    // onAuthStateChange listener will handle setting state (session, user, profile to null, isLoading to false)
  }, []);

  const refreshProfile = useCallback(async () => {
    if (user && session && isMountedRef.current) {
      setIsLoading(true);
      await fetchAndSetProfile(user, session); // Re-fetch profile with current user/session
      if (isMountedRef.current) setIsLoading(false);
    }
  }, [user, session, fetchAndSetProfile]);

  const contextValue = { session, user, profile, isLoading, signOut, refreshProfile };

  if (isLoading && session === undefined) { // Show spinner only on the very initial check before session is known
    console.log("[AuthProvider] Rendering INITIAL global loading spinner (session is undefined).");
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