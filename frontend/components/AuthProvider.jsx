import { useEffect, useState, createContext, useContext } from 'react';
import supabase from '../lib/supabaseClient';

// Define the context type (adjust based on what you provide)
const AuthContext = createContext({
  session: null, // Keep the session object from Supabase
  user: null,    // Keep the user object from Supabase
  profile: null, // Add profile data (including role)
  loading: true, // Add loading state
});

export const AuthProvider = ({ children }) => {
  const [session, setSession] = useState(null);
  const [user, setUser] = useState(null);
  const [profile, setProfile] = useState(null); // Store profile data here
  const [loading, setLoading] = useState(true); // Loading state for initial auth check

  useEffect(() => {
    let isMounted = true; // Prevent state updates on unmounted component

    const fetchSessionAndProfile = async () => {
      setLoading(true); // Start loading
      // 1. Get Session
      const { data: { session: currentSession }, error: sessionError } = await supabase.auth.getSession();

      if (sessionError) {
        console.error("Error getting session:", sessionError);
        if (isMounted) {
            setSession(null);
            setUser(null);
            setProfile(null);
            setLoading(false);
        }
        return;
      }
      
      if (currentSession?.user) {
        const currentUser = currentSession.user;
        // 2. Try Fetching Profile
        const { data: profileData, error: profileError } = await supabase
          .from('profiles')
          .select('*') // Fetch all profile data for now
          .eq('id', currentUser.id)
          .maybeSingle(); // Use maybeSingle to handle null result without error

        if (profileError && profileError.code !== 'PGRST116') { // Ignore 'not found' error code for now
           console.error("Error fetching profile:", profileError);
           // Handle unexpected errors, maybe logout?
            if (isMounted) setLoading(false); // Still stop loading on error
            return;
        }

        if (profileData) {
          // 3a. Profile Exists
           if (isMounted) {
               setSession(currentSession);
               setUser(currentUser);
               setProfile(profileData);
           }
        } else {
          // 3b. Profile Doesn't Exist (First Login for this user) - Create Default Profile
          console.log("Profile not found for user, creating default...");
          const defaultRole = 'patient'; // <<< SET YOUR DEFAULT ROLE HERE
          const { data: newProfile, error: insertError } = await supabase
            .from('profiles')
            .insert({
              id: currentUser.id,
              full_name: currentUser.user_metadata?.full_name, // Get from auth metadata
              email: currentUser.email, // Get from auth user
              role: defaultRole
            })
            .select() // Select the newly created profile
            .single();

          if (insertError) {
            console.error("Error creating default profile:", insertError);
            // Handle failure to create profile - might need to log user out or show error
             if (isMounted) setLoading(false); // Stop loading on error
             return;
          }
          
           if (isMounted) {
               console.log("Default profile created:", newProfile);
               setSession(currentSession);
               setUser(currentUser);
               setProfile(newProfile); // Set the newly created profile
           }
        }

      } else {
        // No user session
         if (isMounted) {
             setSession(null);
             setUser(null);
             setProfile(null);
         }
      }
       if (isMounted) {
           setLoading(false); // Finish loading
       }
    };

    fetchSessionAndProfile();

    // Subscribe to auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, changedSession) => {
        console.log("Auth state changed:", _event, changedSession);
        // Refetch profile when auth state changes (login/logout)
        fetchSessionAndProfile();
      }
    );

    // Cleanup function
    return () => {
      isMounted = false;
      subscription?.unsubscribe();
    };
  }, []); // Run only once on mount

  // Context value includes user info and profile (with role)
  const value = {
    session,
    user,
    profile, // Contains the role
    loading,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Update useUser hook (or create a new one like useAuth)
export const useAuth = () => {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

// Keep original useUser if needed elsewhere, or adapt it
export const useUser = () => {
    const { user } = useContext(AuthContext);
    // This might need adjustment depending on how you used it before.
    // Maybe return user directly or user?.id etc.
    return { user }; // Returning the auth user object for now
};