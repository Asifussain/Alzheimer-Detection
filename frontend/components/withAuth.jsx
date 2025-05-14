import { useRouter } from 'next/router';
import { useEffect } from 'react';
import { useAuth } from './AuthProvider'; // Use the updated hook
import LoadingSpinner from './LoadingSpinner'; // For loading state

const withAuth = (WrappedComponent, allowedRoles = []) => {
  const Wrapper = (props) => {
    const { user, profile, loading, session } = useAuth();
    const router = useRouter();

    useEffect(() => {
      if (!loading) { // Only run check after initial auth load is complete
        if (!user || !session) {
          // If not logged in, redirect to login or home
          // You might want a dedicated login page later
          console.log('User not logged in, redirecting...');
          router.replace('/'); // Redirect to home page for login prompt
        } else if (allowedRoles.length > 0 && !allowedRoles.includes(profile?.role)) {
          // If logged in but role is not allowed, redirect to an unauthorized page or home
           console.log(`User role '${profile?.role}' not in allowed roles: ${allowedRoles.join(', ')}. Redirecting...`);
          router.replace('/'); // Or potentially to a specific '/unauthorized' page
        }
      }
    }, [user, profile, loading, session, router, allowedRoles]); // Add dependencies

    // Show loading spinner while checking auth state
    if (loading || (!user && !session)) {
       // Or a more sophisticated loading screen
        return (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
                <LoadingSpinner />
            </div>
        );
    }

    // If user is authenticated and has the correct role (or no roles specified), render the component
     if (user && session && (allowedRoles.length === 0 || allowedRoles.includes(profile?.role))) {
        return <WrappedComponent {...props} />;
     }

     // Fallback for edge cases or during redirect
      return (
         <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
             <LoadingSpinner /> 
         </div>
     );
  };

  // Set display name for React DevTools
  Wrapper.displayName = `withAuth(${WrappedComponent.displayName || WrappedComponent.name || 'Component'})`;

  return Wrapper;
};

export default withAuth;