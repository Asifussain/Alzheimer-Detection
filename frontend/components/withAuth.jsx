// frontend/components/withAuth.jsx
import { useRouter } from 'next/router';
import { useEffect } from 'react'; // useEffect might still be needed for role checks after loading
import { useAuth, PENDING_ROLE_SELECTION } from './AuthProvider';
import LoadingSpinner from './LoadingSpinner';

const withAuth = (WrappedComponent, allowedRoles = []) => {
  const Wrapper = (props) => {
    const { user, profile, isLoading, session } = useAuth(); // isLoading from context
    const router = useRouter();
    const componentName = WrappedComponent.displayName || WrappedComponent.name || 'Component';

    // The AuthProvider's redirect logic should handle most cases.
    // This useEffect is now more about ensuring, once loading is false,
    // that role conditions are met IF AuthProvider hasn't already redirected.
    useEffect(() => {
        if (isLoading) {
            // console.log(`[withAuth:${componentName}] Waiting: AuthProvider isLoading.`);
            return;
        }
        const currentPath = router.pathname;

        if (!user || !session) {
            // AuthProvider should have redirected to '/' or '/login'
            console.log(`[withAuth:${componentName}] No user/session (should have been redirected by AuthProvider). Path: ${currentPath}`);
            // router.replace('/'); // Potentially redundant, AuthProvider handles this
            return;
        }

        if (!profile) {
            // AuthProvider is still fetching profile, or failed.
            console.log(`[withAuth:${componentName}] User exists, but profile not available. AuthProvider should handle.`);
            return;
        }

        const needsRoleSelection = !profile.role || profile.role === PENDING_ROLE_SELECTION || !profile.role_confirmed;
        if (needsRoleSelection) {
            if (currentPath !== '/select-role') {
                console.log(`[withAuth:${componentName}] Profile needs role selection (should have been redirected by AuthProvider). Path: ${currentPath}`);
                // router.replace('/select-role'); // Potentially redundant
            }
            return;
        }

        if (allowedRoles.length > 0 && !allowedRoles.includes(profile.role)) {
            console.log(`[withAuth:${componentName}] Role '${profile.role}' not in allowed [${allowedRoles.join(', ')}]. Redirecting to / (or an unauthorized page).`);
            router.replace('/'); // Or a dedicated /unauthorized page
            return;
        }
        // console.log(`[withAuth:${componentName}] Access checks passed or handled by AuthProvider.`);

    }, [isLoading, user, session, profile, router, allowedRoles, componentName]);


    if (isLoading) {
    //   console.log(`[withAuth:${componentName}] Rendering loading spinner: Auth context isLoading.`);
      return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', backgroundColor: 'var(--background-start)' }}>
          <LoadingSpinner /> <p style={{ color: 'var(--text-secondary)', marginLeft: '10px' }}>Loading User Session...</p>
        </div>
      );
    }

    // If not loading, and all conditions in useEffect passed (or redirects happened), render.
    // We rely on AuthProvider's redirects to put the user on the right page.
    // If we reach here and the user/profile state is still "wrong" for this page, it's a transient state
    // before a redirect, or the redirect logic needs refinement.
    if (user && profile && profile.role_confirmed && profile.role !== PENDING_ROLE_SELECTION) {
        if (allowedRoles.length === 0 || allowedRoles.includes(profile.role)) {
            // console.log(`[withAuth:${componentName}] Rendering wrapped component.`);
            return <WrappedComponent {...props} />;
        }
    }

    // Fallback: If still here, means not loading, but conditions not met for render (e.g. waiting for redirect)
    // console.log(`[withAuth:${componentName}] Rendering fallback spinner (e.g. waiting for redirect from AuthProvider).`);
    return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', backgroundColor: 'var(--background-start)' }}>
            <LoadingSpinner /> <p style={{ color: 'var(--text-secondary)', marginLeft: '10px' }}>Verifying access...</p>
        </div>
    );
  };

  Wrapper.displayName = `withAuth(${WrappedComponent.displayName || WrappedComponent.name || 'Component'})`;
  return Wrapper;
};

export default withAuth;