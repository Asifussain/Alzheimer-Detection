import { useRouter } from 'next/router';
import { useEffect } from 'react';
import { useAuth, PENDING_ROLE_SELECTION } from './AuthProvider';
import LoadingSpinner from './LoadingSpinner';

const withAuth = (WrappedComponent, allowedRoles = []) => {
    const Wrapper = (props) => {
        const { user, profile, isLoading, session } = useAuth();
        const router = useRouter();
        const componentName = WrappedComponent.displayName || WrappedComponent.name || 'Component';

        useEffect(() => {
                if (isLoading) {
                        return;
                }
                const currentPath = router.pathname;

                if (!user || !session) {
                        return;
                }

                if (!profile) {
                        return;
                }

                const needsRoleSelection = !profile.role || profile.role === PENDING_ROLE_SELECTION || !profile.role_confirmed;
                if (needsRoleSelection) {
                        if (currentPath !== '/select-role') {
                        }
                        return;
                }

                if (allowedRoles.length > 0 && !allowedRoles.includes(profile.role)) {
                        router.replace('/');
                        return;
                }
        }, [isLoading, user, session, profile, router, allowedRoles, componentName]);


        if (isLoading) {
            return (
                <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', backgroundColor: 'var(--background-start)' }}>
                    <LoadingSpinner /> <p style={{ color: 'var(--text-secondary)', marginLeft: '10px' }}>Loading User Session...</p>
                </div>
            );
        }

        if (user && profile && profile.role_confirmed && profile.role !== PENDING_ROLE_SELECTION) {
                if (allowedRoles.length === 0 || allowedRoles.includes(profile.role)) {
                        return <WrappedComponent {...props} />;
                }
        }

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
