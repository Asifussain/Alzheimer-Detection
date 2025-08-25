import { useRouter } from 'next/router';
import { useEffect } from 'react';
import { useAuth } from './AuthProvider';
import LoadingSpinner from './LoadingSpinner';

const withAuth = (WrappedComponent, allowedRoles = [], requireVerification = true) => {
    const Wrapper = (props) => {
        const { user, userProfile, isLoading, session } = useAuth();
        const router = useRouter();
        const componentName = WrappedComponent.displayName || WrappedComponent.name || 'Component';

        useEffect(() => {
            if (isLoading) {
                return;
            }
            
            const currentPath = router.pathname;

            // Not authenticated
            if (!user || !session) {
                const publicPaths = ['/', '/home', '/login', '/landing', '/about', '/contact'];
                if (!publicPaths.includes(currentPath)) {
                    router.replace('/');
                }
                return;
            }

            // No profile exists - needs to complete setup
            if (!userProfile || userProfile.needsSetup) {
                if (currentPath !== '/complete-profile') {
                    router.replace('/complete-profile');
                }
                return;
            }

            // Account pending verification
            if (userProfile.account_status === 'pending') {
                if (currentPath !== '/account-pending') {
                    router.replace('/account-pending');
                }
                return;
            }

            // Account suspended or inactive
            if (['suspended', 'inactive'].includes(userProfile.account_status)) {
                if (currentPath !== '/account-pending') {
                    router.replace('/account-pending');
                }
                return;
            }

            // Phone verification required
            if (requireVerification && userProfile.account_status === 'active' && !userProfile.phone_verified) {
                if (currentPath !== '/verify-phone') {
                    router.replace('/verify-phone');
                }
                return;
            }

            // Role-based access control
            if (allowedRoles.length > 0 && !allowedRoles.includes(userProfile.role)) {
                // Redirect to appropriate dashboard
                router.replace(`/${userProfile.role}/dashboard`);
                return;
            }

            // Additional role-specific verification checks
            if (requireVerification && userProfile.role === 'doctor') {
                const doctorData = userProfile.doctor_profiles?.[0];
                if (doctorData && doctorData.verification_status !== 'verified') {
                    if (currentPath !== '/doctor/verification-pending') {
                        router.replace('/doctor/verification-pending');
                    }
                    return;
                }
            }

            if (requireVerification && userProfile.role === 'patient') {
                const patientData = userProfile.patient_profiles?.[0];
                if (patientData && patientData.verification_status !== 'verified') {
                    if (currentPath !== '/patient/verification-pending') {
                        router.replace('/patient/verification-pending');
                    }
                    return;
                }
            }
        }, [isLoading, user, session, userProfile, router, allowedRoles, requireVerification, componentName]);

        // Show loading spinner
        if (isLoading) {
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
                    <p style={{ color: 'var(--text-secondary)' }}>Loading User Session...</p>
                </div>
            );
        }

        // Check if user meets all requirements
        if (user && userProfile && 
            userProfile.account_status === 'active' && 
            (!requireVerification || userProfile.phone_verified)) {
            
            // Check role permissions
            if (allowedRoles.length === 0 || allowedRoles.includes(userProfile.role)) {
                // Additional verification checks passed
                if (requireVerification) {
                    if (userProfile.role === 'doctor') {
                        const doctorData = userProfile.doctor_profiles?.[0];
                        if (!doctorData || doctorData.verification_status !== 'verified') {
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
                                    <p style={{ color: 'var(--text-secondary)' }}>Verifying credentials...</p>
                                </div>
                            );
                        }
                    }
                    
                    if (userProfile.role === 'patient') {
                        const patientData = userProfile.patient_profiles?.[0];
                        if (!patientData || patientData.verification_status !== 'verified') {
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
                                    <p style={{ color: 'var(--text-secondary)' }}>Verifying patient status...</p>
                                </div>
                            );
                        }
                    }
                }
                
                return <WrappedComponent {...props} />;
            }
        }

        // Fallback loading state
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
                <p style={{ color: 'var(--text-secondary)' }}>Verifying access...</p>
            </div>
        );
    };

    Wrapper.displayName = `withAuth(${WrappedComponent.displayName || WrappedComponent.name || 'Component'})`;
    return Wrapper;
};

export default withAuth;