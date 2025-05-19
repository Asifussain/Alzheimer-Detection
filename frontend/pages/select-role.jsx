// frontend/pages/select-role.jsx
import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
// Import PENDING_ROLE_SELECTION from AuthProvider
import { useAuth, PENDING_ROLE_SELECTION } from '../components/AuthProvider';
import supabase from '../lib/supabaseClient';
import Navbar from '../components/Navbar';
import LoadingSpinner from '../components/LoadingSpinner';
import styles from '../styles/SelectRole.module.css';
import pageStyles from '../styles/PageLayout.module.css';

const ROLES = [
  { id: 'patient', name: 'Patient', description: 'I am seeking analysis for myself or a loved one.' },
  { id: 'technician', name: 'Technician', description: 'I am an EEG technician uploading data for analysis.' },
  { id: 'clinician', name: 'Clinician / Doctor', description: 'I am a healthcare professional reviewing patient data.' },
];

export default function SelectRolePage() {
  const { user, profile, loading: authLoading, refreshProfile } = useAuth();
  const router = useRouter();
  const [selectedRole, setSelectedRole] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (authLoading || !user) {
      return;
    }
    // Use the imported PENDING_ROLE_SELECTION constant
    if (profile && profile.role_confirmed && profile.role !== PENDING_ROLE_SELECTION) {
      const dashboardPath = `/${profile.role}/dashboard`;
      router.replace(dashboardPath);
    }
  }, [user, profile, authLoading, router]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedRole) {
      setError('Please select a role.');
      return;
    }
    setIsSubmitting(true);
    setError('');

    try {
      const { error: updateError } = await supabase
        .from('profiles')
        .update({ role: selectedRole, role_confirmed: true })
        .eq('id', user.id);

      if (updateError) {
        throw updateError;
      }
      await refreshProfile();
      router.replace(`/${selectedRole}/dashboard`);

    } catch (err) {
      console.error('Error updating role:', err);
      setError(err.message || 'Failed to update role. Please try again.');
      setIsSubmitting(false);
    }
  };

  if (authLoading || !user) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', backgroundColor: 'var(--background-start)' }}>
        <LoadingSpinner />
      </div>
    );
  }

  if (user && !profile && !authLoading) {
      return (
          <>
            <Navbar />
            <div className={pageStyles.pageContainer} style={{ textAlign: 'center' }}>
                <LoadingSpinner />
                <p>Loading profile...</p>
            </div>
          </>
      )
  }

  return (
    <>
      <Navbar />
      <div className={`${pageStyles.pageContainer} ${styles.selectRoleContainer}`}>
        <h1 className={pageStyles.pageTitle}>Select Your Role</h1>
        <p className={styles.subheading}>Please choose the role that best describes how you will be using AI4NEURO.</p>

        <form onSubmit={handleSubmit} className={styles.roleForm}>
          <div className={styles.roleOptionsContainer}>
            {ROLES.map((role) => (
              <button
                key={role.id}
                type="button"
                className={`${styles.roleOption} ${selectedRole === role.id ? styles.selected : ''}`}
                onClick={() => setSelectedRole(role.id)}
                aria-pressed={selectedRole === role.id}
              >
                <h3 className={styles.roleName}>{role.name}</h3>
                <p className={styles.roleDescription}>{role.description}</p>
              </button>
            ))}
          </div>

          {error && <p className={styles.errorMessage}>{error}</p>}

          <button type="submit" className={styles.submitButton} disabled={isSubmitting || !selectedRole}>
            {isSubmitting ? <LoadingSpinner /> : 'Confirm Role and Continue'}
          </button>
        </form>
      </div>
    </>
  );
}