import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
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
  const [step, setStep] = useState(1); // 1 for role selection, 2 for details
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  // Form state for additional details
  const [formData, setFormData] = useState({
    // Patient
    date_of_birth: '',
    emergency_contact_name: '',
    emergency_contact_phone: '',
    // Clinician
    clinic_name: '',
    specialization: '',
    license_number: '',
    // Technician
    hospital_affiliation: '',
    certification_id: '',
  });

  useEffect(() => {
    if (authLoading || !user) return;
    if (profile && profile.role_confirmed && profile.role !== PENDING_ROLE_SELECTION) {
      const dashboardPath = `/${profile.role}/dashboard`;
      router.replace(dashboardPath);
    }
  }, [user, profile, authLoading, router]);

  const handleRoleSelect = (roleId) => {
    setSelectedRole(roleId);
    setStep(2); // Move to the details step
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedRole) {
      setError('Please select a role.');
      return;
    }
    setIsSubmitting(true);
    setError('');

    try {
      // 1. Update the main profile with the selected role
      const { error: profileError } = await supabase
        .from('profiles')
        .update({ role: selectedRole, role_confirmed: true })
        .eq('id', user.id);

      if (profileError) throw profileError;

      // 2. Prepare and insert data into the new profile_details table
      const detailsData = {
        profile_id: user.id,
        ...(selectedRole === 'patient' && {
          date_of_birth: formData.date_of_birth,
          emergency_contact_name: formData.emergency_contact_name,
          emergency_contact_phone: formData.emergency_contact_phone,
        }),
        ...(selectedRole === 'clinician' && {
          clinic_name: formData.clinic_name,
          specialization: formData.specialization,
          license_number: formData.license_number,
        }),
        ...(selectedRole === 'technician' && {
          hospital_affiliation: formData.hospital_affiliation,
          certification_id: formData.certification_id,
        }),
      };

      // Use 'upsert' to be safe
      const { error: detailsError } = await supabase
        .from('profile_details')
        .upsert(detailsData, { onConflict: 'profile_id' });

      if (detailsError) throw detailsError;

      // 3. Refresh the auth context and redirect
      await refreshProfile();
      router.replace(`/${selectedRole}/dashboard`);

    } catch (err) {
      console.error('Error updating role and details:', err);
      setError(err.message || 'Failed to update profile. Please try again.');
      setIsSubmitting(false);
    }
  };

  const renderDetailsForm = () => {
    switch (selectedRole) {
      case 'patient':
        return (
          <>
            <input name="date_of_birth" type="date" placeholder="Date of Birth" onChange={handleChange} required />
            <input name="emergency_contact_name" placeholder="Emergency Contact Name" onChange={handleChange} />
            <input name="emergency_contact_phone" placeholder="Emergency Contact Phone" onChange={handleChange} />
          </>
        );
      case 'clinician':
        return (
          <>
            <input name="clinic_name" placeholder="Clinic Name" onChange={handleChange} required />
            <input name="specialization" placeholder="Specialization" onChange={handleChange} required />
            <input name="license_number" placeholder="License Number" onChange={handleChange} required />
          </>
        );
      case 'technician':
        return (
          <>
            <input name="hospital_affiliation" placeholder="Hospital Affiliation" onChange={handleChange} required />
            <input name="certification_id" placeholder="Certification ID" onChange={handleChange} required />
          </>
        );
      default:
        return null;
    }
  };

  if (authLoading || !user) {
    return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}><LoadingSpinner /></div>;
  }

  return (
    <>
      <Navbar />
      <div className={`${pageStyles.pageContainer} ${styles.selectRoleContainer}`}>
        {step === 1 ? (
          <>
            <h1 className={pageStyles.pageTitle}>Select Your Role</h1>
            <p className={styles.subheading}>Choose the role that best describes how you'll use our platform.</p>
            <div className={styles.roleOptionsContainer}>
              {ROLES.map((role) => (
                <button
                  key={role.id}
                  type="button"
                  className={styles.roleOption}
                  onClick={() => handleRoleSelect(role.id)}
                >
                  <h3 className={styles.roleName}>{role.name}</h3>
                  <p className={styles.roleDescription}>{role.description}</p>
                </button>
              ))}
            </div>
          </>
        ) : (
          <>
            <h1 className={pageStyles.pageTitle}>Complete Your Profile</h1>
            <p className={styles.subheading}>Please provide a few more details to get started.</p>
            <form onSubmit={handleSubmit} className={styles.detailsForm}>
              {renderDetailsForm()}
              {error && <p className={styles.errorMessage}>{error}</p>}
              <div className={styles.formActions}>
                <button type="button" onClick={() => setStep(1)} className={styles.backButton}>Back</button>
                <button type="submit" className={styles.submitButton} disabled={isSubmitting}>
                  {isSubmitting ? <LoadingSpinner /> : 'Confirm and Continue'}
                </button>
              </div>
            </form>
          </>
        )}
      </div>
    </>
  );
}