import { useState, useEffect } from 'react';
import { useAuth } from '../../components/AuthProvider';
import supabase from '../../lib/supabaseClient';
import Navbar from '../../components/Navbar';
import LoadingSpinner from '../../components/LoadingSpinner';
import withAuth from '../../components/withAuth';
import styles from '../../styles/Profile.module.css';
import EditProfileModal from '../../components/EditProfileModal'; // Import the new modal

function ProfilePage() {
  const { user, profile, refreshProfile } = useAuth();
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false); // State to control the modal

  useEffect(() => {
    if (!profile) return;

    const fetchProfileDetails = async () => {
      setLoading(true);
      const { data, error } = await supabase
        .from('profile_details')
        .select('*')
        .eq('profile_id', user.id)
        .single();

      if (error && error.code !== 'PGRST116') {
        setError('Could not load profile details.');
      } else {
        setDetails(data);
      }
      setLoading(false);
    };

    fetchProfileDetails();
  }, [profile, user]);

  const handleSaveSuccess = () => {
    refreshProfile();
    setIsModalOpen(false);
  };
  
  const DisplayField = ({ label, value }) => (
    <div className={styles.displayField}>
      <span className={styles.fieldLabel}>{label}</span>
      <span className={styles.fieldValue}>{value || 'Not set'}</span>
    </div>
  );

  const RoleSpecificFields = () => {
    if (!profile) return null;
    switch (profile.role) {
      case 'patient': return <><DisplayField label="Date of Birth" value={details?.date_of_birth} /><DisplayField label="Emergency Contact Name" value={details?.emergency_contact_name} /><DisplayField label="Emergency Contact Phone" value={details?.emergency_contact_phone} /></>;
      case 'clinician': return <><DisplayField label="Clinic Name" value={details?.clinic_name} /><DisplayField label="Specialization" value={details?.specialization} /><DisplayField label="License Number" value={details?.license_number} /></>;
      case 'technician': return <><DisplayField label="Hospital Affiliation" value={details?.hospital_affiliation} /><DisplayField label="Certification ID" value={details?.certification_id} /></>;
      default: return null;
    }
  };

  if (loading) {
    return ( <div className={styles.centeredLoader}><LoadingSpinner /><p>Loading Profile...</p></div> );
  }

  return (
    <>
      <Navbar />
      {isModalOpen && (
        <EditProfileModal
          user={user}
          profile={profile}
          details={details}
          onClose={() => setIsModalOpen(false)}
          onSave={handleSaveSuccess}
        />
      )}
      <div className={styles.profilePage}>
        <div className={styles.profileHeader}>
          <img src={user?.user_metadata?.avatar_url || '/images/default-avatar.png'} alt="Profile Avatar" className={styles.avatar} />
          <h1>{profile?.full_name || user?.email}</h1>
          <p className={styles.roleTag}>{profile?.role}</p>
        </div>

        <div className={styles.profileContent}>
          <div className={styles.detailsSection}>
            <div className={styles.sectionHeader}>
              <h3 className={styles.sectionTitle}>Account Information</h3>
              <button onClick={() => setIsModalOpen(true)} className={styles.inlineEditButton}>✎ Edit</button>
            </div>
            <DisplayField label="Full Name" value={profile?.full_name} />
            <DisplayField label="Email Address" value={profile?.email} />
          </div>
          <div className={styles.detailsSection}>
            <div className={styles.sectionHeader}>
              <h3 className={styles.sectionTitle}>Role-Specific Details</h3>
              <button onClick={() => setIsModalOpen(true)} className={styles.inlineEditButton}>✎ Edit</button>
            </div>
            <RoleSpecificFields />
          </div>
        </div>
      </div>
    </>
  );
}

export default withAuth(ProfilePage);