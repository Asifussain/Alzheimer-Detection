import { useState, useEffect } from 'react';
import { useAuth } from '../../components/AuthProvider';
import { useRouter } from 'next/router';
import supabase from '../../lib/supabaseClient';
import Navbar from '../../components/Navbar';
import LoadingSpinner from '../../components/LoadingSpinner';
import withAuth from '../../components/withAuth';
import styles from '../../styles/Profile.module.css';

function ProfilePage() {
  const { user, userProfile, hospitalData, refreshProfile } = useAuth();
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState({});
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    if (userProfile) {
      setEditData({
        full_name: userProfile.full_name || '',
        phone: userProfile.phone || '',
        address: userProfile.address || ''
      });
      setIsLoading(false);
    }
  }, [userProfile]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setEditData(prev => ({
      ...prev,
      [name]: value
    }));
    setError('');
  };

  const handleSave = async () => {
    try {
      setIsLoading(true);
      setError('');

      // Validate inputs
      if (!editData.full_name.trim()) {
        setError('Full name is required');
        setIsLoading(false);
        return;
      }

      if (!editData.phone.trim()) {
        setError('Phone number is required');
        setIsLoading(false);
        return;
      }

      // Update user profile
      const { error: updateError } = await supabase
        .from('user_profiles')
        .update({
          full_name: editData.full_name.trim(),
          phone: editData.phone.trim(),
          address: editData.address.trim(),
          updated_at: new Date().toISOString()
        })
        .eq('id', user.id);

      if (updateError) throw updateError;

      await refreshProfile();
      setSuccess('Profile updated successfully!');
      setIsEditing(false);

      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(''), 3000);

    } catch (error) {
      console.error('Error updating profile:', error);
      setError('Failed to update profile. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    setEditData({
      full_name: userProfile?.full_name || '',
      phone: userProfile?.phone || '',
      address: userProfile?.address || ''
    });
    setIsEditing(false);
    setError('');
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return '#10b981';
      case 'pending': return '#f59e0b';
      case 'suspended': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getRoleIcon = (role) => {
    switch (role) {
      case 'patient': return '🏥';
      case 'doctor': return '👨‍⚕️';
      case 'admin': return '👑';
      default: return '👤';
    }
  };

  if (isLoading && !userProfile) {
    return (
      <div className={styles.loadingContainer}>
        <LoadingSpinner />
        <p>Loading your profile...</p>
      </div>
    );
  }

  if (!userProfile) {
    return (
      <div className={styles.errorContainer}>
        <h2>Profile Not Found</h2>
        <p>Unable to load your profile information.</p>
        <button onClick={() => router.push('/complete-profile')} className={styles.primaryButton}>
          Complete Profile Setup
        </button>
      </div>
    );
  }

  return (
    <>
      <Navbar />
      <div className={styles.profilePage}>
        <div className={styles.profileContainer}>
          {/* Profile Header */}
          <div className={styles.profileHeader}>
            <div className={styles.avatarSection}>
              <div className={styles.avatar}>
                <span className={styles.avatarText}>
                  {userProfile.full_name?.charAt(0)?.toUpperCase() || '?'}
                </span>
              </div>
              <div className={styles.userInfo}>
                <h1 className={styles.userName}>{userProfile.full_name || 'No Name Set'}</h1>
                <div className={styles.userMeta}>
                  <span className={styles.roleChip}>
                    {getRoleIcon(userProfile.role)} {userProfile.role}
                  </span>
                  <span 
                    className={styles.statusChip}
                    style={{ backgroundColor: getStatusColor(userProfile.account_status) }}
                  >
                    {userProfile.account_status}
                  </span>
                </div>
                <p className={styles.userEmail}>{userProfile.email}</p>
                <p className={styles.userId}>ID: {userProfile.unique_identifier}</p>
              </div>
            </div>

            {!isEditing && (
              <button 
                onClick={() => setIsEditing(true)}
                className={styles.editButton}
              >
                Edit Profile
              </button>
            )}
          </div>

          {/* Success/Error Messages */}
          {success && (
            <div className={styles.successMessage}>
              ✅ {success}
            </div>
          )}

          {error && (
            <div className={styles.errorMessage}>
              ❌ {error}
            </div>
          )}

          {/* Profile Content */}
          <div className={styles.profileContent}>
            {/* Basic Information */}
            <div className={styles.infoCard}>
              <h3 className={styles.cardTitle}>Basic Information</h3>
              <div className={styles.infoGrid}>
                <div className={styles.infoItem}>
                  <label className={styles.infoLabel}>Full Name</label>
                  {isEditing ? (
                    <input
                      type="text"
                      name="full_name"
                      value={editData.full_name}
                      onChange={handleInputChange}
                      className={styles.editInput}
                      placeholder="Enter your full name"
                    />
                  ) : (
                    <span className={styles.infoValue}>
                      {userProfile.full_name || 'Not set'}
                    </span>
                  )}
                </div>

                <div className={styles.infoItem}>
                  <label className={styles.infoLabel}>Email</label>
                  <span className={styles.infoValue}>
                    {userProfile.email}
                  </span>
                </div>

                <div className={styles.infoItem}>
                  <label className={styles.infoLabel}>Phone Number</label>
                  {isEditing ? (
                    <input
                      type="tel"
                      name="phone"
                      value={editData.phone}
                      onChange={handleInputChange}
                      className={styles.editInput}
                      placeholder="Enter your phone number"
                    />
                  ) : (
                    <span className={styles.infoValue}>
                      {userProfile.phone || 'Not set'}
                    </span>
                  )}
                </div>

                <div className={styles.infoItem}>
                  <label className={styles.infoLabel}>Date of Birth</label>
                  <span className={styles.infoValue}>
                    {userProfile.date_of_birth 
                      ? new Date(userProfile.date_of_birth).toLocaleDateString()
                      : 'Not set'
                    }
                  </span>
                </div>

                <div className={styles.infoItem}>
                  <label className={styles.infoLabel}>Address</label>
                  {isEditing ? (
                    <textarea
                      name="address"
                      value={editData.address}
                      onChange={handleInputChange}
                      className={styles.editTextarea}
                      placeholder="Enter your address"
                      rows={3}
                    />
                  ) : (
                    <span className={styles.infoValue}>
                      {userProfile.address || 'Not set'}
                    </span>
                  )}
                </div>

                <div className={styles.infoItem}>
                  <label className={styles.infoLabel}>Hospital</label>
                  <span className={styles.infoValue}>
                    {hospitalData?.name || 'Not specified'}
                  </span>
                </div>
              </div>
            </div>

            {/* Account Status */}
            <div className={styles.infoCard}>
              <h3 className={styles.cardTitle}>Account Status</h3>
              <div className={styles.statusGrid}>
                <div className={styles.statusItem}>
                  <span className={styles.statusLabel}>Account Status</span>
                  <span 
                    className={styles.statusBadge}
                    style={{ backgroundColor: getStatusColor(userProfile.account_status) }}
                  >
                    {userProfile.account_status}
                  </span>
                </div>
                <div className={styles.statusItem}>
                  <span className={styles.statusLabel}>Phone Verified</span>
                  <span className={styles.statusBadge} style={{
                    backgroundColor: userProfile.phone_verified ? '#10b981' : '#ef4444'
                  }}>
                    {userProfile.phone_verified ? '✅ Verified' : '❌ Not Verified'}
                  </span>
                </div>
                <div className={styles.statusItem}>
                  <span className={styles.statusLabel}>Member Since</span>
                  <span className={styles.infoValue}>
                    {new Date(userProfile.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>
            </div>

            {/* Edit Actions */}
            {isEditing && (
              <div className={styles.editActions}>
                <button 
                  onClick={handleSave}
                  disabled={isLoading}
                  className={styles.saveButton}
                >
                  {isLoading ? <LoadingSpinner size={16} /> : '💾'} Save Changes
                </button>
                <button 
                  onClick={handleCancel}
                  className={styles.cancelButton}
                >
                  ❌ Cancel
                </button>
              </div>
            )}

            {/* Quick Actions */}
            {!isEditing && userProfile.role && (
              <div className={styles.quickActions}>
                <button 
                  onClick={() => router.push(`/${userProfile.role}/dashboard`)}
                  className={styles.actionButton}
                >
                  📊 Go to Dashboard
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

export default withAuth(ProfilePage);