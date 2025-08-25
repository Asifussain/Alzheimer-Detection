import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../components/AuthProvider';
import supabase from '../lib/supabaseClient';
import Navbar from '../components/Navbar';
import LoadingSpinner from '../components/LoadingSpinner';
import styles from '../styles/CompleteProfile.module.css';

const ROLES = [
  { 
    id: 'patient', 
    name: 'Patient', 
    description: 'I am seeking medical analysis and care',
    icon: '🏥'
  },
  { 
    id: 'doctor', 
    name: 'Doctor', 
    description: 'I am a healthcare professional providing care',
    icon: '👩‍⚕️'
  },
  { 
    id: 'admin', 
    name: 'Hospital Admin', 
    description: 'I manage hospital operations and staff',
    icon: '👨‍💼'
  },
];

const BLOOD_GROUPS = [
  'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'
];

export default function CompleteProfilePage() {
  const { user, refreshProfile } = useAuth();
  const router = useRouter();
  const [step, setStep] = useState(1); // 1: hospital, 2: role, 3: details
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [hospitals, setHospitals] = useState([]);
  const [bloodGroups, setBloodGroups] = useState([]);
  const [qualifications, setQualifications] = useState([]);
  const [existingProfile, setExistingProfile] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const [formData, setFormData] = useState({
    // Hospital & Basic Info
    hospital_id: '',
    full_name: user?.user_metadata?.full_name || user?.email?.split('@')[0] || '',
    email: user?.email || '',
    phone: '',
    date_of_birth: '',
    address: '',
    role: '',

    // Patient specific
    patient_id: '', // Will be auto-generated
    blood_group_id: '',
    emergency_contact_name: '',
    emergency_contact_phone: '',
    medical_history: '',
    current_medications: '',
    allergies: '',

    // Doctor specific
    medical_license: '',
    qualification_id: '',
    specialization: '',
    experience_years: '',
    consultation_fee: '',

    // Admin specific
    employee_id: '',
    department: '',
  });

  useEffect(() => {
    fetchInitialData();
  }, [user]);

  const fetchInitialData = async () => {
    try {
      setIsLoading(true);

      // Check if user already has a profile - handle potential errors gracefully
      const { data: existingProfileData, error: profileError } = await supabase
        .from('user_profiles')
        .select('*')
        .eq('id', user.id)
        .maybeSingle(); // Use maybeSingle instead of single to avoid errors when no record exists

      // Only set existing profile if we actually found one and there's no error
      if (existingProfileData && !profileError) {
        setExistingProfile(existingProfileData);
        console.log('Found existing profile:', existingProfileData);
        
        // Pre-populate form with existing data
        setFormData(prev => ({
          ...prev,
          hospital_id: existingProfileData.hospital_id || '',
          full_name: existingProfileData.full_name || prev.full_name,
          phone: existingProfileData.phone || '',
          date_of_birth: existingProfileData.date_of_birth || '',
          address: existingProfileData.address || '',
          role: existingProfileData.role || '',
        }));

        // Load role-specific data if exists
        if (existingProfileData.role === 'patient') {
          const { data: patientData } = await supabase
            .from('patient_profiles')
            .select('*')
            .eq('user_id', user.id)
            .maybeSingle();

          if (patientData) {
            setFormData(prev => ({
              ...prev,
              blood_group_id: patientData.blood_group_id || '',
              emergency_contact_name: patientData.emergency_contact_name || '',
              emergency_contact_phone: patientData.emergency_contact_phone || '',
              medical_history: patientData.medical_history || '',
              current_medications: patientData.current_medications || '',
              allergies: patientData.allergies || '',
            }));
          }
        } else if (existingProfileData.role === 'doctor') {
          const { data: doctorData } = await supabase
            .from('doctor_profiles')
            .select('*')
            .eq('user_id', user.id)
            .maybeSingle();

          if (doctorData) {
            setFormData(prev => ({
              ...prev,
              medical_license: doctorData.medical_license || '',
              qualification_id: doctorData.qualification_id || '',
              specialization: doctorData.specialization || '',
              experience_years: doctorData.experience_years?.toString() || '',
              consultation_fee: doctorData.consultation_fee?.toString() || '',
            }));
          }
        } else if (existingProfileData.role === 'admin') {
          const { data: adminData } = await supabase
            .from('admin_profiles')
            .select('*')
            .eq('user_id', user.id)
            .maybeSingle();

          if (adminData) {
            setFormData(prev => ({
              ...prev,
              employee_id: adminData.employee_id || '',
              department: adminData.department || '',
            }));
          }
        }
      } else if (profileError) {
        console.log('No existing profile found or error:', profileError);
        setExistingProfile(null);
      }

      // Fetch hospitals
      const { data: hospitalsData } = await supabase
        .from('hospitals')
        .select('id, name, hospital_code, address')
        .eq('status', 'active')
        .order('name');

      // Fetch blood groups
      const { data: bloodGroupsData } = await supabase
        .from('blood_groups')
        .select('*')
        .order('blood_type');

      // Fetch qualifications
      const { data: qualificationsData } = await supabase
        .from('qualifications')
        .select('*')
        .order('qualification_name');

      setHospitals(hospitalsData || []);
      setBloodGroups(bloodGroupsData || []);
      setQualifications(qualificationsData || []);
    } catch (error) {
      console.error('Error fetching initial data:', error);
      setError('Failed to load required data. Please refresh the page.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setError('');
  };

  const generateUniqueId = async (hospitalCode, role) => {
    try {
      // If updating existing profile, keep the same unique identifier
      if (existingProfile && existingProfile.unique_identifier) {
        return existingProfile.unique_identifier;
      }

      // Get the last sequence number for this hospital and role
      const { data: existingProfiles } = await supabase
        .from('user_profiles')
        .select('unique_identifier')
        .eq('hospital_id', formData.hospital_id)
        .eq('role', role)
        .order('created_at', { ascending: false })
        .limit(1);

      let sequence = 1;
      if (existingProfiles && existingProfiles.length > 0) {
        const lastId = existingProfiles[0].unique_identifier;
        const lastSequence = parseInt(lastId.split('-')[2]) || 0;
        sequence = lastSequence + 1;
      }

      const rolePrefix = {
        'patient': 'PAT',
        'doctor': 'DOC', 
        'admin': 'ADM'
      };
      
      const prefix = rolePrefix[role] || 'USR';
      const paddedSequence = sequence.toString().padStart(4, '0');
      return `${hospitalCode}-${prefix}-${paddedSequence}`;
    } catch (error) {
      console.error('Error generating unique ID:', error);
      return null;
    }
  };

  const handleNext = () => {
    if (step === 1) {
      if (!formData.hospital_id || !formData.full_name || !formData.phone) {
        setError('Please fill in all required fields');
        return;
      }
    } else if (step === 2) {
      if (!formData.role) {
        setError('Please select a role');
        return;
      }
    }
    setStep(step + 1);
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError('');

    try {
      // Double-check for existing profile right before submission
      const { data: currentProfile, error: checkError } = await supabase
        .from('user_profiles')
        .select('*')
        .eq('id', user.id)
        .maybeSingle();

      if (checkError && checkError.code !== 'PGRST116') { // PGRST116 is "no rows returned"
        throw checkError;
      }

      // Update our existing profile state if we found one
      const profileExists = currentProfile && !checkError;
      if (profileExists && !existingProfile) {
        setExistingProfile(currentProfile);
      }

      // Get hospital code for ID generation
      const selectedHospital = hospitals.find(h => h.id === formData.hospital_id);
      if (!selectedHospital) {
        throw new Error('Selected hospital not found');
      }

      // Generate unique identifier (will reuse existing if updating)
      const uniqueId = await generateUniqueId(selectedHospital.hospital_code, formData.role);
      if (!uniqueId) {
        throw new Error('Failed to generate unique identifier');
      }

      const profileData = {
        hospital_id: formData.hospital_id,
        unique_identifier: uniqueId,
        full_name: formData.full_name,
        email: formData.email,
        phone: formData.phone,
        date_of_birth: formData.date_of_birth || null,
        address: formData.address || null,
        role: formData.role,
        account_status: (profileExists ? currentProfile.account_status : null) || 'pending',
        phone_verified: (profileExists ? currentProfile.phone_verified : null) || false
      };

      let userProfileData;

      if (profileExists) {
        console.log('Updating existing profile for user:', user.id);
        // Update existing profile
        const { data: updatedProfile, error: profileError } = await supabase
          .from('user_profiles')
          .update(profileData)
          .eq('id', user.id)
          .select()
          .single();

        if (profileError) {
          console.error('Error updating profile:', profileError);
          throw profileError;
        }
        userProfileData = updatedProfile;
      } else {
        console.log('Creating new profile for user:', user.id);
        // Create new profile with explicit UPSERT to handle race conditions
        const { data: newProfile, error: profileError } = await supabase
          .from('user_profiles')
          .upsert({
            id: user.id,
            ...profileData
          }, { 
            onConflict: 'id',
            ignoreDuplicates: false 
          })
          .select()
          .single();

        if (profileError) {
          console.error('Error creating profile:', profileError);
          throw profileError;
        }
        userProfileData = newProfile;
      }

      // Handle role-specific profiles with proper error handling for RLS
      if (formData.role === 'patient') {
        const patientData = {
          patient_id: uniqueId,
          blood_group_id: formData.blood_group_id || null,
          emergency_contact_name: formData.emergency_contact_name || null,
          emergency_contact_phone: formData.emergency_contact_phone || null,
          medical_history: formData.medical_history || null,
          current_medications: formData.current_medications || null,
          allergies: formData.allergies || null,
          verification_status: 'pending'
        };

        // Check if patient profile exists
        const { data: existingPatient, error: patientCheckError } = await supabase
          .from('patient_profiles')
          .select('user_id')
          .eq('user_id', user.id)
          .maybeSingle();

        if (patientCheckError && patientCheckError.code !== 'PGRST116') {
          console.error('Error checking patient profile:', patientCheckError);
          throw patientCheckError;
        }

        try {
          if (existingPatient) {
            console.log('Updating existing patient profile');
            // Update existing patient profile
            const { error: patientError } = await supabase
              .from('patient_profiles')
              .update(patientData)
              .eq('user_id', user.id);

            if (patientError) {
              console.error('Error updating patient profile:', patientError);
              throw patientError;
            }
          } else {
            console.log('Creating new patient profile');
            // Create new patient profile with explicit user_id
            const { error: patientError } = await supabase
              .from('patient_profiles')
              .insert({
                user_id: user.id,
                ...patientData
              });

            if (patientError) {
              console.error('Error creating patient profile:', patientError);
              
              // If RLS error, try to provide more helpful error message
              if (patientError.code === '42501') {
                throw new Error('Unable to create patient profile. Please ensure you are properly authenticated and have the necessary permissions.');
              }
              throw patientError;
            }
          }
        } catch (error) {
          console.error('Patient profile operation failed:', error);
          throw error;
        }

      } else if (formData.role === 'doctor') {
        const doctorData = {
          medical_license: formData.medical_license,
          qualification_id: formData.qualification_id || null,
          specialization: formData.specialization || null,
          experience_years: parseInt(formData.experience_years) || null,
          consultation_fee: parseFloat(formData.consultation_fee) || null,
          verification_status: 'pending'
        };

        // Check if doctor profile exists
        const { data: existingDoctor, error: doctorCheckError } = await supabase
          .from('doctor_profiles')
          .select('user_id')
          .eq('user_id', user.id)
          .maybeSingle();

        if (doctorCheckError && doctorCheckError.code !== 'PGRST116') {
          throw doctorCheckError;
        }

        if (existingDoctor) {
          console.log('Updating existing doctor profile');
          // Update existing doctor profile
          const { error: doctorError } = await supabase
            .from('doctor_profiles')
            .update(doctorData)
            .eq('user_id', user.id);

          if (doctorError) throw doctorError;
        } else {
          console.log('Creating new doctor profile');
          // Create new doctor profile with UPSERT
          const { error: doctorError } = await supabase
            .from('doctor_profiles')
            .upsert({
              user_id: user.id,
              ...doctorData
            }, {
              onConflict: 'user_id',
              ignoreDuplicates: false
            });

          if (doctorError) throw doctorError;
        }

      } else if (formData.role === 'admin') {
        const adminData = {
          employee_id: formData.employee_id || null,
          department: formData.department || null,
          permissions: {
            manage_doctors: true,
            manage_patients: true,
            view_all_reports: true
          }
        };

        // Check if admin profile exists
        const { data: existingAdmin, error: adminCheckError } = await supabase
          .from('admin_profiles')
          .select('user_id')
          .eq('user_id', user.id)
          .maybeSingle();

        if (adminCheckError && adminCheckError.code !== 'PGRST116') {
          throw adminCheckError;
        }

        if (existingAdmin) {
          console.log('Updating existing admin profile');
          // Update existing admin profile
          const { error: adminError } = await supabase
            .from('admin_profiles')
            .update(adminData)
            .eq('user_id', user.id);

          if (adminError) throw adminError;
        } else {
          console.log('Creating new admin profile');
          // Create new admin profile with UPSERT
          const { error: adminError } = await supabase
            .from('admin_profiles')
            .upsert({
              user_id: user.id,
              ...adminData
            }, {
              onConflict: 'user_id',
              ignoreDuplicates: false
            });

          if (adminError) throw adminError;
        }
      }

      // Refresh profile and redirect
      await refreshProfile();
      router.replace('/account-pending');

    } catch (err) {
      console.error('Error creating/updating profile:', err);
      setError(err.message || 'Failed to save profile. Please try again.');
      setIsSubmitting(false);
    }
  };

  const renderStep1 = () => (
    <div className={styles.stepContent}>
      <h2>Hospital & Basic Information</h2>
      <p className={styles.stepDescription}>
        {existingProfile ? 'Update your basic information and hospital affiliation' : 'First, let\'s get your basic information and hospital affiliation'}
      </p>
      
      <div className={styles.formGroup}>
        <label htmlFor="hospital_id">Select Hospital *</label>
        <select
          id="hospital_id"
          name="hospital_id"
          value={formData.hospital_id}
          onChange={handleChange}
          required
        >
          <option value="">Choose your hospital...</option>
          {hospitals.map(hospital => (
            <option key={hospital.id} value={hospital.id}>
              {hospital.name} ({hospital.hospital_code})
            </option>
          ))}
        </select>
      </div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label htmlFor="full_name">Full Name *</label>
          <input
            type="text"
            id="full_name"
            name="full_name"
            value={formData.full_name}
            onChange={handleChange}
            required
          />
        </div>
        <div className={styles.formGroup}>
          <label htmlFor="phone">Phone Number *</label>
          <input
            type="tel"
            id="phone"
            name="phone"
            value={formData.phone}
            onChange={handleChange}
            placeholder="+1234567890"
            required
          />
        </div>
      </div>

      <div className={styles.formGroup}>
        <label htmlFor="email">Email Address</label>
        <input
          type="email"
          id="email"
          name="email"
          value={formData.email}
          onChange={handleChange}
          disabled
        />
      </div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label htmlFor="date_of_birth">Date of Birth</label>
          <input
            type="date"
            id="date_of_birth"
            name="date_of_birth"
            value={formData.date_of_birth}
            onChange={handleChange}
          />
        </div>
      </div>

      <div className={styles.formGroup}>
        <label htmlFor="address">Address</label>
        <textarea
          id="address"
          name="address"
          value={formData.address}
          onChange={handleChange}
          rows="3"
          placeholder="Your complete address..."
        />
      </div>
    </div>
  );

  const renderStep2 = () => (
    <div className={styles.stepContent}>
      <h2>Select Your Role</h2>
      <p className={styles.stepDescription}>Choose the role that best describes your position</p>
      
      <div className={styles.roleGrid}>
        {ROLES.map((role) => (
          <div
            key={role.id}
            className={`${styles.roleCard} ${formData.role === role.id ? styles.selectedRole : ''}`}
            onClick={() => handleChange({ target: { name: 'role', value: role.id } })}
          >
            <div className={styles.roleIcon}>{role.icon}</div>
            <h3>{role.name}</h3>
            <p>{role.description}</p>
          </div>
        ))}
      </div>
    </div>
  );

  const renderStep3 = () => {
    if (formData.role === 'patient') {
      return (
        <div className={styles.stepContent}>
          <h2>Patient Details</h2>
          <p className={styles.stepDescription}>Please provide your medical information</p>
          
          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label htmlFor="blood_group_id">Blood Group</label>
              <select
                id="blood_group_id"
                name="blood_group_id"
                value={formData.blood_group_id}
                onChange={handleChange}
              >
                <option value="">Select blood group...</option>
                {bloodGroups.map(bg => (
                  <option key={bg.id} value={bg.id}>{bg.blood_type}</option>
                ))}
              </select>
            </div>
          </div>

          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label htmlFor="emergency_contact_name">Emergency Contact Name</label>
              <input
                type="text"
                id="emergency_contact_name"
                name="emergency_contact_name"
                value={formData.emergency_contact_name}
                onChange={handleChange}
              />
            </div>
            <div className={styles.formGroup}>
              <label htmlFor="emergency_contact_phone">Emergency Contact Phone</label>
              <input
                type="tel"
                id="emergency_contact_phone"
                name="emergency_contact_phone"
                value={formData.emergency_contact_phone}
                onChange={handleChange}
              />
            </div>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="medical_history">Medical History</label>
            <textarea
              id="medical_history"
              name="medical_history"
              value={formData.medical_history}
              onChange={handleChange}
              rows="3"
              placeholder="Any relevant medical history..."
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="current_medications">Current Medications</label>
            <textarea
              id="current_medications"
              name="current_medications"
              value={formData.current_medications}
              onChange={handleChange}
              rows="3"
              placeholder="List current medications..."
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="allergies">Allergies</label>
            <textarea
              id="allergies"
              name="allergies"
              value={formData.allergies}
              onChange={handleChange}
              rows="2"
              placeholder="Any known allergies..."
            />
          </div>
        </div>
      );
    } else if (formData.role === 'doctor') {
      return (
        <div className={styles.stepContent}>
          <h2>Doctor Details</h2>
          <p className={styles.stepDescription}>Please provide your professional credentials</p>
          
          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label htmlFor="medical_license">Medical License Number *</label>
              <input
                type="text"
                id="medical_license"
                name="medical_license"
                value={formData.medical_license}
                onChange={handleChange}
                required
              />
            </div>
            <div className={styles.formGroup}>
              <label htmlFor="qualification_id">Qualification</label>
              <select
                id="qualification_id"
                name="qualification_id"
                value={formData.qualification_id}
                onChange={handleChange}
              >
                <option value="">Select qualification...</option>
                {qualifications.map(qual => (
                  <option key={qual.id} value={qual.id}>
                    {qual.qualification_name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label htmlFor="specialization">Specialization</label>
              <input
                type="text"
                id="specialization"
                name="specialization"
                value={formData.specialization}
                onChange={handleChange}
                placeholder="e.g., Neurology, Cardiology"
              />
            </div>
            <div className={styles.formGroup}>
              <label htmlFor="experience_years">Years of Experience</label>
              <input
                type="number"
                id="experience_years"
                name="experience_years"
                value={formData.experience_years}
                onChange={handleChange}
                min="0"
                max="50"
              />
            </div>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="consultation_fee">Consultation Fee ($)</label>
            <input
              type="number"
              id="consultation_fee"
              name="consultation_fee"
              value={formData.consultation_fee}
              onChange={handleChange}
              min="0"
              step="0.01"
            />
          </div>
        </div>
      );
    } else if (formData.role === 'admin') {
      return (
        <div className={styles.stepContent}>
          <h2>Admin Details</h2>
          <p className={styles.stepDescription}>Please provide your administrative information</p>
          
          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label htmlFor="employee_id">Employee ID</label>
              <input
                type="text"
                id="employee_id"
                name="employee_id"
                value={formData.employee_id}
                onChange={handleChange}
              />
            </div>
            <div className={styles.formGroup}>
              <label htmlFor="department">Department</label>
              <input
                type="text"
                id="department"
                name="department"
                value={formData.department}
                onChange={handleChange}
                placeholder="e.g., Administration, IT"
              />
            </div>
          </div>
        </div>
      );
    }
  };

  if (isLoading) {
    return (
      <>
        <Navbar />
        <div className={styles.profileSetup}>
          <div className={styles.setupContainer}>
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '200px' }}>
              <LoadingSpinner size={32} />
            </div>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Navbar />
      <div className={styles.profileSetup}>
        <div className={styles.setupContainer}>
          <div className={styles.setupHeader}>
            <h1>{existingProfile ? 'Update Your Profile' : 'Complete Your Profile'}</h1>
            <div className={styles.progressBar}>
              <div className={styles.progressTrack}>
                <div 
                  className={styles.progressFill}
                  style={{ width: `${(step / 3) * 100}%` }}
                />
              </div>
              <div className={styles.progressSteps}>
                {[1, 2, 3].map(stepNum => (
                  <div
                    key={stepNum}
                    className={`${styles.progressStep} ${stepNum <= step ? styles.activeStep : ''}`}
                  >
                    {stepNum}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <form onSubmit={handleSubmit} className={styles.setupForm}>
            {step === 1 && renderStep1()}
            {step === 2 && renderStep2()}
            {step === 3 && renderStep3()}

            {error && (
              <div className={styles.errorMessage}>
                {error}
              </div>
            )}

            <div className={styles.formActions}>
              {step > 1 && (
                <button
                  type="button"
                  onClick={() => setStep(step - 1)}
                  className={styles.backButton}
                >
                  Back
                </button>
              )}
              
              {step < 3 ? (
                <button
                  type="button"
                  onClick={handleNext}
                  className={styles.nextButton}
                >
                  Next
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className={styles.submitButton}
                >
                  {isSubmitting ? (
                    <>
                      <LoadingSpinner size={16} />
                      {existingProfile ? 'Updating Profile...' : 'Creating Profile...'}
                    </>
                  ) : (
                    existingProfile ? 'Update Profile' : 'Complete Setup'
                  )}
                </button>
              )}
            </div>
          </form>
        </div>
      </div>
    </>
  );
}