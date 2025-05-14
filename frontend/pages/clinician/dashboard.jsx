import Navbar from '../../components/Navbar';
import { useAuth } from '../../components/AuthProvider';
import withAuth from '../../components/withAuth';
import FileUploadSection from '../../components/FileUploadSection';
// Import Hero styles for the layout
import heroStyles from '../../styles/Hero.module.css';
// Keep PageLayout styles for potential container/padding use if needed
import pageStyles from '../../styles/PageLayout.module.css';

function ClinicianDashboard() {
  const { user, profile } = useAuth();

  return (
    <>
      <Navbar />
      {/* Use the hero section as the main layout container */}
      <section className={heroStyles.hero}>

        {/* Left Content Column */}
        <div className={heroStyles.heroContent}>

          {/* 1. Dashboard Title */}
          <h1 className={heroStyles.mainTitle} style={{ marginBottom: '0.5rem' }}>
             Clinician Dashboard
          </h1>

          {/* 2. Welcome Message */}
          <p style={{ marginBottom: '2.5rem', color: '#ccc' }}>
             Welcome, Dr. {profile?.full_name || user?.email}!
          </p>

          {/* 3. File Upload Section */}
          {/* Clinicians might upload for patients? Adapt component if needed later */}
          <FileUploadSection />

          <hr style={{ margin: '3rem 0', borderColor: 'rgba(255, 255, 255, 0.1)' }} />

          {/* 4. Placeholder for Clinician-Specific Reports/Views */}
          <section>
            <h2 style={{ marginBottom: '1.5rem', fontWeight: '500', color: '#eee' }}>
              Patient Reports & Management
            </h2>
            {/* Example: <PatientList /> */}
            {/* Example: Component for viewing individual patient detailed reports */}
            {/* Example: Advanced report generation with comparison features */}
            <p style={{ color: '#aaa' }}>(Clinician-specific patient lists and advanced report views go here)</p>
          </section>

        </div>

        {/* Right Image Column */}
        <div className={heroStyles.heroImageContainer}>
          <img
             src="/images/brain.png"
             alt="Brain Visualization"
             className={heroStyles.heroImage}
           />
        </div>

      </section>
    </>
  );
}

// Apply authentication and role check for 'clinician'
export default withAuth(ClinicianDashboard, ['clinician']);