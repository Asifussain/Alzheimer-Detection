import Navbar from '../components/Navbar';
import styles from '../styles/PageLayout.module.css'; // Reuse the CSS module

export default function AboutPage() {
  return (
    <>
      <Navbar />
      <div className={styles.pageContainer}>
        <h1 className={styles.pageTitle}>About NeuroSynapse</h1>

        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Our Mission</h2>
          <p className={styles.paragraph}>
            Our mission is to harness the power of Artificial Intelligence and EEG analysis to contribute 
            to the early detection of Alzheimer's disease patterns. We aim to provide a valuable tool 
            for individuals and researchers, facilitating awareness and potentially enabling earlier 
            medical consultations.
          </p>
        </section>

        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>The Technology: ADFormer & EEG</h2>
          <p className={styles.paragraph}>
            NeuroSynapse employs the ADFormer model, a specialized Transformer architecture designed 
            for time-series classification tasks like EEG analysis. EEG data provides a rich, non-invasive 
            window into brain activity. By applying advanced AI techniques to this data, we can identify 
            subtle, complex patterns that might correlate with the neurological changes associated with 
            Alzheimer's disease.
          </p>
          <p className={styles.paragraph}>
            Our model is trained on carefully processed datasets relevant to Alzheimer's research, 
            enabling it to learn distinctive features from the EEG signals. 
            {/* Optional: Add link to paper or more details if available */}
            {/* <a href="#" className={styles.link}>Learn more about the ADFormer research</a> */}
          </p>
        </section>

         <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Data Privacy & Security</h2>
          <p className={styles.paragraph}>
            We understand the sensitivity of health-related data. User authentication and data storage 
            are managed securely using Supabase. Uploaded EEG files are processed for analysis and results 
            are linked to your user account, accessible only by you through the "Previous Uploads" page. 
            We are committed to maintaining user privacy and data security according to best practices.
          </p>
        </section>

        {/* Optional: Add sections about the team or future scope */}
        {/*
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Meet the Team</h2>
          <p className={styles.paragraph}>[Placeholder for team information or project background]</p>
        </section>

        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Future Directions</h2>
          <p className={styles.paragraph}>[Placeholder for future plans, e.g., model improvements, additional analysis features]</p>
        </section>
        */}
      </div>
      {/* Optional: Add a Footer component here */}
    </>
  );
}