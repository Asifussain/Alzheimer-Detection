import { useState } from 'react';
import Navbar from '../components/Navbar';
import styles from '../styles/PageLayout.module.css'; // Reuse the CSS module
import formStyles from '../styles/ContactForm.module.css'; // Create this CSS module for form styling

export default function ContactPage() {
  const [formData, setFormData] = useState({ name: '', email: '', subject: '', message: '' });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState(null); // null | 'success' | 'error'

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitStatus(null);

    // **Backend Integration Needed Here**
    // This part requires a backend endpoint (e.g., using Supabase Functions, SendGrid, etc.)
    // to actually process and send the email or store the message.
    // For now, we'll simulate a successful submission.
    console.log("Form data submitted (simulation):", formData);
    await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate network delay

    // Replace with actual API call result
    const success = true; 

    if (success) {
      setSubmitStatus('success');
      setFormData({ name: '', email: '', subject: '', message: '' }); // Clear form
    } else {
      setSubmitStatus('error');
    }
    
    setIsSubmitting(false);
  };

  return (
    <>
      <Navbar />
      <div className={styles.pageContainer}>
        <h1 className={styles.pageTitle}>Contact Us</h1>
        
        <section className={styles.section}>
          <p className={styles.paragraph}>
            Have questions, feedback, or partnership inquiries? Reach out to us!
          </p>
          {/* Add placeholder contact details */}
          <p className={styles.paragraph}>
            <strong>Email:</strong> info@ai4neuro.com
          </p>
        </section>

        <section className={`${styles.section} ${formStyles.contactFormSection}`}>
          <h2 className={styles.sectionTitle}>Send Us a Message</h2>
          <form onSubmit={handleSubmit} className={formStyles.contactForm}>
            <div className={formStyles.formGroup}>
              <label htmlFor="name" className={formStyles.formLabel}>Name:</label>
              <input 
                type="text" 
                id="name" 
                name="name" 
                className={formStyles.formInput} 
                value={formData.name}
                onChange={handleChange}
                required 
              />
            </div>
            <div className={formStyles.formGroup}>
              <label htmlFor="email" className={formStyles.formLabel}>Email:</label>
              <input 
                type="email" 
                id="email" 
                name="email" 
                className={formStyles.formInput} 
                value={formData.email}
                onChange={handleChange}
                required 
               />
            </div>
            <div className={formStyles.formGroup}>
              <label htmlFor="subject" className={formStyles.formLabel}>Subject:</label>
              <input 
                type="text" 
                id="subject" 
                name="subject" 
                className={formStyles.formInput} 
                value={formData.subject}
                onChange={handleChange}
                required 
              />
            </div>
            <div className={formStyles.formGroup}>
              <label htmlFor="message" className={formStyles.formLabel}>Message:</label>
              <textarea 
                id="message" 
                name="message" 
                rows="5" 
                className={formStyles.formTextarea} 
                value={formData.message}
                onChange={handleChange}
                required 
              ></textarea>
            </div>
            
            {submitStatus === 'success' && <p className={formStyles.successMessage}>Message sent successfully!</p>}
            {submitStatus === 'error' && <p className={formStyles.errorMessage}>Failed to send message. Please try again.</p>}

            <button type="submit" className={formStyles.submitButton} disabled={isSubmitting}>
              {isSubmitting ? 'Sending...' : 'Send Message'}
            </button>
          </form>
        </section>
      </div>
      {/* Optional: Add a Footer component here */}
    </>
  );
}