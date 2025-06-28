// frontend/components/PageLayout.jsx
import Navbar from './Navbar';
import styles from '../styles/PageLayout.module.css';

const PageLayout = ({ children }) => {
  return (
    <div className={styles.layoutContainer}>
      <Navbar />
      <main className={styles.mainContent}>
        {children}
      </main>
      {/* You could add a Footer component here later if you want */}
    </div>
  );
};

export default PageLayout;
