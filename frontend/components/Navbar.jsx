import { useEffect, useState, useRef } from 'react';
import Link from 'next/link';
import supabase from '../lib/supabaseClient';
import styles from '../styles/Navbar.module.css';

export default function Navbar() {
  const [user, setUser] = useState(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null); 

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) {
        setUser(session.user);
      }
    });

    const { data: listener } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setUser(session?.user ?? null);
        if (_event === 'SIGNED_OUT' || _event === 'SIGNED_IN') {
             setDropdownOpen(false);
        }
      }
    );
    return () => {
      listener?.subscription?.unsubscribe();
    };
  }, []);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    };
    if (dropdownOpen) {
        document.addEventListener('mousedown', handleClickOutside);
    } else {
        document.removeEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [dropdownOpen]); 

  const handleLogin = async () => {
    await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { /* redirectTo options if needed */ }
    });
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    setDropdownOpen(false); 
  };

  const toggleDropdown = () => {
    setDropdownOpen((prev) => !prev);
  };

  const profileImage = user?.user_metadata?.avatar_url || '/images/default-avatar.png';
  const displayName = user?.user_metadata?.full_name || user?.email || 'User';

  return (
    <nav className={styles.navbar}>
      <div className={styles.navbarBrand}>
        <Link href="/">AI4NEURO</Link>
        </div>
      <ul className={styles.navbarLinks}>
        <li><Link href="/">Home</Link></li>
        <li><Link href="/service">Service</Link></li>
        <li><Link href="/about">About</Link></li>
        <li><Link href="/contact">Contact Us</Link></li>
        <li><Link href="/previous">History</Link></li>
      </ul>
      <div className={styles.rightSection}>
        {user ? (
          <div className={styles.profileContainer} ref={dropdownRef}>
            <img
              src={profileImage}
              alt="Profile"
              className={styles.profilePicture}
              onClick={toggleDropdown} 
              onError={(e) => { e.target.onerror = null; e.target.src='/images/default-avatar.png'}}
            />
            <div className={`${styles.dropdown} ${dropdownOpen ? styles.open : ''}`}>
              <div className={styles.userInfo}>
                <span>{displayName}</span>
              </div>
              <button onClick={handleLogout} className={styles.logoutBtn}>
                Logout
              </button>
            </div>
          </div>
        ) : (
          <button onClick={handleLogin} className={styles.loginBtn}>
            Login
          </button>
        )}
      </div>
    </nav>
  );
}
