import { useEffect, useState, useRef } from 'react';
import Link from 'next/link';
import supabase from '../lib/supabaseClient';
import styles from '../styles/Navbar.module.css';

export default function Navbar() {
  const [user, setUser] = useState(null);
  const [profile, setProfile] = useState(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false); // Restored
  const dropdownRef = useRef(null);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) {
        setUser(session.user);
        fetchProfile(session.user.id);
      }
    });

    const { data: listener } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setUser(session?.user ?? null);
        if (session?.user) {
          fetchProfile(session.user.id);
        } else {
          setProfile(null);
        }
        if (_event === 'SIGNED_OUT' || _event === 'SIGNED_IN') {
          setDropdownOpen(false);
          setMobileMenuOpen(false);
        }
      }
    );

    return () => {
      listener?.subscription?.unsubscribe();
    };
  }, []);

  async function fetchProfile(userId) {
    const { data, error } = await supabase
      .from('profiles')
      .select('*')
      .eq('id', userId)
      .single();

    if (!error) setProfile(data);
  }

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleLogin = async () => {
    await supabase.auth.signInWithOAuth({ provider: 'google' });
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    setDropdownOpen(false);
  };

  const toggleDropdown = () => setDropdownOpen((prev) => !prev);

  // --- FIX IS HERE ---
  // Restored the functions for handling the mobile menu state
  const toggleMobileMenu = () => setMobileMenuOpen((prev) => !prev);
  const closeMobileMenu = () => setMobileMenuOpen(false);
  // --------------------

  const profileImage = user?.user_metadata?.avatar_url || '/images/default-avatar.png';
  const displayName = profile?.full_name || user?.email || 'User';

  return (
    <nav className={styles.navbar}>
      <div className={styles.navbarBrand}>
        <Link href={user ? "/home" : "/landing"}>AI4NEURO</Link>
      </div>

      {/* Restored the mobile menu toggle button */}
      <button
        className={styles.mobileMenuToggle}
        onClick={toggleMobileMenu}
        aria-label="Toggle mobile menu"
      >
        ☰
      </button>

      {/* The onClick handlers will now work correctly */}
      <ul className={`${styles.navbarLinks} ${mobileMenuOpen ? styles.mobileOpen : ''}`}>
        <li><Link href="/home" onClick={closeMobileMenu}>Home</Link></li>
        <li>
          <Link
            href={user && profile?.role ? `/${profile.role}/dashboard` : "/"}
            onClick={closeMobileMenu}
          >
            Dashboard
          </Link>
        </li>
        <li><Link href="/about" onClick={closeMobileMenu}>About</Link></li>
        <li><Link href="/contact" onClick={closeMobileMenu}>Contact Us</Link></li>
        <li><Link href="/previous" onClick={closeMobileMenu}>History</Link></li>
      </ul>

      <div className={styles.rightSection}>
        {user ? (
          <div className={styles.profileContainer} ref={dropdownRef}>
            <img
              src={profileImage}
              alt="Profile"
              className={styles.profilePicture}
              onClick={toggleDropdown}
            />
            <div className={`${styles.dropdown} ${dropdownOpen ? styles.open : ''}`}>
              <div className={styles.userInfo}>
                <span>{displayName}</span>
              </div>
              <Link href="/profile" className={styles.dropdownLink}>
                Profile
              </Link>
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