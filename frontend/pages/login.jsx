// frontend/pages/login.jsx
import { useEffect } from 'react';
import supabase from '../lib/supabaseClient';

export default function LoginPage() {
  useEffect(() => {
    const handleOAuth = async () => {
      await supabase.auth.signInWithOAuth({ provider: 'google' });
    };
    handleOAuth();
  }, []);

  return (
    <div style={{ color: 'white' }}>
      <h1>Redirecting to Google...</h1>
    </div>
  );
}
