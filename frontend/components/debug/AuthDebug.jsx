import { useAuth } from '../AuthProvider';
import emailAuthClient from '@/lib/emailAuthClient';

const AuthDebug = () => {
  const { session, user, userProfile, isLoading } = useAuth();

  const checkToken = async () => {
    const token = localStorage.getItem('auth_token');
    console.log('Stored token:', token ? 'Present' : 'None');
    
    if (token) {
      try {
        const response = await emailAuthClient.verifyToken();
        console.log('Token verification response:', response);
      } catch (error) {
        console.log('Token verification error:', error);
      }
    }
  };

  // Only show in development
  if (process.env.NODE_ENV !== 'development') {
    return null;
  }

  return (
    <div style={{
      position: 'fixed',
      bottom: '10px',
      right: '10px',
      background: 'rgba(0,0,0,0.9)',
      color: 'white',
      padding: '12px',
      borderRadius: '8px',
      fontSize: '11px',
      zIndex: 9999,
      maxWidth: '350px',
      maxHeight: '400px',
      overflow: 'auto',
      border: '1px solid rgba(255,255,255,0.2)'
    }}>
      <strong style={{color: '#60a5fa'}}>🔍 Auth Debug Panel</strong><br/>
      <hr style={{margin: '5px 0', border: '1px solid rgba(255,255,255,0.1)'}}/>
      
      <strong>Core State:</strong><br/>
      Loading: <span style={{color: isLoading ? '#f87171' : '#34d399'}}>{isLoading ? 'Yes' : 'No'}</span><br/>
      User: <span style={{color: user ? '#34d399' : '#f87171'}}>{user ? 'Present' : 'None'}</span><br/>
      Session: <span style={{color: session ? '#34d399' : '#f87171'}}>{session ? 'Present' : 'None'}</span><br/>
      Profile: <span style={{color: userProfile ? '#34d399' : '#f87171'}}>{userProfile ? 'Present' : 'None'}</span><br/>
      
      {userProfile && (
        <>
          <br/><strong>Profile Details:</strong><br/>
          Role: <span style={{color: '#fbbf24'}}>{userProfile.role || 'Not Set'}</span><br/>
          Status: <span style={{color: userProfile.account_status === 'active' ? '#34d399' : '#f87171'}}>{userProfile.account_status}</span><br/>
          Phone Verified: <span style={{color: userProfile.phone_verified ? '#34d399' : '#f87171'}}>{userProfile.phone_verified ? 'Yes' : 'No'}</span><br/>
          Auth Provider: <span style={{color: '#a78bfa'}}>{userProfile.auth_provider}</span><br/>
          Email: <span style={{color: '#60a5fa'}}>{userProfile.email}</span><br/>
          ID: <span style={{fontSize: '9px', color: '#9ca3af'}}>{userProfile.id?.substring(0,8)}...</span><br/>
        </>
      )}
      
      {user && (
        <>
          <br/><strong>User Object:</strong><br/>
          User ID: <span style={{fontSize: '9px', color: '#9ca3af'}}>{user.id?.substring(0,8)}...</span><br/>
          Email: <span style={{color: '#60a5fa'}}>{user.email}</span><br/>
        </>
      )}
      
      <br/>
      <strong>Local Storage:</strong><br/>
      JWT Token: <span style={{color: localStorage.getItem('auth_token') ? '#34d399' : '#f87171'}}>
        {localStorage.getItem('auth_token') ? 'Present' : 'None'}
      </span><br/>
      
      <div style={{marginTop: '8px', display: 'flex', gap: '5px'}}>
        <button onClick={checkToken} style={{
          padding: '3px 6px',
          fontSize: '9px',
          background: '#3b82f6',
          color: 'white',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer'
        }}>
          Check Token
        </button>
        <button onClick={() => console.log('Auth State:', {user, userProfile, session, isLoading})} style={{
          padding: '3px 6px',
          fontSize: '9px',
          background: '#8b5cf6',
          color: 'white',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer'
        }}>
          Log State
        </button>
      </div>
    </div>
  );
};

export default AuthDebug;