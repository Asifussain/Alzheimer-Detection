import { AuthProvider } from '../components/AuthProvider';
import AuthDebug from '../components/debug/AuthDebug';
import '../styles/globals.css';

function MyApp({ Component, pageProps }) {
  return (
    <AuthProvider>
      <Component {...pageProps} />
      <AuthDebug />
    </AuthProvider>
  );
}

export default MyApp;
