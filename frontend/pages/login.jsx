// frontend/pages/login.jsx
import { useAuth } from '@/components/AuthProvider';
import PageLayout from '@/components/PageLayout';
import styles from '@/styles/Home.module.css';
import { useEffect } from 'react';
import { useRouter } from 'next/router';

export default function LoginPage() {
    const { signInWithGoogle, session } = useAuth();
    const router = useRouter();

    // If a user somehow lands here while logged in, redirect them.
    useEffect(() => {
        if (session) {
            router.push('/'); // AuthProvider will handle the rest
        }
    }, [session, router]);

    return (
        <PageLayout>
            <div className={styles.container}>
                <main className={styles.main}>
                    <h1 className={styles.title}>
                        Sign In to AI4NEURO
                    </h1>
                    <p className={styles.description}>
                        Use your Google account to securely access your dashboard.
                    </p>
                    <button onClick={signInWithGoogle} className={styles.button}>
                        Sign In with Google
                    </button>
                </main>
            </div>
        </PageLayout>
    );
}
