import { createContext, useContext, useState, useEffect } from 'react';
import { 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword, 
  signOut, 
  onAuthStateChanged 
} from 'firebase/auth';
import { doc, setDoc, getDoc } from 'firebase/firestore';
import { auth, db } from '../firebase';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);
  const [userRole, setUserRole] = useState(null); // 'student' | 'teacher'
  const [loading, setLoading] = useState(true);

  // MOCK fallback for when Firebase is not configured with real keys yet.
  // This allows the UI to work flawlessly until credentials are provided.
  const isMock = !auth || auth.app.options.apiKey === "YOUR_API_KEY";

  useEffect(() => {
    if (isMock) {
      console.log("🔥 Running with Local Storage Mock Auth (Firebase not configured yet)");
      const storedLocal = localStorage.getItem('mock_user_cast');
      if (storedLocal) {
        const mockUser = JSON.parse(storedLocal);
        setCurrentUser(mockUser);
        setUserRole(mockUser.role || 'student');
      }
      setLoading(false);
      return;
    }

    // Real Firebase listener
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setCurrentUser(user);
      if (user) {
        try {
          const docRef = doc(db, 'users', user.uid);
          const docSnap = await getDoc(docRef);
          if (docSnap.exists() && docSnap.data().role) {
            setUserRole(docSnap.data().role);
          } else {
            setUserRole('student'); // Default fallback
          }
        } catch (e) {
          console.error("Firestore read error", e);
          setUserRole('student'); 
        }
      } else {
        setUserRole(null);
      }
      setLoading(false);
    });
    return unsubscribe;
  }, [isMock]);

  const signup = async (email, password, role) => {
    if (isMock) {
      const mockUser = { uid: Date.now().toString(), email, role };
      localStorage.setItem('mock_user_cast', JSON.stringify(mockUser));
      setCurrentUser(mockUser);
      setUserRole(role);
      return;
    }
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    // Store role in Firestore
    await setDoc(doc(db, 'users', userCredential.user.uid), {
      email,
      role
    });
    setUserRole(role);
  };

  const login = async (email, password) => {
    if (isMock) {
      const storedLocal = localStorage.getItem('mock_user_cast');
      if (!storedLocal) {
        throw new Error("No user found. Please sign up first.");
      }
      const mockUser = JSON.parse(storedLocal);
      setCurrentUser(mockUser);
      setUserRole(mockUser.role || 'student');
      return;
    }
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    // Fetch role after login explicitly just to be safe
    const docRef = doc(db, 'users', userCredential.user.uid);
    const docSnap = await getDoc(docRef);
    if (docSnap.exists() && docSnap.data().role) {
       setUserRole(docSnap.data().role);
    } else {
       setUserRole('student'); // Default fallback
    }
  };

  const logout = async () => {
    if (isMock) {
      localStorage.removeItem('mock_user_cast');
      setCurrentUser(null);
      setUserRole(null);
      return;
    }
    await signOut(auth);
    setCurrentUser(null);
    setUserRole(null);
  };

  const value = {
    currentUser,
    userRole,
    login,
    signup,
    logout
  };

  // Prevent a blank screen by showing a loading status
  if (loading) {
     return (
       <div style={{
         height: '100vh', 
         backgroundColor: '#fff5e4', 
         display: 'flex', 
         flexDirection: 'column',
         alignItems: 'center', 
         justifyContent: 'center',
         fontFamily: 'sans-serif'
       }}>
         <div style={{fontSize: '3rem', marginBottom: '1rem'}}>✨</div>
         <h2 style={{color: '#1e293b', margin: 0}}>Initializing Vardaan...</h2>
         <p style={{color: '#64748b'}}>Connecting to your Firebase project...</p>
       </div>
     );
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
