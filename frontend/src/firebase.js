import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getAnalytics, isSupported } from "firebase/analytics";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyC88JJ8odYgdPO9Q0s7gbvX6IxIcjzCn0o",
  authDomain: "verdent-45e21.firebaseapp.com",
  projectId: "verdent-45e21",
  storageBucket: "verdent-45e21.firebasestorage.app",
  messagingSenderId: "639111507256",
  appId: "1:639111507256:web:98b2c4f4966382892625cd",
  measurementId: "G-E1B8JSMJJY"
};

// Initialize Firebase with safety wrappers
let app, auth, db, analytics;

try {
  app = initializeApp(firebaseConfig);
  auth = getAuth(app);
  db = getFirestore(app);
  
  // Analytics is optional and can fail in many environments
  isSupported().then(yes => {
    if (yes) analytics = getAnalytics(app);
  }).catch(() => {});

} catch (err) {
  console.error("Firebase initialization failed:", err);
}

export { auth, db, analytics };