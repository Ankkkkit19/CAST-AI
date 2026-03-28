import { doc, setDoc, onSnapshot, updateDoc, serverTimestamp } from 'firebase/firestore';
import { db } from '../firebase';

const SESSION_ID = 'VardaanInclusiveClassroom_101';

/**
 * Teacher starts (or re-activates) the class session.
 */
export const startClassSession = async (teacherId) => {
  const sessionRef = doc(db, 'sessions', SESSION_ID);
  await setDoc(sessionRef, {
    teacherId,
    active: true,
    lastSign: null,
    lastSignEmoji: null,
    lastSignAt: serverTimestamp(),
    lastUpdated: serverTimestamp(),
  }, { merge: true });
};

/**
 * Teacher ends the class session.
 */
export const endClassSession = async () => {
  const sessionRef = doc(db, 'sessions', SESSION_ID);
  await updateDoc(sessionRef, {
    active: false,
    lastUpdated: serverTimestamp(),
  });
};

/**
 * Teacher broadcasts a detected sign (with its emoji) to all students via Firestore.
 */
export const broadcastSign = async (sign, emoji = '🤟') => {
  const sessionRef = doc(db, 'sessions', SESSION_ID);
  await updateDoc(sessionRef, {
    lastSign: sign,
    lastSignEmoji: emoji,
    lastSignAt: serverTimestamp(),
    lastUpdated: serverTimestamp(),
  });
};

/**
 * Subscribe to the class session in real-time (used by both teacher and student).
 * onUpdate receives the full session document data.
 */
export const subscribeToClassSession = (onUpdate) => {
  const sessionRef = doc(db, 'sessions', SESSION_ID);
  return onSnapshot(sessionRef, (docSnap) => {
    if (docSnap.exists()) {
      onUpdate(docSnap.data());
    } else {
      // Session document doesn't exist yet – report as inactive
      onUpdate({ active: false, lastSign: null, lastSignEmoji: null });
    }
  });
};
