import { doc, setDoc, getDoc, onSnapshot, collection, query, where } from 'firebase/firestore';
import { db } from '../firebase';

// Save student progress to Firestore
export async function saveProgress(userId, courseId, data) {
  if (!userId || !db) return;
  try {
    const ref = doc(db, 'progress', userId);
    const existing = await getDoc(ref);
    const current = existing.exists() ? existing.data() : {};
    await setDoc(ref, {
      ...current,
      [courseId]: {
        ...((current[courseId]) || {}),
        ...data,
        lastUpdated: new Date().toISOString(),
      },
      email: data.email || current.email || '',
      displayName: data.displayName || current.displayName || '',
    }, { merge: true });
  } catch (err) {
    console.error('Progress save error:', err);
  }
}

// Subscribe to ALL students' progress (for Teacher Dashboard)
export function subscribeToAllProgress(callback) {
  if (!db) return () => {};
  const q = collection(db, 'progress');
  return onSnapshot(q, (snapshot) => {
    const data = snapshot.docs.map(d => ({ id: d.id, ...d.data() }));
    callback(data);
  });
}
