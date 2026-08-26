/**
 * session-store.js — keep a completed live-task session in the browser.
 *
 * WHY THIS EXISTS. The first version handed data to the Analyse page through a
 * one-shot sessionStorage key, consumed on arrival. That is fragile in two ways: a
 * stale cached copy of the receiving page silently drops the payload, and there is no
 * way back to your own results once you navigate away. Persisting the session instead
 * makes the handoff a read rather than a transfer, and lets someone move between pages
 * and return to what they ran.
 *
 * WHERE IT LIVES. localStorage, on this origin, in the visitor's own browser. Nothing
 * is uploaded and nothing is sent anywhere -- the model runs client-side, so there is
 * no server to send it to. One session is kept (the most recent); starting a new task
 * replaces it. clear() removes it, and both pages expose a control that calls clear().
 *
 * WHAT IS STORED. The confusion matrix, the trial-level log, and enough metadata to
 * describe the run. No identifiers beyond the literal string "live", which is the
 * participant label the task writes.
 */
const KEY = "grin.session.v1";

export function save(session) {
  try {
    localStorage.setItem(KEY, JSON.stringify({ ...session, savedAt: Date.now() }));
    return true;
  } catch (e) {
    // private browsing, or the quota is full. The task still works; only the
    // hand-off and resume are unavailable, and callers fall back to a download.
    return false;
  }
}

export function load() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const s = JSON.parse(raw);
    return s && s.counts ? s : null;
  } catch (e) {
    return null;
  }
}

export function clear() {
  try { localStorage.removeItem(KEY); } catch (e) { /* nothing to do */ }
}

/** "3 minutes ago", for telling someone what they are about to reload. */
export function describeAge(ts) {
  if (!ts) return "";
  const s = Math.max(0, (Date.now() - ts) / 1000);
  if (s < 90) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} minute${m === 1 ? "" : "s"} ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} hour${h === 1 ? "" : "s"} ago`;
  const d = Math.round(h / 24);
  return `${d} day${d === 1 ? "" : "s"} ago`;
}
