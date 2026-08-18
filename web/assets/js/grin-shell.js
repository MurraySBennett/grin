/**
 * grin-shell.js — the chrome. Nav injection, theme, and the Konami code.
 *
 * Every page has one `<div id="nav-placeholder"></div>` in its header instead
 * of a copy-pasted <nav>; this fetches web/components/nav.html once and
 * injects it there, matching the same pattern the main site's components.js
 * uses for nav.html/footer.html. One nav to edit, not seven.
 *
 * Theme is driven by ONE class on <html> (`is-dark`, on documentElement, not
 * body). That's deliberate, not incidental: <html> exists the instant the
 * parser starts, before <body> exists at all, which is what lets a tiny
 * SYNCHRONOUS inline <script> in every page's <head> (right after
 * <meta charset>) apply the class before the browser paints anything —
 * eliminating the flash a JS-module-only approach fundamentally cannot,
 * since deferred/module scripts only run after the whole document has
 * already been parsed (and often already painted). This file's own
 * `applyTheme(resolve())` call below re-applies the same logic afterward —
 * redundant with the inline script by design, both as a safety net and
 * because it's what actually redraws the canvases via themeChanged().
 * The CSS and the <canvas> drawing code both read from that single class, so
 * they can never disagree with each other.
 *
 * ES module, side-effecting on import.
 */

import { themeChanged } from "./grt-plot.js";

const KEY = "msb-dark-mode"; // shared with the main site; "dark" | "light" | absent -> follow the OS

async function injectNav() {
  const placeholder = document.getElementById("nav-placeholder");
  if (!placeholder) return; // a page that hasn't been migrated to the shared nav yet
  try {
    const response = await fetch("./components/nav.html");
    if (!response.ok)
      throw new Error(`Failed to load nav.html: ${response.status}`);
    placeholder.outerHTML = await response.text();
  } catch (err) {
    console.error(err);
  }
}

function osPrefersDark() {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

function saved() {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

function resolve() {
  const s = saved();
  if (s === "dark") return true;
  if (s === "light") return false;
  return osPrefersDark();
}

export function applyTheme(dark) {
  document.documentElement.classList.toggle("is-dark", dark);
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.textContent = dark ? "☀" : "☾";
    btn.setAttribute(
      "aria-label",
      dark ? "Switch to light mode" : "Switch to dark mode",
    );
  }
  themeChanged(); // redraw every canvas with the new palette
}

function wireThemeToggle() {
  document.getElementById("theme-toggle")?.addEventListener("click", () => {
    const next = !document.documentElement.classList.contains("is-dark");
    try {
      localStorage.setItem(KEY, next ? "dark" : "light");
    } catch {
      /* private mode */
    }
    applyTheme(next);
  });

  // follow the OS if the user hasn't overridden
  window
    .matchMedia?.("(prefers-color-scheme: dark)")
    .addEventListener?.("change", (e) => {
      if (!saved()) applyTheme(e.matches);
    });
}

/** Matches the main site's convention exactly: a data-nav on each link, a
 * data-page on <body>, the two compared directly — no URL/filename matching,
 * so a page can move or an anchor can carry query params without breaking
 * the highlight. */
function initNav() {
  const pageGroups = {
    "space-builder": "explore",
    "time-attack": "explore",
    independence: "explore",
    validate: "evidence",
    dynamics: "evidence",
  };
  const rawPage = document.body.dataset.page;
  const page = pageGroups[rawPage] ?? rawPage;
  if (!page) return;
  document.querySelectorAll("nav.site [data-nav]").forEach((a) => {
    if (a.dataset.nav === page) a.classList.add("active");
  });

  const toggle = document.getElementById("nav-toggle");
  const nav = document.getElementById("site-nav");
  toggle?.addEventListener("click", () => {
    const open = nav?.classList.toggle("is-open") ?? false;
    toggle.setAttribute("aria-expanded", String(open));
    toggle.textContent = open ? "Close" : "Pages";
  });
  nav?.querySelectorAll("a").forEach((a) =>
    a.addEventListener("click", () => {
      nav.classList.remove("is-open");
      toggle?.setAttribute("aria-expanded", "false");
      if (toggle) toggle.textContent = "Pages";
    }),
  );
}

function initReleaseBanner() {
  const main = document.querySelector("main.container");
  if (!main || document.querySelector(".release-banner")) return;
  const banner = document.createElement("div");
  banner.className = "note warn release-banner";
  banner.innerHTML = `<p><strong>Pre-release browser demo.</strong> The final checkpoint and its provenance manifest are not installed here yet. Use these pages to explore GRIN, not as the release estimator.</p>`;
  main.prepend(banner);
}

/** It's his site. It stays. */
function initKonami() {
  const SEQ = [
    "ArrowUp",
    "ArrowUp",
    "ArrowDown",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "ArrowLeft",
    "ArrowRight",
    "b",
    "a",
  ];
  let i = 0;
  window.addEventListener("keydown", (e) => {
    i = e.key === SEQ[i] || e.key?.toLowerCase() === SEQ[i] ? i + 1 : 0;
    if (i === SEQ.length) {
      document.body.classList.toggle("konami-mode");
      themeChanged();
      i = 0;
    }
  });
}

// Re-apply immediately, synchronously, at module load -- this does NOT wait
// on the async nav fetch, but on its own it still couldn't eliminate the
// flash: this file only runs after the browser has finished parsing (and
// often already painted) the document, same as any deferred/module script.
// The inline <script> in each page's <head> (right after <meta charset>) is
// what actually prevents the flash, by running before <body> is parsed at
// all. This call exists as a fallback (covers any state drift) and because
// it's what triggers themeChanged() to redraw canvases once they exist.
applyTheme(resolve());

async function init() {
  await injectNav();
  // The toggle does not exist until the shared fragment has been injected.
  applyTheme(resolve());
  wireThemeToggle();
  initNav();
  initReleaseBanner();
  initKonami();
}

init();
