/**
 * grin-shell.js — the chrome. Nav/footer injection, theme, and the Konami code.
 *
 * Every page has `<div id="nav-placeholder">` and `<div id="footer-placeholder">`
 * instead of copy-pasted chrome; this fetches web/components/nav.html and
 * footer.html once and injects them, matching the main site's components.js
 * pattern. One place to edit, not eight.
 *
 * Theme is driven by ONE class on <html> (`is-dark`, on documentElement, not
 * body). That's deliberate: <html> exists the instant the parser starts, before
 * <body> exists at all, which is what lets a tiny SYNCHRONOUS inline <script>
 * in every page's <head> (right after <meta charset>) apply the class before
 * the browser paints anything — eliminating the flash a JS-module-only approach
 * cannot. This file's `applyTheme(resolve())` re-applies the same logic
 * afterward — redundant with the inline script by design, both as a safety net
 * and because it's what actually redraws the canvases via themeChanged().
 *
 * ES module, side-effecting on import.
 */

import { themeChanged } from "./grt-plot.js";

const KEY = "msb-dark-mode"; // shared with the main site; "dark" | "light" | absent -> follow the OS

async function injectFragment(placeholderId, url) {
  const placeholder = document.getElementById(placeholderId);
  if (!placeholder) return null;
  try {
    const response = await fetch(url);
    if (!response.ok)
      throw new Error(`Failed to load ${url}: ${response.status}`);
    const html = await response.text();
    placeholder.outerHTML = html;
    return true;
  } catch (err) {
    console.error(err);
    return false;
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

/** Matches the main site's convention: data-nav on each top link, data-page on
 * <body>, compared directly — no URL/filename matching. Explore children use
 * data-nav-sub and also light up the parent Explore item. */
function initNav() {
  const pageGroups = {
    explore: "explore",
    "space-builder": "explore",
    independence: "explore",
    live: "explore",
    validate: "evidence",
    // the tutorials are their own tab; each sub-page lights the parent
    tutorials: "tutorials",
    "tutorial-existing-data": "tutorials",
    "tutorial-adaptive-stopping": "tutorials",
    "tutorial-stimulus-levels": "tutorials",
  };
  const rawPage = document.body.dataset.page;
  const page = pageGroups[rawPage] ?? rawPage;
  if (!page) return;

  document.querySelectorAll("nav.site [data-nav]").forEach((a) => {
    if (a.dataset.nav === page) a.classList.add("active");
  });
  document.querySelectorAll("nav.site [data-nav-sub]").forEach((a) => {
    if (a.dataset.navSub === rawPage) {
      a.classList.add("active");
      a.closest(".has-sub")?.classList.add("is-current");
    }
  });

  const toggle = document.getElementById("nav-toggle");
  const nav = document.getElementById("site-nav");
  const closeMobile = () => {
    nav?.classList.remove("is-open");
    toggle?.setAttribute("aria-expanded", "false");
    if (toggle) toggle.textContent = "Pages";
  };

  toggle?.addEventListener("click", () => {
    const open = nav?.classList.toggle("is-open") ?? false;
    toggle.setAttribute("aria-expanded", String(open));
    toggle.textContent = open ? "Close" : "Pages";
  });

  // Explore submenu: click toggles on coarse pointers; hover works via CSS.
  document.querySelectorAll(".nav-item.has-sub").forEach((item) => {
    const trigger = item.querySelector(":scope > a");
    trigger?.addEventListener("click", (e) => {
      // On narrow layouts, first tap opens the submenu; second follows the link
      // only if already open. Desktop keeps the link as the Explore landing.
      if (window.matchMedia("(max-width: 40rem)").matches) {
        if (!item.classList.contains("is-open")) {
          e.preventDefault();
          item.classList.add("is-open");
          return;
        }
      }
    });
  });

  nav?.querySelectorAll("a").forEach((a) =>
    a.addEventListener("click", () => {
      // Don't collapse mobile nav when only opening a submenu.
      if (a.parentElement?.classList.contains("has-sub") &&
          a.parentElement.classList.contains("is-open") &&
          a.getAttribute("href") === "./explore.html" &&
          window.matchMedia("(max-width: 40rem)").matches) {
        return;
      }
      closeMobile();
    }),
  );
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

// Re-apply immediately at module load — does NOT wait on the async fragment
// fetch. The inline <script> in each page's <head> is what prevents the flash.
applyTheme(resolve());

async function init() {
  await Promise.all([
    injectFragment("nav-placeholder", "./components/nav.html"),
    injectFragment("footer-placeholder", "./components/footer.html"),
  ]);
  // The toggle does not exist until the shared fragment has been injected.
  applyTheme(resolve());
  wireThemeToggle();
  initNav();
  initKonami();
}

init();
