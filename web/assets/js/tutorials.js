/**
 * tutorials.js — language tabs and copy-to-clipboard for the tutorial pages.
 *
 * Progressive enhancement only: with JS disabled every code sample is still in
 * the document, the first one visible and the rest hidden, so the page degrades
 * to "here is the Python version" rather than to nothing.
 *
 * Markup contract:
 *   <div class="codeblock" data-codeblock>
 *     <pre data-lang="Python"><code>…</code></pre>
 *     <pre data-lang="R" hidden><code>…</code></pre>
 *   </div>
 * The tab strip is built from the data-lang attributes, so adding a language
 * means adding a <pre> and nothing else.
 */
const LANG_KEY = "grin-tutorial-lang";

function buildTabs(block) {
  const panes = [...block.querySelectorAll("pre[data-lang]")];
  if (panes.length === 0) return;

  const strip = document.createElement("div");
  strip.className = "tabs";
  strip.setAttribute("role", "tablist");

  const buttons = panes.map((pane, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = pane.dataset.lang;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(i === 0));
    b.addEventListener("click", () => {
      select(pane.dataset.lang);
      // Remember the choice: a reader working in R should not have to re-pick
      // the language on every step of every tutorial.
      try { localStorage.setItem(LANG_KEY, pane.dataset.lang); } catch (e) {}
    });
    strip.appendChild(b);
    return b;
  });

  const spacer = document.createElement("span");
  spacer.className = "spacer";
  strip.appendChild(spacer);

  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "copy";
  copy.textContent = "Copy";
  copy.addEventListener("click", async () => {
    const shown = panes.find((p) => !p.hidden);
    if (!shown) return;
    try {
      await navigator.clipboard.writeText(shown.textContent.trim());
      copy.textContent = "Copied";
      setTimeout(() => (copy.textContent = "Copy"), 1400);
    } catch (e) {
      copy.textContent = "Press Ctrl+C";
      setTimeout(() => (copy.textContent = "Copy"), 1800);
    }
  });
  strip.appendChild(copy);

  block.insertBefore(strip, block.firstChild);

  function select(lang) {
    panes.forEach((p, i) => {
      const on = p.dataset.lang === lang;
      p.hidden = !on;
      buttons[i].setAttribute("aria-selected", String(on));
    });
  }
  block._selectLang = select;
  block._langs = panes.map((p) => p.dataset.lang);
}

function applyRemembered() {
  let want = null;
  try { want = localStorage.getItem(LANG_KEY); } catch (e) {}
  if (!want) return;
  document.querySelectorAll("[data-codeblock]").forEach((b) => {
    if (b._selectLang && b._langs.includes(want)) b._selectLang(want);
  });
}

document.querySelectorAll("[data-codeblock]").forEach(buildTabs);
applyRemembered();
