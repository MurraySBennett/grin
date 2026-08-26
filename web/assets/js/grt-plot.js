/**
 * grt-plot.js — all drawing. Canvas for the perceptual space and RT
 * distributions; DOM for tables and bars.
 *
 * COLOUR CONVENTION (fixed across every page and every exported figure):
 *   SOLID, slate, one shade per stimulus  = the representation you specified,
 *                                           or the raw per-stimulus estimate
 *   DASHED, rose                          = what was RECOVERED from data, or
 *                                           what the fitted model PREDICTS
 * So "how far is the rose from the slate" is always the same question:
 * how much did the inference miss by?
 *
 * Colours are read from CSS custom properties at draw time, so dark mode and
 * Konami mode work on the canvas for free — there is exactly one source of
 * truth for the palette, and it is grin.css.
 *
 * ES module. Depends on grt-core.js.
 */

import { npdf, unpack } from "./grt-core.js";

// --------------------------------------------------------------------------- //
// Theme
// --------------------------------------------------------------------------- //
/**
 * @param {Object} overrides — partial palette to merge on top of the computed
 * site theme, e.g. { stim: ["#000","#000","#000","#000"], predicted: "#000" }.
 * Never touches document.body — this only affects the ONE render call it's
 * passed to, so a custom export palette can never leak into the site's own
 * chrome or into any other canvas.
 */
export function theme(overrides = {}) {
  const cs = getComputedStyle(document.body);
  const v = (n, fallback) => cs.getPropertyValue(n).trim() || fallback;
  return {
    stim: [
      v("--stim-0", "#2a1f5e"),
      v("--stim-1", "#5a4fcc"),
      v("--stim-2", "#7b68ee"),
      v("--stim-3", "#b8aef5"),
    ],
    predicted: v("--predicted", "#b06080"),
    line: v("--line", "#e0e0ff"),
    mute: v("--mute", "#6f6a8f"),
    muteSoft: v("--mute-soft", "#9a95b8"),
    ink: v("--ink", "#1a1a2e"),
    paper: v("--paper", "#ffffff"),
    slate: v("--slate", "#7b68ee"),
    ...overrides,
  };
}

/** A few ready-made palettes for exported figures — print/slide friendly,
 * independent of the site's own light/dark/konami theme. */
/**
 * Ready-made palettes for exported figures — independent of the site's own
 * light/dark/konami theme, which is left completely untouched. `stim`/
 * `predicted`/`line`/`mute`/`muteSoft`/`ink`/`paper` follow theme()'s shape.
 *
 * Two optional extras, used sparingly and only where they can't hurt legibility:
 *   titleFont — a display font applied ONLY to the figure title and the legend
 *               (short strings, safe at small sizes). Axis names and the tiny
 *               A1/A2/B1/B2 corner ticks always stay in the base UI font, on
 *               purpose — a theme should never make the actual data harder to
 *               read.
 *   texture   — "vignette" | "scanlines" | null, a cheap decorative wash drawn
 *               once, after the background fill and before any data.
 */
export const FIGURE_PALETTES = {
  site: null, // no override — use the live site theme, incl. dark/konami

  // --- clean / professional -------------------------------------------- //
  blackOnWhite: {
    stim: ["#000000", "#000000", "#000000", "#000000"],
    predicted: "#000000",
    line: "#bbbbbb",
    mute: "#555555",
    muteSoft: "#999999",
    ink: "#000000",
    paper: "#ffffff",
  },
  whiteOnBlack: {
    stim: ["#ffffff", "#ffffff", "#ffffff", "#ffffff"],
    predicted: "#ffffff",
    line: "#444444",
    mute: "#aaaaaa",
    muteSoft: "#777777",
    ink: "#ffffff",
    paper: "#000000",
  },
  grayscalePrint: {
    // four genuinely distinguishable grays, cheap to photocopy or print in B&W
    stim: ["#1a1a1a", "#595959", "#949494", "#c9c9c9"],
    predicted: "#8a2b2b",
    line: "#dddddd",
    mute: "#666666",
    muteSoft: "#aaaaaa",
    ink: "#1a1a1a",
    paper: "#ffffff",
  },
  colorblindSafe: {
    // Okabe & Ito (2008) qualitative palette — safe under deuteranopia,
    // protanopia, and tritanopia, not just "looks fine to me"
    stim: ["#E69F00", "#56B4E9", "#009E73", "#D55E00"], // orange / sky blue / bluish green / vermillion
    predicted: "#CC79A7", // reddish purple — distinct from all four above
    line: "#dddddd",
    mute: "#555555",
    muteSoft: "#999999",
    ink: "#1a1a1a",
    paper: "#ffffff",
  },
  apaStyle: {
    // Clean and muted in the spirit of APA figure conventions (uncluttered,
    // high-contrast, colour used sparingly) — NOT a certification against the
    // full Publication Manual, which covers far more than a colour palette.
    stim: ["#1c2b4a", "#3a5a8c", "#6a8fc2", "#a8c0e0"],
    predicted: "#8c3a3a",
    line: "#e5e5e5",
    mute: "#595959",
    muteSoft: "#8a8a8a",
    ink: "#1a1a1a",
    paper: "#ffffff",
  },

  // --- just for fun ------------------------------------------------------ //
  trueGrit: {
    // "True GRiT" — old-west wanted-poster sepia
    stim: ["#3b2412", "#6b4423", "#8a5a2e", "#a97638"],
    predicted: "#7a1f1f",
    line: "#c9a876",
    mute: "#5a4225",
    muteSoft: "#8a6d45",
    ink: "#2b1a0e",
    paper: "#e8d5ab",
    titleFont: "Rye",
    texture: "vignette",
  },
  matrixConsole: {
    // "The (Confusion) Matrix" — neon terminal green on black
    stim: ["#00ff41", "#00cc33", "#39ff14", "#00b82e"],
    predicted: "#ffffff",
    line: "#0a3d0a",
    mute: "#33cc55",
    muteSoft: "#1f9938",
    ink: "#00ff41",
    paper: "#000000",
    titleFont: "VT323",
    texture: "scanlines",
  },
  synthwave: {
    // neon pink/cyan on dusk purple — reuses the site's own Press Start 2P,
    // so it stays visually a Murray-site figure even at its most extra
    stim: ["#ff2d95", "#00e5ff", "#ff6ec7", "#7b68ee"],
    predicted: "#ffe600",
    line: "#3d2b5e",
    mute: "#b088e8",
    muteSoft: "#7a5ba8",
    ink: "#f0e6ff",
    paper: "#1a0f2e",
    titleFont: "Press Start 2P",
  },
  chalkboard: {
    stim: ["#f5f0e6", "#f0d878", "#7ec4e8", "#e88a8a"],
    predicted: "#ffffff",
    line: "#3a5a3a",
    mute: "#c9d9c0",
    muteSoft: "#8fa885",
    ink: "#f5f0e6",
    paper: "#1f3d1f",
    titleFont: "Kalam",
  },
  blueprint: {
    stim: ["#ffffff", "#ffffff", "#ffffff", "#ffffff"],
    predicted: "#ffd166",
    line: "#3d6ba8",
    mute: "#a8c5e8",
    muteSoft: "#6d95c9",
    ink: "#ffffff",
    paper: "#1b3a6b",
    titleFont: "SF Mono, DejaVu Sans Mono, ui-monospace, monospace",
  },
  independenceDay: {
    // "Perceptual Independence Day" — official US flag colours: Old Glory Red
    // (#B31942), White, and Old Glory Blue (#002868). Reuses Bungee, since
    // it's one fewer font to fetch. Export a PNG with this one selected for
    // a small surprise.
    stim: ["#B31942", "#B31942", "#B31942", "#B31942"],
    predicted: "#FFFFFF",
    line: "#FFFFFF",
    mute: "#FFFFFF",
    muteSoft: "#cfd8e8",
    ink: "#FFFFFF",
    paper: "#002868",
    titleFont: "Bungee",
    texture: "fireworks",
  },
  senseAndSeparability: {
    // "Sense and separability" — a Regency-era reading, dusty pastels on parchment
    stim: ["#a85c6b", "#7a8f6a", "#8fa8bf", "#c9a876"],
    predicted: "#6b2c3a",
    line: "#d8cbb8",
    mute: "#7a6a58",
    muteSoft: "#b0a48f",
    ink: "#3a2e26",
    paper: "#f5ecd9",
    titleFont: "Playfair Display",
    texture: "vignette",
  },
  perceptualSpace: {
    // the pun was sitting right there — a perceptual space, IN SPACE
    stim: ["#00e5ff", "#ff2d95", "#7b68ee", "#39ff88"],
    predicted: "#ffffff",
    line: "#2a2a5e",
    mute: "#9a9ac9",
    muteSoft: "#5a5a9e",
    ink: "#e8e8ff",
    paper: "#05050f",
    titleFont: "Orbitron",
    texture: "starfield",
  },

  // --- unlocked by typing "Go Spurs Go" (case-insensitive) as the figure
  // title on analyse.html — see the unlock logic there. Not in the normal
  // dropdown list. ---
  spursModern: {
    // official colours: Silver #C4CED4, Black #000000
    stim: ["#000000", "#4a5560", "#8a949c", "#C4CED4"],
    predicted: "#5a6570",
    line: "#C4CED4",
    mute: "#5a5a5a",
    muteSoft: "#8a949c",
    ink: "#000000",
    paper: "#ffffff",
  },
  spursRetroFiesta: {
    // the 90s "Fiesta" alternate: Fuchsia #EF426F, Teal #00B2A9, Orange
    // #FF8200, Silver #8A8D8F, on black
    stim: ["#EF426F", "#00B2A9", "#FF8200", "#8A8D8F"],
    predicted: "#FFFFFF",
    line: "#3a3a3a",
    mute: "#8A8D8F",
    muteSoft: "#5a5a5a",
    ink: "#FFFFFF",
    paper: "#000000",
    titleFont: "Bungee",
  },
};

/** Redraw every registered canvas when the theme changes. */
const redrawers = new Set();
export function onThemeChange(fn) {
  redrawers.add(fn);
  return fn;
}
export function themeChanged() {
  redrawers.forEach((f) => f());
}

// --------------------------------------------------------------------------- //
// Labels
// --------------------------------------------------------------------------- //
export const DEFAULT_LABELS = {
  aName: "Dimension A",
  a1: "A1",
  a2: "A2",
  bName: "Dimension B",
  b1: "B1",
  b2: "B2",
};

/** Stimulus i -> "A1/B1". Canonical: ai = i<2?0:1, bi = i%2. */
export function stimLabel(labels, i) {
  const L = { ...DEFAULT_LABELS, ...labels };
  const a = i < 2 ? L.a1 : L.a2;
  const b = i % 2 === 0 ? L.b1 : L.b2;
  return `${a}/${b}`;
}

export function parseLevels(str, fallback) {
  const parts = String(str || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  return parts.length === 2 ? parts : fallback;
}

// --------------------------------------------------------------------------- //
// Hi-DPI canvas setup
// --------------------------------------------------------------------------- //
/**
 * @param {string|null} background — a CSS colour to fill before drawing, or
 * "transparent"/null/undefined to leave the canvas pixels transparent (the
 * default). The on-screen `background: #fff` in grin.css is CSS chrome only —
 * it never touches the actual pixel data, so an exported PNG is transparent
 * unless a background is explicitly filled here.
 */
function setup(canvas, w, h, background) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  if (background && background !== "transparent") {
    g.fillStyle = background;
    g.fillRect(0, 0, w, h);
  }
  return g;
}

// --------------------------------------------------------------------------- //
// The classic GRT perceptual-space plot
// --------------------------------------------------------------------------- //
const PLOT = 460;
const BAND = 105; // width of the marginal strips
const SCALE = 56; // pixels per unit of z

function drawArrow(g, x1, y1, x2, y2, color) {
  g.strokeStyle = color;
  g.fillStyle = color;
  g.lineWidth = 1.6;
  g.setLineDash([]);
  g.beginPath();
  g.moveTo(x1, y1);
  g.lineTo(x2, y2);
  g.stroke();
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const sz = 6;
  g.beginPath();
  g.moveTo(x2, y2);
  g.lineTo(
    x2 - sz * Math.cos(ang - Math.PI / 6),
    y2 - sz * Math.sin(ang - Math.PI / 6),
  );
  g.lineTo(
    x2 - sz * Math.cos(ang + Math.PI / 6),
    y2 - sz * Math.sin(ang + Math.PI / 6),
  );
  g.closePath();
  g.fill();
}

/** Deterministic pseudo-random in [0,1) from two integers — used for the
 * decorative textures below instead of Math.random() so a theme's stars or
 * sparks sit in the SAME place on every redraw. A randomly-reshuffled sky on
 * every toggle of an unrelated checkbox would read as broken, not charming. */
function hash2(x, y) {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}

/** A cheap decorative wash, drawn once over the whole canvas, after the
 * background fill and before any data — never over the ellipses themselves.
 * Always leaves globalAlpha at 1 on exit, since most of the drawing later in
 * renderSpace does not reset it itself and would otherwise silently inherit
 * whatever a texture left behind. */
function applyTexture(g, w, h, texture) {
  if (texture === "vignette") {
    const grad = g.createRadialGradient(
      w / 2,
      h / 2,
      Math.min(w, h) * 0.25,
      w / 2,
      h / 2,
      Math.max(w, h) * 0.72,
    );
    grad.addColorStop(0, "rgba(0,0,0,0)");
    grad.addColorStop(1, "rgba(20,10,0,0.28)");
    g.fillStyle = grad;
    g.fillRect(0, 0, w, h);
  } else if (texture === "scanlines") {
    g.fillStyle = "rgba(255,255,255,0.035)";
    for (let y = 0; y < h; y += 3) g.fillRect(0, y, w, 1);
  } else if (texture === "starfield") {
    const cell = 14;
    for (let gx = 0; gx < w; gx += cell) {
      for (let gy = 0; gy < h; gy += cell) {
        const r1 = hash2(gx, gy);
        if (r1 <= 0.84) continue; // most cells stay empty — a sky, not a wall
        const px = gx + hash2(gx + 1, gy) * cell;
        const py = gy + hash2(gx, gy + 1) * cell;
        const size = 0.5 + r1 * 1.6;
        g.globalAlpha = 0.35 + hash2(gy, gx) * 0.55;
        g.fillStyle = "#ffffff";
        g.beginPath();
        g.arc(px, py, size, 0, 2 * Math.PI);
        g.fill();
      }
    }
  } else if (texture === "sunset") {
    // A full beach-sunset sky. Deliberately opaque — it fully repaints
    // whatever background fill ran before it, since a translucent sunset
    // wash over a transparent PNG doesn't read as a sunset at all.
    const grad = g.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "#2a1a4a");
    grad.addColorStop(0.35, "#6a3a6a");
    grad.addColorStop(0.65, "#e8703a");
    grad.addColorStop(1, "#f4c95a");
    g.fillStyle = grad;
    g.fillRect(0, 0, w, h);
  } else if (texture === "fireworks") {
    const bursts = [
      { x: w * 0.14, y: h * 0.12, color: "#ff5a5a" },
      { x: w * 0.86, y: h * 0.16, color: "#5a8dff" },
      { x: w * 0.5, y: h * 0.07, color: "#ffe14d" },
    ];
    for (const b of bursts) {
      for (let i = 0; i < 16; i++) {
        const ang = (i / 16) * Math.PI * 2 + hash2(b.x, i) * 0.4;
        const len = 9 + hash2(i, b.y) * 15;
        g.globalAlpha = 0.6;
        g.strokeStyle = b.color;
        g.lineWidth = 1.3;
        g.beginPath();
        g.moveTo(b.x, b.y);
        g.lineTo(b.x + Math.cos(ang) * len, b.y + Math.sin(ang) * len);
        g.stroke();
      }
    }
  }
  g.globalAlpha = 1;
}

/** T.titleFont may be a single font name ("Rye") needing quotes + a fallback,
 * or an already-complete stack ("SF Mono, ui-monospace, monospace") that must
 * NOT be wrapped in quotes as a whole — that would make it one invalid name. */
function resolveFont(titleFont, weightPx, fallback) {
  if (!titleFont) return fallback;
  const family = titleFont.includes(",")
    ? titleFont
    : `"${titleFont}", Inter, system-ui, sans-serif`;
  return `${weightPx} ${family}`;
}

/**
 * One equal-density contour (1 SD) of a bivariate normal with unit variances.
 *
 * @param {Object} extra
 *   markerImage an HTMLImageElement drawn (aspect-preserved) at the mean
 *               instead of the small dot. Solid-ellipses-only — the dashed
 *               comparison overlay never gets a marker, same as the plain dot.
 */
function drawEllipse(g, cx, cy, sc, zx, zy, rho, color, dashed, extra = {}) {
  const r = Math.min(0.98, Math.max(-0.98, rho));
  const x = cx + zx * sc;
  const y = cy - zy * sc;
  // eigen-decomposition of [[1,r],[r,1]] -> axes sqrt(1±r) at 45 degrees
  const ax = Math.sqrt(1 + r) * sc;
  const bx = Math.sqrt(1 - r) * sc;
  const c45 = Math.SQRT1_2;

  g.strokeStyle = color;
  g.lineWidth = extra.lineWidth ?? (dashed ? 1.5 : 2.1);
  g.setLineDash(extra.dash ?? (dashed ? [5, 4] : []));
  g.beginPath();
  for (let t = 0; t <= 2 * Math.PI + 0.02; t += 0.08) {
    const ex = ax * Math.cos(t);
    const ey = bx * Math.sin(t);
    const rx = (ex - ey) * c45;
    const ry = (ex + ey) * c45;
    if (t === 0) g.moveTo(x + rx, y - ry);
    else g.lineTo(x + rx, y - ry);
  }
  g.closePath();
  g.stroke();
  g.setLineDash([]);

  if (!dashed) {
    if (extra.markerImage) {
      const img = extra.markerImage;
      const h = 18,
        w = h * (img.naturalWidth / img.naturalHeight || 1);
      g.drawImage(img, x - w / 2, y - h / 2, w, h);
    } else {
      g.fillStyle = color;
      g.beginPath();
      g.arc(x, y, 3.2, 0, 2 * Math.PI);
      g.fill();
    }
  }
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {Object} opts
 *   stimuli   [{zx,zy,rho}] x4   — SOLID, T.stim[i]
 *   predicted [{zx,zy,rho}] x4   — DASHED, T.predicted (optional)
 *   labels    {aName,a1,a2,bName,b1,b2}
 *   showMarginals   boolean
 *   theme           partial palette override (see theme()) — e.g. for export
 *                    to print/slide colours without touching the site theme
 *   background      CSS colour to fill, or "transparent" (default)
 *   title           string, drawn centred above the plot; reserves its own
 *                    space rather than overlapping anything
 *   showAxisNames   default true — the "Dimension A" / "Dimension B" text
 *   showLevelTicks  default true — the A1/A2/B1/B2 corner labels
 *   legend          {stimuli:[4 strings], predictedLabel?:string} — if given,
 *                    draws a legend directly on the canvas so the exported PNG
 *                    is self-contained (the DOM legend from buildLegend() is
 *                    NOT part of the exported image on its own)
 *   stimMarkerImage an HTMLImageElement drawn at each solid ellipse's mean
 *                    instead of the small dot — an easter egg (the spur)
 *
 * MARGINALS. Each dimension's marginal is drawn as two colours (one per level
 * of THAT dimension), with SOLID vs DOTTED marking the level of the OTHER
 * dimension. This is the grtools convention and it makes non-separability
 * visible at a glance: if the solid and dotted curves of the same colour do not
 * lie on top of each other, that dimension is NOT separable from the other one.
 */
/** The coordinate system renderSpace draws into — pulled out so any other
 * function that needs to draw ellipses in register with it (the fading-trail
 * layer function, below) computes it from the exact same formula and can
 * never quietly drift out of alignment. */
function spaceGeometry(opts) {
  const show = !!opts.showMarginals;
  const size = PLOT + (show ? BAND : 0);
  const titleH = opts.title ? 32 : 0;
  const legendRows = opts.legend
    ? opts.legend.stimuli.length + (opts.legend.predictedLabel ? 1 : 0)
    : 0;
  const legendH = legendRows ? 10 + legendRows * 16 : 0;
  return {
    size,
    titleH,
    legendH,
    sc: SCALE,
    cx: PLOT / 2,
    cy: (show ? BAND : 0) + PLOT / 2,
  };
}

export function renderSpace(canvas, opts) {
  const T = theme(opts.theme);
  const L = { ...DEFAULT_LABELS, ...(opts.labels || {}) };
  const show = !!opts.showMarginals;
  const showAxisNames = opts.showAxisNames !== false;
  const showLevelTicks = opts.showLevelTicks !== false;
  const { size, titleH, legendH, sc, cx, cy } = spaceGeometry(opts);

  const g = setup(canvas, size, size + titleH + legendH, opts.background);
  applyTexture(g, size, size + titleH + legendH, T.texture);

  if (opts.title) {
    g.fillStyle = T.ink;
    g.font = resolveFont(
      T.titleFont,
      "600 17px",
      "600 13px Inter, system-ui, sans-serif",
    );
    g.textAlign = "center";
    g.fillText(opts.title, size / 2, 21);
  }

  g.save();
  g.translate(0, titleH);

  const top = cy - PLOT / 2,
    bottom = cy + PLOT / 2;
  const left = 0,
    right = PLOT;

  // the decision bounds (fixed at 0 under DS) — these ARE the model, not decoration.
  // The vertical line gets a wider inset than the horizontal one specifically
  // so it doesn't run into the B1/B2 labels near the top and bottom edges —
  // A1/A2 sit well clear of the horizontal line already (offset above it).
  g.strokeStyle = T.line;
  g.setLineDash([6, 5]);
  g.lineWidth = 1.4;
  g.beginPath();
  g.moveTo(cx, top + 20);
  g.lineTo(cx, bottom - 20);
  g.moveTo(left + 6, cy);
  g.lineTo(right - 6, cy);
  g.stroke();
  g.setLineDash([]);

  // orientation key — the arrows and their names are one unit; either both
  // show or neither does, so a toggle can't leave an unlabelled arrow behind.
  if (showAxisNames) {
    const ox = left + 24,
      oy = bottom - 24;
    drawArrow(g, ox, oy, ox + 40, oy, T.muteSoft);
    drawArrow(g, ox, oy, ox, oy - 40, T.muteSoft);
    g.fillStyle = T.mute;
    g.font = "11px Inter, system-ui, sans-serif";
    g.textAlign = "left";
    g.fillText(L.aName, ox + 44, oy + 4);
    g.save();
    g.translate(ox - 6, oy - 44);
    g.rotate(-Math.PI / 2);
    g.fillText(L.bName, 0, 0);
    g.restore();
  }

  // level ticks (A1/A2/B1/B2). Deliberately, ALWAYS the palette's display
  // font when one exists — not left to whatever g.font happens to still be
  // set to from an earlier draw call, which is what made this silently
  // depend on showAxisNames before: that block was the only thing resetting
  // g.font back to the base font, so turning axis names off left the title's
  // font bleeding into these ticks by accident rather than by design.
  if (showLevelTicks) {
    g.fillStyle = T.muteSoft;
    g.font = resolveFont(
      T.titleFont,
      "13px",
      "11px Inter, system-ui, sans-serif",
    );
    g.textAlign = "left";
    g.fillText(L.a1, left + 6, cy - 7);
    g.textAlign = "right";
    g.fillText(L.a2, right - 6, cy - 7);
    g.textAlign = "center";
    g.fillText(L.b2, cx, top + 13);
    g.fillText(L.b1, cx, bottom - 5);
  }

  const stimExtra = { markerImage: opts.stimMarkerImage || null };
  opts.stimuli.forEach((s, i) =>
    drawEllipse(g, cx, cy, sc, s.zx, s.zy, s.rho, T.stim[i], false, stimExtra),
  );
  if (opts.predicted)
    opts.predicted.forEach((s) =>
      drawEllipse(g, cx, cy, sc, s.zx, s.zy, s.rho, T.predicted, true),
    );

  if (show) {
    const MC = [T.stim[0], T.stim[2]];
    const H = 175;

    const curve = (project) => (meanZ, color, dotted) => {
      g.strokeStyle = color;
      g.lineWidth = dotted ? 1.4 : 2;
      g.setLineDash(dotted ? [2, 3] : []);
      g.beginPath();
      let first = true;
      for (let z = -3.6; z <= 3.6; z += 0.06) {
        const [px, py] = project(meanZ + z, npdf(z) * H);
        if (first) {
          g.moveTo(px, py);
          first = false;
        } else g.lineTo(px, py);
      }
      g.stroke();
      g.setLineDash([]);
    };

    // top strip: dimension A (x). Colour = A-level; dotted = paired with B2.
    const baseY = BAND - 8;
    const curveA = curve((z, d) => [cx + z * sc, baseY - d]);
    curveA(opts.stimuli[0].zx, MC[0], false); // A1 with B1
    curveA(opts.stimuli[1].zx, MC[0], true); // A1 with B2
    curveA(opts.stimuli[2].zx, MC[1], false); // A2 with B1
    curveA(opts.stimuli[3].zx, MC[1], true); // A2 with B2
    g.strokeStyle = T.line;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(0, baseY);
    g.lineTo(PLOT, baseY);
    g.stroke();

    // right strip: dimension B (y). Colour = B-level; dotted = paired with A2.
    const baseX = PLOT + 8;
    const curveB = curve((z, d) => [baseX + d, cy - z * sc]);
    curveB(opts.stimuli[0].zy, MC[0], false); // B1 with A1
    curveB(opts.stimuli[2].zy, MC[0], true); // B1 with A2
    curveB(opts.stimuli[1].zy, MC[1], false); // B2 with A1
    curveB(opts.stimuli[3].zy, MC[1], true); // B2 with A2
    g.strokeStyle = T.line;
    g.beginPath();
    g.moveTo(baseX, top);
    g.lineTo(baseX, bottom);
    g.stroke();
  }

  g.restore(); // undo the title-offset translate — legend below is in absolute coords

  if (opts.legend) {
    let ly = titleH + size + 8;
    g.font = resolveFont(
      T.titleFont,
      "13px",
      "11px Inter, system-ui, sans-serif",
    );
    g.textAlign = "left";
    const row = (color, text) => {
      g.fillStyle = color;
      g.beginPath();
      g.arc(10, ly - 4, 4, 0, 2 * Math.PI);
      g.fill();
      g.fillStyle = T.ink;
      g.fillText(text, 22, ly);
      ly += 16;
    };
    opts.legend.stimuli.forEach((text, i) => row(T.stim[i], text));
    if (opts.legend.predictedLabel)
      row(T.predicted, opts.legend.predictedLabel);
  }
}

/** Convenience: a 12-vector -> the {zx,zy,rho} array renderSpace wants. */
export function toStimuli(params) {
  const { zx, zy, rho } = unpack(params);
  return [0, 1, 2, 3].map((i) => ({ zx: zx[i], zy: zy[i], rho: rho[i] }));
}

// --------------------------------------------------------------------------- //
// Confusion matrix
// --------------------------------------------------------------------------- //
function heatColor(v, T) {
  const t = Math.max(0, Math.min(1, v));
  const [r0, g0, b0] = hexRGB(T.paper);
  const [r1, g1, b1] = hexRGB(T.slate);
  const mix = (a, b) => Math.round(a + (b - a) * t);
  return `rgb(${mix(r0, r1)},${mix(g0, g1)},${mix(b0, b1)})`;
}
function hexRGB(h) {
  const s = h.replace("#", "").trim();
  const n =
    s.length === 3
      ? s
          .split("")
          .map((c) => c + c)
          .join("")
      : s;
  return [0, 2, 4].map((i) => parseInt(n.slice(i, i + 2), 16) || 0);
}

// --------------------------------------------------------------------------- //
// Fading-trail figure: many fits over the course of an experiment, layered
// onto ONE canvas, older fits pushed down the alpha channel as newer ones draw
// over them — a visual history of convergence, not a snapshot. Shared core
// loop under every "Dynamics" page idea (adaptive selection, early stopping,
// power planning, drift tracking): all of them are "refit repeatedly, watch
// what changes," this is just the rendering half of that.
// --------------------------------------------------------------------------- //
/**
 * Draw ONE layer of ellipses onto an EXISTING canvas, without clearing it and
 * without touching the axes/title/legend — those get drawn once, up front, by
 * a separate renderSpace() call. Uses the exact same coordinate formula
 * renderSpace itself uses (spaceGeometry), so a trail can never drift out of
 * register with the chrome it's layered onto.
 *
 * @param {HTMLCanvasElement} canvas — must already be sized (a prior
 *   renderSpace call did this); this function does not resize or clear it.
 * @param {{zx,zy,rho}[]} stimuli
 * @param {number} alpha — 0..1
 * @param {Object} opts — must match the geometry-affecting opts (theme,
 *   showMarginals, title, legend) passed to the renderSpace call that drew
 *   the chrome, or the layer will land in the wrong place.
 */
/**
 * One layer of the trail.
 *
 * `opts.dashed` / `opts.lineWidth` exist so a caller can distinguish the CURRENT fit
 * from the ones behind it. Opacity alone is a weak cue once there are more than a few
 * layers -- several faded ellipses overlapping read as one darker shape, and the eye
 * cannot tell which is the live estimate. Dashing the history and drawing the newest
 * solid and thicker makes it unambiguous without adding a colour to the scheme.
 */
export function drawFadeLayer(canvas, stimuli, alpha, opts = {}) {
  const T = theme(opts.theme);
  const { titleH, sc, cx, cy } = spaceGeometry(opts);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0); // reapplies cleanly; does NOT clear pixels

  g.save();
  g.translate(0, titleH);
  g.globalAlpha = Math.max(0, Math.min(1, alpha));
  stimuli.forEach((s, i) =>
    drawEllipse(g, cx, cy, sc, s.zx, s.zy, s.rho, T.stim[i], !!opts.dashed, {
      lineWidth: opts.lineWidth,
      dash: opts.dash,
    }),
  );
  g.globalAlpha = 1;
  g.restore();
}

/** The fade curve itself — pulled out so the instant composite and the
 * animated step-through below can never compute it differently. */
function fadeAlpha(
  i,
  n,
  { minAlpha = 0.08, maxAlpha = 1.0, curve = "linear" } = {},
) {
  const t = n <= 1 ? 1 : i / (n - 1); // 0 (oldest) .. 1 (newest)
  const shaped = curve === "exp" ? t * t : t;
  return minAlpha + (maxAlpha - minAlpha) * shaped;
}

/**
 * The full fading-trail figure, drawn INSTANTLY as one static composite: draws
 * the static chrome once, then layers every checkpoint on top, oldest to
 * newest, fading in as they go. Good for a still figure (a slide, an export);
 * for an on-screen animation that reveals checkpoint by checkpoint, see
 * animateFadeTrail below, which shares this exact same fade curve.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{stimuli:{zx,zy,rho}[], label?:string}[]} checkpoints — oldest first
 * @param {Object} opts — same shape as renderSpace's opts (theme, labels,
 *   showMarginals, title, background, etc.) MINUS `stimuli`/`predicted`,
 *   which this function supplies itself as it steps through the checkpoints
 * @param {Object} fade
 *   minAlpha default 0.08 — the oldest checkpoint's opacity
 *   maxAlpha default 1.0  — the newest checkpoint's opacity
 *   curve    default "linear" | "exp" — exp emphasises recent fits more
 *     strongly, useful when there are many checkpoints and the early ones
 *     are mostly noise you don't want competing visually with the converged
 *     result
 */
export function renderFadeTrail(canvas, checkpoints, opts = {}, fade = {}) {
  renderSpace(canvas, { ...opts, stimuli: [], predicted: null });
  const last = checkpoints.length - 1;
  checkpoints.forEach((c, i) => {
    const current = i === last;
    drawFadeLayer(canvas, c.stimuli, fadeAlpha(i, checkpoints.length, fade), {
      ...opts,
      // history: dashed and thin. current: solid, thicker, full weight.
      dashed: !current,
      lineWidth: current ? 2.6 : 1.2,
    });
  });
}

/**
 * Same figure, revealed one checkpoint at a time rather than composited
 * instantly — the chrome draws once, then each layer draws with a pause in
 * between, so it reads as a fit converging (or drifting) over the session
 * rather than a single static image appearing all at once.
 *
 * @param {Object} anim
 *   delayMs    default 90  — pause between layers, in ms. Set 0 to draw as
 *              fast as the event loop allows while still yielding each frame
 *              (keeps the tab responsive even with many checkpoints).
 *   onProgress optional (i, n, checkpoint) => void, called after each layer
 *              draws — wire this to a progress bar or a live readout.
 *   signal     optional AbortSignal — check `signal.aborted` between frames
 *              to let the caller cancel a run in progress (e.g. the user
 *              changed a slider and clicked Run again).
 * @returns {Promise<void>}
 */
export async function animateFadeTrail(
  canvas,
  checkpoints,
  opts = {},
  fade = {},
  anim = {},
) {
  const { delayMs = 90, onProgress = null, signal = null } = anim;
  renderSpace(canvas, { ...opts, stimuli: [], predicted: null });
  const n = checkpoints.length;
  for (let i = 0; i < n; i++) {
    if (signal?.aborted) return;
    drawFadeLayer(canvas, checkpoints[i].stimuli, fadeAlpha(i, n, fade), {
      ...opts,
      dashed: i !== n - 1,
      lineWidth: i === n - 1 ? 2.6 : 1.2,
    });
    onProgress?.(i + 1, n, checkpoints[i]);
    // always yield at least one frame, even at delayMs=0 -- this is what
    // keeps a long trail from reading as a browser freeze: the event loop
    // gets control back after every single layer, not just at the end.
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
}

/**
 * @param {HTMLTableElement} table
 * @param {number[][]} P — 4x4 PROPORTIONS (not counts)
 * @param {Object} labels
 * @param {{counts?:number[][]}} opts — if given, cells show "0.84" over "(67)"
 */
export function renderCM(table, P, labels, opts = {}) {
  const T = theme();
  const head = [0, 1, 2, 3]
    .map((j) => `<th>${stimLabel(labels, j)}</th>`)
    .join("");
  let html = `<tr><th></th>${head}</tr>`;
  for (let i = 0; i < 4; i++) {
    html += `<tr><th>${stimLabel(labels, i)}</th>`;
    for (let j = 0; j < 4; j++) {
      const v = P[i][j];
      const bg = heatColor(v, T);
      const fg = v > 0.55 ? "#fff" : T.ink;
      const sub = opts.counts
        ? `<div style="font-size:.62rem;opacity:.75">${opts.counts[i][j]}</div>`
        : "";
      html += `<td style="background:${bg};color:${fg}">${v.toFixed(2)}${sub}</td>`;
    }
    html += "</tr>";
  }
  table.innerHTML = html;
}

export function cmAccuracy(P) {
  return [0, 1, 2, 3].reduce((a, i) => a + P[i][i], 0) / 4;
}

// --------------------------------------------------------------------------- //
// RT distributions
// --------------------------------------------------------------------------- //
/**
 * Defective RT densities: one curve per stimulus, over the RTs of CORRECT
 * responses, with the five quantiles GRIN actually reads marked as ticks.
 *
 * "Defective" is the right word and worth saying out loud: each curve is scaled
 * by that stimulus's accuracy, so the area under it is P(correct). A curve that
 * is both LOW and FAST is the signature of guessing, not of speed.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {Object} opts
 *   cellRTs  [stim][resp] -> sorted RTs
 *   counts   4x4
 *   labels
 *   quantiles number[] — the ticks to mark (default the trained 5)
 *   maxRT    number
 */
export function renderRT(canvas, opts) {
  const T = theme();
  const W = 460,
    H = 300;
  const g = setup(canvas, W, H);
  const pad = { l: 42, r: 12, t: 12, b: 34 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const quantiles = opts.quantiles || [0.1, 0.3, 0.5, 0.7, 0.9];

  const allRT = opts.cellRTs.flat(2);
  if (!allRT.length) {
    g.fillStyle = T.mute;
    g.font = "12px Inter, system-ui, sans-serif";
    g.textAlign = "center";
    g.fillText("No response times yet — run an experiment.", W / 2, H / 2);
    return;
  }
  const maxRT =
    opts.maxRT ??
    Math.min(
      4,
      quantileOf(
        allRT.slice().sort((a, b) => a - b),
        0.98,
      ) * 1.15,
    );
  const x = (t) => pad.l + (Math.min(t, maxRT) / maxRT) * plotW;

  // kernel-density each stimulus's CORRECT-response RTs, scaled by accuracy
  const curves = [];
  let peak = 0;
  for (let s = 0; s < 4; s++) {
    const rts = opts.cellRTs[s][s];
    const n = opts.counts[s].reduce((a, b) => a + b, 0) || 1;
    const pCorrect = rts.length / n;
    if (rts.length < 2) {
      curves.push(null);
      continue;
    }
    const sd = stdev(rts);
    const bw = Math.max(0.03, 1.06 * sd * Math.pow(rts.length, -0.2)); // Silverman
    const pts = [];
    for (let i = 0; i <= 120; i++) {
      const t = (i / 120) * maxRT;
      let d = 0;
      for (const r of rts) d += npdf((t - r) / bw);
      d = (d / (rts.length * bw)) * pCorrect; // <- defective scaling
      pts.push([t, d]);
      peak = Math.max(peak, d);
    }
    curves.push({ pts, rts, pCorrect });
  }
  const y = (d) => pad.t + plotH - (d / (peak || 1)) * plotH * 0.92;

  // axes
  g.strokeStyle = T.line;
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(pad.l, pad.t);
  g.lineTo(pad.l, pad.t + plotH);
  g.lineTo(pad.l + plotW, pad.t + plotH);
  g.stroke();
  g.fillStyle = T.mute;
  g.font = "10px Inter, system-ui, sans-serif";
  g.textAlign = "center";
  for (let t = 0; t <= maxRT + 1e-9; t += maxRT / 4) {
    const px = x(t);
    g.beginPath();
    g.moveTo(px, pad.t + plotH);
    g.lineTo(px, pad.t + plotH + 4);
    g.stroke();
    g.fillText(`${t.toFixed(1)}s`, px, pad.t + plotH + 16);
  }
  g.fillText("response time", pad.l + plotW / 2, H - 4);
  g.save();
  g.translate(11, pad.t + plotH / 2);
  g.rotate(-Math.PI / 2);
  g.fillText("density × P(correct)", 0, 0);
  g.restore();

  // curves + quantile ticks
  curves.forEach((c, s) => {
    if (!c) return;
    g.strokeStyle = T.stim[s];
    g.lineWidth = 2;
    g.beginPath();
    c.pts.forEach(([t, d], i) =>
      i ? g.lineTo(x(t), y(d)) : g.moveTo(x(t), y(d)),
    );
    g.stroke();

    // the 5 quantiles the network actually sees
    const sorted = c.rts;
    g.fillStyle = T.stim[s];
    for (const q of quantiles) {
      const v = quantileOf(sorted, q);
      const px = x(v);
      g.fillRect(px - 0.75, pad.t + plotH - 6 - s * 3, 1.5, 6);
    }
  });
}

/** Nearest-rank, matching the training-time featurisation. */
function quantileOf(sorted, q) {
  if (!sorted.length) return 0;
  const k = sorted.length;
  const i = Math.min(k - 1, Math.max(0, Math.round(q * (k - 1))));
  return sorted[i];
}
function stdev(v) {
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  return Math.sqrt(
    v.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, v.length - 1),
  );
}

// --------------------------------------------------------------------------- //
// DOM widgets
// --------------------------------------------------------------------------- //
export function buildLegend(el, labels, opts = {}) {
  const T = theme(opts.theme);
  const items = [0, 1, 2, 3].map(
    (i) =>
      `<span><span class="dot" style="background:${T.stim[i]}"></span>${stimLabel(labels, i)}</span>`,
  );
  if (opts.predictedLabel)
    items.push(
      `<span><span class="dot" style="background:${T.predicted}"></span>${opts.predictedLabel}</span>`,
    );
  el.innerHTML = items.join("");
}

/**
 * An estimate with a 90% interval. `range` is the axis; a zero line is drawn
 * when the range spans zero, because "does this interval contain zero" is the
 * whole question for a correlation.
 */
export function estRow(label, value, sd, range) {
  const [a, b] = range;
  const pct = (v) => Math.max(0, Math.min(100, (100 * (v - a)) / (b - a)));
  const lo = pct(value - 1.645 * sd);
  const hi = pct(value + 1.645 * sd);
  const zero =
    a < 0 && b > 0 ? `<div class="zero" style="left:${pct(0)}%"></div>` : "";
  return `<div class="est">
    <div class="lbl">${label}</div>
    <div class="track">
      ${zero}
      <div class="ci" style="left:${lo}%;width:${Math.max(0.6, hi - lo)}%"></div>
      <div class="pt" style="left:${pct(value)}%"></div>
    </div>
    <div class="num">${value.toFixed(2)} ± ${sd.toFixed(2)}</div>
  </div>`;
}

/** A probability bar. Dimmed when the probability is not decisive; pass
 * highlight:true to mark the one bar that matters most in a sequence (e.g.
 * the first checkpoint in an early-stopping track that actually crossed the
 * stopping threshold) with a distinct border/background rather than relying
 * on the reader to compare numbers down a column. */
export function pbar(label, p, { dimBelow = 0.6, highlight = false } = {}) {
  const cls = p < dimBelow ? "fill dim" : "fill";
  return `<div class="pbar${highlight ? " pbar-hit" : ""}">
    <div class="lbl">${label}</div>
    <div class="track"><div class="${cls}" style="width:${(100 * p).toFixed(1)}%"></div></div>
    <div class="num">${p.toFixed(2)}</div>
  </div>`;
}

export function exportPNG(canvas, filename) {
  const a = document.createElement("a");
  a.download = filename || "grin_plot.png";
  a.href = canvas.toDataURL("image/png");
  a.click();
}
