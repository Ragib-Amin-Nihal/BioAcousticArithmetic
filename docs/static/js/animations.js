/* Lightweight, dependency-free SVG/JS animations for the project page.
 * Each init function is guarded so a missing container never throws.
 * Concept animations mirror the Manim hero scenes (graceful fallback).
 */
(function () {
  "use strict";
  const SVGNS = "http://www.w3.org/2000/svg";
  const TEAL = "#0f7b8a", TEAL_D = "#0a5963", AMBER = "#e8973a", RED = "#c0392b";
  const GROUP_COLORS = ["#0f7b8a", "#2e8b57", "#e8973a", "#8e44ad", "#3477b8"];
  const GROUPS = ["G1 Passerines", "G2 Non-passerines", "G3 Raptors/water",
                  "G4 Marine mammals", "G5 Amphibians"];

  const el = (tag, attrs) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in (attrs || {})) n.setAttribute(k, attrs[k]);
    return n;
  };
  const byId = (id) => document.getElementById(id);

  // ------------------------------------------------------------------
  // HERO: animated triptych (acoustic niche -> orthogonal TVs -> compose)
  // ------------------------------------------------------------------
  function heroTriptych() {
    const host = byId("hero-anim");
    if (!host) return;
    const W = 960, H = 300;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Acoustic niche partitioning produces near-orthogonal task vectors that compose without interference." });
    host.appendChild(svg);

    const panelW = 300, gap = 30, pad = 14;
    const titles = ["(a) Acoustic niche", "(b) Near-orthogonal task vectors", "(c) Composition"];
    const px = [0, panelW + gap, 2 * (panelW + gap)];

    titles.forEach((t, i) => {
      svg.appendChild(el("rect", { x: px[i], y: 26, width: panelW, height: H - 40,
        rx: 12, fill: "#ffffff", stroke: "#e4e8ea" }));
      const lab = el("text", { x: px[i] + panelW / 2, y: 18, "text-anchor": "middle",
        "font-size": 14, "font-weight": 700, fill: TEAL_D });
      lab.textContent = t; svg.appendChild(lab);
    });

    // (a) frequency bands, non-overlapping energy blobs
    const ax = px[0], aTop = 42, aH = H - 70, bandH = aH / 5;
    for (let i = 0; i < 5; i++) {
      const y = aTop + i * bandH;
      svg.appendChild(el("rect", { x: ax + pad, y: y + 3, width: panelW - 2 * pad,
        height: bandH - 6, rx: 6, fill: GROUP_COLORS[i], opacity: 0.10 }));
      // energy "hump" placed in a distinct horizontal (frequency) slot per group
      const cx = ax + pad + (panelW - 2 * pad) * (0.15 + 0.16 * i);
      const blob = el("ellipse", { cx, cy: y + bandH / 2, rx: 26, ry: bandH * 0.32,
        fill: GROUP_COLORS[i], opacity: 0.85 });
      blob.appendChild(animate("opacity", "0.35;0.9;0.35", 3.2, i * 0.3));
      svg.appendChild(blob);
    }
    const fLab = el("text", { x: ax + panelW / 2, y: H - 18, "text-anchor": "middle",
      "font-size": 11, fill: "#5b6770" }); fLab.textContent = "frequency →"; svg.appendChild(fLab);

    // (b) near-orthogonal arrows from a shared origin
    const bx = px[1], oy = aTop + aH / 2 + 6, ox = bx + panelW / 2 - 20;
    defsArrow(svg);
    const angles = [-72, -36, 8, 46, 80]; // spread, all far from collinear
    const lens = [70, 58, 48, 30, 40];     // magnitude ~ dataset size (G1 largest)
    angles.forEach((deg, i) => {
      const r = deg * Math.PI / 180;
      const x2 = ox + Math.cos(r) * lens[i], y2 = oy + Math.sin(r) * lens[i];
      const ln = el("line", { x1: ox, y1: oy, x2: ox, y2: oy, stroke: GROUP_COLORS[i],
        "stroke-width": 3, "marker-end": "url(#arrow)" });
      svg.appendChild(ln);
      animateAttr(ln, "x2", ox, x2, 1.0, 0.5 + i * 0.25);
      animateAttr(ln, "y2", oy, y2, 1.0, 0.5 + i * 0.25);
    });
    svg.appendChild(dot(ox, oy, 4, "#1a1f24"));
    const cosLab = el("text", { x: bx + panelW / 2, y: H - 18, "text-anchor": "middle",
      "font-size": 11, fill: "#5b6770" });
    cosLab.textContent = "cosine ≈ 0.01–0.09 (≈ 90°)"; svg.appendChild(cosLab);

    // (c) compose: base + sum of small vectors -> merged star
    const cx0 = px[2] + 70, cy0 = oy;
    svg.appendChild(dot(cx0, cy0, 5, "#1a1f24"));
    const b0 = el("text", { x: cx0, y: cy0 + 22, "text-anchor": "middle", "font-size": 11,
      fill: "#1a1f24", "font-weight": 700 }); b0.textContent = "θ₀"; svg.appendChild(b0);
    let cx = cx0, cy = cy0;
    const steps = [[34, -30], [30, 26], [40, 4], [18, 36], [24, -16]];
    steps.forEach((d, i) => {
      const nx = cx + d[0], ny = cy + d[1];
      const seg = el("line", { x1: cx, y1: cy, x2: cx, y2: cy, stroke: GROUP_COLORS[i],
        "stroke-width": 3, opacity: 0.9, "marker-end": "url(#arrow)" });
      svg.appendChild(seg);
      animateAttr(seg, "x2", cx, nx, 0.6, 1.0 + i * 0.6);
      animateAttr(seg, "y2", cy, ny, 0.6, 1.0 + i * 0.6);
      cx = nx; cy = ny;
    });
    const star = el("path", { d: starPath(cx, cy, 11, 5), fill: AMBER, stroke: "#b9701f",
      "stroke-width": 1, opacity: 0 });
    star.appendChild(animate("opacity", "0;0;1", 6, 4.2, "freeze"));
    svg.appendChild(star);
    const mLab = el("text", { x: cx, y: cy + 24, "text-anchor": "middle", "font-size": 11,
      fill: "#b9701f", "font-weight": 700, opacity: 0 });
    mLab.appendChild(animate("opacity", "0;0;1", 6, 4.2, "freeze"));
    mLab.textContent = "θ_merged"; svg.appendChild(mLab);
    const cLab = el("text", { x: px[2] + panelW / 2, y: H - 18, "text-anchor": "middle",
      "font-size": 11, fill: "#5b6770" });
    cLab.textContent = "661-species classifier — no shared data"; svg.appendChild(cLab);
  }

  // ------------------------------------------------------------------
  // Task-vector arithmetic widget (play button)
  // ------------------------------------------------------------------
  function tvArithmetic() {
    const host = byId("tv-arith");
    if (!host) return;
    const W = 560, H = 360, ox = 120, oy = 250;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });
    host.appendChild(svg); defsArrow(svg);
    grid(svg, W, H);
    // axes
    svg.appendChild(el("line", { x1: 30, y1: oy, x2: W - 20, y2: oy, stroke: "#cfd6da" }));
    svg.appendChild(el("line", { x1: ox, y1: 30, x2: ox, y2: H - 20, stroke: "#cfd6da" }));
    svg.appendChild(dot(ox, oy, 5, "#1a1f24"));
    const t0 = el("text", { x: ox - 8, y: oy + 22, "text-anchor": "end", "font-size": 13,
      "font-weight": 700 }); t0.textContent = "θ₀ (base)"; svg.appendChild(t0);

    const taus = [[95, -150], [150, -55], [120, -110], [60, -40]];
    const N = taus.length, scale = 1 / N;
    const segLayer = el("g", {}); svg.appendChild(segLayer);
    const ghostLayer = el("g", { opacity: 0.5 }); svg.appendChild(ghostLayer);
    // faint full-size individual task vectors for reference
    taus.forEach((t, i) => {
      ghostLayer.appendChild(el("line", { x1: ox, y1: oy, x2: ox + t[0], y2: oy + t[1],
        stroke: GROUP_COLORS[i], "stroke-width": 1.5, "stroke-dasharray": "4 4" }));
    });
    const merged = el("path", { d: "", fill: AMBER, stroke: "#b9701f", "stroke-width": 1, opacity: 0 });
    svg.appendChild(merged);
    const mTxt = el("text", { x: 0, y: 0, "font-size": 13, "font-weight": 700, fill: "#b9701f", opacity: 0 });
    svg.appendChild(mTxt);

    function play() {
      while (segLayer.firstChild) segLayer.removeChild(segLayer.firstChild);
      merged.setAttribute("opacity", 0); mTxt.setAttribute("opacity", 0);
      let cx = ox, cy = oy, i = 0;
      const step = () => {
        if (i >= N) {
          merged.setAttribute("d", starPath(cx, cy, 10, 5));
          fade(merged); fade(mTxt);
          mTxt.setAttribute("x", cx + 12); mTxt.setAttribute("y", cy - 6);
          mTxt.textContent = "θ_merged = θ₀ + (1/N) Σ τᵢ";
          return;
        }
        const nx = cx + taus[i][0] * scale, ny = cy + taus[i][1] * scale;
        const seg = el("line", { x1: cx, y1: cy, x2: cx, y2: cy, stroke: GROUP_COLORS[i],
          "stroke-width": 3.5, "marker-end": "url(#arrow)" });
        segLayer.appendChild(seg);
        tween(420, (p) => {
          seg.setAttribute("x2", cx + (nx - cx) * p);
          seg.setAttribute("y2", cy + (ny - cy) * p);
        }, () => { cx = nx; cy = ny; i++; step(); });
      };
      step();
    }
    byId("tv-play").addEventListener("click", play);
    play();
  }

  // ------------------------------------------------------------------
  // Cosine "near-orthogonal" fan (slider across the empirical gradient)
  // ------------------------------------------------------------------
  function cosineFan() {
    const host = byId("cos-fan");
    if (!host) return;
    const W = 460, H = 320, ox = W / 2, oy = H - 60, R = 150;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });
    host.appendChild(svg); defsArrow(svg);
    // 90-degree reference wedge
    svg.appendChild(el("path", { d: `M ${ox} ${oy} L ${ox} ${oy - R} A ${R} ${R} 0 0 1 ${ox + R} ${oy} Z`,
      fill: TEAL, opacity: 0.05 }));
    const fixed = el("line", { x1: ox, y1: oy, x2: ox + R, y2: oy, stroke: "#7a868d",
      "stroke-width": 3, "marker-end": "url(#arrow)" });
    svg.appendChild(fixed);
    const moving = el("line", { x1: ox, y1: oy, x2: ox, y2: oy - R, stroke: TEAL,
      "stroke-width": 3.5, "marker-end": "url(#arrow)" });
    svg.appendChild(moving);
    svg.appendChild(dot(ox, oy, 4, "#1a1f24"));

    const regimes = [
      { name: "Cross-taxa (bird ↔ other)", cos: 0.014 },
      { name: "Intra-avian (bird ↔ bird)", cos: 0.090 },
      { name: "Regional (region ↔ region)", cos: 0.120 },
    ];
    const slider = byId("cos-slider"), out = byId("cos-readout");
    function update() {
      const r = regimes[+slider.value];
      const ang = Math.acos(r.cos);            // radians from fixed arrow
      const x2 = ox + Math.cos(ang) * R, y2 = oy - Math.sin(ang) * R;
      moving.setAttribute("x2", x2); moving.setAttribute("y2", y2);
      out.innerHTML = `${r.name} — cosine = <span class="readout">${r.cos.toFixed(3)}</span>` +
        ` &nbsp;(angle ≈ <span class="readout">${(ang * 180 / Math.PI).toFixed(1)}°</span>)`;
    }
    slider.addEventListener("input", update);
    update();
  }

  // ------------------------------------------------------------------
  // Merging-method comparison (why sign-conflict resolution hurts here)
  // ------------------------------------------------------------------
  function mergingMethods() {
    const host = byId("merge-methods");
    if (!host) return;
    const W = 520, H = 250;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });
    host.appendChild(svg);
    const out = byId("merge-readout");
    // 8x8 parameter grid; columns colored by sign of a task vector
    const cols = 16, rows = 8, cw = (W - 60) / cols, ch = 22, gx = 30, gy = 20;
    const cells = [];
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const rect = el("rect", { x: gx + c * cw, y: gy + r * ch, width: cw - 2, height: ch - 2,
        rx: 2, fill: TEAL, opacity: 0.85 });
      svg.appendChild(rect); cells.push(rect);
    }
    const cap = el("text", { x: W / 2, y: H - 8, "text-anchor": "middle", "font-size": 12,
      fill: "#5b6770" }); svg.appendChild(cap);

    function render(method) {
      cells.forEach((cell) => {
        let keep = true, color = TEAL;
        if (method === "ties") {
          // near-chance sign agreement -> ~half of params discarded at random
          keep = Math.random() > 0.5; color = keep ? TEAL : "#dfe6e9";
        } else if (method === "dare") {
          keep = Math.random() > 0.3; color = keep ? AMBER : "#f4e7d3"; // drop + rescale
        }
        cell.setAttribute("fill", color);
        cell.setAttribute("opacity", keep ? 0.9 : 1);
      });
      if (method === "simple") {
        cap.textContent = "Simple average: every parameter of every τ contributes — optimal here";
        out.innerHTML = `<b>Simple average</b> — all parameters kept. <span class="stat">Best accuracy (59.2%, 86% of joint)</span>.`;
      } else if (method === "ties") {
        cap.textContent = "TIES sign-election: ~half of each τ discarded (sign agreement ≈ 0.5)";
        out.innerHTML = `<b>TIES</b> — majority-vote sign election is ~random under orthogonality, discarding ~50% of each vector. <span class="stat" style="color:#c0392b">−1 to −6 pp accuracy</span>.`;
      } else {
        cap.textContent = "DARE: random drop + rescale — no sign conflict to resolve here";
        out.innerHTML = `<b>DARE</b> — random dropout with rescaling; competitive but adds variance without benefit when τ are orthogonal.`;
      }
    }
    const scope = host.closest(".widget") || document;
    scope.querySelectorAll("[data-method]").forEach((b) => {
      b.addEventListener("click", () => {
        scope.querySelectorAll("[data-method]").forEach((x) =>
          x.classList.remove("active"));
        b.classList.add("active");
        render(b.getAttribute("data-method"));
      });
    });
    render("simple");
  }

  // ------------------------------------------------------------------
  // helpers
  // ------------------------------------------------------------------
  function defsArrow(svg) {
    if (svg.querySelector("#arrow")) return;
    const defs = el("defs", {});
    const m = el("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 8, refY: 5,
      markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
    m.appendChild(el("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "context-stroke" }));
    defs.appendChild(m); svg.appendChild(defs);
  }
  function grid(svg, W, H) {
    const g = el("g", { stroke: "#eef2f4", "stroke-width": 1 });
    for (let x = 0; x <= W; x += 28) g.appendChild(el("line", { x1: x, y1: 0, x2: x, y2: H }));
    for (let y = 0; y <= H; y += 28) g.appendChild(el("line", { x1: 0, y1: y, x2: W, y2: y }));
    svg.appendChild(g);
  }
  function dot(cx, cy, r, fill) { return el("circle", { cx, cy, r, fill }); }
  function starPath(cx, cy, R, n) {
    let d = ""; const r = R * 0.45;
    for (let i = 0; i < n * 2; i++) {
      const rad = (i % 2 ? r : R), a = Math.PI / n * i - Math.PI / 2;
      d += (i ? "L" : "M") + (cx + Math.cos(a) * rad) + " " + (cy + Math.sin(a) * rad) + " ";
    }
    return d + "Z";
  }
  function animate(attr, values, dur, begin, fill) {
    const a = el("animate", { attributeName: attr, values, dur: dur + "s",
      begin: (begin || 0) + "s", repeatCount: fill ? "1" : "indefinite" });
    if (fill) a.setAttribute("fill", fill);
    return a;
  }
  function animateAttr(node, attr, from, to, dur, begin) {
    node.appendChild(el("animate", { attributeName: attr, from, to, dur: dur + "s",
      begin: begin + "s", fill: "freeze", repeatCount: "1" }));
  }
  function tween(ms, onStep, onDone) {
    const t0 = performance.now();
    function frame(t) {
      const p = Math.min(1, (t - t0) / ms);
      onStep(p < 1 ? 1 - Math.pow(1 - p, 3) : 1);
      if (p < 1) requestAnimationFrame(frame); else onDone && onDone();
    }
    requestAnimationFrame(frame);
  }
  function fade(node) { tween(500, (p) => node.setAttribute("opacity", p)); }

  document.addEventListener("DOMContentLoaded", function () {
    tvArithmetic();
    cosineFan();
    mergingMethods();
  });
})();
