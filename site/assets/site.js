/* vindex — site behaviour. No dependencies. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------- red-pen marks: inline the <symbol>s so strokes can animate ---------------- */
  const inlineMarks = (root = document) => {
    $$("svg > use", root).forEach((use) => {
      const sym = document.querySelector(use.getAttribute("href"));
      const svg = use.parentNode;
      if (!sym) return;
      svg.setAttribute("viewBox", sym.getAttribute("viewBox"));
      const par = sym.getAttribute("preserveAspectRatio");
      if (par) svg.setAttribute("preserveAspectRatio", par);
      svg.innerHTML = sym.innerHTML;
    });
  };
  inlineMarks();

  /* ---------------- nav ---------------- */
  const nav = $(".nav");
  if (nav) {
    const onScroll = () => nav.classList.toggle("scrolled", scrollY > 8);
    addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  /* ---------------- copy buttons ---------------- */
  const ICON_COPY = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
  const ICON_OK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';
  const copyText = async (text, btn) => {
    try { await navigator.clipboard.writeText(text); } catch (_) {
      const t = document.createElement("textarea");
      t.value = text; document.body.appendChild(t); t.select();
      try { document.execCommand("copy"); } catch (_) {}
      t.remove();
    }
    btn.classList.add("done"); btn.innerHTML = ICON_OK;
    setTimeout(() => { btn.classList.remove("done"); btn.innerHTML = ICON_COPY; }, 1400);
  };
  $$("[data-copy]").forEach((btn) => {
    btn.innerHTML = ICON_COPY;
    btn.setAttribute("aria-label", "Copy");
    btn.addEventListener("click", () => copyText(btn.dataset.copy, btn));
  });
  $$("pre[data-copyable]").forEach((pre) => {
    const btn = document.createElement("button");
    btn.className = "copy"; btn.type = "button"; btn.innerHTML = ICON_COPY; btn.setAttribute("aria-label", "Copy code");
    btn.addEventListener("click", () => copyText(pre.innerText.replace(/^\$ /gm, "").trim(), btn));
    pre.appendChild(btn);
  });

  /* ---------------- reveal (also triggers the pen strokes) ---------------- */
  if ("IntersectionObserver" in window && !reduced) {
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }), { rootMargin: "0px 0px -12% 0px" });
    $$(".reveal").forEach((el) => io.observe(el));
  } else {
    $$(".reveal").forEach((el) => el.classList.add("in"));
  }

  /* ---------------- tabs ---------------- */
  $$("[data-tabs]").forEach((tabs) => {
    const buttons = $$('[role="tab"]', tabs);
    const select = (btn) => buttons.forEach((b) => {
      const on = b === btn;
      b.setAttribute("aria-selected", on);
      b.tabIndex = on ? 0 : -1;
      $("#" + b.getAttribute("aria-controls")).hidden = !on;
    });
    buttons.forEach((b, i) => {
      b.addEventListener("click", () => select(b));
      b.addEventListener("keydown", (e) => {
        const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (!d) return;
        const next = buttons[(i + d + buttons.length) % buttons.length];
        next.focus(); select(next);
      });
    });
  });

  /* ---------------- playground: JS port of vindex.script_adherence ---------------- */
  const SCRIPTS = {
    devanagari: [0x0900, 0x097f], bengali: [0x0980, 0x09ff], gurmukhi: [0x0a00, 0x0a7f],
    gujarati: [0x0a80, 0x0aff], odia: [0x0b00, 0x0b7f], tamil: [0x0b80, 0x0bff],
    telugu: [0x0c00, 0x0c7f], kannada: [0x0c80, 0x0cff], malayalam: [0x0d00, 0x0d7f],
  };
  // Same order as vindex.script.SCRIPT_RANGES (ties go to the earlier script).
  const ORDER = ["devanagari", "gurmukhi", "gujarati", "odia", "tamil", "telugu", "kannada", "malayalam", "bengali"];
  const HINDI = new Set(["hai", "hain", "kya", "nahi", "mera", "aap", "ka", "ki", "ke", "se", "mein", "tha", "hoga", "raha"]);

  const scriptOf = (cp) => {
    for (const name of ORDER) { const [a, b] = SCRIPTS[name]; if (cp >= a && cp <= b) return name; }
    return null;
  };
  const count = (text) => {
    const c = { latin: 0, letters: {} };
    ORDER.forEach((n) => { c[n] = 0; c.letters[n] = 0; });
    for (const ch of text) {
      if (/[A-Za-z]/.test(ch)) { c.latin++; continue; }
      const s = scriptOf(ch.codePointAt(0));
      if (!s) continue;
      c[s]++;
      if (!/\p{Nd}/u.test(ch) && ch !== "।" && ch !== "॥") c.letters[s]++;
    }
    return c;
  };
  const classify = (text) => {
    if (!text || !text.trim()) return "empty";
    const c = count(text);
    let dom = ORDER[0], n = c[dom];
    for (const s of ORDER) if (c[s] > n) { dom = s; n = c[s]; }
    if (n > c.latin) return dom;
    if (c.latin > n * 2) return "roman";
    return "mixed";
  };
  const noSignal = (text) => {
    const c = count(text);
    return c.latin === 0 && ORDER.every((s) => c.letters[s] === 0);
  };
  const looksHinglish = (text) => (text.match(/[A-Za-z]+/g) || []).some((w) => HINDI.has(w.toLowerCase()));
  const bucketOf = (prompt) => {
    const l = classify(prompt);
    if (ORDER.includes(l)) return "native-script";
    if (l === "mixed") return "code-mixed";
    if (l === "roman") return looksHinglish(prompt) ? "code-mixed" : "romanized";
    return "empty";
  };
  const R = (passed, label, reason) => ({ passed, label, reason });
  const scriptAdherence = (prompt, response, strict) => {
    const pl = classify(prompt), rl = classify(response);
    if (pl === "empty" || rl === "empty") return R(false, "empty", "prompt or response is empty.");
    if (noSignal(prompt)) return R(false, "no_script_signal", "prompt has no letters in any recognized script.");
    const bucket = bucketOf(prompt);
    if (noSignal(response)) return R(false, "no_script_signal", "response has no letters in any recognized script — only digits, emoji or punctuation.");
    if (bucket === "native-script") {
      if (rl === pl) return R(true, "matched", `prompt and response both in ${pl}.`);
      if (rl === "mixed") return R(true, "mixed", `prompt in ${pl}; response is code-mixed.`);
      return R(false, "script_mismatch", `prompt in ${pl}; response in ${rl}.`);
    }
    if (rl === "roman" || rl === "mixed") {
      if (strict && bucket === "code-mixed" && rl === "roman" && looksHinglish(prompt) && !looksHinglish(response))
        return R(false, "language_mismatch", "prompt is Romanized Hindi; response is Roman-script English.");
      return rl === "roman"
        ? R(true, "matched", `prompt is ${bucket}; response is Roman script, as expected.`)
        : R(true, "mixed", `prompt is ${bucket}; response is ${rl}.`);
    }
    return R(false, "script_mismatch", `prompt is ${bucket}; response in ${rl}, not Roman script.`);
  };

  const pg = $("#playground");
  if (pg) {
    const P = $("#pg-prompt"), A = $("#pg-response"), S = $("#pg-strict");
    const CROSS = '<path d="M9 8 C 15 15, 24 24, 32 33"/><path d="M31 7 C 24 15, 16 24, 8 34"/>';
    const TICK = '<path d="M5 22 C 9 25, 12 29, 15 34 C 20 22, 27 12, 36 4"/>';
    const update = () => {
      const r = scriptAdherence(P.value, A.value, S.checked);
      $("#pg-q").textContent = P.value || "—";
      $("#pg-a").textContent = A.value || "—";
      $("#pg-p-script").textContent = classify(P.value);
      $("#pg-r-script").textContent = classify(A.value);
      const v = $("#pg-verdict");
      v.classList.toggle("pass", r.passed);
      $("svg", v).innerHTML = r.passed ? TICK : CROSS;
      $("#pg-word").textContent = r.passed ? "correct script" : "wrong";
      $("#pg-reason").textContent = r.reason;
      $("#pg-label").textContent = r.label;
      $("#pg-label").style.color = r.passed ? "var(--ok)" : "";
      $("#pg-score").textContent = r.passed ? "1/1" : "0/1";
    };
    [P, A].forEach((el) => el.addEventListener("input", () => { $$(".presets button").forEach((b) => b.classList.remove("on")); update(); }));
    S.addEventListener("change", update);
    $$(".presets button").forEach((b) => b.addEventListener("click", () => {
      P.value = b.dataset.p; A.value = b.dataset.r; S.checked = b.dataset.strict === "1";
      $$(".presets button").forEach((x) => x.classList.toggle("on", x === b));
      update();
    }));
    update();
  }

  /* ---------------- docs scrollspy ---------------- */
  const side = $(".sidebar");
  if (side && "IntersectionObserver" in window) {
    const links = new Map($$("a[href^='#']", side).map((a) => [a.getAttribute("href").slice(1), a]));
    const visible = new Set();
    const spy = new IntersectionObserver((entries) => {
      entries.forEach((e) => (e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id)));
      const first = $$(".prose [id]").find((el) => visible.has(el.id) && links.has(el.id));
      if (first) links.forEach((a, id) => a.classList.toggle("active", id === first.id));
    }, { rootMargin: "-80px 0px -65% 0px" });
    $$(".prose h2[id], .prose h3[id]").forEach((h) => spy.observe(h));
  }
})();
