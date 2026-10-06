/* vindex — site behaviour. No dependencies. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------- theme ---------------- */
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("vindex-theme");
    if (saved) root.dataset.theme = saved;
  } catch (_) {}
  $$("[data-theme-toggle]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const dark = root.dataset.theme
        ? root.dataset.theme === "dark"
        : matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("vindex-theme", root.dataset.theme); } catch (_) {}
    })
  );

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
  // Copy buttons on every <pre> in docs/tabs.
  $$("pre[data-copyable]").forEach((pre) => {
    const btn = document.createElement("button");
    btn.className = "copy"; btn.type = "button"; btn.innerHTML = ICON_COPY; btn.setAttribute("aria-label", "Copy code");
    btn.addEventListener("click", () => copyText(pre.innerText.replace(/^\$ /gm, "").trim(), btn));
    pre.appendChild(btn);
  });

  /* ---------------- kolam ornament ---------------- */
  // A pulli kolam: dot grid with a single looping line woven around it.
  const kolam = (n = 7, step = 100) => {
    const size = n * step, c = size / 2, r = step / 2;
    let dots = "", loops = "";
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x = r + i * step, y = r + j * step;
      const d = Math.abs(x - c) + Math.abs(y - c);
      if (d > c) continue;
      dots += `<circle cx="${x}" cy="${y}" r="5"/>`;
      loops += `<rect x="${x - r * .72}" y="${y - r * .72}" width="${r * 1.44}" height="${r * 1.44}" rx="${r * .3}" transform="rotate(45 ${x} ${y})"/>`;
    }
    const petals = Array.from({ length: 16 }, (_, k) =>
      `<ellipse cx="${c}" cy="${c - size * .44}" rx="${step * .32}" ry="${step * .9}" transform="rotate(${k * 22.5} ${c} ${c})"/>`).join("");
    return `<svg viewBox="0 0 ${size} ${size}" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true">
      <g fill="currentColor" stroke="none">${dots}</g><g>${loops}</g>
      <circle cx="${c}" cy="${c}" r="${size * .47}" /><circle cx="${c}" cy="${c}" r="${size * .49}" stroke-dasharray="2 10"/>
      <g opacity=".7">${petals}</g></svg>`;
  };
  $$("[data-kolam]").forEach((el) => { el.innerHTML = kolam(+el.dataset.kolam || 7); });

  /* ---------------- reveal on scroll ---------------- */
  if ("IntersectionObserver" in window && !reduced) {
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }), { rootMargin: "0px 0px -8% 0px" });
    $$(".reveal").forEach((el) => io.observe(el));
  } else {
    $$(".reveal").forEach((el) => el.classList.add("in"));
  }

  /* ---------------- tabs ---------------- */
  $$("[data-tabs]").forEach((tabs) => {
    const buttons = $$('[role="tab"]', tabs);
    const select = (btn) => {
      buttons.forEach((b) => {
        const on = b === btn;
        b.setAttribute("aria-selected", on);
        b.tabIndex = on ? 0 : -1;
        $("#" + b.getAttribute("aria-controls")).hidden = !on;
      });
    };
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

  /* ---------------- hero terminal ---------------- */
  const term = $("#term-body");
  if (term) {
    const lines = [
      ["cmd", "vindex run evals/support_bot.jsonl"],
      ["", ""],
      ["", '  <span class="p">vindex</span> <span class="d">0.6.0</span>  <span class="b">support_bot.jsonl</span> <span class="d">· 240 cases</span>'],
      ["", ""],
      ["", '  <span class="r">✗</span> script_adherence       <span class="y">█████████████░░</span> <span class="b"> 86.7%</span>  <span class="d">208/240</span>'],
      ["", '  <span class="g">✓</span> check_trace            <span class="g">███████████████</span> <span class="b">100.0%</span>  <span class="d">64/64</span>'],
      ["", '  <span class="g">✓</span> indic_judge            <span class="g">██████████████░</span> <span class="b"> 96.3%</span>  <span class="d">231/240</span>'],
      ["", ""],
      ["", '  <span class="b">Failures</span>'],
      ["", ""],
      ["", '  <span class="r">●</span> <span class="b">refund-017</span> <span class="d">·</span> script_adherence <span class="d">·</span> <span class="y">script_mismatch</span>'],
      ["", '    <span class="d">prompt  </span> Mera refund kab tak aayega?'],
      ["", '    <span class="d">response</span> आपका रिफंड 5-7 कार्यदिवसों में आ जाएगा।'],
      ["", '    <span class="d">reason  </span> prompt is code-mixed; response in devanagari,'],
      ["", '             not Roman script.'],
      ["", '  <span class="d">  … 31 more</span>'],
      ["", ""],
      ["", '  <span class="r">FAILED</span> script_adherence below 95%'],
    ];
    const render = (upto, typed) => {
      let html = "";
      for (let i = 0; i < upto; i++) {
        const [kind, text] = lines[i];
        html += kind === "cmd" ? `<span class="p">$</span> ${text}\n` : `${text}\n`;
      }
      if (typed !== undefined) html += `<span class="p">$</span> ${typed}<span class="cursor"></span>`;
      term.innerHTML = html;
    };
    if (reduced) { render(lines.length); term.innerHTML += '<span class="p">$</span> <span class="cursor"></span>'; }
    else {
      const cmd = lines[0][1];
      let k = 0;
      const typeCmd = () => {
        render(0, cmd.slice(0, k));
        if (k++ < cmd.length) setTimeout(typeCmd, 28 + Math.random() * 40);
        else setTimeout(() => showLines(1), 380);
      };
      const showLines = (n) => {
        render(n);
        if (n < lines.length) setTimeout(() => showLines(n + 1), n < 3 ? 160 : 70);
        else term.innerHTML += '<span class="p">$</span> <span class="cursor"></span>';
      };
      setTimeout(typeCmd, 500);
    }
  }

  /* ---------------- playground: JS port of vindex.script_adherence ---------------- */
  const SCRIPTS = {
    devanagari: [0x0900, 0x097f], bengali: [0x0980, 0x09ff], gurmukhi: [0x0a00, 0x0a7f],
    gujarati: [0x0a80, 0x0aff], odia: [0x0b00, 0x0b7f], tamil: [0x0b80, 0x0bff],
    telugu: [0x0c00, 0x0c7f], kannada: [0x0c80, 0x0cff], malayalam: [0x0d00, 0x0d7f],
  };
  // Iteration order matches vindex.script.SCRIPT_RANGES (ties go to the earlier script).
  const ORDER = ["devanagari", "gurmukhi", "gujarati", "odia", "tamil", "telugu", "kannada", "malayalam", "bengali"];
  const COLORS = {
    roman: "#8a7e72", devanagari: "#e8890c", bengali: "#c8361d", gurmukhi: "#7a4fd0",
    gujarati: "#2d7a4c", odia: "#c43a7a", tamil: "#27335e", telugu: "#1f8a99",
    kannada: "#b8860b", malayalam: "#5b8c2a",
  };
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
  const R = (passed, label, reason) => ({ passed, label, reason, score: passed ? 1 : 0 });
  const scriptAdherence = (prompt, response, strict) => {
    const pl = classify(prompt), rl = classify(response);
    if (pl === "empty" || rl === "empty") return R(false, "empty", "prompt or response is empty.");
    if (noSignal(prompt)) return R(false, "no_script_signal", "prompt has no alphabetic characters in any recognized script, so no script-adherence verdict can be made against it.");
    const bucket = bucketOf(prompt);
    if (noSignal(response)) return R(false, "no_script_signal", "response has no alphabetic characters in any recognized script (e.g. emoji, digits, or punctuation only).");
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
    const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    const py = (s) => JSON.stringify(s).slice(1, -1);
    const barFor = (text) => {
      const c = count(text);
      const parts = [["roman", c.latin], ...ORDER.map((s) => [s, c[s]])].filter(([, v]) => v > 0);
      const total = parts.reduce((a, [, v]) => a + v, 0) || 1;
      return {
        bar: parts.map(([k, v]) => `<i style="width:${(v / total) * 100}%;background:${COLORS[k]}"></i>`).join(""),
        legend: parts.map(([k, v]) => `<span><b style="background:${COLORS[k]}"></b>${k} ${Math.round((v / total) * 100)}%</span>`).join(""),
        label: classify(text),
      };
    };
    const update = () => {
      const r = scriptAdherence(P.value, A.value, S.checked);
      const v = $("#pg-verdict");
      v.textContent = r.passed ? "PASS" : "FAIL";
      v.className = "big-verdict " + (r.passed ? "pass" : "fail");
      $("#pg-label").textContent = r.label;
      $("#pg-reason").textContent = r.reason;
      [["p", P.value], ["r", A.value]].forEach(([k, t]) => {
        const b = barFor(t);
        $(`#pg-${k}-bar`).innerHTML = b.bar;
        $(`#pg-${k}-legend`).innerHTML = b.legend;
        $(`#pg-${k}-script`).textContent = b.label;
      });
      $("#pg-code").innerHTML =
        `<span class="k">from</span> vindex <span class="k">import</span> script_adherence\n\n` +
        `r = script_adherence(\n    <span class="s">"${esc(py(P.value))}"</span>,\n    <span class="s">"${esc(py(A.value))}"</span>,` +
        (S.checked ? `\n    strict_language_check=<span class="n">True</span>,` : "") +
        `\n)\nr.passed  <span class="c"># ${r.passed ? "True" : "False"}</span>\nr.label   <span class="c"># "${r.label}"</span>`;
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
