// nightshift demo. Every number comes from run_artifacts.json (latest full run) and journal.json (all runs).
(async () => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const P = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
  const N = (x) => (x == null ? "–" : Math.round(x).toLocaleString("en-US"));
  const mean = (xs) => { const v = xs.filter((x) => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  const COL = { base: "#64748B", rag: "#818CF8", sft: "#10B981", sdft: "#F59E0B", rose: "#F43F5E" };
  const NAME = { base: "Base", rag: "RAG", sft: "Same-model SFT", sdft: "On-policy SDFT" };
  const SHORT = { base: "base", rag: "rag", sft: "sft", sdft: "sdft" };
  const ARMS = ["base", "rag", "sft", "sdft"];

  let A, J;
  try {
    [A, J] = await Promise.all([fetch("run_artifacts.json").then((r) => r.json()), fetch("journal.json").then((r) => r.json())]);
  } catch (e) {
    document.querySelector("main").innerHTML = `<p class="lede" style="padding:80px 0">Could not load the run data (${esc(e.message)}).</p>`;
    return;
  }

  const ev = (arm) => A.arms[arm].days.map((d) => d.eval);
  const releases = A.meta.releases;
  const modOfDay = (d) => releases.find((r) => r.day === d).module;
  const shortMod = (m) => m.replace("tessel.", "");
  const fullRun = [...J.runs].reverse().find((r) => r.kind === "full");
  const hw = (fullRun?.hardware.match(/A100|H100|L40S/) || ["GPU"])[0];
  $("modelPill").textContent = `${A.meta.model.split("/").pop().replace("-Instruct", "").replace("Qwen2.5", "Qwen2.5")} · ${hw}`;
  $("labCount").textContent = `(${J.runs.length})`;
  $("footMeta").textContent = `Latest data: ${A.meta.model}, generated ${A.meta.created}. Source: results/ in the repo.`;

  // ------------------------------------------------------------------ python highlighting
  const PY = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|\b(\d+(?:\.\d+)?)\b|\b(def|return|lambda|for|in|if|else|elif|import|from|as|not|and|or|is|None|True|False|while|class|with|yield|try|except|raise|pass|break|continue)\b|\b(print|len|range|sorted|sum|min|max|list|dict|set|tuple|str|int|float|enumerate|zip|abs|any|all|isinstance|reversed)\b|\b([A-Z][A-Za-z0-9_]*)\b|\.([a-z_][A-Za-z0-9_]*)(?=\()/g;
  function hl(src) {
    let out = "", last = 0, m;
    PY.lastIndex = 0;
    while ((m = PY.exec(src))) {
      out += esc(src.slice(last, m.index));
      const [t, c, s, n, k, b, ty, f] = m;
      out += c ? `<span class="tk-c">${esc(c)}</span>` : s ? `<span class="tk-s">${esc(s)}</span>` : n ? `<span class="tk-n">${n}</span>`
        : k ? `<span class="tk-k">${k}</span>` : b ? `<span class="tk-b">${b}</span>` : ty ? `<span class="tk-t">${ty}</span>`
        : f ? `.<span class="tk-f">${f}</span>` : esc(t);
      last = m.index + t.length;
    }
    return out + esc(src.slice(last));
  }
  function code(text) {
    let t = String(text || "").trim();
    if (t.includes("```")) {
      const parts = t.split("```");
      t = parts.length >= 3 ? parts[parts.length - 2] : parts[parts.length - 1];
      if (t.startsWith("python")) t = t.slice(6);
    }
    return t.replace(/^\n+|\s+$/g, "");
  }

  // ------------------------------------------------------------------ chart engine
  function sizeOf(svg) {
    const W = Math.max(280, svg.parentElement.clientWidth || 800);
    const H = Math.round(Math.min(380, Math.max(230, W * 0.4)));
    return { W, H };
  }
  function lineChart(svg, tip, o) {
    // o: {series:[{label,color,values,dash,dim,hidden}], yMax, yMin, ticks, fmt, log, active, onPick, band, xLabel}
    const { W, H } = sizeOf(svg);
    const m = { l: 46, r: 14, t: 12, b: 30 };
    const n = o.n || 10;
    const x = (i) => m.l + (i * (W - m.l - m.r)) / (n - 1);
    const lo = o.yMin ?? 0, hi = o.yMax;
    const yv = (v) => (o.log ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo));
    const y = (v) => m.t + (1 - Math.max(0, Math.min(1, yv(v)))) * (H - m.t - m.b);
    let g = "";
    if (o.band) { const bx = x(o.band - 1), w = (W - m.l - m.r) / (n - 1); g += `<rect x="${bx - w / 2}" y="${m.t}" width="${w}" height="${H - m.t - m.b}" fill="${COL.rose}" opacity="0.08" rx="6"/>`; }
    for (const t of o.ticks) g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}"/><text x="${m.l - 10}" y="${y(t) + 4}" text-anchor="end">${o.fmt(t)}</text>`;
    for (let i = 0; i < n; i++) g += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle"${o.active === i + 1 ? ' style="fill:#E7EAF0"' : ""}>${o.xLabel ? o.xLabel(i + 1) : i + 1}</text>`;
    if (o.active) g += `<line class="scrub" x1="${x(o.active - 1)}" x2="${x(o.active - 1)}" y1="${m.t}" y2="${H - m.b}"/>`;
    for (const s of o.series) {
      if (s.hidden) continue;
      let d = "", pen = false;
      s.values.forEach((v, i) => { if (v == null) { pen = false; return; } d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(o.log ? Math.max(lo, v) : v).toFixed(1)}`; pen = true; });
      g += `<path class="line" d="${d}" stroke="${s.color}" ${s.dash ? 'stroke-dasharray="6 5"' : ""} opacity="${s.dim ? 0.28 : 1}"/>`;
      if (o.active && s.values[o.active - 1] != null && !s.dim)
        g += `<circle cx="${x(o.active - 1)}" cy="${y(o.log ? Math.max(lo, s.values[o.active - 1]) : s.values[o.active - 1])}" r="5" fill="${s.color}" stroke="#11141A" stroke-width="2.5"/>`;
    }
    g += `<rect class="hit" x="${m.l - 10}" y="0" width="${W - m.l - m.r + 20}" height="${H}" fill="transparent" style="cursor:${o.onPick ? "pointer" : "crosshair"}"/>`;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("height", H);
    svg.innerHTML = g;
    const hit = svg.querySelector(".hit");
    const idx = (ev) => { const r = svg.getBoundingClientRect(); const px = ((ev.clientX - r.left) / r.width) * W; return Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / (W - m.l - m.r)) * (n - 1)))); };
    hit.onmousemove = (e) => {
      const i = idx(e);
      const rows = o.series.filter((s) => !s.hidden).map((s) => `<div class="r"><i style="--c:${s.color}">${esc(s.label)}</i><b>${o.fmt(s.values[i], true)}</b></div>`).join("");
      tip.innerHTML = `<div class="t">${o.xLabel ? o.xLabel(i + 1, true) : `Day ${i + 1}`}</div>${rows}`;
      tip.hidden = false;
      const r = svg.getBoundingClientRect(), px = (x(i) / W) * r.width;
      tip.style.top = "8px";
      tip.style.left = px > r.width / 2 ? `${px - tip.offsetWidth - 14}px` : `${px + 14}px`;
    };
    hit.onmouseleave = () => { tip.hidden = true; };
    hit.onclick = (e) => o.onPick && o.onPick(idx(e) + 1);
  }
  const rose_ = (a, b) => a != null && b != null && b > a * 1.5 && b - a > 0.005; // loss rose by a real amount
  function dumbbell(svg, tip, o) {
    // o: {rows:[{label,color,pairs:[[start,end],...]}], lo, hi, ticks}; log scale; red when end > start
    const { W, H } = sizeOf(svg);
    const m = { l: 46, r: 14, t: 12, b: 30 }, n = o.rows[0].pairs.length;
    const x = (i) => m.l + 18 + (i * (W - m.l - m.r - 36)) / (n - 1);
    const y = (v) => { const t = (Math.log10(Math.max(o.lo, v)) - Math.log10(o.lo)) / (Math.log10(o.hi) - Math.log10(o.lo)); return m.t + (1 - Math.min(1, t)) * (H - m.t - m.b); };
    let g = "";
    if (o.band) { const w = (W - m.l - m.r - 36) / (n - 1); g += `<rect x="${x(o.band - 1) - w / 2}" y="${m.t}" width="${w}" height="${H - m.t - m.b}" fill="${COL.rose}" opacity="0.08" rx="6"/>`; }
    for (const t of o.ticks) g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}"/><text x="${m.l - 10}" y="${y(t) + 4}" text-anchor="end">${t}</text>`;
    for (let i = 0; i < n; i++) g += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">N${i + 1}</text>`;
    o.rows.forEach((r, j) => {
      const dx = (j - (o.rows.length - 1) / 2) * 9;
      r.pairs.forEach(([a, b], i) => {
        if (a == null || b == null) return;
        const up = rose_(a, b), c = up ? COL.rose : r.color, X = x(i) + dx;
        g += `<line x1="${X}" x2="${X}" y1="${y(a)}" y2="${y(b)}" stroke="${c}" stroke-width="3" stroke-linecap="round" opacity="${up ? 1 : 0.75}"/>`;
        g += `<circle cx="${X}" cy="${y(a)}" r="3.5" fill="#11141A" stroke="${c}" stroke-width="2"/><circle cx="${X}" cy="${y(b)}" r="4.2" fill="${c}"/>`;
      });
    });
    g += `<rect class="hit" x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}" fill="transparent"/>`;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("height", H); svg.innerHTML = g;
    const hit = svg.querySelector(".hit");
    hit.onmousemove = (e) => {
      const r = svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * W;
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l - 18) / (W - m.l - m.r - 36)) * (n - 1))));
      tip.innerHTML = `<div class="t">Night ${i + 1}: loss start → end</div>` + o.rows.map((row) => { const [a, b] = row.pairs[i];
        return `<div class="r"><i style="--c:${rose_(a, b) ? COL.rose : row.color}">${esc(row.label)}</i><b>${a == null ? "–" : `${a.toFixed(3)} → ${b.toFixed(3)}`}</b></div>`; }).join("");
      tip.hidden = false; tip.style.top = "8px";
      const p = (x(i) / W) * r.width; tip.style.left = p > r.width / 2 ? `${p - tip.offsetWidth - 14}px` : `${p + 14}px`;
    };
    hit.onmouseleave = () => { tip.hidden = true; };
  }
  const redraws = [];
  new ResizeObserver(() => redraws.forEach((f) => f())).observe(document.querySelector("main"));

  // ------------------------------------------------------------------ OVERVIEW: hero + KPIs
  const means = Object.fromEntries(ARMS.map((a) => [a, mean(ev(a).map((e) => e.all_acc))]));
  const ragTok = mean(ev("rag").map((e) => e.memory_tokens));
  const lastRagTok = ev("rag")[9].memory_tokens;
  $("lede").innerHTML = `<b style="color:var(--text)">tessel</b> is a Python library we invented: ${releases.length} modules, released one per simulated day, so no model could have seen it in pretraining. By day the model practices on it; by night it trains what it learned into its own weights. Every answer is graded by running the code.`;
  const kpi = (arm, meta, extra = "") => `<div class="kpi ${extra}" style="--c:${COL[arm]}">
      <div class="lab"><span class="dot"></span>${NAME[arm]}</div>
      <div class="valrow"><div class="val">${P(means[arm])}</div>${extra === "win" ? `<span class="badge">0 doc tokens</span>` : ""}</div>
      <div class="meta">${meta}</div></div>`;
  $("kpis").innerHTML =
    kpi("base", "no docs · no training") +
    kpi("rag", `docs in every prompt · <b>${N(ragTok)}</b> tokens avg`) +
    kpi("sft", `closed-book · within <b>${Math.round(100 * (means.rag - means.sft))} pts</b> of RAG`, "win") +
    kpi("sdft", `closed-book · <b>0</b> doc tokens`);
  $("kpiNote").textContent = `10-day mean on held-out tasks from every module released so far (${ev("sdft")[0].n_eval_tasks} tasks on day 1, ${ev("sdft")[9].n_eval_tasks} on day 10). RAG's prompt grows to ${N(lastRagTok)} doc tokens by day 10; SFT and SDFT answer with none.`;

  // ------------------------------------------------------------------ OVERVIEW: day stage
  const METRICS = [
    { k: "all_acc", label: "All modules", desc: "held-out tasks from every module released so far" },
    { k: "unseen_family_acc", label: "Unseen task types", desc: "task types never practiced or trained on" },
    { k: "retention_acc", label: "Older modules", desc: "modules released on earlier days: the forgetting check" },
    { k: "reasoning_acc", label: "Reasoning", desc: "general reasoning problems unrelated to tessel" },
    { k: "memory_tokens", label: "Doc tokens", desc: "docs and lessons pasted into each prompt", tokens: true },
  ];
  const st = { day: 10, metric: "all_acc", hidden: new Set(), playing: null, mod: modOfDay(10), base: false, ver: "v1", ex: 0, ovMetric: "all", sig: "loss" };
  $("metricPills").innerHTML = METRICS.map((m) => `<button type="button" data-k="${m.k}">${m.label}</button>`).join("");
  $("ticks").innerHTML = Array.from({ length: 10 }, (_, i) => `<span>${i + 1}</span>`).join("");
  $("legend").innerHTML = ARMS.map((a) => `<span class="it" data-a="${a}"><span class="sw" style="--c:${COL[a]}"></span>${NAME[a]}</span>`).join("");

  function drawMain() {
    const M = METRICS.find((m) => m.k === st.metric);
    const series = ARMS.map((a) => ({ label: NAME[a], color: COL[a], values: ev(a).map((e) => e[M.k]), hidden: st.hidden.has(a) }));
    const maxTok = Math.max(...ev("rag").map((e) => e.memory_tokens));
    const yMax = M.tokens ? Math.ceil(maxTok / 500) * 500 : 1;
    lineChart($("mainChart"), $("mainTip"), {
      series, yMax, active: st.day, onPick: (d) => setDay(d, true),
      ticks: M.tokens ? Array.from({ length: yMax / 500 + 1 }, (_, i) => i * 500) : [0, 0.25, 0.5, 0.75, 1],
      fmt: (v, tipMode) => (v == null ? "–" : M.tokens ? (tipMode ? `${N(v)} tok` : N(v)) : P(v)),
      xLabel: (d, long) => (long ? `Day ${d} · ${modOfDay(d)}` : `D${d}`),
    });
    document.querySelectorAll("#metricPills button").forEach((b) => b.classList.toggle("on", b.dataset.k === st.metric));
    document.querySelectorAll("#legend .it").forEach((b) => b.classList.toggle("off", st.hidden.has(b.dataset.a)));
  }
  function setDay(d, user) {
    st.day = d;
    if (user) stopPlay();
    st.mod = modOfDay(d);
    const s = $("day"); s.value = d; s.style.setProperty("--p", `${((d - 1) / 9) * 100}%`);
    document.querySelectorAll("#ticks span").forEach((t, i) => t.classList.toggle("on", i + 1 === d));
    const e = ev("sdft")[d - 1];
    $("dayTitle").textContent = `Day ${d}`;
    $("daySub").innerHTML = `<span class="mono" style="color:var(--text)">${esc(modOfDay(d))}</span> released · ${e.n_eval_tasks} held-out tasks across ${d} module${d > 1 ? "s" : ""} · ${esc(METRICS.find((m) => m.k === st.metric).desc)}`;
    $("inspDay").textContent = `day ${d}`;
    drawMain();
    drawInspector();
  }
  $("day").addEventListener("input", (e) => setDay(+e.target.value, true));
  $("metricPills").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; st.metric = b.dataset.k; setDay(st.day); };
  $("legend").onclick = (e) => { const it = e.target.closest(".it"); if (!it) return; const a = it.dataset.a; st.hidden.has(a) ? st.hidden.delete(a) : st.hidden.add(a); drawMain(); };
  function stopPlay() { if (st.playing) { clearInterval(st.playing); st.playing = null; } $("play").querySelector(".ico").textContent = "▶"; $("play").querySelector(".lbl").textContent = "Play timelapse"; }
  $("play").onclick = () => {
    if (st.playing) return stopPlay();
    if (st.day >= 10) setDay(1);
    $("play").querySelector(".ico").textContent = "❚❚"; $("play").querySelector(".lbl").textContent = "Pause";
    st.playing = setInterval(() => { if (st.day >= 10) return stopPlay(); setDay(st.day + 1); if (st.day >= 10) stopPlay(); }, 1000);
  };
  redraws.push(() => { if (!$("view-overview").hidden) drawMain(); });

  // ------------------------------------------------------------------ OVERVIEW: code inspector
  const traceOf = (arm, day, mod) => A.traces.find((t) => t.arm === arm && t.day === day && t.module === mod);
  const taskOf = (id) => A.showcase_tasks.find((t) => t.id === id);
  function drawInspector() {
    const d = st.day;
    $("modChips").innerHTML = releases.filter((r) => r.day <= d).map((r) =>
      `<button type="button" data-m="${r.module}" class="${r.module === st.mod ? "on" : ""}"><span class="d">D${r.day}</span>${esc(shortMod(r.module))}</button>`).join("");
    const arms = st.base ? ["base", "rag", "sft", "sdft"] : ["rag", "sft", "sdft"];
    const t0 = traceOf("sft", d, st.mod);
    const task = t0 && taskOf(t0.task_id);
    $("taskText").innerHTML = `<span class="lbl">Held-out task · ${esc(st.mod)} · never practiced</span>${esc(task ? task.prompt : t0 ? t0.task_id : "")}`;
    $("codeGrid").style.setProperty("--n", arms.length);
    $("codeGrid").innerHTML = arms.map((a) => {
      const t = traceOf(a, d, st.mod);
      if (!t) return "";
      const ok = t.grade.passed;
      const tok = a === "rag" ? `<span class="tok many" title="Average docs + lessons tokens per prompt on day ${d}">≈${N(ev("rag")[d - 1].memory_tokens)} doc tokens</span>`
        : a === "base" ? `<span class="tok">0 doc tokens</span>` : `<span class="tok zero">0 doc tokens</span>`;
      return `<div class="code-card"><div class="ch"><span class="nm" style="--c:${COL[a]}"><span class="dot"></span>${NAME[a]}</span>
        <span class="badges">${tok}<span class="pf ${ok ? "pass" : "fail"}">${ok ? "PASS" : "FAIL"}</span></span></div>
        <pre class="code">${hl(code(t.output))}</pre>${ok ? "" : `<div class="fb err">${esc(t.grade.feedback.slice(0, 220))}</div>`}</div>`;
    }).join("");
  }
  $("modChips").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; st.mod = b.dataset.m; drawInspector(); };
  $("showBase").onchange = (e) => { st.base = e.target.checked; drawInspector(); };

  // prompt drawer
  let drawerArm = "rag";
  function drawDrawer() {
    const arms = ["rag", "sft", "sdft", "base"];
    $("drawerSeg").innerHTML = arms.map((a) => `<button type="button" data-a="${a}" class="${a === drawerArm ? "on" : ""}">${NAME[a]}</button>`).join("");
    const t = traceOf(drawerArm, st.day, st.mod);
    const note = drawerArm === "rag" ? `RAG pays for this: the docs for every module so far (plus lessons it wrote itself) are pasted into each prompt, about ${N(ev("rag")[st.day - 1].memory_tokens)} tokens on day ${st.day}.`
      : `No docs anywhere: this is the entire prompt. Everything ${NAME[drawerArm]} knows about tessel has to be in its weights.`;
    $("drawerTitle").textContent = `Exact prompt · day ${st.day} · ${st.mod}`;
    $("drawerBody").innerHTML = t ? t.messages.map((m) => `<div class="msg"><div class="role">${esc(m.role)}</div><pre>${esc(m.content)}</pre></div>`).join("") + `<div class="msg"><div class="note">${esc(note)}</div></div>` : "";
  }
  $("openPrompt").onclick = () => { drawDrawer(); $("drawer").hidden = false; document.body.style.overflow = "hidden"; };
  $("drawerSeg").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; drawerArm = b.dataset.a; drawDrawer(); };
  const closeDrawer = () => { $("drawer").hidden = true; document.body.style.overflow = ""; };
  $("drawer").onclick = (e) => { if (e.target.closest("[data-close]")) closeDrawer(); };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("drawer").hidden) closeDrawer(); });

  // why SFT won
  const m10 = (arm, k) => mean(ev(arm).map((e) => e[k]));
  $("why").innerHTML = `
    <div class="card" style="--c:${COL.sft}"><div class="kicker">Why SFT held up</div><h3><span class="dot"></span>Same-model SFT + replay</h3>
      <div class="big">${P(means.sft)}</div><div class="fine">10-day mean · older modules ${P(m10("sft", "retention_acc"))} · unseen types ${P(m10("sft", "unseen_family_acc"))}</div>
      <ul><li>Targets are verified solutions written by the same base model reading the docs: on-distribution, not someone else's demonstrations.</li>
      <li>Each night also replays older-module practice and general reasoning, so new modules don't push out old ones.</li>
      <li>Learns straight from correct code, even on API names it starts at 0% on.</li></ul></div>
    <div class="card" style="--c:${COL.sdft}"><div class="kicker">Why SDFT trailed</div><h3><span class="dot"></span>On-policy SDFT</h3>
      <div class="big">${P(means.sdft)}</div><div class="fine">10-day mean · older modules ${P(m10("sdft", "retention_acc"))} · unseen types ${P(m10("sdft", "unseen_family_acc"))}</div>
      <ul><li>Learns from its own attempts, and on a brand-new module those attempts start out wrong: a cold start.</li>
      <li>The signal is a token-by-token pull toward the teacher along those attempts, weaker than training on correct code.</li>
      <li>Same tasks, replay and nightly schedule as SFT, so the main difference between the two arms is the learning method itself.</li></ul></div>
    <div class="why-note">Our best explanation, not separately tested. Eval sets are small (${ev("sdft")[0].n_eval_tasks}–${ev("sdft")[9].n_eval_tasks} tasks a day), so gaps of a few points are noise.</div>`;

  // ------------------------------------------------------------------ AUTOPSY
  const V = J.runs.find((r) => r.kind === "versus");
  const AU = (J.autopsy && J.autopsy.examples) || [];
  const row = (arm, ver) => V && V.rows.find((r) => r.arm === arm && r.ver === ver);
  function drawAutopsyStatic() {
    if (!V) return;
    const s1 = row("sdft", "v1"), s2 = row("sdft", "v2"), f1 = row("sft", "v1"), f2 = row("sft", "v2");
    $("autoLede").textContent = `On day 5 (${modOfDay(5)} released), v1's nightly update scrambled both trained models: garbled imports, invented names, one call repeated until the token limit. v2 changed only the update itself: lower learning rate, fewer passes, cosine decay. Same tasks, same seed, same teacher.`;
    const dl = (lab, a, b) => `<div class="delta"><div class="lab">${lab}</div><div class="vals"><span class="a">${a}</span><span class="arrow">→</span><span class="b">${b}</span></div></div>`;
    $("deltas").innerHTML =
      dl("SDFT unparseable answers on day 5", s1.day5_syntax, s2.day5_syntax) +
      dl("SDFT older-module score on day 5", P(s1.old[4]), P(s2.old[4])) +
      dl("SDFT reasoning on day 5", P(s1.reasoning[4]), P(s2.reasoning[4])) +
      dl("SFT 10-day mean, all modules", P(mean(f1.all)), P(mean(f2.all)));
    $("ovPills").innerHTML = [["all", "All modules"], ["old", "Older modules"], ["unseen", "Unseen types"], ["reasoning", "Reasoning"]]
      .map(([k, l]) => `<button type="button" data-k="${k}">${l}</button>`).join("");
  }
  function drawAutopsy() {
    if (!V) return;
    document.querySelectorAll("#verSeg button").forEach((b) => b.classList.toggle("on", b.dataset.v === st.ver));
    $("deltas").className = `deltas ${st.ver}`;
    // snippet
    const e = AU[st.ex];
    if (e) {
      const side = e[st.ver];
      $("snipTitle").innerHTML = `<span style="color:${COL[e.arm]}">${NAME[e.arm]}</span> · <span class="mono">${esc(e.task_id.split("#")[0])}</span>`;
      $("snipSub").textContent = `${e.module} task on day 5. ${st.ver === "v1" ? "v1 answer: failed to parse." : side && side.passed ? "v2 answer to the same task: passes." : "v2 answer to the same task: still fails."}`;
      $("snipIdx").textContent = `${st.ex + 1} / ${AU.length}`;
      const ok = side && side.passed;
      $("snip").innerHTML = `<div class="prompt">${esc(e.prompt.split("\n").filter(Boolean).slice(-6).join("\n"))}</div>
        <div class="code-card ${ok ? "good" : "bad"}"><div class="ch"><span class="nm" style="--c:${COL[e.arm]}"><span class="dot"></span>${st.ver} · day 5</span>
        <span class="badges"><span class="pf ${ok ? "pass" : "fail"}">${ok ? "PASS" : "FAIL"}</span></span></div>
        <pre class="code wrap">${side ? hl(side.code) : ""}</pre>${ok || !side ? "" : `<div class="fb err">${esc(side.feedback)}</div>`}</div>`;
    }
    // overlay chart
    const lab = { all: "Score on every module so far", old: "Older modules only", unseen: "Unseen task types", reasoning: "General reasoning" };
    $("ovTitle").textContent = lab[st.ovMetric];
    document.querySelectorAll("#ovPills button").forEach((b) => b.classList.toggle("on", b.dataset.k === st.ovMetric));
    const ser = [];
    for (const arm of ["sft", "sdft"]) for (const ver of ["v1", "v2"]) {
      const r = row(arm, ver);
      ser.push({ label: `${SHORT[arm]} ${ver}`, color: COL[arm], values: r[st.ovMetric], dash: ver === "v1", dim: ver !== st.ver });
    }
    lineChart($("ovChart"), $("ovTip"), { series: ser, yMax: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => P(v), band: 5,
      xLabel: (d, long) => (long ? `Day ${d} · ${modOfDay(d)}` : `D${d}`) });
    $("ovLegend").innerHTML = ser.map((s) => `<span class="it"><span class="sw ${s.dash ? "dash" : ""}" style="--c:${s.color}"></span>${esc(s.label)}</span>`).join("") +
      `<span class="it"><span class="sw" style="--c:${COL.rose};height:10px;width:10px;opacity:.5;border-radius:3px"></span>day 5</span>`;
    drawSignal();
  }
  function drawSignal() {
    const pills = `<div class="pills small" id="sigPills" style="margin:10px 0 4px"><button type="button" data-s="loss" class="${st.sig === "loss" ? "on" : ""}">Loss within each night</button><button type="button" data-s="grad" class="${st.sig === "grad" ? "on" : ""}">Gradient norm (v2)</button></div>`;
    $("sigKicker").textContent = "Inside each night's training";
    if (st.sig === "loss") {
      const rows = ["sft", "sdft"].map((arm) => ({ label: SHORT[arm], color: COL[arm], pairs: row(arm, st.ver).loss }));
      const rose = rows.map((r) => { const ns = r.pairs.map((p, i) => (rose_(p[0], p[1]) ? i + 1 : 0)).filter(Boolean); return ns.length ? `${r.label} ${ns.join(", ")}` : null; }).filter(Boolean);
      $("sigTitle").textContent = `${st.ver}: loss at the start → end of each night`;
      $("sigSub").innerHTML = pills + `Each bar runs from the loss at the start of a night (hollow dot) to the end (solid). It should fall. <b style="color:var(--rose)">Red</b>: it rose${rose.length ? ` (${esc(rose.join("; "))})` : ""}. Log scale.`;
      dumbbell($("sigChart"), $("sigTip"), { rows, lo: 0.001, hi: 2, ticks: [0.001, 0.01, 0.1, 1], band: 5 });
      $("sigLegend").innerHTML = rows.map((r) => `<span class="it"><span class="sw" style="--c:${r.color}"></span>${esc(r.label)}</span>`).join("") +
        `<span class="it"><span class="sw" style="--c:${COL.rose}"></span>loss rose during the night</span>`;
    } else {
      $("sigTitle").textContent = "v2: gradient norm per night";
      $("sigSub").innerHTML = pills + `Largest single step (solid) and average (dashed), before clipping at 1.0. v1 didn't log gradient norms.`;
      const ser = [];
      for (const arm of ["sft", "sdft"]) {
        const r = row(arm, "v2");
        ser.push({ label: `${SHORT[arm]} largest`, color: COL[arm], values: r.grad_max });
        ser.push({ label: `${SHORT[arm]} average`, color: COL[arm], values: r.grad_mean, dash: true });
      }
      const top = Math.ceil(Math.max(...ser.flatMap((s) => s.values.filter((v) => v != null))));
      lineChart($("sigChart"), $("sigTip"), { series: ser, yMax: top, ticks: Array.from({ length: top + 1 }, (_, i) => i),
        fmt: (v, tip) => (v == null ? "–" : tip ? v.toFixed(2) : String(v)), xLabel: (d, long) => (long ? `Night ${d}` : `N${d}`) });
      $("sigLegend").innerHTML = ser.map((s) => `<span class="it"><span class="sw ${s.dash ? "dash" : ""}" style="--c:${s.color}"></span>${esc(s.label)}</span>`).join("");
    }
  }
  $("verSeg").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; st.ver = b.dataset.v; drawAutopsy(); };
  $("snipPrev").onclick = () => { st.ex = (st.ex - 1 + AU.length) % AU.length; drawAutopsy(); };
  $("snipNext").onclick = () => { st.ex = (st.ex + 1) % AU.length; drawAutopsy(); };
  $("ovPills").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; st.ovMetric = b.dataset.k; drawAutopsy(); };
  $("sigSub").onclick = (e) => { const b = e.target.closest("button[data-s]"); if (!b) return; st.sig = b.dataset.s; drawSignal(); };
  redraws.push(() => { if (!$("view-autopsy").hidden) drawAutopsy(); });

  // ------------------------------------------------------------------ LAB LOG
  const bars = (rows) => `<div class="bars">${rows.map((r) => {
    const p = r.frac && r.frac.n ? Math.round((100 * r.frac.k) / r.frac.n) : r.pct ?? 0;
    return `<div class="bar" style="--c:${r.color}"><div class="bl"><b>${esc(r.label)}</b><span>${esc(r.sub || "")}</span></div>
      <div class="bt"><div class="bf" style="width:${p}%"></div></div><div class="bv">${p}% ${r.frac ? `<span>${r.frac.k}/${r.frac.n}</span>` : ""}</div></div>`;
  }).join("")}</div>`;
  const heat = (rows, n = 10) => `<div class="heat"><table><tr><th></th>${Array.from({ length: n }, (_, i) => `<th>D${i + 1}</th>`).join("")}<th>mean</th></tr>${rows.map((r) =>
    `<tr><th style="color:${r.color}">${esc(r.label)}</th>${r.values.map((v) => `<td style="background:${r.color}${Math.round(20 + (v ?? 0) * 150).toString(16).padStart(2, "0")}">${P(v)}</td>`).join("")}<td><b>${P(mean(r.values))}</b></td></tr>`).join("")}</table></div>`;
  const lastDay = (run, arm) => run.arms[arm] && run.arms[arm][run.arms[arm].length - 1];
  const takeaway = (t) => (t ? `<div class="takeaway"><b>Takeaway.</b> ${esc(t)}</div>` : "");
  function body(r) {
    if (r.kind === "compare") {
      const col = (f) => (f.k === f.n ? COL.sft : f.k ? COL.sdft : COL.rose);
      return `<div class="h4">Student after one night, no docs: all held-out tasks</div>${bars(r.rows.map((x) => ({ label: x.label, sub: `${x.opt_steps} optimizer steps · reasoning ${P(x.reasoning)}`, frac: x.acc, color: col(x.acc) })))}
        <div class="h4">Of those, the task type it never practiced</div>${bars(r.rows.map((x) => ({ label: x.label, sub: "never practiced or trained on", frac: x.unseen, color: col(x.unseen) })))}
        <div class="h4">The unseen task, solved closed-book</div><div class="ba" style="--n:${r.rows.length}">${r.rows.map((x) => `<div class="code-card"><div class="ch"><span class="nm">${esc(x.label)}</span><span class="pf ${x.unseen_passed ? "pass" : "fail"}">${x.unseen_passed ? "PASS" : "FAIL"}</span></div><pre class="code">${hl(x.unseen_code || "")}</pre>${x.unseen_passed ? "" : `<div class="fb err">${esc(x.unseen_result)}</div>`}</div>`).join("")}</div>`;
    }
    if (r.kind === "full") {
      const rows = ARMS.filter((a) => r.arms[a]).map((a) => ({ label: SHORT[a], color: COL[a], values: r.arms[a].map((d) => d.all_acc) }));
      const d10 = (k) => bars(ARMS.filter((a) => r.arms[a]).map((a) => ({ label: SHORT[a], sub: "", pct: Math.round(100 * (lastDay(r, a)[k] ?? 0)), color: COL[a] })));
      const cfg = r.arm_config ? Object.entries(r.arm_config).filter(([a]) => a === "sft" || a === "sdft").map(([a, c]) => `${a}: lr ${c.lr} × ${c.epochs}${c.lr_schedule ? ` ${c.lr_schedule}` : ""}`).join(" · ") : "";
      return `<div class="h4">All modules so far, day 1 → 10</div>${heat(rows)}
        <div class="h4">Day 10: older modules only</div>${d10("retention_acc")}
        <div class="h4">Day 10: never-practiced task types</div>${d10("unseen_family_acc")}
        ${cfg ? `<p class="fine" style="margin-top:14px">Nightly update per arm: ${esc(cfg)}</p>` : ""}`;
    }
    if (r.kind === "versus") {
      const rows = (k) => r.rows.map((x) => ({ label: `${x.arm} ${x.ver}`, color: COL[x.arm], values: x[k] }));
      return `<div class="h4">All modules so far</div>${heat(rows("all"))}<div class="h4">Older modules only</div>${heat(rows("old"))}
        <div class="h4">Day 5</div><div class="heat"><table><tr><th></th><th>older modules</th><th>reasoning</th><th>unparseable</th><th>nights flagged</th></tr>${r.rows.map((x) =>
          `<tr><th style="color:${COL[x.arm]}">${x.arm} ${x.ver}</th><td>${P(x.old[4])}</td><td>${P(x.reasoning[4])}</td><td>${x.day5_syntax}</td><td>${x.flagged}/10</td></tr>`).join("")}</table></div>`;
    }
    // single-day run
    const base = lastDay(r, "base"), rag = lastDay(r, "rag"), s = lastDay(r, "sdft");
    const days = r.arms.sdft ? r.arms.sdft.length : 1;
    let h = bars([base && { label: "base", sub: "no docs, no training", frac: base.acc, color: COL.base }, rag && { label: "rag", sub: "docs in prompt", frac: rag.acc, color: COL.rag },
      s && s.teacher && { label: "teacher", sub: "frozen model + docs", frac: s.teacher, color: "#C792EA" },
      s && { label: "sdft student", sub: `no docs, after ${days} night${days > 1 ? "s" : ""}`, frac: s.acc, color: COL.sdft }].filter(Boolean));
    const types = (r.teacher_by_type || []).filter((t) => t.n);
    if (types.some((t) => t.pass < t.n) && types.some((t) => t.pass > 0)) {
      h += `<div class="h4">Teacher by task type</div>${bars(types.map((t) => ({ label: t.type.replace("run.", ""), frac: { k: t.pass, n: t.n }, color: t.pass === t.n ? COL.sft : t.pass ? COL.sdft : COL.rose })))}`;
      if (r.teacher_fail_example) h += `<div class="h4">The one mistake, every time</div><div class="code-card"><pre class="code">${hl(r.teacher_fail_example.code)}</pre><div class="fb err">${esc(r.teacher_fail_example.error)}</div></div>`;
    }
    if (r.before_after && r.id === "20261125-215708-preflight") {
      const b = r.before_after;
      h += `<div class="h4">Same held-out task, before and after night 1</div><div class="ba"><div class="code-card"><div class="ch"><span class="nm">before (base)</span><span class="pf fail">FAIL</span></div><pre class="code">${hl(b.before)}</pre><div class="fb err">${esc(b.before_error)}</div></div>
        <div class="code-card"><div class="ch"><span class="nm">after night 1 (sdft)</span><span class="pf fail">FAIL</span></div><pre class="code">${hl(b.after)}</pre><div class="fb err">${esc(b.after_error)}</div></div></div>`;
    }
    return h;
  }
  $("acc").innerHTML = J.runs.map((r, i) => `<details class="entry"><summary><span class="no">${String(i + 1).padStart(2, "0")}</span>
      <div><div class="tt">${esc(r.title)}</div><div class="mm">${esc(r.model)} · ${esc(r.hardware)}</div></div><span class="chev">›</span></summary>
      <div class="bd"><p>${esc(r.what)}</p>${body(r)}${takeaway(r.lesson)}</div></details>`).join("");

  // ------------------------------------------------------------------ routing
  const VIEWS = ["overview", "autopsy", "lab"];
  let autopsyReady = false;
  function route() {
    const v = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "overview";
    VIEWS.forEach((k) => { $(`view-${k}`).hidden = k !== v; });
    document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("on", a.dataset.tab === v));
    if (v !== "overview") stopPlay();
    if (v === "overview") setDay(st.day);
    if (v === "autopsy") { if (!autopsyReady) { drawAutopsyStatic(); autopsyReady = true; } drawAutopsy(); }
  }
  window.addEventListener("hashchange", () => { route(); window.scrollTo({ top: 0 }); });
  route();
})();
