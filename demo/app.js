const ARMS = ["base", "rag", "sft", "sdft"];
const COLOR = { base: "var(--base)", rag: "var(--rag)", sft: "var(--sft)", sdft: "var(--sdft)" };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);

let A, armsPresent, day;

fetch("run_artifacts.json").then((r) => r.json()).then((data) => {
  A = data;
  armsPresent = ARMS.filter((a) => A.arms[a]);
  const maxDay = Math.max(...armsPresent.map((a) => A.arms[a].days.length));
  $("day").max = maxDay;
  $("day").value = maxDay;
  if (A.meta.placeholder) {
    $("banner").innerHTML = `<div class="banner">DRY-RUN PLACEHOLDER: these numbers come from a scripted stub, not a trained model.
      ${esc(A.meta.placeholder_reason)}</div>`;
  }
  $("meta").textContent = `model: ${A.meta.model} · engine: ${A.meta.engine} · generated ${A.meta.created}` +
    (A.meta.approx_tokens ? " · token counts approximate" : "");
  $("legend").innerHTML = armsPresent.map((a) => `<span><span class="dot" style="background:${COLOR[a]}"></span> ${a}</span>`).join("");
  $("day").addEventListener("input", render);
  $("module").addEventListener("change", renderCompare);
  render();
});

function ev(arm, d) { return A.arms[arm]?.days[d - 1]?.eval; }

function render() {
  day = +$("day").value;
  const rel = A.meta.releases.find((r) => r.day === day);
  $("dayLabel").textContent = `Day ${day}`;
  $("dayNote").textContent = `${rel.module} ships today. ${day} module${day > 1 ? "s" : ""} released so far.`;
  renderCards();
  charts();
  const sel = $("module"), prev = sel.value;
  sel.innerHTML = A.meta.releases.filter((r) => r.day <= day).map((r) => `<option>${r.module}</option>`).join("");
  sel.value = A.meta.releases.some((r) => r.module === prev && r.day <= day) ? prev : rel.module;
  renderCompare();
  renderLessons();
  renderNights();
}

function renderCards() {
  $("cards").innerHTML = armsPresent.map((a) => {
    const e = ev(a, day);
    const n = A.arms[a].days[day - 1]?.night;
    const tok = e.memory_tokens;
    return `<div class="panel">
      <div class="arm"><span class="dot" style="background:${COLOR[a]}"></span>${a}
        ${n ? `<span class="badge ${n.skipped ? "" : n.accepted ? "pass" : "fail"}" style="margin-left:auto">${n.skipped ? "no update" : n.accepted ? "update kept" : "rolled back"}</span>` : ""}</div>
      <div class="kpis">
        <div class="kpi"><div class="v">${pct(e.unseen_family_acc)}</div><div class="l">unseen task types</div></div>
        <div class="kpi"><div class="v">${pct(e.all_acc)}</div><div class="l">all held-out tasks</div></div>
        <div class="kpi"><div class="v">${pct(e.new_module_acc)}</div><div class="l">today's module</div></div>
        <div class="kpi"><div class="v">${pct(e.retention_acc)}</div><div class="l">earlier modules</div></div>
        <div class="kpi"><div class="v">${pct(e.reasoning_acc)}</div><div class="l">reasoning</div></div>
        <div class="kpi" style="grid-column:span 2"><div class="v" style="color:${tok > 0 ? "var(--rag)" : "var(--sdft)"}">${Math.round(tok)}</div>
          <div class="l">memory tokens per prompt</div></div>
      </div></div>`;
  }).join("");
}

function lineChart(el, key, { max = 1, fmt = pct } = {}) {
  const W = 360, H = 170, L = 34, R = 8, T = 10, B = 22;
  const n = Math.max(...armsPresent.map((a) => A.arms[a].days.length));
  if (key === "tokens") max = Math.max(1, ...armsPresent.flatMap((a) => A.arms[a].days.map((d) => d.eval.memory_tokens)));
  const x = (d) => L + ((d - 1) / Math.max(1, n - 1)) * (W - L - R);
  const y = (v) => T + (1 - v / max) * (H - T - B);
  let s = `<svg viewBox="0 0 ${W} ${H}">`;
  for (const g of [0, 0.5, 1]) {
    s += `<line x1="${L}" x2="${W - R}" y1="${y(g * max)}" y2="${y(g * max)}" stroke="var(--line)"/>` +
         `<text x="${L - 6}" y="${y(g * max) + 4}" fill="var(--muted)" font-size="10" text-anchor="end" font-family="var(--mono)">${key === "tokens" ? Math.round(g * max) : fmt(g * max)}</text>`;
  }
  for (let d = 1; d <= n; d++) s += `<text x="${x(d)}" y="${H - 6}" fill="var(--muted)" font-size="10" text-anchor="middle" font-family="var(--mono)">${d}</text>`;
  s += `<line x1="${x(day)}" x2="${x(day)}" y1="${T}" y2="${H - B}" stroke="var(--muted)" stroke-dasharray="3 3" opacity=".6"/>`;
  armsPresent.forEach((a, i) => {
    const pts = A.arms[a].days.map((d, j) => {
      const v = key === "tokens" ? d.eval.memory_tokens : d.eval[key];
      return v == null ? null : [x(j + 1), y(v) + (i - 1.5) * 1.2];
    }).filter(Boolean);
    if (!pts.length) return;
    s += `<polyline fill="none" stroke="${COLOR[a]}" stroke-width="2" points="${pts.map((p) => p.join(",")).join(" ")}"/>`;
    pts.forEach((p) => (s += `<circle cx="${p[0]}" cy="${p[1]}" r="2.4" fill="${COLOR[a]}"/>`));
  });
  el.innerHTML = s + "</svg>";
}

function charts() {
  lineChart($("c1"), "unseen_family_acc");
  lineChart($("c2"), "retention_acc");
  lineChart($("c3"), "reasoning_acc");
  lineChart($("c4"), "tokens");
  lineChart($("c5"), "new_module_acc");
  lineChart($("c6"), "day1_module_acc");
}

function renderCompare() {
  const mod = $("module").value;
  const task = A.showcase_tasks.find((t) => t.module === mod);
  $("taskText").innerHTML = task ? `<b>Task:</b> ${esc(task.prompt)} <span class="sub">(${esc(task.id)}, ${task.tests.length} hidden tests)</span>` : "";
  $("compare").innerHTML = armsPresent.map((a) => {
    const tr = A.traces.find((t) => t.arm === a && t.day === day && t.module === mod && t.phase === "eval");
    if (!tr) return `<div class="panel"><div class="arm"><span class="dot" style="background:${COLOR[a]}"></span>${a}</div><div class="sub">no trace</div></div>`;
    const e = ev(a, day);
    const mem = Math.round(e.memory_tokens);
    const id = `raw-${a}`;
    return `<div class="panel">
      <div class="arm"><span class="dot" style="background:${COLOR[a]}"></span>${a}
        <span class="badge ${tr.grade.passed ? "pass" : "fail"}" style="margin-left:auto">${tr.grade.passed ? "PASS" : "FAIL"} ${tr.grade.n_pass}/${tr.grade.n_total}</span></div>
      <div style="margin-top:8px"><span class="badge ${mem > 0 ? "tok" : "zero"}">${mem > 0 ? `+${mem} memory tokens` : "0 memory tokens"}</span></div>
      ${tr.grade.passed ? "" : `<div class="sub" style="margin-top:8px;color:var(--bad)">${esc(tr.grade.feedback)}</div>`}
      <pre>${esc(tr.output)}</pre>
      <button class="link" onclick="const p=document.getElementById('${id}');p.hidden=!p.hidden">View raw prompt &amp; trace</button>
      <pre id="${id}" hidden>${esc(tr.messages.map((m) => `<${m.role}>\n${m.content}`).join("\n\n"))}</pre>
    </div>`;
  }).join("");
}

function renderLessons() {
  const arm = A.arms.sdft ? "sdft" : armsPresent.find((a) => A.arms[a].lessons?.length);
  const ls = (A.arms[arm]?.lessons || []).filter((l) => l.day <= day).sort((a, b) => b.priority - a.priority).slice(0, 25);
  $("lessons").innerHTML = `<tr><th>module</th><th>lesson (written by the model)</th><th>verified</th><th>surprise</th><th>recurs</th><th>priority</th><th>into weights?</th><th>sources</th></tr>` +
    (ls.length ? ls.map((l) => `<tr><td class="m">${esc(l.module)}</td><td>${esc(l.text)}</td>
      <td class="m">${l.verified ? "✓" : "✗"}</td><td class="m">${l.surprise.toFixed(2)}</td><td class="m">${l.recurrence}</td>
      <td class="m">${l.priority.toFixed(2)}</td><td class="m" style="color:${l.promoted ? "var(--sdft)" : "var(--muted)"}">${l.promoted ? "promoted" : "memory only"}</td>
      <td class="m">${l.sources.length} task(s)</td></tr>`).join("") : `<tr><td colspan="8" class="sub">No lessons yet.</td></tr>`);
}

function renderNights() {
  const rows = [];
  for (const a of ["sft", "sdft"].filter((a) => A.arms[a])) {
    A.arms[a].days.slice(0, day).forEach((d, i) => {
      const n = d.night, p = d.practice;
      if (!n) return;
      const loss = n.train.loss_mean == null ? "–" : n.train.loss_mean.toFixed(4);
      rows.push(`<tr><td class="m">${a}</td><td class="m">${i + 1}</td><td class="m">${pct(p.practice_pass_rate)} of ${p.n_practice}</td>
        <td class="m">${p.authored.filter((x) => x.accepted).length}/${p.authored.length}</td>
        <td class="m">${pct(n.teacher_pass_rate)}</td><td class="m">${n.train_items}</td><td class="m">${loss}</td>
        <td class="m">${n.gate_after ? pct(n.gate_after.reasoning) : "–"}</td>
        <td class="m" style="color:${n.skipped ? "var(--muted)" : n.accepted ? "var(--sdft)" : "var(--bad)"}">${n.skipped ? "skipped (nothing verified)" : n.accepted ? "kept" : "ROLLED BACK: " + esc(n.rejected_because.join("; "))}</td></tr>`);
    });
  }
  $("nights").innerHTML = `<tr><th>arm</th><th>day</th><th>practice pass</th><th>self-made tasks</th><th>teacher pass</th><th>train items</th><th>loss</th><th>gate: reasoning</th><th>update</th></tr>` + rows.join("");
}
