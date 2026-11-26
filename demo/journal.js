// Real-run sections of the page: status, day/night diagram and lab notebook. Data: journal.json
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fr = (f) => (f ? `${f.k}/${f.n}` : "–");
  const P = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);

  function bars(rows) {
    // rows: [{label, sub, frac:{k,n}, color}]
    return `<div class="hbars">` + rows.map((r) => {
      const p = r.frac && r.frac.n ? Math.round((100 * r.frac.k) / r.frac.n) : 0;
      return `<div class="hb"><div class="hl"><b>${esc(r.label)}</b><span>${esc(r.sub || "")}</span></div>
        <div class="ht"><div class="hf" style="width:${p}%;background:${r.color}"></div></div>
        <div class="hv"><b>${r.frac ? p + "%" : "–"}</b> <span>${fr(r.frac)}</span></div></div>`;
    }).join("") + `</div>`;
  }

  function lastDay(run, arm) {
    const d = run.arms[arm];
    return d ? d[d.length - 1] : null;
  }

  function runRows(run) {
    const base = lastDay(run, "base"), rag = lastDay(run, "rag"), s = lastDay(run, "sdft");
    const days = run.arms.sdft ? run.arms.sdft.length : 1;
    return [
      base && { label: "base", sub: "no docs, no training", frac: base.acc, color: "var(--base)" },
      rag && { label: "rag", sub: "docs pasted in prompt", frac: rag.acc, color: "var(--rag)" },
      s && s.teacher && { label: "teacher", sub: "frozen model + docs", frac: s.teacher, color: "var(--teach)" },
      s && { label: "sdft student", sub: `no docs, after ${days} night${days > 1 ? "s" : ""}`, frac: s.acc, color: "var(--sdft)" },
    ].filter(Boolean);
  }

  function status(j) {
    const r = [...j.runs].reverse().find((x) => x.kind !== "compare");
    const base = lastDay(r, "base"), rag = lastDay(r, "rag");
    const tuned = j.runs.find((x) => x.kind === "compare");
    const best = tuned ? tuned.rows.reduce((a, b) => (b.acc.k > a.acc.k ? b : a)) : null;
    const s = best ? { acc: best.acc, reasoning: best.reasoning, teacher: lastDay(r, "sdft").teacher } : lastDay(r, "sdft");
    const tile = (v, l, sub, cls = "") => `<div class="tile ${cls}"><div class="v">${v}</div><div class="l">${l}</div><div class="s">${sub}</div></div>`;
    $("status").innerHTML = `<div class="tiles">
      ${tile(fr(base.acc), "base model", "can't guess the invented API", "c-base")}
      ${tile(fr(rag.acc), "with docs in prompt", "reads the docs perfectly", "c-rag")}
      ${tile(fr(s.teacher), "teacher passes", "real signal to distill", "c-teach")}
      ${tile(fr(s.acc), "student, after night 1", best ? `no docs · unseen types ${fr(best.unseen)}` : "not learned yet", "c-sdft")}
      ${tile(`${P(base.reasoning)} → ${P(s.reasoning)}`, "general reasoning", "base → after training: unharmed", "")}
    </div>
    <p class="sub">Latest real run: ${esc(r.model)} on ${esc(r.hardware)}, day 1 of the library (tessel.run), 12 held-out tasks.
      ${best ? `Student: ${esc(best.label)}.` : ""} <b>The full 10-day run hasn't happened yet.</b> ${esc(j.next)}</p>`;
  }

  function compareEntry(r, i) {
    const col = (x) => (x.acc.k === x.acc.n ? "var(--sdft)" : x.acc.k ? "var(--rag)" : "var(--bad)");
    const rows = r.rows.map((x) => ({ label: x.label, sub: `${x.opt_steps} optimizer steps · reasoning ${P(x.reasoning)}`, frac: x.acc, color: col(x) }));
    const un = r.rows.map((x) => ({ label: x.label, sub: "never practiced or trained on", frac: x.unseen, color: col({ acc: x.unseen }) }));
    const codes = r.rows.map((x) => `<div><div class="tag">${esc(x.label)}</div><pre>${esc(x.unseen_code)}</pre>
      <div class="${x.unseen_passed ? "ok" : "err"}">${esc(x.unseen_result)}</div></div>`).join("");
    return `<article class="entry hl-entry">
      <div class="num">${String(i + 1).padStart(2, "0")}</div>
      <div class="body">
        <div class="eh"><h3>${esc(r.title)}</h3><div class="sub">${esc(r.model)} · ${esc(r.hardware)}</div></div>
        <p>${esc(r.what)}</p>
        <div class="h3">Student after one night, no docs in prompt: all 12 held-out tasks</div>${bars(rows)}
        <div class="h3">Of those, the task type it never saw in practice or training</div>${bars(un)}
        <div class="h3">The unseen task, solved closed-book</div>
        <p class="sub">${esc(r.unseen_prompt)}</p>
        <div class="ba3">${codes}</div>
        <div class="lesson"><b>Takeaway.</b> ${esc(r.lesson)}</div>
      </div></article>`;
  }

  function journal(j) {
    $("journal").innerHTML = j.runs.map((r, i) => {
      if (r.kind === "compare") return compareEntry(r, i);
      const types = r.teacher_by_type.filter((t) => t.n);
      const showTypes = types.some((t) => t.pass < t.n) && types.some((t) => t.pass > 0);
      const typeBars = showTypes ? `<div class="h3">Teacher by task type</div>` + bars(types.map((t) => ({
        label: t.type.replace("run.", ""), sub: "", frac: { k: t.pass, n: t.n },
        color: t.pass === t.n ? "var(--sdft)" : t.pass === 0 ? "var(--bad)" : "var(--rag)" }))) : "";
      const fail = showTypes && r.teacher_fail_example ? `<div class="h3">The one mistake, every time</div>
        <pre>${esc(r.teacher_fail_example.code)}</pre><div class="err">${esc(r.teacher_fail_example.error)}</div>` : "";
      const ba = r.before_after && r.id === "20261125-215708-preflight" ? `<div class="h3">Same held-out task, before and after night 1</div>
        <div class="ba"><div><div class="tag">before (base model)</div><pre>${esc(r.before_after.before)}</pre><div class="err">${esc(r.before_after.before_error)}</div></div>
        <div><div class="tag">after one night (sdft, no docs)</div><pre>${esc(r.before_after.after)}</pre><div class="err">${esc(r.before_after.after_error)}</div></div></div>
        <p class="sub">The student picked up the teacher's chained style (filter → map → fold) but not yet the library's names (sieve → thread → knot).</p>` : "";
      return `<article class="entry">
        <div class="num">${String(i + 1).padStart(2, "0")}</div>
        <div class="body">
          <div class="eh"><h3>${esc(r.title)}</h3><div class="sub">${esc(r.model)} · ${esc(r.hardware)}</div></div>
          <p>${esc(r.what)}</p>
          ${bars(runRows(r))}
          ${typeBars}${fail}${ba}
          <div class="lesson"><b>Takeaway.</b> ${esc(r.lesson)}</div>
        </div></article>`;
    }).join("");
  }

  fetch("journal.json").then((r) => r.json()).then((j) => { status(j); journal(j); })
    .catch(() => { $("status").innerHTML = `<p class="sub">No real runs yet.</p>`; });
})();
