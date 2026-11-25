// TAAT vs DAAT step-by-step simulation on real postings from /api/trace.
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const K = 3; // top-k kept by DAAT's heap
  const el = (tag, props = {}, ...kids) => {
    const n = Object.assign(document.createElement(tag), props);
    kids.flat().forEach(k => k != null && k !== false && n.append(k));
    return n;
  };
  const f2 = x => x.toFixed(2);

  let data = null, taat = [], daat = [], step = 0, timer = null;
  // read before app.js rewrites the URL with search params
  let initialStep = +new URLSearchParams(location.search).get("simstep") || 0;

  // ---------------- precompute both algorithms ----------------
  function buildTaat(lists) {
    const steps = [];
    const acc = new Map();
    lists.forEach((L, i) => L.postings.forEach((p, j) => {
      const delta = p.tf * L.idf;
      const isNew = !acc.has(p.doc);
      acc.set(p.doc, (acc.get(p.doc) || 0) + delta);
      steps.push({ i, j, doc: p.doc, tf: p.tf, delta, isNew, total: acc.get(p.doc),
                   acc: new Map(acc), read: steps.length + 1 });
    }));
    return steps;
  }

  function buildDaat(lists) {
    const steps = [];
    const ptr = lists.map(() => 0);
    const heap = []; // [{doc, score}], kept sorted desc, size <= K
    let read = 0, cmps = 0;
    for (;;) {
      const active = lists.map((L, i) => i).filter(i => ptr[i] < lists[i].postings.length);
      if (!active.length) break;
      cmps += active.length; // find the smallest doc under the pointers
      const doc = Math.min(...active.map(i => lists[i].postings[ptr[i]].doc));
      const before = [...ptr];
      const matched = [];
      let score = 0;
      for (const i of active) {
        const p = lists[i].postings[ptr[i]];
        cmps += 1; // compare head with the current doc
        if (p.doc === doc) {
          matched.push(i);
          score += p.tf * lists[i].idf;
          ptr[i]++; read++;
        }
      }
      let verdict;
      if (heap.length < K) { heap.push({ doc, score }); verdict = "added"; }
      else if (score > heap[heap.length - 1].score) {
        verdict = `replaced d${heap[heap.length - 1].doc}`;
        heap[heap.length - 1] = { doc, score };
      } else verdict = "dropped";
      heap.sort((a, b) => b.score - a.score);
      steps.push({ doc, before, after: [...ptr], matched, score, verdict,
                   heap: heap.map(h => ({ ...h })), read, cmps });
    }
    return steps;
  }

  // ---------------- rendering ----------------
  const title = d => data.docs[d] || `doc ${d}`;
  const docChip = d => el("span", { className: "dchip", title: title(d), textContent: `d${d}` });

  function listsView(cellClass) {
    return el("div", { className: "plists" }, data.lists.map((L, i) =>
      el("div", { className: "prow" },
        el("div", { className: "pterm" },
          el("b", { textContent: L.term }),
          el("small", { textContent: `idf ${f2(L.idf)} · df ${L.df}` })),
        el("div", { className: "pcells" }, L.postings.map((p, j) =>
          el("span", { className: `cell ${cellClass(i, j)}`, title: `${title(p.doc)} · tf=${p.tf}` },
            el("b", { textContent: `d${p.doc}` }), el("small", { textContent: `tf ${p.tf}` })))))));
  }

  function counters(items) {
    return el("div", { className: "pcount" }, items.map(([k, v, hint]) =>
      el("div", { title: hint || "" }, el("span", { textContent: k }), el("b", { textContent: v }))));
  }

  function renderTaat() {
    const n = Math.min(step, taat.length);
    const s = n ? taat[n - 1] : null;
    const flatIdx = (i, j) => data.lists.slice(0, i).reduce((a, L) => a + L.postings.length, 0) + j;
    const cls = (i, j) => {
      const k = flatIdx(i, j);
      return k === n - 1 ? "cur" : k < n - 1 ? "done" : "";
    };
    const acc = s ? [...s.acc.entries()].sort((a, b) => b[1] - a[1]) : [];
    const finished = n === taat.length;
    const termIdx = s ? s.i : 0;

    const narr = !s
      ? `Ready. TAAT walks one postings list completely before starting the next, adding tf × idf into a score accumulator per document.`
      : `List "${data.lists[s.i].term}": d${s.doc} has tf ${s.tf} → ${s.isNew ? "new accumulator" : "acc"}[d${s.doc}] += ${s.tf} × ${f2(data.lists[s.i].idf)} = ${f2(s.total)}` +
        (finished ? `. All lists done: sort ${acc.length} accumulators and take the top ${K}.` : "");

    return el("div", { className: "panel" },
      el("div", { className: "phead" }, el("h3", { textContent: "TAAT · term-at-a-time" }),
        el("span", { className: "pstatus", textContent: finished ? "done" : s ? `term ${termIdx + 1}/${data.lists.length}` : "" })),
      counters([
        ["postings read", s ? s.read : 0],
        ["accumulators in memory", acc.length, "TAAT keeps a partial score for every doc seen so far"],
        ["comparisons", 0, "TAAT never compares doc IDs across lists: it just scans each list sequentially"],
      ]),
      listsView(cls),
      el("div", { className: "pstate" },
        el("div", { className: "plabel", textContent: finished ? `Accumulators → top ${K}` : "Score accumulators" }),
        acc.length ? el("div", { className: "accs" }, acc.map(([d, v], r) =>
          el("div", { className: `acc ${s && d === s.doc ? "hot" : ""} ${finished && r < K ? "win" : ""}` },
            docChip(d), el("span", { className: "t", textContent: title(d) }), el("b", { textContent: f2(v) }))))
          : el("div", { className: "muted small", textContent: "empty" })),
      el("p", { className: "narr", textContent: narr }));
  }

  function renderDaat() {
    // shared clock = postings read, so neither algorithm looks faster just by step granularity
    const n = daat.filter(d => d.read <= step).length;
    const s = n ? daat[n - 1] : null;
    const ptr = s ? s.after : data.lists.map(() => 0);
    const cls = (i, j) => {
      if (s && s.matched.includes(i) && j === s.before[i]) return "cur";
      if (j < ptr[i]) return "done";
      if (j === ptr[i]) return "ptr";
      return "";
    };
    const finished = n === daat.length;
    const narr = !s
      ? `Ready. DAAT keeps one pointer per list (▼), always scores the smallest doc under the pointers completely, then keeps only the best ${K} in a heap.`
      : `Smallest doc under the pointers is d${s.doc}; found in ${s.matched.map(i => `"${data.lists[i].term}"`).join(" + ")} → score ${f2(s.score)} → ${s.verdict}` +
        (finished ? `. All pointers exhausted: the heap already holds the top ${K}.` : "");

    return el("div", { className: "panel" },
      el("div", { className: "phead" }, el("h3", { textContent: "DAAT · document-at-a-time" }),
        el("span", { className: "pstatus", textContent: finished ? "done" : s ? `doc ${n}` : "" })),
      counters([
        ["postings read", s ? s.read : 0],
        ["scores in memory", s ? s.heap.length : 0, `DAAT only keeps a heap of the best ${K}`],
        ["comparisons", s ? s.cmps : 0, "finding the min doc across list heads + matching each head: interpreter overhead in Python"],
      ]),
      listsView(cls),
      el("div", { className: "pstate" },
        el("div", { className: "plabel", textContent: `Top-${K} heap` }),
        s && s.heap.length ? el("div", { className: "accs" }, s.heap.map(h =>
          el("div", { className: `acc ${h.doc === s.doc ? "hot" : ""} ${finished ? "win" : ""}` },
            docChip(h.doc), el("span", { className: "t", textContent: title(h.doc) }), el("b", { textContent: f2(h.score) }))))
          : el("div", { className: "muted small", textContent: "empty" })),
      el("p", { className: "narr", textContent: narr }));
  }

  function render() {
    const max = taat.length;
    $("#simStage").replaceChildren(renderTaat(), renderDaat());
    $("#simStep").textContent = `postings read ${step} / ${max}`;
    $("#simBack").disabled = step === 0;
    $("#simNext").disabled = step >= max;
    $("#simPlay").textContent = timer ? "❚❚ Pause" : step >= max ? "↺ Replay" : "▶ Play";
    if (step >= max) {
      const top = arr => arr.slice(0, K).map(x => x.doc ?? x[0]).join(",");
      const tTop = [...taat[taat.length - 1].acc.entries()].sort((a, b) => b[1] - a[1]);
      const same = top(tTop) === top(daat[daat.length - 1].heap);
      const d = daat[daat.length - 1];
      $("#simVerdict").hidden = false;
      $("#simVerdict").replaceChildren(
        el("b", { textContent: same ? "Same top-3, different work. " : "Done. " }),
        `Both read ${d.read} postings. TAAT needed ${tTop.length} accumulators but zero cross-list comparisons; ` +
        `DAAT held only ${K} scores but made ${d.cmps} comparisons juggling pointers. ` +
        `In Python every one of those comparisons is interpreted bytecode, which is why TAAT won the full benchmark (9.47 ms vs 16.12 ms P95). ` +
        `In C++/Java, where comparisons are cheap and memory is the bottleneck, DAAT usually wins.`);
    } else $("#simVerdict").hidden = true;
  }

  // ---------------- controls ----------------
  const stop = () => { clearInterval(timer); timer = null; };
  function play() {
    const max = taat.length;
    if (timer) { stop(); render(); return; }
    if (step >= max) step = 0;
    timer = setInterval(() => {
      step++;
      if (step >= max) stop();
      render();
    }, +$("#simSpeed").value);
    render();
  }

  async function load(q) {
    stop();
    $("#simStage").replaceChildren(el("p", { className: "muted", textContent: "Loading postings…" }));
    const r = await fetch(`/api/trace?q=${encodeURIComponent(q)}`);
    data = await r.json();
    if (!data.lists.length) {
      $("#simStage").replaceChildren(el("p", { className: "muted", textContent: "None of those terms are in the index. Try another query." }));
      $("#simVerdict").hidden = true;
      return;
    }
    taat = buildTaat(data.lists);
    daat = buildDaat(data.lists);
    step = Math.min(initialStep, taat.length);
    initialStep = 0;
    $("#simNote").textContent =
      `Real postings for ${data.terms.map(t => `"${t}"`).join(", ")} from the live index, sampled to ${Object.keys(data.docs).length} of ` +
      `${data.sampled_from.toLocaleString()} matching docs (a mix of docs shared across terms and docs unique to one list). d = document number. Hover a cell for its title.`;
    render();
  }

  $("#simForm").addEventListener("submit", e => { e.preventDefault(); load($("#simQ").value.trim() || "quantum physics"); });
  $("#simPlay").addEventListener("click", play);
  $("#simNext").addEventListener("click", () => { stop(); step++; render(); });
  $("#simBack").addEventListener("click", () => { stop(); step = Math.max(0, step - 1); render(); });
  $("#simReset").addEventListener("click", () => { stop(); step = 0; render(); });
  $("#simSpeed").addEventListener("input", () => { if (timer) { stop(); play(); } });
  document.querySelectorAll("[data-simq]").forEach(b => b.addEventListener("click", () => {
    $("#simQ").value = b.dataset.simq; load(b.dataset.simq);
  }));

  load($("#simQ").value);
})();
