(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const fmt = new Intl.NumberFormat("en-US");
  const state = { model: "tfidf", mode: "taat" };
  const LABEL = { boolean: "Boolean", tf: "TF", tfidf: "TF-IDF", taat: "TAAT", daat: "DAAT" };

  const el = (tag, props = {}, ...kids) => {
    const n = Object.assign(document.createElement(tag), props);
    kids.flat().forEach(k => k != null && n.append(k));
    return n;
  };

  // ---------- stats ----------
  fetch("/api/stats").then(r => r.json()).then(s => {
    $$("#stats dd").forEach(dd => (dd.textContent = fmt.format(s[dd.dataset.k])));
    $("#buildInfo").textContent =
      `Index build at startup: TF-IDF ${s.build_seconds.tfidf}s; the TF and Boolean indexes share its postings ` +
      `to fit a 512 MB container (Boolean derived in ${s.build_seconds.boolean}s). ${fmt.format(s.tokens)} tokens after preprocessing.`;
  });

  // ---------- segmented controls ----------
  function setSeg(name, value) {
    state[name] = value;
    $$(`.seg[data-name="${name}"] button`).forEach(b =>
      b.setAttribute("aria-checked", String(b.dataset.v === value)));
    updateHint();
  }
  $$(".seg").forEach(seg => seg.addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    setSeg(seg.dataset.name, b.dataset.v);
    if ($("#q").value.trim()) search();
  }));

  function updateHint() {
    $("#hint").textContent = state.model === "boolean"
      ? 'Boolean mode: AND, OR, NOT, PHRASE and ( ) are supported. Plain words are ANDed together. Results are unranked set matches.'
      : `${LABEL[state.model]} mode: free text, ranked by ${state.model === "tf" ? "Σ term frequency" : "Σ tf × idf"}. Boolean operators are ignored.`;
  }

  // ---------- search ----------
  let ctrl;
  async function search() {
    const q = $("#q").value.trim();
    if (!q) return;
    ctrl?.abort();
    ctrl = new AbortController();
    const params = new URLSearchParams({ q, model: state.model, mode: state.mode, k: 10 });
    history.replaceState(null, "", `?${params}#search`);
    try {
      const r = await fetch(`/api/search?${params}`, { signal: ctrl.signal });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      render(await r.json());
    } catch (err) {
      if (err.name !== "AbortError") renderError(err.message);
    }
    $("#compare").hidden = true;
  }

  function render(d) {
    const meta = $("#meta");
    meta.hidden = false;
    meta.replaceChildren(...[
      el("span", { className: "badge", textContent: `${d.latency_ms.toFixed(2)} ms` }),
      el("span", { textContent: `${LABEL[d.model]} · ${LABEL[d.mode]}` }),
      d.total_matches != null ? el("span", { textContent: `${fmt.format(d.total_matches)} matching docs` }) : null,
      el("span", {}, "Executed: ", el("code", { textContent: d.executed_query || "—" })),
      d.terms.length ? el("span", {}, "Stems: ", el("code", { textContent: d.terms.join(" ") })) : null,
    ].filter(Boolean)
    );
    const list = $("#results");
    if (!d.results.length) {
      list.replaceChildren(el("li", { className: "empty", textContent: "No documents matched. Try a broader query, or switch to TF-IDF for ranked free-text search." }));
      return;
    }
    list.replaceChildren(...d.results.map(r => el("li", {},
      el("div", { className: "r-head" },
        el("span", { className: "r-rank", textContent: r.rank }),
        el("a", { className: "r-title", href: r.url, target: "_blank", rel: "noopener", textContent: r.title }),
        r.score != null ? el("span", { className: "r-score", textContent: `score ${r.score}` }) : null,
      ),
      el("p", { className: "r-snip" }, r.snippet.map(s => s.hit ? el("mark", { textContent: s.text }) : s.text)),
    )));
  }

  function renderError(msg) {
    $("#meta").hidden = true;
    $("#results").replaceChildren(el("li", { className: "empty", textContent: `Error: ${msg}` }));
  }

  $("#form").addEventListener("submit", e => { e.preventDefault(); search(); });
  let t;
  $("#q").addEventListener("input", () => { clearTimeout(t); t = setTimeout(search, 350); });

  $$(".chip").forEach(c => c.addEventListener("click", () => {
    $("#q").value = c.dataset.q;
    setSeg("model", c.dataset.m);
    search();
  }));

  // ---------- compare ----------
  $("#compareBtn").addEventListener("click", async () => {
    const q = $("#q").value.trim() || "roman empire history";
    $("#q").value = q;
    const btn = $("#compareBtn");
    btn.disabled = true;
    try {
      const d = await (await fetch(`/api/compare?q=${encodeURIComponent(q)}`)).json();
      const max = Math.max(...d.rows.map(r => r.median_ms), 0.001);
      const box = $("#compare");
      box.hidden = false;
      box.replaceChildren(
        el("p", { className: "muted", style: "margin:0 0 12px;font-size:14px" },
          `Median of 7 runs per configuration for "${q}". Lower is better.`),
        el("div", { className: "bars" }, d.rows.flatMap(r => [
          el("span", { textContent: `${LABEL[r.model]} · ${LABEL[r.mode]}` }),
          el("div", { className: `bar ${r.mode}`, style: `width:${(r.median_ms / max) * 100}%`,
                      title: r.top.length ? `Top: ${r.top.join(", ")}` : "no results" }),
          el("span", { className: "v", textContent: `${r.median_ms.toFixed(2)} ms` }),
        ])),
      );
    } finally {
      btn.disabled = false;
    }
  });

  // ---------- benchmarks table ----------
  const decode = id => {
    const m = id.match(/SelfIndex_i(\d)d(\d)c(\d)o(\w+?)_q(TAAT|DAAT)/);
    if (!m) {
      const cache = id.split("_").pop();
      return ["BM25", "Lucene", "Lucene default", cache.charAt(0) + cache.slice(1).toLowerCase() + " cache"];
    }
    const idx = { 1: "Boolean", 2: "TF", 3: "TF-IDF" }[m[1]] + (m[4] === "sp" ? " + skip ptrs" : "");
    return [idx, { 1: "JSON", 2: "SQLite" }[m[2]], { 1: "None", 2: "Elias γ/δ", 3: "Zlib" }[m[3]], m[5]];
  };
  fetch("/api/benchmarks").then(r => r.json()).then(rows => {
    const best = k => Math[k === "qps" ? "max" : "min"](...rows.map(r => r[k]).filter(Boolean));
    const b = { p95_ms: best("p95_ms"), qps: best("qps") };
    $("#benchTable tbody").replaceChildren(...rows.map(r => {
      const [idx, store, comp, mode] = decode(r.id);
      return el("tr", { className: r.system === "Elasticsearch" ? "es" : "" },
        el("td", { className: "id", textContent: r.system === "Elasticsearch" ? "Elasticsearch" : r.id.replace(/_q\w+$/, "") }),
        el("td", { textContent: idx }), el("td", { textContent: store }), el("td", { textContent: comp }),
        el("td", { textContent: mode }),
        el("td", { className: "n", textContent: r.p50_ms }),
        el("td", { className: `n ${r.p95_ms === b.p95_ms ? "best" : ""}`, textContent: r.p95_ms }),
        el("td", { className: "n", textContent: r.p99_ms }),
        el("td", { className: `n ${r.qps === b.qps ? "best" : ""}`, textContent: fmt.format(r.qps) }),
        el("td", { className: "n", textContent: r.disk_mb ? fmt.format(r.disk_mb) : "—" }),
      );
    }));
  });

  // ---------- lightbox ----------
  $$(".gallery img").forEach(img => img.addEventListener("click", () => {
    const lb = el("div", { className: "lightbox" }, el("img", { src: img.src, alt: img.alt }));
    lb.addEventListener("click", () => lb.remove());
    document.body.append(lb);
  }));
  document.addEventListener("keydown", e => e.key === "Escape" && $(".lightbox")?.remove());

  // ---------- initial state from URL ----------
  const p = new URLSearchParams(location.search);
  if (p.get("model")) setSeg("model", p.get("model"));
  if (p.get("mode")) setSeg("mode", p.get("mode"));
  updateHint();
  $("#q").value = p.get("q") || "quantum physics";
  search();
})();
