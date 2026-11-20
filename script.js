"use strict";

const examples = [
  { label: "Public bulletin", text: "The public health office published its weekly COVID-19 bulletin and linked the official report." },
  { label: "Viral rumor", text: "An anonymous viral post claims the COVID-19 report was invented and asks readers to share immediately." },
  { label: "Mixed signals", text: "A viral post shares a weekly report, but the original source could not be found." }
];
const corpus = [
  { fake: false, text: examples[0].text },
  { fake: false, text: "The health office published an official report with weekly case numbers and links to the original source." },
  { fake: false, text: "A public bulletin listed regional data and explained a correction to the weekly COVID-19 report." },
  { fake: false, text: "Researchers published weekly statistics and shared the original data with the public." },
  { fake: false, text: "A local agency updated its report and provided a public link to the source." },
  { fake: true, text: examples[1].text },
  { fake: true, text: "A viral post cites an anonymous source and asks people to share immediately." },
  { fake: true, text: "An anonymous account shared a fabricated notice without links to supporting data." },
  { fake: true, text: "The invented bulletin spread through a viral message before readers could find the original source." },
  { fake: true, text: "A viral post presents a made-up report as an official public notice." }
];
const models = [
  { label: "SVM", slope: 1.46, bias: .08 },
  { label: "Random Forest", slope: 1.09, bias: -.15 },
  { label: "Logistic Regression", slope: 1.27, bias: -.04 },
  { label: "KNN", slope: .89, bias: .12 },
  { label: "Multinomial NB", slope: 1.62, bias: -.03 }
];
const steps = [
  { label: "Text", focus: "text" },
  { label: "Features", focus: "features" },
  { label: "Level 1", focus: "V1" },
  { label: "Level 2", focus: "V7" },
  { label: "Level 3", focus: "V9" }
];
const nodes = {
  V1: { family: "BoW", n: 1 }, V2: { family: "BoW", n: 2 }, V3: { family: "BoW", n: 3 },
  V4: { family: "TF-IDF", n: 1 }, V5: { family: "TF-IDF", n: 2 }, V6: { family: "TF-IDF", n: 3 },
  V7: { family: "BoW", from: ["V1", "V2", "V3"] },
  V8: { family: "TF-IDF", from: ["V4", "V5", "V6"] },
  V9: { from: ["V7", "V8"] }
};
const el = id => document.getElementById(id);
const state = {
  text: examples[0].text, selected: "V9", feature: "V1", featureTerm: null, expandedFeatures: new Set(),
  threshold: 50, bowWeight: 50, overrides: {}
};
const featureFamilies = [
  { id: "BoW", title: "Bag-of-Words", metric: "Count", voters: ["V1", "V2", "V3"] },
  { id: "TF-IDF", title: "TF‑IDF", metric: "Weight", voters: ["V4", "V5", "V6"] }
];
const initialChartRows = 5;
const inlineFeatureDetail = matchMedia("(max-width: 950px)");
const stopWords = new Set("a an and are as at be but by for from in is it its of on or that the their to was were with".split(" "));
const percent = value => value === null ? "—" : (value * 100).toFixed(1) + "%";
const decimal = value => value.toFixed(4);
const safe = value => String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
let trace;
let drawFrame;

function tokenize(text) {
  return (text.toLowerCase().match(/[\p{L}\p{N}]+(?:-[\p{L}\p{N}]+)*/gu) || [])
    .filter(word => !stopWords.has(word));
}

function countGrams(tokens, n) {
  const counts = new Map();
  for (let index = 0; index + n <= tokens.length; index++) {
    const term = tokens.slice(index, index + n).join(" ");
    counts.set(term, (counts.get(term) || 0) + 1);
  }
  return counts;
}

const corpusGrams = corpus.map(doc => {
  const tokens = tokenize(doc.text);
  return { fake: doc.fake, grams: [1, 2, 3].map(n => countGrams(tokens, n)) };
});

function buildFeatures(tokens, family, n) {
  const counts = countGrams(tokens, n);
  const features = [];
  const total = Math.max(0, tokens.length - n + 1);
  for (const [term, count] of counts) {
    const present = corpusGrams.filter(doc => doc.grams[n - 1].has(term));
    const fake = present.filter(doc => doc.fake).length;
    const real = present.length - fake;
    const df = present.length;
    const tf = count / total;
    const idf = Math.log((corpus.length + 1) / (df + 1)) + 1;
    const weight = family === "BoW" ? count : tf * idf;
    features.push({ term, count, total, df, tf, idf, weight, signal: Math.log((fake + 1) / (real + 1)) });
  }
  return features.sort((a, b) => b.weight - a.weight || a.term.localeCompare(b.term));
}

function simulate() {
  const tokens = tokenize(state.text);
  const features = {}, base = {}, votes = {};
  for (const [id, node] of Object.entries(nodes)) {
    if (node.n) {
      const vector = buildFeatures(tokens, node.family, node.n);
      features[id] = vector;
      const total = vector.reduce((sum, item) => sum + item.weight, 0);
      const evidence = total ? vector.reduce((sum, item) => sum + item.weight * item.signal, 0) / total : 0;
      base[id] = models.map((model, index) => state.overrides[id]?.[index] ??
        (total ? 1 / (1 + Math.exp(-(model.slope * evidence + model.bias))) : .5));
      votes[id] = tokens.length ? base[id].reduce((sum, value) => sum + value, 0) / base[id].length : null;
    } else {
      const weights = nodeWeights(id);
      votes[id] = tokens.length ? node.from.reduce((sum, child, i) => sum + votes[child] * weights[i], 0) : null;
    }
  }
  return { tokens, features, base, votes };
}

function nodeWeights(id) {
  if (id === "V9") return [state.bowWeight / 100, (100 - state.bowWeight) / 100];
  return nodes[id].n ? models.map(() => 1 / models.length) : nodes[id].from.map(() => 1 / nodes[id].from.length);
}

function votingStage() {
  if (state.selected === "text" || state.selected === "features") return state.selected === "text" ? 0 : 1;
  if (nodes[state.selected].n) return 2;
  return state.selected === "V9" ? 4 : 3;
}

function connectedEdges() {
  const edges = new Set();
  const collectInputs = id => {
    for (const child of nodes[id]?.from || []) {
      edges.add(child + id);
      collectInputs(child);
    }
  };
  if (nodes[state.selected]) {
    collectInputs(state.selected);
    let child = state.selected;
    for (const [parent, node] of Object.entries(nodes)) {
      if (node.from?.includes(child)) {
        edges.add(child + parent);
        child = parent;
      }
    }
  }
  return edges;
}

function drawWires() {
  cancelAnimationFrame(drawFrame);
  drawFrame = requestAnimationFrame(() => {
    const canvas = el("wires");
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * scale);
    canvas.height = Math.round(rect.height * scale);
    const ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    const active = connectedEdges();
    const links = Object.entries(nodes).flatMap(([parent, node]) => (node.from || []).map(child => [child, parent]));
    links.sort(([a, b], [c, d]) => Number(active.has(a + b)) - Number(active.has(c + d)));
    for (const [child, parent] of links) {
      const from = el("node-" + child).getBoundingClientRect();
      const to = el("node-" + parent).getBoundingClientRect();
      const x1 = from.right - rect.left, x2 = to.left - rect.left;
      const y1 = from.top + from.height / 2 - rect.top, y2 = to.top + to.height / 2 - rect.top;
      const branchRight = nodes[child].n ? el(nodes[child].family === "BoW" ? "bow-voters" : "tfidf-voters").getBoundingClientRect().right - rect.left : x1;
      const bend = branchRight + (x2 - branchRight) * .53;
      const slotIndex = nodes[parent].from.indexOf(child);
      const inset = (slotIndex - (nodes[parent].from.length - 1) / 2) * 10;
      const isActive = active.has(child + parent);
      ctx.strokeStyle = isActive ? "#7966b3" : "#abb1ba";
      ctx.lineWidth = isActive ? 1.7 : 1;
      ctx.beginPath();
      if (nodes[child].n) {
        const rail = from.bottom - rect.top + 3 + slotIndex * 2;
        ctx.moveTo((from.left + from.right) / 2 - rect.left, from.bottom - rect.top);
        ctx.lineTo((from.left + from.right) / 2 - rect.left, rail);
        ctx.lineTo(bend, rail);
        ctx.lineTo(bend, y2 + inset);
        ctx.lineTo(x2, y2 + inset);
      } else {
        ctx.moveTo(x1, y1);
        ctx.bezierCurveTo(bend, y1, bend, y2 + inset, x2, y2 + inset);
      }
      ctx.stroke();
    }
  });
}

function makeGraph() {
  for (const [id, node] of Object.entries(nodes)) {
    const button = document.createElement("button");
    button.type = "button";
    button.id = "node-" + id;
    button.className = "node" + (id === "V9" ? " merge result-node" : node.n ? "" : " merge");
    button.innerHTML = '<span class="node-name"><strong>' + id + '</strong><small>' +
      (node.n ? node.n + "‑gram" : id === "V9" ? "Final vote" : node.family) +
      '</small></span><output></output><span class="node-meter" aria-hidden="true"><i></i></span>';
    const parent = node.n ? (node.family === "BoW" ? "bow-voters" : "tfidf-voters") : id.toLowerCase();
    el(parent).append(button);
    button.addEventListener("click", () => {
      select(id);
      if (matchMedia("(max-width: 860px)").matches) el("inspector").scrollIntoView({ block: "nearest" });
    });
  }
}

function renderGraph() {
  const active = connectedEdges();
  for (const [id, node] of Object.entries(nodes)) {
    const button = el("node-" + id);
    const value = trace.votes[id];
    const name = node.n ? node.family + " " + node.n + "-gram" : node.family || "final vote";
    button.querySelector("output").textContent = percent(value);
    button.querySelector("i").style.flexBasis = (value ?? 0) * 100 + "%";
    button.querySelector(".node-meter").hidden = value === null;
    button.setAttribute("aria-label", id + ", " + name + ", " + (value === null ? "no score" : percent(value) + " simulated probability of fake"));
    button.setAttribute("aria-pressed", String(state.selected === id));
    button.dataset.path = String([...active].some(edge => edge.includes(id)));
  }
  const stage = votingStage();
  el("steps").querySelectorAll("button").forEach((button, index) => {
    if (stage === index) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  el("prev").disabled = stage === 0;
  el("next").disabled = stage === steps.length - 1;
  drawWires();
}

function setInspector(title, badge, description, content) {
  el("inspector").innerHTML = '<div class="detail-title"><h2>' + title + '</h2><span class="tag">' + badge +
    '</span></div><p class="detail-copy">' + description + '</p>' + content;
}

function renderTokens() {
  const content = trace.tokens.length
    ? '<div class="caption">' + trace.tokens.length + ' retained tokens</div><div class="token-list">' +
      trace.tokens.map(token => '<span class="token">' + safe(token) + '</span>').join("") + '</div>'
    : '<p class="hint">Enter text with at least one retained word to start.</p>';
  setInspector("Cleaned text", "Input", "The demo lowercases the input, extracts words, and removes common stop words.", content);
}

function featureValue(value, family) {
  if (family === "BoW") return String(value);
  return value < .001 && value > 0 ? value.toPrecision(3) : value.toFixed(3);
}

function featureCard(id, max) {
  const node = nodes[id];
  const items = trace.features[id];
  const expanded = state.expandedFeatures.has(id);
  const visible = expanded ? items : items.slice(0, initialChartRows);
  const rows = visible.map((item, index) =>
    '<button class="feature-bar" type="button" data-chart="' + id + '" data-feature-index="' + index + '" aria-pressed="false">' +
    '<span class="feature-bar-label"><span class="feature-bar-term">' + safe(item.term) + '</span>' +
    '<span class="feature-bar-value">' + featureValue(item.weight, node.family) + '</span></span>' +
    '<span class="feature-track" aria-hidden="true"><i style="width:' + item.weight / max * 100 + '%"></i></span></button>').join("");
  const more = items.length > initialChartRows
    ? '<button class="feature-more" type="button" data-expand="' + id + '" aria-controls="features-' + id +
      '" aria-expanded="' + expanded + '">' + (expanded ? 'Show top ' + initialChartRows : 'Show all ' + items.length) +
      ' n‑grams</button>'
    : "";
  const body = items.length ? rows : '<p class="hint">Needs ' + node.n + ' retained ' + (node.n === 1 ? 'token' : 'tokens') + '.</p>';
  return '<article class="feature-card" id="card-' + id + '" aria-labelledby="feature-heading-' + id + '">' +
    '<h4 id="feature-heading-' + id + '">' + node.n + '‑gram <span>' + id + '</span></h4>' +
    '<div class="feature-rows" id="features-' + id + '">' + body + '</div>' + more + '</article>';
}

function featureScale(family) {
  // Each n-gram list is sorted descending by weight.
  return Math.max(...family.voters.map(id => trace.features[id][0]?.weight || 0));
}

function featureFamily(family) {
  const max = featureScale(family);
  const scale = max ? family.metric + ' · shared scale 0–' + featureValue(max, family.id) : family.metric;
  return '<section class="feature-family' + (family.id === "TF-IDF" ? ' tfidf' : '') + '">' +
    '<div class="family-heading"><h3>' + family.title + '</h3><span class="family-scale">' + scale + '</span></div>' +
    '<div class="feature-grid">' + family.voters.map(id => featureCard(id, max)).join("") + '</div></section>';
}

function matchingTokens(tokens, n, term) {
  const marked = new Set();
  for (let i = 0; i + n <= tokens.length; i++) {
    if (tokens.slice(i, i + n).join(" ") !== term) continue;
    for (let j = i; j < i + n; j++) marked.add(j);
  }
  return marked;
}

function updateFeatureSelection() {
  const id = state.feature;
  const node = nodes[id];
  const items = trace.features[id];
  const selected = items.find(item => item.term === state.featureTerm) || items[0];
  state.featureTerm = selected?.term ?? null;
  const marked = selected ? matchingTokens(trace.tokens, node.n, selected.term) : new Set();

  const tokenStrip = el("feature-source-tokens");
  tokenStrip.innerHTML = trace.tokens.map((token, index) => marked.has(index)
    ? '<mark class="token">' + safe(token) + '</mark>'
    : '<span class="token">' + safe(token) + '</span>').join("");
  tokenStrip.hidden = !trace.tokens.length;

  const explorer = el("feature-explorer");
  explorer.querySelectorAll(".feature-card").forEach(card => { card.dataset.active = String(card.id === "card-" + id); });
  explorer.querySelectorAll("[data-chart]").forEach(button => {
    const chart = button.dataset.chart;
    const item = trace.features[chart][Number(button.dataset.featureIndex)];
    const match = !!selected && item.term === selected.term && nodes[chart].n === node.n;
    button.setAttribute("aria-pressed", String(match && chart === id));
    button.dataset.linked = String(match && chart !== id);
  });

  let detail;
  if (!selected) {
    detail = '<p class="hint">' + (trace.tokens.length ? 'Choose an n‑gram from a populated chart.' :
      'Enter news text with at least one retained token, then run the trace.') + '</p>';
  } else {
    const bow = node.family === "BoW";
    const scores = [
      ["Count in this text", selected.count],
      ...(!bow ? [
        ["Total " + node.n + "‑grams", selected.total],
        ["Term frequency (TF)", featureValue(selected.tf, node.family)],
        ["Toy corpus documents (df)", selected.df + ' / ' + corpus.length],
        ["Inverse document frequency (IDF)", featureValue(selected.idf, node.family)]
      ] : [])
    ];
    detail = '<p class="caption">' + id + ' · ' + node.family + ' · ' + node.n + '‑gram</p>' +
      '<p class="feature-term">' + safe(selected.term) + '</p>' +
      '<dl class="feature-breakdown">' + scores.map(([label, value]) =>
        '<div><dt>' + label + '</dt><dd>' + value + '</dd></div>').join("") + '</dl>' +
      (bow ? '<p class="feature-formula">BoW = count = ' + selected.count + '</p>' :
        '<p class="feature-formula">TF = ' + selected.count + ' ÷ ' + selected.total +
        '<br>IDF = ln((' + corpus.length + ' + 1) ÷ (' + selected.df + ' + 1)) + 1' +
        '<br>TF × IDF ≈ ' + featureValue(selected.weight, node.family) + '</p>');
  }
  el("feature-detail-content").innerHTML = detail;
  const open = el("open-feature-voter");
  open.textContent = "Inspect " + id + " voter →";
  open.dataset.voter = id;
  positionFeatureDetail();
}

function positionFeatureDetail() {
  if (state.selected !== "features") return;
  const detail = el("feature-detail");
  const target = inlineFeatureDetail.matches ? el("card-" + state.feature) : el("feature-detail-slot");
  if (detail.parentElement !== target) target.append(detail);
}

function renderFeatures() {
  const count = trace.tokens.length;
  el("feature-explorer").innerHTML =
    '<div class="feature-intro"><h2 id="feature-title" tabindex="-1">Feature extraction</h2>' +
    '<p class="hint">Select a bar to highlight every occurrence and see how its score is calculated.</p></div>' +
    '<section class="feature-source" aria-label="Cleaned text">' +
    '<p class="caption"><strong>Cleaned text</strong> · ' + count + ' retained ' + (count === 1 ? 'token' : 'tokens') + '</p>' +
    '<div class="token-list" id="feature-source-tokens"></div></section>' +
    '<div class="feature-layout"><div>' + featureFamilies.map(featureFamily).join("") + '</div>' +
    '<div class="feature-detail-slot" id="feature-detail-slot"><aside class="feature-detail" id="feature-detail" aria-labelledby="feature-detail-title"><h3 id="feature-detail-title">Selected n‑gram</h3>' +
    '<div id="feature-detail-content" aria-live="polite"></div>' +
    '<button class="row-link" type="button" id="open-feature-voter"></button></aside></div></div>';
  updateFeatureSelection();
}

function renderVoter() {
  const id = state.selected, node = nodes[id];
  const title = id === "V9" ? "V9 · Final vote" : id + " · " + node.family + (node.n ? " " + node.n + "‑gram" : " fusion");
  const level = id === "V9" ? 3 : node.n ? 1 : 2;
  const copy = node.n ? "Average five illustrative classifier probabilities for this feature representation." :
    id === "V9" ? "Combine the two feature families and apply the decision threshold." :
    "Average the three n‑gram voters for this feature family.";
  if (trace.votes[id] === null) {
    setInspector(title, "Level " + level, copy, '<p class="hint">Run a trace with at least one retained token to see scores.</p>');
    return;
  }

  let content = "";
  if (node.n) {
    content = '<button class="plain features-link" type="button" data-jump="features">Inspect ' + node.n + '‑gram features</button>';
    content += models.map((model, index) =>
      '<div class="base-row"><label for="base-' + index + '">' + model.label +
      '</label><input type="range" min="0" max="100" step="1" id="base-' + index + '" data-base="' +
      index + '" value="' + Math.round(trace.base[id][index] * 100) + '" aria-label="' + model.label +
      ' simulated fake probability"><output id="base-value-' + index + '" for="base-' + index + '"></output></div>').join("");
    content += '<button class="row-link" type="button" data-reset="base"><span>Reset classifier scores</span><span>↺</span></button>';
    if (!trace.features[id].length) content += '<p class="hint">No n‑grams of this size. Default scores are 50%.</p>';
  } else {
    content = node.from.map(child => '<button class="row-link" type="button" data-jump="' + child + '"><span>' + child +
      ' · ' + (id === "V9" ? nodes[child].family : nodes[child].n + "‑gram") + '</span><span>' +
      percent(trace.votes[child]) + ' ↗</span></button>').join("");
  }

  if (id === "V9") {
    content = '<div class="output-pair"><div class="fake"><strong id="fake-score"></strong><span class="caption">P(fake)</span></div>' +
      '<div class="real"><strong id="real-score"></strong><span class="caption">P(real)</span></div></div>' +
      '<div class="meter" aria-hidden="true"><i id="final-meter"></i></div><p class="class-result" id="decision"></p>' + content;
  }
  content += '<div class="math">Soft-vote calculation <output class="equation" id="equation"></output></div>';
  if (id === "V9") {
    content += range("bow", "V7 weight", state.bowWeight, 0, 100) + range("threshold", "Decision threshold", state.threshold, 0, 100) +
      '<p class="hint threshold-note">The demo assigns “fake” when the unrounded P(fake) meets or exceeds the threshold.</p>';
  }
  setInspector(title, "Level " + level, copy, content);
  syncVoter();
}

function range(id, label, value, min, max) {
  return '<div class="control"><div class="control-label"><label for="' + id + '">' + label + '</label><output id="' +
    id + '-value" for="' + id + '"></output></div><input type="range" id="' + id + '" min="' + min +
    '" max="' + max + '" step="1" value="' + value + '"></div>';
}

function syncVoter() {
  const id = state.selected, node = nodes[id];
  const value = trace.votes[id];
  const children = node.n ? trace.base[id] : node.from.map(child => trace.votes[child]);
  const calculation = id === "V9"
    ? nodeWeights(id).map((weight, i) => weight.toFixed(2) + " × " + decimal(children[i])).join(" + ")
    : "(" + children.map(decimal).join(" + ") + ") ÷ " + children.length;
  el("equation").textContent = calculation + " = " + percent(value);
  if (node.n) children.forEach((score, index) => { el("base-value-" + index).textContent = percent(score); });
  if (id === "V9") {
    el("fake-score").textContent = percent(value);
    el("real-score").textContent = percent(1 - value);
    el("final-meter").style.flexBasis = (value * 100) + "%";
    el("decision").textContent = "Simulated decision · " + (value >= state.threshold / 100 ? "Fake" : "Real");
    el("bow-value").textContent = state.bowWeight + "% · V8 " + (100 - state.bowWeight) + "%";
    el("threshold-value").textContent = state.threshold + "%";
  }
}

function renderInspector() {
  if (state.selected === "text") renderTokens();
  else renderVoter();
}

function select(focus) {
  const exploring = focus === "features";
  if (exploring && nodes[state.selected]?.n && state.feature !== state.selected) {
    state.feature = state.selected;
    state.featureTerm = null;
  }
  state.selected = focus;
  el("feature-explorer").hidden = !exploring;
  el("voting-workspace").hidden = exploring;
  renderGraph();
  if (exploring) renderFeatures();
  else renderInspector();
}

function run(text, focus) {
  if (state.text !== text) {
    state.featureTerm = null;
    state.expandedFeatures.clear();
  }
  state.text = text;
  state.overrides = {};
  trace = simulate();
  select(focus);
  el("run-status").textContent = trace.tokens.length ? "Trace ready" : "No usable tokens";
  el("examples").querySelectorAll("button").forEach((button, index) =>
    button.setAttribute("aria-pressed", String(text === examples[index].text)));
}

examples.forEach((example, index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "choice";
  button.textContent = example.label;
  button.dataset.example = index;
  el("examples").append(button);
  button.addEventListener("click", () => {
    el("claim").value = example.text;
    run(example.text, state.selected);
  });
});

steps.forEach(({ label, focus }, index) => {
  const item = document.createElement("li");
  if (index) item.innerHTML = '<span class="step-chevron" aria-hidden="true">›</span>';
  const button = document.createElement("button");
  button.type = "button";
  button.className = "step";
  button.textContent = label;
  button.addEventListener("click", () => select(focus));
  item.append(button);
  el("steps").append(item);
});

el("claim").value = state.text;
el("claim").addEventListener("input", () => {
  el("run-status").textContent = el("claim").value.trim() === state.text ? "Trace ready" : "Run to update the trace";
});
el("input-form").addEventListener("submit", event => {
  event.preventDefault();
  run(el("claim").value.trim(), state.selected === "features" ? "features" : "text");
});
el("prev").addEventListener("click", () => select(steps[votingStage() - 1].focus));
el("next").addEventListener("click", () => select(steps[votingStage() + 1].focus));
el("inspector").addEventListener("click", event => {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.dataset.jump) {
    select(button.dataset.jump);
    const heading = state.selected === "features" ? el("feature-title") : el("inspector").querySelector("h2");
    heading.setAttribute("tabindex", "-1");
    heading.focus({ preventScroll: true });
  }
  if (button.dataset.reset) {
    delete state.overrides[state.selected];
    trace = simulate();
    renderGraph();
    renderVoter();
    el("inspector").querySelector("[data-reset]").focus();
  }
});
el("feature-explorer").addEventListener("click", event => {
  const button = event.target.closest("button");
  if (!button) return;
  const { chart, featureIndex, expand, voter } = button.dataset;
  if (chart) {
    state.feature = chart;
    state.featureTerm = trace.features[chart][Number(featureIndex)].term;
    updateFeatureSelection();
  } else if (expand) {
    if (state.expandedFeatures.has(expand)) state.expandedFeatures.delete(expand);
    else state.expandedFeatures.add(expand);
    const family = featureFamilies.find(section => section.id === nodes[expand].family);
    const card = el("card-" + expand);
    const detail = el("feature-detail");
    if (detail.parentElement === card) el("feature-detail-slot").append(detail);
    card.outerHTML = featureCard(expand, featureScale(family));
    updateFeatureSelection();
    el("card-" + expand).querySelector("[data-expand]").focus();
  } else if (voter) {
    select(voter);
    const heading = el("inspector").querySelector("h2");
    heading.setAttribute("tabindex", "-1");
    heading.focus();
  }
});
el("inspector").addEventListener("input", event => {
  const target = event.target, value = Number(target.value);
  if (target.id === "threshold") state.threshold = value;
  else if (target.id === "bow") state.bowWeight = value;
  else if (target.dataset.base !== undefined) {
    state.overrides[state.selected] ??= {};
    state.overrides[state.selected][target.dataset.base] = value / 100;
  } else return;
  trace = simulate();
  renderGraph();
  syncVoter();
});

makeGraph();
run(state.text, state.selected);
inlineFeatureDetail.addEventListener("change", positionFeatureDetail);
new ResizeObserver(drawWires).observe(el("flow"));
