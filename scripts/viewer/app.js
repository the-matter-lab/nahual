// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Sidebar: a source (data split or evaluation task) and its examples. Main: one prompt and its responses.
'use strict';
const $ = id => document.getElementById(id);
const STYLE = {stick: {radius: 0.2}, sphere: {scale: 0.2}};
const PAGE = 100;
let source = null, count = 0, first = 0, loaded = 0, busy = false, selected = -1, epoch = 0, drawn = 0;
let prefix = '';  // template text shared by a source's prompts, left out of the list
let view = null;  // {prompt, message, refs, ref, samples, sample, gens, gen, pending, error}

async function api(url, body) {
  const r = await fetch(url, body === undefined ? undefined : {method: 'POST', body: JSON.stringify(body)});
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function remember() {
  try { localStorage.setItem('nahual-viewer', JSON.stringify({source: source, index: selected})); } catch (e) { /* private mode */ }
}
function recall() {
  try { return JSON.parse(localStorage.getItem('nahual-viewer')) || {}; } catch (e) { return {}; }
}

// ---- Structures: one offscreen viewer draws every tile to an image; one live viewer follows the mouse.
let painter = null, live = null, liveTile = null;
function show(v, s) {
  v.clear(); v.setView([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]);  // reset the rotation left by the previous structure
  v.addModel(s.xyz, 'xyz'); v.setStyle({}, STYLE);
  if (s.cell) {
    // Lattice vectors a, b, c as red, green, blue arrows, and the cell as a wireframe box.
    const o = s.cell.origin, vec = s.cell.vectors, P = (x, y, z) => ({x: x, y: y, z: z});
    ['red', 'green', 'blue'].forEach((color, i) => v.addArrow({start: P(...o), radius: 0.2, radiusRatio: 2, mid: 0.92, color: color,
      end: P(o[0] + vec[i][0], o[1] + vec[i][1], o[2] + vec[i][2])}));
    const corner = n => P(...[0, 1, 2].map(k => o[k] + [0, 1, 2].reduce((t, i) => t + ((n >> i) & 1) * vec[i][k], 0)));
    for (let n = 0; n < 8; n++) for (let i = 0; i < 3; i++)
      if (!((n >> i) & 1)) v.addCylinder({start: corner(n), end: corner(n | (1 << i)), radius: 0.05});
    if (s.cell.vacuum !== null) {  // a slab: turn its vacuum axis upright, look slightly down, and frame the atoms
      const u = vec[s.cell.vacuum], h = Math.hypot(u[0], u[2]);
      if (h > 1e-6) v.rotate(Math.atan2(h, u[1]) * 180 / Math.PI, {x: -u[2] / h, y: 0, z: u[0] / h});
      v.rotate(20, 'x'); v.zoomTo({model: -1}); v.zoom(0.9);
      return;
    }
  }
  v.zoomTo(); v.zoom(0.9);
}
function paint(tile) {
  if (!painter) {
    const div = el('div');
    div.style.cssText = 'position:fixed;left:-10000px;top:0;width:200px;height:200px';
    document.body.appendChild(div);
    painter = $3Dmol.createViewer(div, {backgroundColor: 'white'});
  }
  show(painter, tile.s); painter.render();
  tile.s.png = tile.img.src = painter.pngURI();  // kept on the structure, so it is drawn once
}
function park() {  // turn the live tile back into an image, keeping its orientation
  if (!liveTile) return;
  liveTile.s.view = live.getView(); live.render();
  liveTile.s.png = liveTile.img.src = live.pngURI();
  live.container.remove(); liveTile = null;
}
function activate(tile) {
  if (liveTile === tile || !window.$3Dmol) return;
  park();
  if (live) tile.appendChild(live.container);
  else live = $3Dmol.createViewer(tile.appendChild(el('div', 'live')), {backgroundColor: 'white'});
  live.resize(); show(live, tile.s);
  if (tile.s.view) live.setView(tile.s.view);
  live.render(); liveTile = tile;
}
function tile(s) {
  const t = el('div', 'mol');
  t.s = s; t.img = t.appendChild(el('img'));
  if (s.png) t.img.src = s.png;
  t.appendChild(el('small', '', s.formula));
  t.onpointerenter = () => activate(t);
  return t;
}
function flow(segs) {
  const f = el('div', 'flow');
  for (const s of segs) f.appendChild(s.type === 'text' ? el('div', 'text', s.text) : tile(s));
  return f;
}
async function paintAll() {
  const mine = ++drawn;
  for (const t of document.querySelectorAll('.mol')) {
    if (!window.$3Dmol || mine !== drawn) return;  // no 3Dmol.js, or a newer render took over
    if (t.s.png) continue;
    paint(t);
    await new Promise(requestAnimationFrame);
  }
}

// ---- Thread: one user message and the responses to it
function message(label, segs) {
  const m = el('div', 'bot');
  m.appendChild(typeof label === 'string' ? el('div', 'label', label) : label);
  m.appendChild(segs.length ? flow(segs) : el('div', 'text', '(no output)'));
  return m;
}
function pager(name, key) {  // one response at a time, with arrows through the alternatives
  const list = view[key + 's'], k = view[key], label = el('div', 'label');
  if (!list.length) return null;
  if (list.length > 1) {
    const prev = label.appendChild(el('button', '', '‹')), next = el('button', '', '›');
    prev.onclick = () => turn(key, -1); next.onclick = () => turn(key, 1);
    label.append(el('span', '', `${name} ${k + 1} / ${list.length}`), next);
  } else {
    label.appendChild(el('span', '', name));
  }
  if (list[k].note) label.appendChild(el('span', 'metric', list[k].note));
  return message(label, list[k].segments);
}
function turn(key, d) {
  const n = view[key + 's'].length;
  view[key] = (view[key] + d + n) % n;
  render();
}
function render(scroll) {
  park();
  const t = $('thread');
  t.innerHTML = '';
  if (!view) return;
  const user = t.appendChild(el('div', 'user')), bubble = user.appendChild(el('div', 'bubble'));
  bubble.appendChild(flow(view.prompt));
  if (view.message && !view.pending) {
    const use = user.appendChild(el('button', 'use', 'Use this prompt'));
    use.title = 'Copy this prompt into the message box to generate responses to it';
    use.onclick = () => { $('input').value = view.message; grow(); $('input').focus(); };
  }
  for (const [name, key] of [['Reference Response', 'ref'], ['Sample', 'sample'], ['Nahual', 'gen']]) {
    const m = pager(name, key);
    if (m) t.appendChild(m);
  }
  if (view.pending) t.appendChild(el('div', 'bot')).appendChild(el('div', 'typing', 'Generating'));
  if (view.error) t.appendChild(message('Error', [{type: 'text', text: view.error}]));
  if (scroll === 'bottom') t.scrollTop = t.scrollHeight;
  else if (scroll === 'top') t.scrollTop = 0;
  paintAll();
}
function note(text) {
  park(); view = null;
  $('thread').innerHTML = '';
  $('thread').appendChild(el('div', 'note', text));
}

// ---- Sidebar
function common(titles) {  // the shared leading words of a page of titles
  if (titles.length < 2) return '';
  let p = titles.reduce((a, b) => { let i = 0; while (i < a.length && a[i] === b[i]) i++; return a.slice(0, i); });
  p = p.slice(0, p.lastIndexOf(' ') + 1);
  return p.length > 12 ? p : '';
}
function item(i, text) {
  const d = el('div', 'item' + (i === selected ? ' on' : ''));
  d.dataset.i = i; d.title = text;
  d.append(el('span', '', i), el('b', '', text.startsWith(prefix) ? text.slice(prefix.length) : text));
  d.onclick = () => open(i);
  return d;
}
function mark() {
  for (const d of $('list').children) d.classList.toggle('on', +d.dataset.i === selected);
  const on = $('list').querySelector('.on');
  if (on) on.scrollIntoView({block: 'nearest'});
}
async function page(offset, limit) {
  const mine = source;
  const r = await api(`/api/items?source=${encodeURIComponent(source)}&offset=${offset}&limit=${limit}`);
  if (mine !== source) return null;
  count = r.count;
  $('count').textContent = count.toLocaleString() + (count === 1 ? ' example' : ' examples');
  return r;
}
async function more() {  // the next page of titles, when scrolled to the end
  if (busy || loaded >= count) return;
  busy = true;
  try {
    const r = await page(loaded, PAGE);
    if (r && !$('list').children.length) prefix = common(r.titles);
    if (r) { r.titles.forEach((t, k) => $('list').appendChild(item(r.offset + k, t))); loaded = r.offset + r.titles.length; }
  } finally { busy = false; }
}
async function earlier() {  // the previous page, when scrolled to the top after a jump
  if (busy || first <= 0) return;
  busy = true;
  try {
    const start = Math.max(0, first - PAGE), r = await page(start, first - start);
    if (r) {
      const list = $('list'), height = list.scrollHeight, items = document.createDocumentFragment();
      r.titles.forEach((t, k) => items.appendChild(item(start + k, t)));
      list.prepend(items); list.scrollTop += list.scrollHeight - height; first = start;
    }
  } finally { busy = false; }
}
async function choose(id, index) {
  source = id; $('source').value = id;
  selected = -1; count = 1; first = loaded = Math.max(0, index - 10);
  $('list').innerHTML = ''; $('count').textContent = 'Loading…';
  note('Loading…');
  try {
    await more();
    await open(Math.min(index, count - 1));
  } catch (e) { note(e.message); $('count').textContent = ''; }
}
async function open(index) {
  if (index < 0) return;
  if (index < first || index >= loaded) return choose(source, index);  // outside the listed window
  const mine = ++epoch;
  selected = index; mark();
  try {
    const r = await api(`/api/example?source=${encodeURIComponent(source)}&index=${index}`);
    if (mine !== epoch) return;
    view = {prompt: r.prompt, message: r.message, samples: r.samples, sample: 0, gens: [], gen: 0,
            refs: r.references.filter(x => x.length).map(x => ({segments: x, note: ''})), ref: 0};
    remember(); render('top');
  } catch (e) { if (mine === epoch) note(e.message); }
}

// ---- Model and message box
function checkpoint() {
  return $('ckpt').value === 'custom' ? $('custom').value.trim() : $('ckpt').value;
}
async function status(state) {  // show the model state; poll while it loads
  const s = state || await api('/api/model').catch(e => ({state: 'error', error: e.message}));
  $('mstate').className = 'state ' + s.state;
  $('mstate').textContent = {loading: 'loading…', ready: 'ready', error: 'unavailable', none: ''}[s.state] || '';
  $('mstate').title = s.error || '';
  if (s.state === 'loading') setTimeout(status, 2000);
}
async function load(path) {
  if (path) status(await api('/api/model', {checkpoint: path}).catch(e => ({state: 'error', error: e.message})));
}
function grow() { $('input').style.height = ''; $('input').style.height = $('input').scrollHeight + 'px'; }

$('composer').onsubmit = async e => {
  e.preventDefault();
  const text = $('input').value.trim();
  if (!text || (view && view.pending)) return;
  if (!checkpoint()) { view = {prompt: [], refs: [], samples: [], gens: [], error: 'Choose a checkpoint below the message box.'}; return render(); }
  // Single turn: the message replaces the thread. A prompt used unchanged keeps its reference responses.
  const same = view && view.message === text;
  view = {prompt: [{type: 'text', text: text.replace(/```xyz\n[\s\S]*?```/g, '[structure]')}], message: text,
          refs: same ? view.refs : [], ref: 0, samples: [], sample: 0, gens: [], gen: 0, pending: true};
  const mine = view;
  $('input').value = ''; grow(); render('top');
  try {
    const out = await api('/api/generate', {message: text, checkpoint: checkpoint(), samples: +$('ns').value,
                                            steps: +$('steps').value, temperature: +$('temp').value});
    mine.prompt = out.prompt; mine.gens = out.samples;
  } catch (err) { mine.error = err.message; }
  mine.pending = false;
  if (view === mine) render();
  status();
};
$('input').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('composer').requestSubmit(); } };
$('input').oninput = grow;
$('ckpt').onchange = () => {
  const custom = $('ckpt').value === 'custom';
  $('custom').hidden = !custom;
  if (custom) $('custom').focus(); else load(checkpoint());
};
$('custom').onchange = () => load(checkpoint());

// ---- Navigation
$('source').onchange = () => choose($('source').value, 0);
$('jump').onchange = () => { $('jump').blur(); open(Math.min(Math.max(+$('jump').value || 0, 0), count - 1)); };
$('list').onscroll = () => {
  const l = $('list');
  if (l.scrollTop + l.clientHeight > l.scrollHeight - 200) more();
  if (l.scrollTop < 50) earlier();
};
document.onkeydown = e => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) || !view) return;
  if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && selected >= 0) {
    e.preventDefault();
    open(Math.min(Math.max(selected + (e.key === 'ArrowDown' ? 1 : -1), 0), count - 1));
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    const key = ['gen', 'sample', 'ref'].find(k => (view[k + 's'] || []).length > 1);
    if (key) turn(key, e.key === 'ArrowRight' ? 1 : -1);
  }
};

api('/api/sources').then(r => {
  for (const c of r.checkpoints) $('ckpt').appendChild(el('option', '', c.label)).value = c.path;
  $('ckpt').appendChild(el('option', '', 'Custom path…')).value = 'custom';
  if (r.default) $('ckpt').value = r.default;
  else { $('ckpt').value = 'custom'; $('custom').hidden = false; }
  status();
  const groups = {};
  for (const s of r.sources) {
    if (!groups[s.group]) { groups[s.group] = $('source').appendChild(el('optgroup')); groups[s.group].label = s.group; }
    groups[s.group].appendChild(el('option', '', s.label)).value = s.id;
  }
  if (!r.sources.length) {
    return note('No data or samples found. Download the data with python -m nahual.prepare download, '
                + 'or pass --samples with a folder of evaluation results.');
  }
  const last = recall(), known = r.sources.some(s => s.id === last.source);
  choose(known ? last.source : r.source, known ? Math.max(last.index, 0) : 0);
}).catch(e => note(e.message));
