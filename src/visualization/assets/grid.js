// The page's behaviour. Every handler is delegated from `document`, so the
// lightbox's cloned cell works without rebinding anything.

// --- source pills: a URL is a plain link, a local path copies to the clipboard ---
function flashCopied(pill) {
  var original = pill.textContent;
  pill.textContent = 'copied';
  pill.classList.add('copied');
  setTimeout(function () {
    pill.textContent = original;
    pill.classList.remove('copied');
  }, 1200);
}

function copySource(pill) {
  var text = pill.dataset.copy || '';
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(function () { flashCopied(pill); }, function () {});
    return;
  }
  var staging = document.createElement('textarea');
  staging.value = text;
  staging.style.position = 'fixed';
  staging.style.opacity = '0';
  document.body.appendChild(staging);
  staging.select();
  try {
    document.execCommand('copy');
    flashCopied(pill);
  } catch (err) { /* an unreachable clipboard is not worth an alert */ }
  document.body.removeChild(staging);
}

// --- layer toggling: sidebar checkboxes drive overlays through shared data-keys ---
// A key ends in a class name, which comes from the data. Splicing that into a
// selector unescaped throws on the first quote or backslash, and the throw escapes
// the delegated change handler — so one odd class name kills every checkbox on the
// page, not just its own row.
function setHidden(key, hidden) {
  document.querySelectorAll('.layer[data-key=' + CSS.escape(key) + ']').forEach(function (layer) {
    layer.classList.toggle('hidden', hidden);
  });
}

function leavesOf(prefix) {
  return Array.prototype.filter.call(document.querySelectorAll('input.cls'), function (box) {
    return box.dataset.key.indexOf(prefix) === 0;
  });
}

function refreshGroups() {
  document.querySelectorAll('input.grp').forEach(function (group) {
    var leaves = leavesOf(group.dataset.prefix);
    var on = leaves.filter(function (box) { return box.checked; }).length;
    group.checked = on === leaves.length && leaves.length > 0;
    group.indeterminate = on > 0 && on < leaves.length;
  });
}

// --- filters: sample-wide, per task, and a band per measured score; all combine with AND ---
var TOLERANCE = 1e-6;
var wantedSamples = 'all';
var bands = {};

function round3(value) {
  return String(parseFloat(value.toFixed(3)));
}

// A sample is correct only when every task it was judged on matched. One wrong
// task makes the whole sample a mistake; a sample judged on nothing is neither,
// and stays out of both narrowed views.
function passesSampleFilter(verdicts) {
  if (wantedSamples === 'all') { return true; }
  var judged = Object.keys(verdicts);
  if (!judged.length) { return false; }
  var missed = judged.some(function (task) { return verdicts[task] !== 'correct'; });
  return wantedSamples === 'mistakes' ? missed : !missed;
}

// A band is keyed by task AND metric, so a task measured two ways gets two of
// them. An untouched slider filters nothing; once narrowed it also drops samples
// that never earned that score, because "iou below 0.4" cannot include a sample
// without an iou.
function passesBands(scores) {
  return Object.keys(bands).every(function (key) {
    var band = bands[key];
    if (!band.narrowed) { return true; }
    var value = scores[key];
    return value !== undefined && value >= band.low - TOLERANCE && value <= band.high + TOLERANCE;
  });
}

function applyFilters() {
  var cells = document.querySelectorAll('.grid > .cell');
  var shown = 0;
  cells.forEach(function (cell) {
    var verdicts = JSON.parse(cell.dataset.verdicts || '{}');
    var scores = JSON.parse(cell.dataset.scores || '{}');
    var visible = passesSampleFilter(verdicts) && passesBands(scores);
    if (visible) { shown += 1; }
    cell.classList.toggle('hidden', !visible);
  });
  var readout = document.getElementById('shown');
  if (readout) { readout.textContent = shown + ' / ' + cells.length + ' shown'; }
  var empty = document.getElementById('empty');
  if (empty) { empty.classList.toggle('hidden', shown > 0 || cells.length === 0); }
}

// A filter combination that hides everything should offer the way back, not just
// an empty page that reads as a broken one.
function resetFilters() {
  wantedSamples = 'all';
  document.querySelectorAll('input.sample-verdict').forEach(function (radio) {
    radio.checked = radio.value === 'all';
  });
  document.querySelectorAll('.range').forEach(function (wrap) {
    var lowEdge = wrap.querySelector('.edge.low');
    var highEdge = wrap.querySelector('.edge.high');
    lowEdge.value = lowEdge.dataset.low;
    highEdge.value = highEdge.dataset.high;
    refreshRange(wrap);
  });
  applyFilters();
}

function refreshRange(wrap) {
  var lowEdge = wrap.querySelector('.edge.low');
  var highEdge = wrap.querySelector('.edge.high');
  var floor = parseFloat(lowEdge.dataset.low);
  var ceiling = parseFloat(lowEdge.dataset.high);
  var low = parseFloat(lowEdge.value);
  var high = parseFloat(highEdge.value);
  var span = ceiling - floor || 1;
  var fill = wrap.querySelector('.fill');
  fill.style.left = ((low - floor) / span * 100) + '%';
  fill.style.right = ((ceiling - high) / span * 100) + '%';
  wrap.querySelector('.bounds').textContent = round3(low) + ' – ' + round3(high);
  bands[wrap.dataset.key] = {
    low: low,
    high: high,
    narrowed: low > floor + TOLERANCE || high < ceiling - TOLERANCE
  };
}

document.addEventListener('input', function (event) {
  var edge = event.target.closest('.edge');
  if (!edge) { return; }
  var wrap = edge.closest('.range');
  var lowEdge = wrap.querySelector('.edge.low');
  var highEdge = wrap.querySelector('.edge.high');
  // The handles may not cross; the one being dragged stops at the other.
  if (parseFloat(lowEdge.value) > parseFloat(highEdge.value)) {
    if (edge.classList.contains('low')) { lowEdge.value = highEdge.value; }
    else { highEdge.value = lowEdge.value; }
  }
  refreshRange(wrap);
  applyFilters();
});

// --- lightbox: clicking a cell clones it large; sidebar keys keep working on the clone ---
var lightbox = document.getElementById('lb');
var holder = document.getElementById('lb-holder');
var counter = document.getElementById('lb-count');
var current = -1;

function visibleCells() {
  return Array.prototype.filter.call(document.querySelectorAll('.grid > .cell'), function (cell) {
    return !cell.classList.contains('hidden');
  });
}

function showCell(index) {
  var cells = visibleCells();
  if (!cells.length) { return; }
  current = (index + cells.length) % cells.length;
  var clone = cells[current].cloneNode(true);
  clone.querySelectorAll('.chip').forEach(function (chip) {
    if (chip.dataset.full) { chip.textContent = chip.dataset.full; }
  });
  holder.innerHTML = '';
  holder.appendChild(clone);
  counter.textContent = (current + 1) + ' / ' + cells.length;
  lightbox.classList.remove('hidden');
}

function hideLightbox() {
  lightbox.classList.add('hidden');
  current = -1;
}

// --- one click handler for the page ---
document.addEventListener('click', function (event) {
  var pill = event.target.closest('.src');
  if (pill) {
    // A source pill belongs to its cell but is not a way into the lightbox.
    event.stopPropagation();
    if (pill.classList.contains('copy')) { copySource(pill); }
    return;
  }
  var handle = event.target.closest('.node > .header > .caret, .node > .header > .title');
  if (handle) {
    var node = handle.closest('.node');
    var open = node.classList.toggle('open');
    node.querySelector('.caret').setAttribute('aria-expanded', open ? 'true' : 'false');
    return;
  }
  if (event.target.id === 'reset') {
    resetFilters();
    return;
  }
  var cell = event.target.closest('.grid > .cell');
  if (cell) {
    showCell(visibleCells().indexOf(cell));
    return;
  }
  if (event.target === lightbox) { hideLightbox(); }
});

document.addEventListener('change', function (event) {
  var box = event.target;
  if (box.classList.contains('cls')) {
    setHidden(box.dataset.key, !box.checked);
    refreshGroups();
  } else if (box.classList.contains('grp')) {
    leavesOf(box.dataset.prefix).forEach(function (leaf) {
      leaf.checked = box.checked;
      setHidden(leaf.dataset.key, !box.checked);
    });
    refreshGroups();
  } else if (box.classList.contains('sample-verdict')) {
    wantedSamples = box.value;
    applyFilters();
  }
});

document.getElementById('lb-close').addEventListener('click', hideLightbox);
document.getElementById('lb-prev').addEventListener('click', function () { showCell(current - 1); });
document.getElementById('lb-next').addEventListener('click', function () { showCell(current + 1); });

document.addEventListener('keydown', function (event) {
  if (lightbox.classList.contains('hidden')) { return; }
  if (event.key === 'Escape') { hideLightbox(); }
  if (event.key === 'ArrowLeft') { showCell(current - 1); }
  if (event.key === 'ArrowRight') { showCell(current + 1); }
});

document.querySelectorAll('.range').forEach(refreshRange);
applyFilters();
