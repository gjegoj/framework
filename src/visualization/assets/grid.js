function setHidden(key, hidden) {
  document.querySelectorAll('.layer[data-key="' + key + '"]').forEach(function (c) {
    c.classList.toggle('hidden', hidden);
  });
}
function leavesOf(prefix) {
  return Array.prototype.filter.call(document.querySelectorAll('input.cls'),
    function (c) { return c.dataset.key.indexOf(prefix) === 0; });
}
function refreshGroups() {
  document.querySelectorAll('input.grp').forEach(function (g) {
    var leaves = leavesOf(g.dataset.prefix);
    var on = leaves.filter(function (c) { return c.checked; }).length;
    g.checked = on === leaves.length && leaves.length > 0;
    g.indeterminate = on > 0 && on < leaves.length;
  });
}
document.querySelectorAll('input.cls').forEach(function (cb) {
  cb.addEventListener('change', function () { setHidden(cb.dataset.key, !cb.checked); refreshGroups(); });
});
document.querySelectorAll('input.grp').forEach(function (cb) {
  cb.addEventListener('change', function () {
    leavesOf(cb.dataset.prefix).forEach(function (child) {
      child.checked = cb.checked;
      setHidden(child.dataset.key, !cb.checked);
    });
    refreshGroups();
  });
});
document.querySelectorAll('.caret, .title').forEach(function (el) {
  el.addEventListener('click', function () { el.closest('.node').classList.toggle('open'); });
});

// Lightbox: click a cell to inspect it large. The clone keeps each layer's data-key and
// current .hidden state, so the sidebar's setHidden toggles the zoomed view live.
(function () {
  var cells = Array.prototype.slice.call(document.querySelectorAll('.grid > .cell'));
  var lb = document.getElementById('lb');
  if (!lb || !cells.length) { return; }
  var holder = document.getElementById('lb-holder');
  var count = document.getElementById('lb-count');
  var idx = -1;
  function show(i) {
    idx = (i + cells.length) % cells.length;
    holder.innerHTML = '';
    holder.appendChild(cells[idx].cloneNode(true));
    count.textContent = (idx + 1) + ' / ' + cells.length;
    lb.classList.remove('hidden');
  }
  function close() { lb.classList.add('hidden'); holder.innerHTML = ''; }
  cells.forEach(function (cell, i) { cell.addEventListener('click', function () { show(i); }); });
  document.getElementById('lb-prev').addEventListener('click', function (e) { e.stopPropagation(); show(idx - 1); });
  document.getElementById('lb-next').addEventListener('click', function (e) { e.stopPropagation(); show(idx + 1); });
  document.getElementById('lb-close').addEventListener('click', function (e) { e.stopPropagation(); close(); });
  lb.addEventListener('click', function (e) { if (e.target === lb) { close(); } });
  document.addEventListener('keydown', function (e) {
    if (lb.classList.contains('hidden')) { return; }
    if (e.key === 'Escape') { close(); }
    else if (e.key === 'ArrowLeft') { show(idx - 1); }
    else if (e.key === 'ArrowRight') { show(idx + 1); }
  });
})();
