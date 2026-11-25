/* Shared light/dark background theme (24 Nov 2026).
   One preference in localStorage['trilo_theme'] = 'dark' (default) | 'light', applied as
   <html data-theme="...">. The :root[data-theme=light] block in each page flips the CSS tokens
   (--bg/--ink/...); drawing surfaces (the THREE 3D viewer, the prism, the p5 editor, the server
   blueprint) subscribe via TriloTheme.onChange and recolor themselves — CSS alone can't reach a
   canvas or a WebGL clear colour.
   The choice travels: a page pushes it to its own iframes and to its parent (the studio embeds the
   3D viewer and the prism), and across tabs via the storage event. Embedded views (?only3d / ?embed)
   follow the parent and show no switch of their own. */
(function () {
  var KEY = 'trilo_theme';
  var root = document.documentElement;
  var subs = [];

  function get() { try { return localStorage.getItem(KEY) === 'light' ? 'light' : 'dark'; } catch (_) { return 'dark'; } }
  function apply(t) { root.setAttribute('data-theme', t); }
  function notify(t) { for (var i = 0; i < subs.length; i++) { try { subs[i](t); } catch (_) {} } }

  apply(get());                                   // in <head>, before first paint — no flash

  function relay(t, except) {                     // hand the choice to any iframes we host and to our parent
    try { var fr = document.getElementsByTagName('iframe');
      for (var i = 0; i < fr.length; i++) { var w = fr[i].contentWindow; if (w && w !== except) w.postMessage({ type: 'trilo-theme', theme: t }, '*'); } } catch (_) {}
    try { if (window.parent && window.parent !== window && window.parent !== except) window.parent.postMessage({ type: 'trilo-theme', theme: t }, '*'); } catch (_) {}
  }

  function set(t, fromRemote, source) {
    t = (t === 'light') ? 'light' : 'dark';
    if (t === get()) { if (!fromRemote) relay(t, source); return; }   // still relay a no-op so late-mounted iframes catch up
    try { localStorage.setItem(KEY, t); } catch (_) {}
    apply(t); notify(t); relay(t, source);
    var sw = document.getElementById('trilo-theme-sw');
    if (sw) { var b = sw.getElementsByTagName('button'); for (var i = 0; i < b.length; i++) b[i].className = (b[i].getAttribute('data-t') === t) ? 'on' : ''; }
  }

  window.TriloTheme = {
    get: get,
    isLight: function () { return get() === 'light'; },
    set: function (t) { set(t, false); },
    toggle: function () { set(get() === 'light' ? 'dark' : 'light', false); },
    onChange: function (f) { subs.push(f); return f; }
  };

  window.addEventListener('message', function (e) {
    var d = e.data; if (d && d.type === 'trilo-theme' && (d.theme === 'light' || d.theme === 'dark')) set(d.theme, true, e.source);
  });
  window.addEventListener('storage', function (e) { if (e.key === KEY && e.newValue) set(e.newValue, true); });

  // ---- the switch: a small segmented DARK | LIGHT control, injected bottom-left (opt-in per page via
  //      <html data-theme-switch="on">), hidden in embedded views where the parent owns the choice.
  function mount() {
    if (root.getAttribute('data-theme-switch') !== 'on') return;
    var u = new URLSearchParams(location.search);
    if (u.has('only3d') || u.has('embed')) return;                   // an embedded surface: follow the host, no switch
    if (document.getElementById('trilo-theme-sw')) return;

    var st = document.createElement('style');
    var bottom = root.getAttribute('data-theme-sw-bottom') || '12';
    st.textContent =
      '#trilo-theme-sw{position:fixed;left:14px;bottom:' + bottom + 'px;z-index:9999;display:flex;' +
      'border:1px solid var(--line,#262626);background:var(--bg,#000);font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;' +
      'font-size:10px;letter-spacing:.1em;text-transform:uppercase;user-select:none;opacity:.9}' +
      '#trilo-theme-sw:hover{opacity:1}' +
      '#trilo-theme-sw button{background:transparent;color:var(--mid,#9a9a9a);border:none;border-right:1px solid var(--line,#262626);' +
      'padding:6px 10px;font:inherit;letter-spacing:inherit;text-transform:inherit;cursor:pointer}' +
      '#trilo-theme-sw button:last-child{border-right:none}' +
      '#trilo-theme-sw button.on{background:var(--ink,#fff);color:var(--bg,#000)}';
    document.head.appendChild(st);

    var box = document.createElement('div'); box.id = 'trilo-theme-sw'; box.title = 'light / dark background';
    var cur = get();
    [['dark', 'dark'], ['light', 'light']].forEach(function (p) {
      var b = document.createElement('button'); b.setAttribute('data-t', p[0]); b.textContent = p[1];
      if (p[0] === cur) b.className = 'on';
      b.onclick = function () { set(p[0], false); };
      box.appendChild(b);
    });
    document.body.appendChild(box);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount); else mount();
})();
