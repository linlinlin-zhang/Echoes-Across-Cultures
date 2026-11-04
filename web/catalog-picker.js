/* Browse the public, cloud-hosted catalog without paid API calls. */
(() => {
  window.EchoCatalogPicker = {
    mount({onSelect}) {
      const t = (key, params) => window.EchoI18n.t(`music.catalog.${key}`, params);
      const dialog = document.createElement('dialog');
      dialog.className = 'catalog-picker';
      dialog.setAttribute('aria-labelledby', 'catalog-heading');
      dialog.innerHTML = `
        <div class="catalog-heading"><h2 id="catalog-heading" data-i18n="music.catalog.choose"></h2>
          <button class="pill" type="button" data-close data-i18n="music.catalog.close"></button></div>
        <form class="catalog-search" role="search">
          <input id="catalog-query" type="search" maxlength="200" autocomplete="off"
            data-i18n-placeholder="music.catalog.query" data-i18n-aria-label="music.catalog.query">
          <button class="pill" type="submit" data-i18n="music.catalog.search"></button>
          <button class="pill" type="button" data-shuffle data-i18n="music.catalog.shuffle"></button>
        </form>
        <p class="catalog-status" role="status" aria-live="polite"></p>
        <button class="pill" type="button" data-retry hidden data-i18n="music.catalog.retry"></button>
        <ul class="catalog-results"></ul>`;
      document.body.append(dialog);
      window.EchoI18n.apply(dialog);
      const query = dialog.querySelector('input');
      const list = dialog.querySelector('ul');
      const status = dialog.querySelector('[role="status"]');
      const retry = dialog.querySelector('[data-retry]');
      let controller, timer, items = [], total = 0, state = 'idle', seed = 42;
      function render() {
        status.textContent = state === 'ready'
          ? (items.length ? t('count', {shown: items.length, total}) : t('empty'))
          : state === 'idle' ? '' : t(state);
        retry.hidden = state !== 'error';
        list.replaceChildren();
        items.forEach(item => {
          const li = document.createElement('li');
          const button = document.createElement('button');
          button.type = 'button';
          const title = document.createElement('strong');
          title.textContent = item.title || '—';
          const subtitle = document.createElement('span');
          subtitle.textContent = [item.artist, item.source_dataset].filter(Boolean).join(' · ');
          button.append(title, subtitle);
          button.addEventListener('click', () => {onSelect(item); dialog.close();});
          li.append(button); list.append(li);
        });
      }
      async function load() {
        clearTimeout(timer);
        controller?.abort();
        const current = controller = new AbortController();
        items = []; state = 'loading'; render();
        const timeout = setTimeout(() => current.abort(), 20000);
        try {
          const params = new URLSearchParams({q: query.value.trim(), limit: '24', random_seed: String(seed)});
          const response = await fetch(`/api/mainline/catalog?${params}`, {signal: current.signal});
          if (!response.ok) throw new Error(`Catalog HTTP ${response.status}`);
          const data = await response.json();
          if (controller !== current || !dialog.open) return;
          if (!Array.isArray(data.items)) throw new Error('Invalid catalog response');
          items = data.items; total = data.total_available ?? items.length; state = 'ready'; render();
        } catch (error) {
          if (controller !== current || !dialog.open) return;
          state = 'error'; render();
        } finally {clearTimeout(timeout);}
      }
      document.getElementById('catalog-button').addEventListener('click', () => {
        dialog.showModal(); query.focus(); load();
      });
      dialog.querySelector('[data-close]').addEventListener('click', () => dialog.close());
      dialog.addEventListener('close', () => {clearTimeout(timer); controller?.abort(); controller = null;});
      dialog.querySelector('form').addEventListener('submit', event => {event.preventDefault(); seed = 42; load();});
      query.addEventListener('input', () => {
        controller?.abort(); controller = null; clearTimeout(timer);
        items = []; state = 'loading'; render();
        timer = setTimeout(() => {seed = 42; load();}, 300);
      });
      dialog.querySelector('[data-shuffle]').addEventListener('click', () => {seed = Date.now(); load();});
      retry.addEventListener('click', load);
      window.addEventListener('echo-language-change', () => {window.EchoI18n.apply(dialog); render();});
    }
  };
})();
