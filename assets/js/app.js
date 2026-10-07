(() => {
  const PDFJS_WORKER_URL =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

  const state = {
    documents: [],
    filteredDocs: [],
    activeDoc: null,
    pdfDoc: null,
    currentPage: 1,
    totalPages: 1,
    zoom: 1.0,
    basePageWidth: 800,
    pageAspectRatio: 1.294,
    renderedPages: new Set(),
    renderingPages: new Map(),
    pageObserver: null,
    scrollObserver: null,
    invertPages: localStorage.getItem('thebox_invert_pages') === '1',
    isProgrammaticScroll: false
  };

  const els = {
    boxView: document.getElementById('box-view'),
    readerView: document.getElementById('reader-view'),
    docList: document.getElementById('doc-list'),
    docCount: document.getElementById('doc-count'),
    searchInput: document.getElementById('search-input'),
    syncStatus: document.getElementById('sync-status'),
    btnBack: document.getElementById('btn-back'),
    readerTitle: document.getElementById('reader-title'),
    btnPrevPage: document.getElementById('btn-prev-page'),
    btnNextPage: document.getElementById('btn-next-page'),
    pageInput: document.getElementById('page-input'),
    pageTotal: document.getElementById('page-total'),
    btnZoomOut: document.getElementById('btn-zoom-out'),
    btnZoomReset: document.getElementById('btn-zoom-reset'),
    btnZoomIn: document.getElementById('btn-zoom-in'),
    btnInvert: document.getElementById('btn-invert'),
    btnRaw: document.getElementById('btn-raw'),
    readerViewport: document.getElementById('reader-viewport'),
    readerStatus: document.getElementById('reader-status'),
    pagesContainer: document.getElementById('pages-container')
  };

  function getSavedPage(docId) {
    const val = parseInt(localStorage.getItem(`thebox_page_${docId}`), 10);
    return Number.isFinite(val) && val > 0 ? val : 1;
  }

  function savePage(docId, pageNum) {
    if (!docId || !pageNum) return;
    localStorage.setItem(`thebox_page_${docId}`, String(pageNum));
  }

  function slugToTitle(filename) {
    const base = filename.replace(/\.[^.]+$/, '');
    return base
      .replace(/[-_]+/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return '';
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(0)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
  }

  async function fetchGithubFolderFallback(existingDocs) {
    const host = window.location.hostname;
    if (!host.endsWith('.github.io')) return existingDocs;

    const owner = host.replace('.github.io', '');
    const pathParts = window.location.pathname.split('/').filter(Boolean);
    const repo = pathParts[0] || `${owner}.github.io`;

    try {
      const res = await fetch(
        `https://api.github.com/repos/${owner}/${repo}/contents/documents`
      );
      if (!res.ok) return existingDocs;
      const items = await res.json();
      if (!Array.isArray(items)) return existingDocs;

      const byFile = new Map(existingDocs.map((d) => [d.filename, d]));
      for (const item of items) {
        if (item.type !== 'file') continue;
        const extMatch = item.name.match(/\.(pdf|md|txt)$/i);
        if (!extMatch) continue;
        if (!byFile.has(item.name)) {
          const ext = extMatch[1].toLowerCase();
          const id = item.name.replace(/\.[^.]+$/, '').toLowerCase();
          byFile.set(item.name, {
            id,
            title: slugToTitle(item.name),
            subtitle: '',
            filename: item.name,
            path: `documents/${item.name}`,
            type: ext,
            pages: null,
            size: formatBytes(item.size),
            added: ''
          });
        }
      }
      return Array.from(byFile.values());
    } catch (_) {
      return existingDocs;
    }
  }

  async function loadDocuments() {
    let docs = [];
    try {
      const res = await fetch(`documents/manifest.json?t=${Date.now()}`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data.documents)) {
          docs = data.documents;
        }
      }
    } catch (_) {
      // fallback if manifest fetch fails
    }

    docs = await fetchGithubFolderFallback(docs);
    state.documents = docs;
    state.filteredDocs = docs;
    renderDocList();
    handleHashRoute();
  }

  function renderDocList() {
    const docs = state.filteredDocs;
    els.docCount.textContent = String(docs.length);
    els.docList.innerHTML = '';

    if (docs.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.textContent = state.documents.length
        ? 'No matching documents'
        : 'No documents in documents/';
      els.docList.appendChild(empty);
      return;
    }

    for (const doc of docs) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'doc-item';
      btn.setAttribute('role', 'listitem');

      const savedPage = getSavedPage(doc.id);
      const showResume = savedPage > 1 && (!doc.pages || savedPage <= doc.pages);

      const main = document.createElement('div');
      main.className = 'doc-main';

      const titleRow = document.createElement('div');
      titleRow.className = 'doc-title-row';

      const titleEl = document.createElement('span');
      titleEl.className = 'doc-title';
      titleEl.textContent = doc.title || doc.filename;
      titleRow.appendChild(titleEl);
      main.appendChild(titleRow);

      if (doc.subtitle) {
        const subEl = document.createElement('span');
        subEl.className = 'doc-subtitle';
        subEl.textContent = doc.subtitle;
        main.appendChild(subEl);
      }

      const meta = document.createElement('div');
      meta.className = 'doc-meta';

      if (showResume) {
        const resumePill = document.createElement('span');
        resumePill.className = 'meta-pill meta-resume';
        resumePill.textContent = `P. ${savedPage}`;
        meta.appendChild(resumePill);
      }

      if (doc.pages) {
        const pagesSpan = document.createElement('span');
        pagesSpan.textContent = `${doc.pages}p`;
        meta.appendChild(pagesSpan);
      }

      if (doc.size) {
        const sizeSpan = document.createElement('span');
        sizeSpan.textContent = doc.size;
        meta.appendChild(sizeSpan);
      }

      const typePill = document.createElement('span');
      typePill.className = 'meta-pill';
      typePill.textContent = (doc.type || 'pdf').toUpperCase();
      meta.appendChild(typePill);

      const arrow = document.createElement('span');
      arrow.className = 'doc-arrow';
      arrow.innerHTML =
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';
      meta.appendChild(arrow);

      btn.appendChild(main);
      btn.appendChild(meta);

      btn.addEventListener('click', () => {
        openDocument(doc);
      });

      els.docList.appendChild(btn);
    }
  }

  function filterDocuments(query) {
    const q = query.trim().toLowerCase();
    if (!q) {
      state.filteredDocs = state.documents;
    } else {
      state.filteredDocs = state.documents.filter((doc) => {
        const hay = `${doc.title || ''} ${doc.subtitle || ''} ${doc.filename || ''} ${doc.category || ''}`.toLowerCase();
        return hay.includes(q);
      });
    }
    renderDocList();
  }

  function computeBaseWidth() {
    const vw = els.readerViewport.clientWidth || window.innerWidth;
    const isMobile = window.innerWidth <= 640;
    const horizontalPad = isMobile ? 8 : 36;
    const maxDesktopWidth = 840;
    return Math.max(280, Math.min(vw - horizontalPad, maxDesktopWidth));
  }

  function updateInvertUI() {
    if (state.invertPages) {
      els.pagesContainer.classList.add('invert-pages');
      els.btnInvert.classList.add('active');
    } else {
      els.pagesContainer.classList.remove('invert-pages');
      els.btnInvert.classList.remove('active');
    }
  }

  function cleanupReader() {
    if (state.pageObserver) {
      state.pageObserver.disconnect();
      state.pageObserver = null;
    }
    if (state.scrollObserver) {
      state.scrollObserver.disconnect();
      state.scrollObserver = null;
    }
    state.renderedPages.clear();
    state.renderingPages.clear();
    if (state.pdfDoc) {
      state.pdfDoc.destroy();
      state.pdfDoc = null;
    }
    els.pagesContainer.innerHTML = '';
  }

  async function openDocument(doc, initialPage) {
    cleanupReader();
    state.activeDoc = doc;
    state.zoom = 1.0;
    els.btnZoomReset.textContent = '100%';

    els.boxView.classList.add('hidden');
    els.readerView.classList.remove('hidden');

    els.readerTitle.textContent = doc.title || doc.filename;
    els.btnRaw.href = doc.path;
    document.title = `${doc.title || doc.filename} — TheBox`;

    const targetPage = initialPage || getSavedPage(doc.id) || 1;
    updateHash(doc.id, targetPage);
    updateInvertUI();

    const ext = (doc.type || 'pdf').toLowerCase();
    if (ext === 'md' || ext === 'txt') {
      await renderTextDocument(doc);
    } else {
      await renderPdfDocument(doc, targetPage);
    }
  }

  async function renderTextDocument(doc) {
    els.readerStatus.textContent = 'Loading document...';
    els.readerStatus.classList.remove('hidden');
    els.pageInput.value = '1';
    els.pageTotal.textContent = '1';

    try {
      const res = await fetch(doc.path);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      els.readerStatus.classList.add('hidden');

      const surface = document.createElement('article');
      surface.className = 'text-doc-surface';
      surface.textContent = text;
      els.pagesContainer.appendChild(surface);
    } catch (err) {
      els.readerStatus.textContent = `Unable to load document (${err.message})`;
    }
  }

  async function waitForPdfJs() {
    let attempts = 0;
    while (!window.pdfjsLib && attempts < 50) {
      await new Promise((r) => setTimeout(r, 100));
      attempts++;
    }
    if (!window.pdfjsLib) {
      throw new Error('PDF engine unavailable');
    }
    window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_URL;
  }

  async function renderPdfDocument(doc, startPage) {
    els.readerStatus.textContent = 'Loading PDF...';
    els.readerStatus.classList.remove('hidden');

    try {
      await waitForPdfJs();
      const loadingTask = window.pdfjsLib.getDocument({
        url: doc.path,
        cMapUrl: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/cmaps/',
        cMapPacked: true
      });

      const pdfDoc = await loadingTask.promise;
      if (state.activeDoc !== doc) return;

      state.pdfDoc = pdfDoc;
      state.totalPages = pdfDoc.numPages;
      els.pageTotal.textContent = String(pdfDoc.numPages);
      els.pageInput.max = String(pdfDoc.numPages);

      const firstPage = await pdfDoc.getPage(1);
      const vp1 = firstPage.getViewport({ scale: 1 });
      state.pageAspectRatio = vp1.height / vp1.width;
      state.basePageWidth = computeBaseWidth();

      els.readerStatus.classList.add('hidden');
      buildPageSlots(pdfDoc.numPages);
      setupObservers();

      const validPage = Math.min(Math.max(1, startPage), pdfDoc.numPages);
      state.currentPage = validPage;
      els.pageInput.value = String(validPage);

      if (validPage > 1) {
        requestAnimationFrame(() => {
          jumpToPage(validPage, false);
        });
      }
    } catch (err) {
      els.readerStatus.innerHTML = `Could not render inline PDF. <a href="${doc.path}" target="_blank" rel="noopener" style="color:#ededf0;text-decoration:underline;">Open PDF directly</a>`;
    }
  }

  function buildPageSlots(numPages) {
    els.pagesContainer.innerHTML = '';
    const targetWidth = Math.round(state.basePageWidth * state.zoom);
    const targetHeight = Math.round(targetWidth * state.pageAspectRatio);

    const frag = document.createDocumentFragment();
    for (let i = 1; i <= numPages; i++) {
      const slot = document.createElement('div');
      slot.className = 'pdf-page-slot';
      slot.dataset.page = String(i);
      slot.style.width = `${targetWidth}px`;
      slot.style.height = `${targetHeight}px`;

      const wm = document.createElement('span');
      wm.className = 'page-number-watermark';
      wm.textContent = String(i);
      slot.appendChild(wm);

      frag.appendChild(slot);
    }
    els.pagesContainer.appendChild(frag);
  }

  function setupObservers() {
    if (state.pageObserver) state.pageObserver.disconnect();
    if (state.scrollObserver) state.scrollObserver.disconnect();

    state.pageObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const pageNum = parseInt(entry.target.dataset.page, 10);
            renderPageCanvas(pageNum, entry.target);
          }
        }
      },
      {
        root: els.readerViewport,
        rootMargin: '900px 0px 900px 0px',
        threshold: 0.01
      }
    );

    state.scrollObserver = new IntersectionObserver(
      (entries) => {
        if (state.isProgrammaticScroll) return;
        let bestPage = null;
        let bestRatio = 0;
        for (const entry of entries) {
          if (entry.isIntersecting && entry.intersectionRatio > bestRatio) {
            bestRatio = entry.intersectionRatio;
            bestPage = parseInt(entry.target.dataset.page, 10);
          }
        }
        if (bestPage && bestPage !== state.currentPage) {
          state.currentPage = bestPage;
          els.pageInput.value = String(bestPage);
          if (state.activeDoc) {
            savePage(state.activeDoc.id, bestPage);
            updateHash(state.activeDoc.id, bestPage, true);
          }
        }
      },
      {
        root: els.readerViewport,
        threshold: [0.15, 0.4, 0.7]
      }
    );

    const slots = els.pagesContainer.querySelectorAll('.pdf-page-slot');
    slots.forEach((slot) => {
      state.pageObserver.observe(slot);
      state.scrollObserver.observe(slot);
    });
  }

  async function renderPageCanvas(pageNum, slotEl) {
    if (!state.pdfDoc) return;
    const renderKey = `${pageNum}@${state.zoom.toFixed(2)}@${state.basePageWidth}`;
    if (state.renderedPages.has(renderKey) || state.renderingPages.has(pageNum)) {
      return;
    }

    state.renderingPages.set(pageNum, renderKey);
    try {
      const page = await state.pdfDoc.getPage(pageNum);
      if (!state.pdfDoc) return;

      const unscaledVp = page.getViewport({ scale: 1 });
      const targetCssWidth = Math.round(state.basePageWidth * state.zoom);
      const targetCssHeight = Math.round(
        targetCssWidth * (unscaledVp.height / unscaledVp.width)
      );

      slotEl.style.width = `${targetCssWidth}px`;
      slotEl.style.height = `${targetCssHeight}px`;

      const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
      const scale = (targetCssWidth / unscaledVp.width) * dpr;
      const viewport = page.getViewport({ scale });

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d', { alpha: false });
      canvas.width = Math.floor(viewport.width);
      canvas.height = Math.floor(viewport.height);

      await page.render({
        canvasContext: ctx,
        viewport
      }).promise;

      const oldCanvas = slotEl.querySelector('canvas');
      if (oldCanvas) oldCanvas.remove();
      slotEl.appendChild(canvas);

      state.renderedPages.add(renderKey);
    } catch (_) {
      // ignore cancelled renders
    } finally {
      state.renderingPages.delete(pageNum);
    }
  }

  function applyZoom(newZoom) {
    if (!state.pdfDoc) return;
    const clamped = Math.min(2.5, Math.max(0.5, Math.round(newZoom * 100) / 100));
    if (clamped === state.zoom) return;

    const prevPage = state.currentPage;
    state.zoom = clamped;
    els.btnZoomReset.textContent = `${Math.round(state.zoom * 100)}%`;
    state.renderedPages.clear();

    const targetWidth = Math.round(state.basePageWidth * state.zoom);
    const targetHeight = Math.round(targetWidth * state.pageAspectRatio);

    const slots = els.pagesContainer.querySelectorAll('.pdf-page-slot');
    slots.forEach((slot) => {
      slot.style.width = `${targetWidth}px`;
      slot.style.height = `${targetHeight}px`;
    });

    jumpToPage(prevPage, false);
    setupObservers();
  }

  function jumpToPage(pageNum, smooth = true) {
    if (!state.pdfDoc) return;
    const target = Math.min(Math.max(1, pageNum), state.totalPages);
    state.currentPage = target;
    els.pageInput.value = String(target);

    if (state.activeDoc) {
      savePage(state.activeDoc.id, target);
      updateHash(state.activeDoc.id, target, true);
    }

    const slot = els.pagesContainer.querySelector(
      `.pdf-page-slot[data-page="${target}"]`
    );
    if (slot) {
      state.isProgrammaticScroll = true;
      slot.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' });
      setTimeout(() => {
        state.isProgrammaticScroll = false;
      }, smooth ? 450 : 60);
    }
  }

  function closeReader() {
    cleanupReader();
    state.activeDoc = null;
    els.readerView.classList.add('hidden');
    els.boxView.classList.remove('hidden');
    document.title = 'TheBox';
    history.replaceState(null, '', window.location.pathname + window.location.search);
    renderDocList();
  }

  function updateHash(docId, pageNum, replace = false) {
    const hash = `#doc=${encodeURIComponent(docId)}&page=${pageNum}`;
    if (replace) {
      history.replaceState(null, '', hash);
    } else {
      history.pushState(null, '', hash);
    }
  }

  function handleHashRoute() {
    const raw = window.location.hash.replace(/^#/, '');
    if (!raw) {
      if (state.activeDoc) closeReader();
      return;
    }
    const params = new URLSearchParams(raw);
    const docKey = params.get('doc');
    const pageParam = parseInt(params.get('page'), 10);
    if (!docKey) return;

    const found = state.documents.find(
      (d) => d.id === docKey || d.filename === docKey
    );
    if (found) {
      if (state.activeDoc !== found) {
        openDocument(found, Number.isFinite(pageParam) ? pageParam : undefined);
      } else if (Number.isFinite(pageParam) && pageParam !== state.currentPage) {
        jumpToPage(pageParam, false);
      }
    }
  }

  // Event Listeners
  els.searchInput.addEventListener('input', (e) => {
    filterDocuments(e.target.value);
  });

  els.btnBack.addEventListener('click', () => {
    closeReader();
  });

  els.btnPrevPage.addEventListener('click', () => {
    jumpToPage(state.currentPage - 1, true);
  });

  els.btnNextPage.addEventListener('click', () => {
    jumpToPage(state.currentPage + 1, true);
  });

  els.pageInput.addEventListener('change', () => {
    const val = parseInt(els.pageInput.value, 10);
    if (Number.isFinite(val)) {
      jumpToPage(val, false);
    } else {
      els.pageInput.value = String(state.currentPage);
    }
  });

  els.pageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.target.blur();
    }
  });

  els.btnZoomIn.addEventListener('click', () => {
    applyZoom(state.zoom + 0.2);
  });

  els.btnZoomOut.addEventListener('click', () => {
    applyZoom(state.zoom - 0.2);
  });

  els.btnZoomReset.addEventListener('click', () => {
    state.basePageWidth = computeBaseWidth();
    applyZoom(1.0);
  });

  els.btnInvert.addEventListener('click', () => {
    state.invertPages = !state.invertPages;
    localStorage.setItem('thebox_invert_pages', state.invertPages ? '1' : '0');
    updateInvertUI();
  });

  window.addEventListener('hashchange', () => {
    handleHashRoute();
  });

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    if (!state.pdfDoc) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      const nextBase = computeBaseWidth();
      if (Math.abs(nextBase - state.basePageWidth) > 12) {
        state.basePageWidth = nextBase;
        state.renderedPages.clear();
        const targetWidth = Math.round(state.basePageWidth * state.zoom);
        const targetHeight = Math.round(targetWidth * state.pageAspectRatio);
        els.pagesContainer.querySelectorAll('.pdf-page-slot').forEach((slot) => {
          slot.style.width = `${targetWidth}px`;
          slot.style.height = `${targetHeight}px`;
        });
        setupObservers();
      }
    }, 180);
  });

  window.addEventListener('keydown', (e) => {
    const tag = document.activeElement ? document.activeElement.tagName : '';
    const inInput = tag === 'INPUT' || tag === 'TEXTAREA';

    if (!state.activeDoc) {
      if (e.key === '/' && !inInput) {
        e.preventDefault();
        els.searchInput.focus();
      } else if (e.key === 'Escape' && inInput) {
        els.searchInput.value = '';
        filterDocuments('');
        els.searchInput.blur();
      }
      return;
    }

    if (inInput) return;

    if (e.key === 'Escape') {
      e.preventDefault();
      closeReader();
    } else if (e.key === 'ArrowRight' || e.key === 'PageDown') {
      e.preventDefault();
      jumpToPage(state.currentPage + 1, true);
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      e.preventDefault();
      jumpToPage(state.currentPage - 1, true);
    } else if (e.key === '+' || e.key === '=') {
      e.preventDefault();
      applyZoom(state.zoom + 0.2);
    } else if (e.key === '-' || e.key === '_') {
      e.preventDefault();
      applyZoom(state.zoom - 0.2);
    } else if (e.key === '0') {
      e.preventDefault();
      applyZoom(1.0);
    } else if (e.key.toLowerCase() === 'i') {
      e.preventDefault();
      els.btnInvert.click();
    }
  });

  loadDocuments();
})();

