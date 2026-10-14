export function nextCaseIndex(key, current, count) {
  if (key === 'ArrowRight') return (current + 1) % count;
  if (key === 'ArrowLeft') return (current - 1 + count) % count;
  if (key === 'Home') return 0;
  if (key === 'End') return count - 1;
  return current;
}

export function paperHref(value = '') {
  if (!value.trim()) return null;
  try {
    const url = new URL(value.trim());
    return ['https:', 'http:'].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

export function activeSectionIndex(tops, readingLine, atPageEnd = false) {
  if (atPageEnd) return tops.length - 1;
  for (let index = tops.length - 1; index >= 0; index -= 1) {
    if (tops[index] <= readingLine) return index;
  }
  return -1;
}

function setupNavigation() {
  const header = document.querySelector('.site-header');
  const links = [...header.querySelectorAll('a[href^="#"]')];
  const sections = links.map(link => document.getElementById(link.hash.slice(1)));
  let scheduled = false;
  let active = -1;
  let scrollOffset = 0;

  const update = () => {
    scheduled = false;
    const headerBottom = header.getBoundingClientRect().bottom;
    const offset = Math.ceil(headerBottom + 24);
    if (offset !== scrollOffset) {
      document.documentElement.style.setProperty('--scroll-offset', `${offset}px`);
      scrollOffset = offset;
    }
    const readingLine = headerBottom + Math.min(window.innerHeight * .2, 160);
    const atPageEnd = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2;
    const next = activeSectionIndex(sections.map(section => section.getBoundingClientRect().top), readingLine, atPageEnd);
    if (next === active) return;
    links.forEach((link, index) => {
      if (index === next) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
    active = next;
  };
  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(update);
  };

  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  window.addEventListener('hashchange', schedule);
  window.addEventListener('load', schedule);
  if (typeof ResizeObserver !== 'undefined') {
    const observer = new ResizeObserver(schedule);
    observer.observe(header);
    observer.observe(document.querySelector('main'));
  }
  document.fonts?.ready.then(schedule);
  update();
}

function setupPaper() {
  const link = document.getElementById('paper-link');
  const href = paperHref(link.dataset.url);
  if (href) {
    link.href = href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.removeAttribute('aria-disabled');
    link.setAttribute('aria-label', 'Paper');
  } else {
    link.addEventListener('click', event => event.preventDefault());
    link.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') event.preventDefault();
    });
  }
}

function setupCases() {
  const tabs = [...document.querySelectorAll('[data-case]')];
  const panels = tabs.map(tab => document.getElementById(tab.dataset.case));
  document.querySelector('.case-tabs').setAttribute('role', 'tablist');
  tabs.forEach((tab, index) => {
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', panels[index].id);
    panels[index].setAttribute('role', 'tabpanel');
    panels[index].setAttribute('aria-labelledby', tab.id);
    panels[index].tabIndex = 0;
  });
  const select = index => {
    tabs.forEach((tab, i) => {
      tab.setAttribute('aria-selected', String(i === index));
      tab.tabIndex = i === index ? 0 : -1;
      panels[i].hidden = i !== index;
    });
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', event => {
      event.preventDefault();
      select(index);
    });
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = nextCaseIndex(event.key, index, tabs.length);
      select(next);
      tabs[next].focus();
    });
  });
  const selectHash = () => {
    const index = panels.findIndex(panel => `#${panel.id}` === window.location.hash);
    if (index >= 0) select(index);
  };
  select(0);
  selectHash();
  window.addEventListener('hashchange', selectHash);
}

function setupFigures() {
  const dialog = document.getElementById('figure-dialog');
  if (typeof dialog.showModal !== 'function') return;
  const enlarged = document.getElementById('dialog-image');
  let trigger;
  document.querySelectorAll('[data-zoom]').forEach(link => {
    link.setAttribute('aria-haspopup', 'dialog');
    link.addEventListener('click', event => {
      event.preventDefault();
      const image = link.querySelector('img');
      trigger = link;
      enlarged.src = image.currentSrc || image.src;
      enlarged.alt = image.alt;
      enlarged.classList.toggle('portrait-figure', image.naturalWidth / image.naturalHeight < 1.2);
      document.getElementById('dialog-caption').textContent = link.closest('figure').querySelector('figcaption').textContent;
      dialog.showModal();
      document.body.classList.add('modal-open');
    });
  });
  dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.body.classList.remove('modal-open');
    trigger?.focus({ preventScroll: true });
  });
}

function setupCitation() {
  const button = document.getElementById('copy-citation');
  const citation = document.getElementById('bibtex');
  const status = document.getElementById('copy-status');
  button.hidden = false;
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(citation.textContent.trim());
      status.textContent = 'BibTeX copied.';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(citation);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = 'Automatic copying is unavailable. The citation is selected for you to copy manually.';
    }
  });
}

if (typeof document !== 'undefined') {
  setupPaper();
  setupCases();
  setupFigures();
  setupCitation();
  setupNavigation();
}
