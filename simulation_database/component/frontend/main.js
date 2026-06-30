// Simulation database panel — tree, search, workspace

let tokensStyleEl = null;

function injectTokensCss(css) {
  if (!css) return;
  if (!tokensStyleEl) {
    tokensStyleEl = document.createElement("style");
    tokensStyleEl.setAttribute("data-design-tokens", "1");
    document.head.appendChild(tokensStyleEl);
  }
  tokensStyleEl.textContent = css;
}

let lastArgs = null;
let focusedPathId = null;
let contextTarget = null;
let searchDebounce = null;
let clickTimer = null;
const CLICK_DELAY_MS = 180;
const SEARCH_DEBOUNCE_MS = 50;
const SEARCH_MAX_RESULTS = 80;
const MIN_INVERTED_TOKEN_LEN = 2;
const CATALOG_STORAGE_PREFIX = "sim_db_search_catalog:";
let lastAddSentAt = 0;
let lastFocusSentAt = 0;
let suppressClickUntil = 0;
let resyncRequestedFor = null;
let searchCatalog = { entries: [], inverted: {} };

const elPanel = document.getElementById("panel");
const elBrowser = document.getElementById("panel-browser");
const elWorkspacePanel = document.getElementById("panel-workspace");
const elTree = document.getElementById("tree-area");
const elSearchArea = document.getElementById("search-area");
const elSearchInput = document.getElementById("search-input");
const elWorkspace = document.getElementById("workspace-area");
const elWorkspaceHelp = document.getElementById("workspace-help");
const elBrowserHelp = document.getElementById("browser-help");
const elContextMenu = document.getElementById("context-menu");
const elDownloadToggle = document.getElementById("download-toggle");
const elClearMaterials = document.getElementById("clear-materials");
const elHelpTooltip = document.getElementById("help-tooltip");

function hideHelpTooltip() {
  if (!elHelpTooltip) return;
  elHelpTooltip.classList.add("hidden");
  elHelpTooltip.textContent = "";
}

function bindHelpIcon(el, helpText) {
  if (!el) return;
  if (!helpText) {
    el.classList.add("hidden");
    return;
  }
  el.classList.remove("hidden");
  el.onmouseenter = () => {
    elHelpTooltip.textContent = helpText;
    elHelpTooltip.classList.remove("hidden");
    const rect = el.getBoundingClientRect();
    elHelpTooltip.style.left = Math.min(rect.right + 8, window.innerWidth - 300) + "px";
    elHelpTooltip.style.top = Math.max(8, rect.top - 4) + "px";
  };
  el.onmouseleave = () => {
    elHelpTooltip.classList.add("hidden");
  };
}

function catalogStorageKey(fingerprint) {
  return CATALOG_STORAGE_PREFIX + String(fingerprint || "");
}

function persistSearchCatalog(fingerprint) {
  if (!fingerprint) return;
  try {
    sessionStorage.setItem(catalogStorageKey(fingerprint), JSON.stringify(searchCatalog));
  } catch (_) {}
}

function restoreSearchCatalogFromStorage(fingerprint) {
  if (!fingerprint) return false;
  try {
    const raw = sessionStorage.getItem(catalogStorageKey(fingerprint));
    if (!raw) return false;
    const cached = JSON.parse(raw);
    if (!cached || !Array.isArray(cached.entries) || !cached.entries.length) return false;
    searchCatalog = {
      entries: cached.entries,
      inverted: cached.inverted || {},
    };
    return true;
  } catch (_) {
    return false;
  }
}

function installSearchCatalog(catalog) {
  if (!catalog) return;
  const fingerprint = String(catalog.fingerprint || "");
  if (Array.isArray(catalog.entries) && catalog.entries.length) {
    searchCatalog = {
      entries: catalog.entries,
      inverted: catalog.inverted || {},
    };
    persistSearchCatalog(fingerprint);
    resyncRequestedFor = null;
    return;
  }
  if (restoreSearchCatalogFromStorage(fingerprint)) {
    resyncRequestedFor = null;
    return;
  }
  if (fingerprint && resyncRequestedFor !== fingerprint) {
    resyncRequestedFor = fingerprint;
    sendAction({ action: "resync_search_catalog" });
  }
}

function tokenizeSearchText(text) {
  const lower = String(text || "").toLowerCase();
  const tokens = new Set();
  for (const part of lower.split(/[^a-z0-9_]+/)) {
    if (part.length >= MIN_INVERTED_TOKEN_LEN) tokens.add(part);
  }
  for (const part of lower.replace(/_/g, " ").replace(/\./g, " ").split(/[^a-z0-9]+/)) {
    if (part.length >= MIN_INVERTED_TOKEN_LEN) tokens.add(part);
  }
  return [...tokens];
}

function candidateEntryIndices(query) {
  const key = String(query || "").trim();
  if (!key) return [];
  const qLower = key.toLowerCase();
  const entries = searchCatalog.entries;
  if (!entries.length) return [];

  let tokens = tokenizeSearchText(qLower);
  if (!tokens.length && qLower.length >= MIN_INVERTED_TOKEN_LEN) {
    tokens = [qLower];
  }
  if (!tokens.length) {
    return entries.map((_, index) => index);
  }

  let candidateSet = null;
  for (const token of tokens) {
    const hits = searchCatalog.inverted[token];
    if (!hits || !hits.length) return [];
    const tokenSet = new Set(hits);
    if (candidateSet === null) candidateSet = tokenSet;
    else {
      candidateSet = new Set([...candidateSet].filter((index) => tokenSet.has(index)));
    }
    if (!candidateSet.size) return [];
  }
  return candidateSet ? [...candidateSet].sort((a, b) => a - b) : entries.map((_, index) => index);
}

function searchLocal(query) {
  const key = String(query || "").trim();
  if (!key) return [];
  const qLower = key.toLowerCase();
  const entries = searchCatalog.entries;
  if (!entries.length) return [];

  const candidates = candidateEntryIndices(key);
  const results = [];
  for (const index of candidates) {
    const row = entries[index];
    if (!row) continue;
    const pathKeys = row[0];
    const pathId = row[1];
    const leafType = row[2];
    const label = row[3];
    const hay = `${label} ${pathId}`.toLowerCase();
    if (!hay.includes(qLower)) continue;
    results.push({
      path_keys: pathKeys,
      path_id: pathId,
      leaf_type: leafType,
      label,
    });
    if (results.length >= SEARCH_MAX_RESULTS) break;
  }
  return results;
}

function currentSearchQuery() {
  return elSearchInput.value.trim();
}

function updateSearchView() {
  const query = currentSearchQuery();
  const searchMode = Boolean(query);
  elTree.classList.toggle("hidden", searchMode);
  elSearchArea.classList.toggle("hidden", !searchMode);
  if (searchMode) {
    renderSearchResults(searchLocal(query));
  } else {
    renderTree();
  }
}

function bindWorkspaceHelp(helpText) {
  bindHelpIcon(elWorkspaceHelp, helpText);
}

function sendAction(payload) {
  const now = Date.now();
  if (payload.action === "add") {
    if (now - lastAddSentAt < 400) return;
    lastAddSentAt = now;
  }
  if (payload.action === "focus") {
    if (now - lastFocusSentAt < 350) return;
    lastFocusSentAt = now;
  }
  Streamlit.setComponentValue({ ...payload, ts: Date.now() });
}

let lastDownloadKey = "";

function triggerDownload(base64, filename) {
  if (!base64 || !filename) return;
  const topWin = window.top || window;
  const dedupeKey = `${filename}:${base64.length}:${base64.slice(0, 32)}`;
  if (topWin.__simDbLastDownloadKey === dedupeKey) return;
  topWin.__simDbLastDownloadKey = dedupeKey;
  lastDownloadKey = dedupeKey;
  try {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchorTargets = [window.top, window.parent, window];
    let triggered = false;
    for (const win of anchorTargets) {
      if (!win || !win.document) continue;
      try {
        const a = win.document.createElement("a");
        a.href = url;
        a.download = filename;
        a.style.display = "none";
        a.rel = "noopener";
        win.document.body.appendChild(a);
        a.click();
        a.remove();
        triggered = true;
        break;
      } catch (_) {
        /* try next frame */
      }
    }
    if (!triggered) {
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  } catch (e) {
    console.error("download failed", e);
  }
}

function iconForNode(node) {
  if (node.is_leaf) {
    return node.leaf_type === "spectrum" ? "◆" : "●";
  }
  return "▣";
}

function leafPreviewPayload(node) {
  return {
    action: "preview",
    path_keys: node.path_keys,
    path_id: node.path_id,
    leaf_type: node.leaf_type,
  };
}

function leafAddPayload(node) {
  return {
    action: "add",
    path_keys: node.path_keys,
    path_id: node.path_id,
    leaf_type: node.leaf_type,
  };
}

function bindLeafClick(row, node) {
  row.addEventListener("dblclick", (ev) => {
    ev.preventDefault();
    suppressClickUntil = Date.now() + 400;
    focusedPathId = node.path_id;
    clearTimeout(clickTimer);
    clickTimer = null;
    sendAction(leafAddPayload(node));
  });
  row.addEventListener("click", (ev) => {
    if (Date.now() < suppressClickUntil) return;
    if (ev.detail > 1) return;
    focusedPathId = node.path_id;
    clearTimeout(clickTimer);
    clickTimer = setTimeout(() => {
      if (Date.now() < suppressClickUntil) return;
      sendAction(leafPreviewPayload(node));
    }, CLICK_DELAY_MS);
  });
}

function renderTreeNode(node, depth) {
  const wrap = document.createElement("div");
  wrap.className = "tree-node";
  wrap.dataset.pathId = node.path_id;

  const row = document.createElement("div");
  row.className = "tree-row";
  if (focusedPathId === node.path_id) row.classList.add("focused");

  const chevron = document.createElement("span");
  chevron.className = "tree-chevron" + (node.is_leaf ? " empty" : "");
  const expanded = lastArgs.expanded_paths && lastArgs.expanded_paths.includes(node.path_id);
  chevron.textContent = node.is_leaf ? "" : expanded ? "▾" : "▸";
  if (!node.is_leaf) {
    chevron.addEventListener("click", (ev) => {
      ev.stopPropagation();
      sendAction({
        action: expanded ? "collapse" : "expand",
        path_id: node.path_id,
        path_keys: node.path_keys,
      });
    });
  }

  const icon = document.createElement("span");
  icon.className = "tree-icon";
  icon.textContent = iconForNode(node);

  const label = document.createElement("span");
  label.className = "tree-label";
  label.textContent = node.key;

  const count = document.createElement("span");
  count.className = "tree-count";
  if (!node.is_leaf && node.child_count) {
    count.textContent = String(node.child_count);
  }

  row.appendChild(chevron);
  row.appendChild(icon);
  row.appendChild(label);
  row.appendChild(count);

  if (node.is_leaf) {
    bindLeafClick(row, node);
  } else {
    row.addEventListener("click", (ev) => {
      focusedPathId = node.path_id;
      if (ev.detail >= 2) {
        sendAction({
          action: "expand",
          path_id: node.path_id,
          path_keys: node.path_keys,
        });
      }
    });
  }

  row.addEventListener("contextmenu", (ev) => {
    ev.preventDefault();
    contextTarget = node;
    showContextMenu(ev.clientX, ev.clientY, node);
  });

  wrap.appendChild(row);

  if (!node.is_leaf && expanded && node.children && node.children.length) {
    const childrenWrap = document.createElement("div");
    childrenWrap.className = "tree-children";
    node.children.forEach((child) => childrenWrap.appendChild(renderTreeNode(child, depth + 1)));
    wrap.appendChild(childrenWrap);
  }

  return wrap;
}

function renderTree() {
  elTree.innerHTML = "";
  const nodes = lastArgs.tree_nodes || [];
  if (!nodes.length) {
    elTree.innerHTML = '<div class="empty-tree">无可用数据库</div>';
    return;
  }
  nodes.forEach((n) => elTree.appendChild(renderTreeNode(n, 0)));
}

function bindSearchResultClick(div, r) {
  div.addEventListener("dblclick", (ev) => {
    ev.preventDefault();
    clearTimeout(clickTimer);
    sendAction({
      action: "add",
      path_keys: r.path_keys,
      path_id: r.path_id,
      leaf_type: r.leaf_type,
    });
  });
  div.addEventListener("click", (ev) => {
    clearTimeout(clickTimer);
    clickTimer = setTimeout(() => {
      sendAction({
        action: "preview",
        path_keys: r.path_keys,
        path_id: r.path_id,
        leaf_type: r.leaf_type,
      });
    }, CLICK_DELAY_MS);
  });
}

function renderSearchResults(results) {
  elSearchArea.innerHTML = "";
  const list = results || [];
  if (!list.length) {
    elSearchArea.innerHTML = '<div class="empty-tree">无匹配结果</div>';
    return;
  }
  list.forEach((r) => {
    const div = document.createElement("div");
    div.className = "search-result";
    const badge = document.createElement("span");
    badge.className = "badge " + (r.leaf_type === "spectrum" ? "badge-spectrum" : "badge-material");
    badge.textContent = r.leaf_type === "spectrum" ? "光谱" : "材料";
    const path = document.createElement("div");
    path.className = "search-result-path";
    path.textContent = r.label || r.path_id;
    div.appendChild(badge);
    div.appendChild(path);
    bindSearchResultClick(div, r);
    elSearchArea.appendChild(div);
  });
}

function renderWorkspace() {
  const ws = lastArgs.workspace || {};
  const focus = ws.focus;
  hideHelpTooltip();
  elWorkspace.innerHTML = "";

  const helpText = ws.help_text || "";
  const rangeWarnText = ws.range_warn_text || helpText;
  bindWorkspaceHelp(helpText);

  const spectrumSection = document.createElement("div");
  spectrumSection.className = "spectrum-section";
  const specTitle = document.createElement("div");
  specTitle.className = "materials-title";
  const spectra = ws.spectra || [];
  specTitle.innerHTML = '光谱 <span class="count-badge">' + spectra.length + "</span>";
  spectrumSection.appendChild(specTitle);

  const specList = document.createElement("div");
  specList.className = "materials-list";
  if (!spectra.length) {
    const empty = document.createElement("div");
    empty.className = "slot-empty slot-empty-minimal";
    specList.appendChild(empty);
  } else {
    spectra.forEach((s) => {
      const uniqueName = s.unique_name;
      const card = makeCard(
        uniqueName,
        s.node_path || "",
        "spectrum",
        "spectrum",
        uniqueName,
        Boolean(s.warn),
        rangeWarnText
      );
      if (focus && focus.kind === "spectrum" && focus.unique_name === uniqueName) {
        card.classList.add("focused");
      }
      specList.appendChild(card);
    });
  }
  spectrumSection.appendChild(specList);

  const matSection = document.createElement("div");
  matSection.className = "materials-section";
  const matTitle = document.createElement("div");
  matTitle.className = "materials-title";
  matTitle.innerHTML = '材料 <span class="count-badge">' + (ws.materials ? ws.materials.length : 0) + "</span>";
  matSection.appendChild(matTitle);

  const list = document.createElement("div");
  list.className = "materials-list";
  const materials = ws.materials || [];
  if (!materials.length) {
    const empty = document.createElement("div");
    empty.className = "slot-empty slot-empty-minimal";
    list.appendChild(empty);
  } else {
    materials.forEach((m) => {
      const uniqueName = m.unique_name;
      const card = makeCard(
        uniqueName,
        m.node_path || "",
        "material",
        "material",
        uniqueName,
        Boolean(m.warn),
        rangeWarnText
      );
      if (focus && focus.kind === "material" && focus.unique_name === uniqueName) {
        card.classList.add("focused");
      }
      list.appendChild(card);
    });
  }
  matSection.appendChild(list);

  elWorkspace.appendChild(spectrumSection);
  elWorkspace.appendChild(matSection);
}

function makeCard(displayName, nodePath, cardClass, kind, itemUniqueName, warn, helpText) {
  const card = document.createElement("div");
  card.className = "slot-card " + cardClass;
  if (warn) {
    card.classList.add("warn-card");
  }

  const body = document.createElement("div");
  body.className = "slot-body";
  const nameEl = document.createElement("div");
  nameEl.className = "slot-name";
  nameEl.textContent = displayName;
  const pathEl = document.createElement("div");
  pathEl.className = "slot-path";
  pathEl.textContent = nodePath || "";
  body.appendChild(nameEl);
  if (nodePath) body.appendChild(pathEl);

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "btn-remove";
  remove.title = "移除";
  remove.textContent = "×";
  remove.addEventListener("click", (ev) => {
    ev.stopPropagation();
    hideHelpTooltip();
    card.querySelector(".card-warn-tooltip")?.remove();
    card.classList.remove("warn-card");
    sendAction({ action: "remove", kind, unique_name: itemUniqueName });
  });

  card.appendChild(body);
  card.appendChild(remove);

  if (warn && helpText) {
    const tip = document.createElement("div");
    tip.className = "card-warn-tooltip";
    tip.setAttribute("role", "tooltip");
    tip.textContent = helpText;
    card.appendChild(tip);
  }

  card.addEventListener("dblclick", (ev) => {
    if (ev.target.closest(".btn-remove")) return;
    ev.preventDefault();
    suppressClickUntil = Date.now() + 250;
    clearTimeout(clickTimer);
    sendAction({ action: "focus", kind, unique_name: itemUniqueName });
  });
  card.addEventListener("click", (ev) => {
    if (ev.target.closest(".btn-remove")) return;
    if (Date.now() < suppressClickUntil) return;
    clearTimeout(clickTimer);
    clickTimer = setTimeout(() => {
      sendAction({ action: "focus", kind, unique_name: itemUniqueName });
    }, CLICK_DELAY_MS);
  });

  return card;
}

function showContextMenu(x, y, node) {
  elContextMenu.innerHTML = "";
  const items = [];
  if (node.is_leaf) {
    items.push({ label: "预览", action: "preview" });
    items.push({ label: "加入工作区", action: "add" });
  } else {
    items.push({ label: "展开", action: "expand" });
  }
  items.forEach((item) => {
    const el = document.createElement("div");
    el.className = "context-item";
    el.textContent = item.label;
    el.addEventListener("click", () => {
      hideContextMenu();
      const payload = {
        action: item.action,
        path_keys: node.path_keys,
        path_id: node.path_id,
        leaf_type: node.leaf_type,
      };
      sendAction(payload);
    });
    elContextMenu.appendChild(el);
  });
  elContextMenu.classList.remove("hidden");
  elContextMenu.style.left = Math.min(x, window.innerWidth - 180) + "px";
  elContextMenu.style.top = Math.min(y, window.innerHeight - 120) + "px";
}

function hideContextMenu() {
  elContextMenu.classList.add("hidden");
  contextTarget = null;
}

document.addEventListener("click", () => hideContextMenu());

elSearchInput.addEventListener("input", () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(updateSearchView, SEARCH_DEBOUNCE_MS);
});

elSearchInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") {
    elSearchInput.value = "";
    updateSearchView();
  }
});

elDownloadToggle.addEventListener("change", () => {
  sendAction({ action: "download_toggle", enabled: elDownloadToggle.checked });
});

elClearMaterials.addEventListener("click", () => {
  hideHelpTooltip();
  sendAction({ action: "clear_workspace" });
});

function applySection(section) {
  elPanel.classList.remove("section-browser", "section-workspace", "section-all");
  const mode = section || "all";
  elPanel.classList.add("section-" + mode);
}

function render(args) {
  injectTokensCss(args.tokens_css || "");
  lastArgs = args;
  applySection(args.section || "all");
  bindHelpIcon(elBrowserHelp, args.browser_help_text || "");
  elDownloadToggle.checked = Boolean(args.download_on_action);
  installSearchCatalog(args.search_catalog || {});

  if (document.activeElement !== elSearchInput && args.search_query) {
    elSearchInput.value = args.search_query;
  }

  updateSearchView();
  renderWorkspace();

  if (
    args.auto_download_base64 &&
    args.auto_download_filename &&
    (args.section === "browser" || args.section === "workspace")
  ) {
    triggerDownload(args.auto_download_base64, args.auto_download_filename);
  }

  const h = args.height || Math.max(480, window.innerHeight - 24);
  Streamlit.setFrameHeight(h);
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, (event) => {
  render(event.detail.args);
});

Streamlit.setComponentReady();
Streamlit.setFrameHeight(520);
