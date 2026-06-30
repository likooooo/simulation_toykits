// Panel section header — same help tooltip behavior as simulation_db_panel

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

const elHeader = document.getElementById("section-header");
const elTitle = document.getElementById("section-title");
const elHelp = document.getElementById("section-help");
const elLinkWrap = document.getElementById("section-link-wrap");
const elLink = document.getElementById("section-link");
const elHelpTooltip = document.getElementById("help-tooltip");

const GLOBAL_TOOLTIP_ID = "panel-section-head-tooltip";
const GLOBAL_TOOLTIP_STYLE_ID = "panel-section-head-tooltip-style";

let globalTooltip = null;
let globalTooltipReady = false;

function getHostWindow() {
  const candidates = [window.top, window.parent, window];
  for (const win of candidates) {
    if (!win) continue;
    try {
      if (win.document) return win;
    } catch (_) {
      /* cross-origin frame */
    }
  }
  return window;
}

function injectTooltipStyles(hostDoc) {
  if (!hostDoc || hostDoc.getElementById(GLOBAL_TOOLTIP_STYLE_ID)) return;
  const style = hostDoc.createElement("style");
  style.id = GLOBAL_TOOLTIP_STYLE_ID;
  style.textContent = `
    #${GLOBAL_TOOLTIP_ID} {
      position: fixed;
      z-index: 999999;
      max-width: 280px;
      padding: 8px 10px;
      border-radius: var(--radius-sm, 6px);
      border: 1px solid var(--color-border, #E5E7EB);
      background: var(--color-surface, #FFFFFF);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
      font-family: var(--font-ui, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif);
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-line;
      color: var(--color-text, #111827);
      pointer-events: none;
    }
    #${GLOBAL_TOOLTIP_ID}.hidden {
      display: none;
    }
  `;
  hostDoc.head.appendChild(style);
}

function ensureGlobalTooltip() {
  if (globalTooltipReady) return globalTooltip;
  globalTooltipReady = true;

  const hostWin = getHostWindow();
  if (!hostWin || hostWin === window) return null;
  try {
    const hostDoc = hostWin.document;
    if (!hostDoc.body) return null;
    injectTooltipStyles(hostDoc);
    let tooltip = hostDoc.getElementById(GLOBAL_TOOLTIP_ID);
    if (!tooltip) {
      tooltip = hostDoc.createElement("div");
      tooltip.id = GLOBAL_TOOLTIP_ID;
      tooltip.className = "hidden";
      tooltip.setAttribute("role", "tooltip");
      hostDoc.body.appendChild(tooltip);
    }
    globalTooltip = tooltip;
    return globalTooltip;
  } catch (_) {
    return null;
  }
}

function hideLocalHelpTooltip() {
  if (!elHelpTooltip) return;
  elHelpTooltip.classList.add("hidden");
  elHelpTooltip.textContent = "";
}

function hideGlobalHelpTooltip() {
  const tooltip = globalTooltip || ensureGlobalTooltip();
  if (!tooltip) return;
  tooltip.classList.add("hidden");
  tooltip.textContent = "";
}

function hideHelpTooltip() {
  hideGlobalHelpTooltip();
  hideLocalHelpTooltip();
}

function getHostAnchorRect(el) {
  const rect = el.getBoundingClientRect();
  let left = rect.left;
  let top = rect.top;
  let w = window;

  while (w !== w.top) {
    const frame = w.frameElement;
    if (!frame) break;
    const frameRect = frame.getBoundingClientRect();
    left += frameRect.left;
    top += frameRect.top;
    try {
      w = w.parent;
    } catch (_) {
      break;
    }
  }

  return {
    top,
    left,
    right: left + rect.width,
    bottom: top + rect.height,
    width: rect.width,
    height: rect.height,
  };
}

function positionTooltip(tooltip, anchorEl, helpText, viewportWidth, useHostCoords) {
  const rect = useHostCoords ? getHostAnchorRect(anchorEl) : anchorEl.getBoundingClientRect();
  tooltip.textContent = helpText;
  tooltip.style.left = Math.min(rect.right + 8, viewportWidth - 300) + "px";
  tooltip.style.top = Math.max(8, rect.top - 4) + "px";
  tooltip.classList.remove("hidden");
}

function bindHelpIcon(el, helpText) {
  if (!el) return;
  if (!helpText) {
    el.classList.add("hidden");
    el.onmouseenter = null;
    el.onmouseleave = null;
    return;
  }
  el.classList.remove("hidden");
  el.onmouseenter = () => {
    hideHelpTooltip();
    const hostWin = getHostWindow();
    const viewportWidth = hostWin ? hostWin.innerWidth : window.innerWidth;
    const tooltip = ensureGlobalTooltip();
    if (tooltip) {
      positionTooltip(tooltip, el, helpText, viewportWidth, true);
      return;
    }
    if (elHelpTooltip) {
      positionTooltip(elHelpTooltip, el, helpText, viewportWidth, false);
    }
  };
  el.onmouseleave = () => {
    hideHelpTooltip();
  };
}

function render(args) {
  injectTokensCss(args.tokens_css || "");
  elTitle.textContent = args.title || "";
  bindHelpIcon(elHelp, args.help_text || "");

  if (args.help_url) {
    elLink.href = args.help_url;
    elLink.textContent = args.help_url_label || "— 使用说明";
    elLinkWrap.classList.remove("hidden");
  } else {
    elLinkWrap.classList.add("hidden");
  }

  elHeader.classList.toggle("align-right", args.align === "right");

  const h = Math.max(36, elHeader.offsetHeight + 2);
  Streamlit.setFrameHeight(h);
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, (event) => {
  render(event.detail.args || {});
});

Streamlit.setComponentReady();
Streamlit.setFrameHeight(36);
