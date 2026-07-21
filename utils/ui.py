"""Visual system + theming hub shared by all Streamlit pages.

All colors flow from the token dicts below into CSS custom properties
(`:root`). Component CSS and page-level inline styles reference those
variables (`var(--surface)`, `var(--ink)`, ...), so a single theme switch
re-paints the whole app. Plotly charts are themed via `apply_plotly_theme`.
"""

import base64
import hashlib
import re
from functools import lru_cache
from pathlib import Path

import streamlit as st


# --- Brand + legacy exports (pages import ORANGE) ---------------------------
ORANGE = "#ee4d2d"
INK = "#172033"
MUTED = "#687086"
LOGO_PATH = Path(__file__).resolve().parent.parent / "Logo Trustee.svg"
PRODUCT_IMAGE_PATHS = tuple(
    Path(__file__).resolve().parent.parent / f"netflix {index}.{extension}"
    for index, extension in (
        (1, "png"), (2, "jpg"), (3, "png"), (4, "png"), (5, "jpg"),
        (6, "png"), (7, "jpg"), (8, "jpg"), (9, "png"), (10, "jpg"),
    )
)

# --- Design tokens ----------------------------------------------------------
LIGHT = {
    "bg": "#f5f6f9",
    "surface": "#ffffff",
    "surface-2": "#eef1f6",
    "border": "#e6e9f0",
    "ink": "#141b2b",
    "muted": "#5c6576",
    "muted-2": "#8a91a3",
    "brand": "#ee4d2d",
    "brand-hover": "#d83f20",
    "brand-tint": "#fff3ef",
    "brand-ink": "#c43e24",
    "secondary": "#27364f",
    "ok-bg": "#e8f7ef",
    "ok-ink": "#1f7a4d",
    "danger-bg": "#fdeee9",
    "danger-ink": "#d13d20",
    "info-bg": "#eef2ff",
    "info-border": "#c7d2fe",
    "info-ink": "#3730a3",
    "hero-from": "#ee4d2d",
    "hero-to": "#ff7f5c",
    "shadow": "0 8px 26px rgba(20,27,43,.06)",
    "shadow-sm": "0 2px 8px rgba(20,27,43,.05)",
    "chart-font": "#3a4256",
    "chart-grid": "#e6e9f0",
}
DARK = {
    "bg": "#0e1320",
    "surface": "#161d2c",
    "surface-2": "#1e2636",
    "border": "#2b3446",
    "ink": "#eef1f7",
    "muted": "#a2abbd",
    "muted-2": "#7c8698",
    "brand": "#ff6a45",
    "brand-hover": "#ff7f5c",
    "brand-tint": "rgba(238,77,45,.16)",
    "brand-ink": "#ff9575",
    "secondary": "#8aa0cf",
    "ok-bg": "rgba(31,122,77,.22)",
    "ok-ink": "#5fe0a0",
    "danger-bg": "rgba(209,61,32,.24)",
    "danger-ink": "#ff9b82",
    "info-bg": "rgba(99,102,241,.16)",
    "info-border": "rgba(129,140,248,.42)",
    "info-ink": "#b7c0ff",
    "hero-from": "#ee4d2d",
    "hero-to": "#ff7f5c",
    "shadow": "0 10px 30px rgba(0,0,0,.45)",
    "shadow-sm": "0 2px 10px rgba(0,0,0,.35)",
    "chart-font": "#c3cbdb",
    "chart-grid": "#2b3446",
}


def is_dark() -> bool:
    return st.session_state.get("theme") == "dark"


def _tokens() -> dict:
    return DARK if is_dark() else LIGHT


def chart_secondary() -> str:
    """Discrete secondary series color that stays legible per theme."""
    return _tokens()["secondary"]


def _root_vars() -> str:
    lines = ";".join(f"--{name}:{value}" for name, value in _tokens().items())
    return "<style>:root{" + lines + "}</style>"


# Static stylesheet — every color routes through a var() so the theme switch
# repaints everything. Uses .format-free plain string (no dynamic braces).
_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif; color:var(--ink)}
.msi {font-family:'Material Symbols Rounded'; font-weight:normal; font-style:normal;
  line-height:1; vertical-align:-4px; -webkit-font-feature-settings:'liga'; font-feature-settings:'liga'}
.msi.lg {font-size:44px}
.topnav {display:flex; align-items:center; gap:4px; flex-wrap:wrap; padding:2px 0 6px}
[data-testid="stPageLink"] {min-width:0}
[data-testid="stPageLink"] a {width:100%; box-sizing:border-box; justify-content:center;
  border-radius:10px; padding:7px 6px; color:var(--muted); font-size:14px;
  font-weight:600; white-space:nowrap}
[data-testid="stPageLink"] a:hover {background:var(--surface-2); color:var(--ink)}
.nav-user {display:flex; align-items:center; justify-content:flex-end; gap:6px; min-height:40px;
  color:var(--ink); font-weight:600; white-space:nowrap; line-height:1}
.nav-user .msi {color:var(--muted); font-size:22px; vertical-align:0}
.brand-logo {display:block; width:130px; max-width:100%; height:auto}
.stApp {background:var(--bg); color:var(--ink)}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none !important}
[data-testid="stHeader"] {display:none}
.block-container {max-width:1240px; padding-top:1rem; padding-bottom:3rem}
h1,h2,h3 {letter-spacing:-.025em; color:var(--ink)}
h4,h5,h6,p,span,label,li {color:var(--ink)}
[data-testid="stCaptionContainer"], .stCaption, small {color:var(--muted) !important}
hr, [data-testid="stDivider"] hr {border-color:var(--border)}

[data-testid="stMetric"] {background:var(--surface); border:1px solid var(--border);
  border-radius:16px; padding:18px 20px; box-shadow:var(--shadow); transition:transform .15s ease,box-shadow .15s ease}
[data-testid="stMetric"]:hover {transform:translateY(-2px); box-shadow:var(--shadow)}
[data-testid="stMetricValue"] {color:var(--ink); font-weight:700}
[data-testid="stMetricLabel"] {color:var(--muted)}

div.stButton > button, div.stFormSubmitButton > button {border-radius:10px; background:var(--brand); color:#fff; border:0;
  font-weight:600; transition:background .15s ease, transform .1s ease}
div.stButton > button *, div.stFormSubmitButton > button * {color:inherit}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {background:var(--brand-hover); color:#fff; transform:translateY(-1px)}
div.stButton > button:active, div.stFormSubmitButton > button:active {transform:translateY(0)}
div.stButton > button[kind="tertiary"] {background:transparent; color:var(--muted)}
div.stButton > button[kind="tertiary"]:hover {background:var(--surface-2); color:var(--ink); transform:none}

/* Inputs adapt to theme */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"],
.stTextInput input, .stNumberInput input, .stTextArea textarea, [data-testid="stTextArea"] textarea {background:var(--surface) !important;
  border-color:var(--border) !important; color:var(--ink) !important}
[data-baseweb="select"] *, [data-baseweb="input"] * {color:var(--ink)}
[data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {background:var(--surface) !important; border:1px solid var(--border) !important}
[data-baseweb="popover"] li, li[role="option"] {background:var(--surface) !important; color:var(--ink) !important}
[data-baseweb="popover"] li:hover, li[role="option"]:hover, li[role="option"][aria-selected="true"] {background:var(--surface-2) !important; color:var(--ink) !important}
.stSlider [data-baseweb="slider"] div[role="slider"] {background:var(--brand)}
[data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {color:var(--muted)}

/* Info / warning boxes */
[data-testid="stAlert"] {background:var(--info-bg); border:1px solid var(--info-border);
  color:var(--ink); border-radius:12px}

/* Cards / containers with border */
[data-testid="stVerticalBlockBorderWrapper"] {background:var(--surface);
  border-color:var(--border) !important; border-radius:16px; box-shadow:var(--shadow-sm)}
[data-testid="stDataFrame"] {border:1px solid var(--border); border-radius:12px}
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"], [data-testid="stDataFrameGlideDataEditor"] {
  --gdg-bg-cell:var(--surface); --gdg-bg-cell-medium:var(--surface-2);
  --gdg-bg-header:var(--surface-2); --gdg-bg-header-has-focus:var(--surface-2); --gdg-bg-header-hovered:var(--surface-2);
  --gdg-text-dark:var(--ink); --gdg-text-medium:var(--muted); --gdg-text-light:var(--muted-2); --gdg-text-header:var(--ink);
  --gdg-border-color:var(--border); --gdg-horizontal-border-color:var(--border); --gdg-bg-bubble:var(--surface)}

/* ---- Design-system classes ---- */
.hero {background:linear-gradient(120deg,var(--hero-from),var(--hero-to)); color:#fff;
  padding:38px; border-radius:24px; box-shadow:0 16px 40px rgba(238,77,45,.22); margin-bottom:24px}
.hero h1 {margin:0 0 10px; font-size:2.25rem; color:#fff}
.hero p {margin:0; opacity:.92; max-width:720px; color:#fff}
.eyebrow {font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--brand)}
.panel {background:var(--surface); border:1px solid var(--border); border-radius:16px;
  padding:20px 22px; margin:8px 0 16px; box-shadow:var(--shadow-sm)}
.pill {display:inline-block; padding:6px 12px; border-radius:99px; background:var(--brand-tint);
  color:var(--brand-ink); font-size:.74rem; font-weight:700; letter-spacing:.04em}
.product-title {height:72px; overflow:hidden; display:-webkit-box; -webkit-line-clamp:3;
  -webkit-box-orient:vertical; font-size:16px; font-weight:700; line-height:1.45; color:var(--ink); margin:8px 0 4px}
.product-meta {height:26px; color:var(--muted-2); font-size:13px; display:flex; align-items:center}
.prod-thumb {height:165px; border-radius:12px; background:linear-gradient(135deg,var(--brand-tint),var(--surface-2));
  display:block; width:100%; object-fit:cover}
.product-photo-detail {height:340px; border-radius:20px; display:block; width:100%;
  object-fit:cover; box-shadow:var(--shadow-sm)}
.product-image-placeholder {min-height:340px; border:1px solid var(--border); border-radius:20px;
  background:linear-gradient(145deg,var(--surface),var(--surface-2)); display:flex;
  flex-direction:column; align-items:center; justify-content:center; gap:14px; padding:28px;
  box-sizing:border-box; text-align:center; box-shadow:var(--shadow-sm)}
.product-image-placeholder .msi {font-size:54px; color:var(--muted-2); vertical-align:0}
.product-image-placeholder .placeholder-label {font-size:15px; line-height:1.45;
  font-weight:600; color:var(--muted)}
.trust-row {height:42px; box-sizing:border-box; color:var(--brand-ink); padding:10px 2px;
  border-top:1px solid var(--border); font-size:12px; font-weight:700; display:flex;
  align-items:center; justify-content:space-between}
.stat-box {background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:12px 14px}
.stat-box .k {font-size:12px; color:var(--muted)}
.stat-box .v {font-size:24px; font-weight:700; color:var(--ink)}
.info-card {background:var(--info-bg); border:1px solid var(--info-border); padding:16px;
  border-radius:12px; color:var(--info-ink)}
.review-card {background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:16px 18px; margin:8px 0 14px; box-shadow:var(--shadow-sm)}
.review-head {display:flex; justify-content:space-between; gap:12px; align-items:center}
.review-head b {color:var(--ink)}
.badge {padding:5px 11px; border-radius:99px; font-size:12px; font-weight:700}
.badge-fake {background:var(--danger-bg); color:var(--danger-ink)}
.badge-ok {background:var(--ok-bg); color:var(--ok-ink)}
.review-stars {color:#fbbf24; margin:8px 0}
.review-body {color:var(--ink)}
.review-conf {color:var(--muted-2); font-size:12px; margin-top:10px}

@media (max-width: 768px) {
  .block-container {padding-top:.75rem}
  .st-key-top_navigation > div > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
    flex-wrap:wrap; gap:.45rem .75rem}
  .st-key-top_navigation > div > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:nth-child(1),
  .st-key-top_navigation > div > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:nth-child(2) {flex:0 0 100%; width:100%}
  .st-key-top_navigation > div > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:nth-child(3),
  .st-key-top_navigation > div > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:nth-child(4) {flex:1 1 calc(50% - .4rem); width:calc(50% - .4rem)}
  .st-key-top_navigation [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"] {
    flex-wrap:nowrap; gap:.25rem; overflow-x:auto; scrollbar-width:none; padding-bottom:2px}
  .st-key-top_navigation [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"]::-webkit-scrollbar {display:none}
  .st-key-top_navigation [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"] {flex:0 0 auto; width:auto; min-width:max-content}
  .st-key-top_navigation [data-testid="stPageLink"] a {padding:7px 10px}
  .st-key-top_navigation .nav-user {justify-content:flex-start}
  .st-key-top_navigation div.stButton > button {width:auto; padding-left:12px; padding-right:12px}
  .brand-logo {width:112px}
  .st-key-product_back {margin:.25rem 0 .5rem}
  .st-key-product_back div.stButton > button {width:auto; padding-left:0; padding-right:10px}
}
</style>
"""


NAV_ITEMS = (
    ("app.py", "Marketplace", "storefront"),
    ("pages/2_Review_Analysis.py", "Review", "reviews"),
    ("pages/3_Seller_Analysis.py", "Seller", "storefront"),
    ("pages/4_User_Behavior.py", "User", "group"),
    ("pages/5_Model_Evaluation.py", "Model", "analytics"),
    ("pages/6_Simulation.py", "Simulasi", "science"),
)


def setup_page(title: str, icon: str = "dashboard", show_eyebrow: bool = True) -> None:
    st.set_page_config(page_title=f"{title} · ShopAI", page_icon="🛍️", layout="wide")
    st.session_state.setdefault("theme", "light")
    st.markdown(_root_vars() + _STYLE, unsafe_allow_html=True)
    _top_bar(title if show_eyebrow else "", icon if show_eyebrow else "")


def icon(name: str, size: str = "") -> str:
    cls = f"msi {size}".strip()
    return f'<span class="{cls}">{name}</span>'


@lru_cache(maxsize=len(PRODUCT_IMAGE_PATHS))
def _image_data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def product_image(item_id: object, shop_id: object, detail: bool = False) -> str:
    """Return a stable, pseudo-random product image as an HTML element."""
    product_key = f"{item_id}:{shop_id}".encode("utf-8")
    image_index = int.from_bytes(hashlib.sha256(product_key).digest()[:4], "big")
    image_path = PRODUCT_IMAGE_PATHS[image_index % len(PRODUCT_IMAGE_PATHS)]
    css_class = "product-photo-detail" if detail else "prod-thumb"
    return (
        f'<img class="{css_class}" src="{_image_data_uri(image_path)}" '
        f'alt="Foto produk" loading="lazy">'
    )


@lru_cache(maxsize=1)
def trustee_logo() -> str:
    """Return the vector logo with its exported white canvas removed."""
    svg = LOGO_PATH.read_text(encoding="utf-8")
    svg = re.sub(
        r'<g clip-path="url\(#[^)]+\)"><path fill="#ffffff" '
        r'd="M 0 0\.0546875 L 531\.5 0\.0546875 L 531\.5 233\.945312 '
        r'L 0 233\.945312 Z M 0 0\.0546875 "[^>]*/></g>',
        "",
        svg,
        count=1,
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (
        f'<img class="brand-logo" src="data:image/svg+xml;base64,{encoded}" '
        f'alt="Trustee — Trusted Review Intelligence">'
    )


def _top_bar(eyebrow_text: str, icon_name: str) -> None:
    with st.container(key="top_navigation"):
        brand_col, nav_col, account_col, toggle_col = st.columns(
            [1.0, 4.5, 0.8, 1.0], vertical_alignment="center"
        )
        with brand_col:
            st.markdown(trustee_logo(), unsafe_allow_html=True)
        with nav_col:
            cols = st.columns(len(NAV_ITEMS))
            for col, (path, label, _) in zip(cols, NAV_ITEMS):
                col.page_link(path, label=label)
        with account_col:
            st.markdown(
                f'<div class="nav-user">{icon("account_circle")}<span>Budi</span></div>',
                unsafe_allow_html=True,
            )
        with toggle_col:
            if is_dark():
                clicked = st.button("Terang", icon=":material/light_mode:", type="tertiary",
                                    key="shopai_theme_toggle", use_container_width=True)
            else:
                clicked = st.button("Gelap", icon=":material/dark_mode:", type="tertiary",
                                    key="shopai_theme_toggle", use_container_width=True)
    if clicked:
        st.session_state["theme"] = "light" if is_dark() else "dark"
        st.rerun()
    if eyebrow_text:
        subtitle = f" · {icon_name}" if icon_name else ""
        st.markdown(
            f'<div class="eyebrow">{eyebrow_text}{subtitle}</div>',
            unsafe_allow_html=True,
        )


def marketplace_header() -> None:
    return None


def apply_plotly_theme(fig):
    """Make a Plotly figure adopt the active theme (bg, font, gridlines)."""
    t = _tokens()
    fig.update_layout(
        template="plotly_dark" if is_dark() else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["chart-font"], family="DM Sans, sans-serif"),
        legend=dict(font=dict(color=t["chart-font"])),
    )
    fig.update_xaxes(gridcolor=t["chart-grid"], zerolinecolor=t["chart-grid"], color=t["chart-font"])
    fig.update_yaxes(gridcolor=t["chart-grid"], zerolinecolor=t["chart-grid"], color=t["chart-font"])
    return fig


def prediction_notice(df) -> None:
    if "fallback" in set(df.get("prediction_source", [])):
        st.info("Mode demo aktif: indikator risiko menggunakan aturan sederhana karena model .pkl belum tersedia.")


def page_bounds(total: int, key: str, page_size: int = 12) -> tuple[int, int]:
    if total <= 0:
        return 0, 0
    page_count = max(1, -(-total // page_size))
    state_key = f"_page_{key}"
    current = min(st.session_state.get(state_key, 1), page_count)
    st.session_state[state_key] = current
    start = (current - 1) * page_size
    return start, min(start + page_size, total)


def page_controls(total: int, key: str, page_size: int = 12) -> None:
    if total <= page_size:
        return
    page_count = max(1, -(-total // page_size))
    state_key = f"_page_{key}"
    current = min(st.session_state.get(state_key, 1), page_count)
    start = (current - 1) * page_size
    end = min(start + page_size, total)
    prev_col, info_col, next_col = st.columns([1, 2, 1], vertical_alignment="center")
    if prev_col.button("Sebelumnya", key=f"{key}_prev", use_container_width=True, disabled=current <= 1):
        st.session_state[state_key] = current - 1
        st.rerun()
    info_col.markdown(
        f'<div style="text-align:center;color:var(--muted);font-weight:600">'
        f'Halaman {current} / {page_count} · {start + 1}-{end} dari {total:,}</div>',
        unsafe_allow_html=True,
    )
    if next_col.button("Berikutnya", key=f"{key}_next", use_container_width=True, disabled=current >= page_count):
        st.session_state[state_key] = current + 1
        st.rerun()
