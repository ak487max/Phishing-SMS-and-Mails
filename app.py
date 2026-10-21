
import html
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import show_dashboard


# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "message_classifier.joblib"

st.set_page_config(
    page_title="PhishGuard Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CYBER UI THEME
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg: #080B16;
        --panel: #101629;
        --cyan: #00F5D4;
        --blue: #54A8FF;
        --purple: #B58AFF;
        --pink: #FF70C5;
        --green: #52F5A9;
        --amber: #FFC857;
        --red: #FF6482;
        --muted: #9AA9C5;
        --border: rgba(130, 165, 220, 0.19);
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #142447 0%, transparent 34%),
            radial-gradient(ellipse at 95% 22%, #27163F 0%, transparent 28%),
            var(--bg);
        font-family: 'Space Grotesk', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10172B, #090D19);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        font-family: 'Space Grotesk', sans-serif;
    }

    h1, h2, h3 {
        letter-spacing: -0.7px;
    }

    .cyber-hero {
        position: relative;
        overflow: hidden;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
        padding: 30px;
        margin-bottom: 24px;
        border: 1px solid rgba(0,245,212,.25);
        border-radius: 22px;
        background:
            linear-gradient(115deg, rgba(15,32,57,.98), rgba(25,17,52,.94));
        box-shadow: 0 15px 55px rgba(0,0,0,.2);
    }

    .cyber-hero:before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        opacity: .22;
        background-image:
            linear-gradient(rgba(84,168,255,.14) 1px, transparent 1px),
            linear-gradient(90deg, rgba(84,168,255,.14) 1px, transparent 1px);
        background-size: 27px 27px;
        mask-image: linear-gradient(90deg, black, transparent 80%);
    }

    .hero-content {
        position: relative;
        z-index: 2;
    }

    .eyebrow {
        color: var(--cyan);
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 3px;
        font-weight: 600;
    }

    .hero-title {
        font-size: clamp(29px, 4vw, 46px);
        line-height: 1.15;
        font-weight: 700;
        margin: 10px 0;
        background: linear-gradient(90deg, #FFFFFF, #00F5D4, #B58AFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #AAB9D3;
        font-size: 15px;
        max-width: 600px;
        line-height: 1.65;
    }

    .radar {
        position: relative;
        flex-shrink: 0;
        width: 145px;
        height: 145px;
        border-radius: 50%;
        border: 1px solid rgba(0,245,212,.55);
        background:
            radial-gradient(circle, transparent 18%, rgba(0,245,212,.12) 19%, transparent 20%),
            radial-gradient(circle, transparent 39%, rgba(0,245,212,.25) 40%, transparent 41%),
            radial-gradient(circle, transparent 62%, rgba(0,245,212,.22) 63%, transparent 64%),
            linear-gradient(90deg, transparent 49.5%, rgba(0,245,212,.3) 50%, transparent 50.5%),
            linear-gradient(0deg, transparent 49.5%, rgba(0,245,212,.3) 50%, transparent 50.5%),
            rgba(0,245,212,.035);
        box-shadow: 0 0 32px rgba(0,245,212,.12), inset 0 0 30px rgba(0,245,212,.08);
    }

    .radar:before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 50%;
        background: conic-gradient(from 0deg, transparent 72%, rgba(0,245,212,.42));
        animation: radarSweep 4s linear infinite;
    }

    .radar:after {
        content: "";
        position: absolute;
        width: 8px;
        height: 8px;
        top: 33%;
        left: 64%;
        border-radius: 50%;
        background: var(--pink);
        box-shadow: 0 0 13px var(--pink);
        animation: pulseDot 1.8s infinite;
    }

    @keyframes radarSweep {
        to { transform: rotate(360deg); }
    }

    @keyframes pulseDot {
        50% { opacity: .4; transform: scale(.7); }
    }

    .section-label {
        color: var(--muted);
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 2px;
        margin: 8px 0 13px;
        text-transform: uppercase;
    }

    .glass-card {
        background: linear-gradient(145deg, rgba(21,32,56,.95), rgba(13,19,35,.95));
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 19px;
        margin-bottom: 12px;
        transition: border-color .2s ease, transform .2s ease;
    }

    .glass-card:hover {
        border-color: rgba(0,245,212,.48);
        transform: translateY(-2px);
    }

    .mini-label {
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .big-number {
        font-size: 30px;
        font-weight: 700;
        margin: 5px 0;
    }

    .neon-text {
        color: var(--cyan);
        text-shadow: 0 0 15px rgba(0,245,212,.25);
    }

    .risk-chip {
        display: inline-block;
        border-radius: 30px;
        padding: 5px 11px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .7px;
        margin-bottom: 8px;
    }

    .chip-red {
        color: #FF92A5;
        border: 1px solid #8F354D;
        background: rgba(255,100,130,.12);
    }

    .chip-orange {
        color: #FFD18A;
        border: 1px solid #855B25;
        background: rgba(255,200,87,.10);
    }

    .chip-purple {
        color: #D1B7FF;
        border: 1px solid #67469A;
        background: rgba(181,138,255,.12);
    }

    .chip-cyan {
        color: #8FFFF0;
        border: 1px solid #176B69;
        background: rgba(0,245,212,.09);
    }

    .message-box {
        background: rgba(8,13,27,.85);
        border: 1px solid #293753;
        border-radius: 14px;
        padding: 18px;
        line-height: 2.1;
        overflow-wrap: anywhere;
        color: #E7EDFF;
        white-space: pre-wrap;
    }

    .hl-red {
        background: rgba(255,75,105,.22);
        border-bottom: 2px solid #FF6482;
        color: #FFB3C1;
        border-radius: 3px;
        padding: 2px 3px;
    }

    .hl-orange {
        background: rgba(255,176,65,.19);
        border-bottom: 2px solid #FFC857;
        color: #FFD996;
        border-radius: 3px;
        padding: 2px 3px;
    }

    .hl-purple {
        background: rgba(181,138,255,.19);
        border-bottom: 2px solid #B58AFF;
        color: #DDC8FF;
        border-radius: 3px;
        padding: 2px 3px;
    }

    .hl-pink {
        background: rgba(255,112,197,.17);
        border-bottom: 2px solid #FF70C5;
        color: #FFB8E3;
        border-radius: 3px;
        padding: 2px 3px;
    }

    .stTextArea textarea {
        background: #0B1222 !important;
        color: #E7EDFF !important;
        border: 1px solid #334363 !important;
        border-radius: 13px !important;
        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 0 1px var(--cyan) !important;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 42px;
        transition: all .2s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #00D9C0, #54A8FF, #B58AFF);
        background-size: 180% 180%;
        color: #07111D;
        border: none;
    }

    .stButton > button[kind="primary"]:hover {
        background-position: 100% 0;
        box-shadow: 0 0 24px rgba(0,245,212,.22);
        transform: translateY(-2px);
    }

    div[data-testid="stMetric"] {
        background: rgba(17,26,46,.85);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 15px;
    }

    div[data-testid="stMetricValue"] {
        color: var(--cyan);
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(15,24,42,.7);
    }

    hr {
        border-color: var(--border);
    }

    .footer {
        margin-top: 32px;
        padding-top: 16px;
        border-top: 1px solid var(--border);
        color: #7889A6;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 1px;
    }

    @media (max-width: 650px) {
        .radar { width: 82px; height: 82px; }
        .cyber-hero { padding: 20px; }
    }

    @media (prefers-reduced-motion: reduce) {
        .radar:before, .radar:after { animation: none; }
        * { transition: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD EXISTING MODEL
# =========================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run python train_model.py first."
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()


# =========================================================
# RISK INDICATOR RULES
# =========================================================
RISK_RULES = {
    "Suspicious link": {
        "patterns": [
            r"https?://\S+",
            r"\bwww\.\S+",
            r"\bbit\.ly/\S*",
            r"\btinyurl\.com/\S*",
            r"\bt\.co/\S*",
            r"\brb\.gy/\S*",
            r"\bcutt\.ly/\S*",
        ],
        "description": (
            "A URL was found. Inspect its destination before opening it. "
            "A URL or HTTPS alone does not establish maliciousness."
        ),
        "points": 2,
        "color": "red",
    },
    "Urgency or threats": {
        "patterns": [
            r"\burgent\b",
            r"\bimmediately\b",
            r"\bact now\b",
            r"\baccount.{0,25}(?:suspend|block|restrict|terminat)",
            r"\bwithin \d+ hours?\b",
            r"\bexpires? today\b",
            r"\blast warning\b",
        ],
        "description": (
            "Pressure or threats can be used to rush someone into "
            "clicking a link or disclosing information."
        ),
        "points": 2,
        "color": "orange",
    },
    "Sensitive information request": {
        "patterns": [
            r"\bpassword\b",
            r"\b(?:otp|one[- ]time password)\b",
            r"\bpin\b",
            r"\bcvv\b",
            r"\bbank details\b",
            r"\bverify.{0,30}(?:account|identity|details)\b",
            r"\bconfirm.{0,25}(?:password|credentials|pin)\b",
        ],
        "description": (
            "The message mentions credentials or financial information. "
            "Never disclose OTPs, passwords, PINs, or CVVs in response "
            "to an unverified message."
        ),
        "points": 3,
        "color": "red",
    },
    "Prize or reward claim": {
        "patterns": [
            r"\byou (?:have )?won\b",
            r"\bclaim (?:your|the) (?:prize|reward|gift)\b",
            r"\bfree gift\b",
            r"\blottery\b",
            r"\bcongratulations.{0,40}(?:winner|prize|reward)\b",
        ],
        "description": (
            "Prize and lottery claims can be used to encourage clicking "
            "links or paying fraudulent fees."
        ),
        "points": 2,
        "color": "pink",
    },
    "Payment or money request": {
        "patterns": [
            r"\bpay immediately\b",
            r"\bpayment.{0,25}(?:required|pending|failed|overdue)\b",
            r"\btransfer (?:money|funds)\b",
            r"\bpay now\b",
            r"\bprocessing fee\b",
            r"\brefund.{0,25}(?:claim|click|verify)\b",
        ],
        "description": (
            "The message mentions a payment, fee, refund, or money "
            "transfer. Independently verify unexpected payment requests."
        ),
        "points": 2,
        "color": "orange",
    },
    "Possible impersonation": {
        "patterns": [
            r"\b(?:bank|government|police|tax department)"
            r".{0,50}(?:verify|suspend|penalty|account|payment)\b",
            r"\b(?:customer support|security team|admin)"
            r".{0,40}(?:password|verify|account|otp)\b",
            r"\bkyc\b",
        ],
        "description": (
            "The message may be invoking an organisation or support "
            "identity. Verify through a known official channel."
        ),
        "points": 2,
        "color": "purple",
    },
}


def detect_risks(message):
    findings = []

    for name, rule in RISK_RULES.items():
        matched_patterns = [
            pattern
            for pattern in rule["patterns"]
            if re.search(pattern, message, flags=re.IGNORECASE)
        ]

        if matched_patterns:
            findings.append({
                "name": name,
                "description": rule["description"],
                "points": rule["points"],
                "color": rule["color"],
                "patterns": matched_patterns,
            })

    return findings


def risk_level(score):
    if score >= 6:
        return "HIGH", "#FF6482"
    if score >= 3:
        return "MEDIUM", "#FFC857"
    if score > 0:
        return "LOW", "#54A8FF"
    return "NO RULE MATCH", "#52F5A9"


def spam_probability(message):
    probabilities = model.predict_proba([message])[0]
    classes = [str(label).lower() for label in model.classes_]

    if "spam" not in classes:
        return None

    return float(probabilities[classes.index("spam")])


# =========================================================
# URL EXTRACTION
# =========================================================
def extract_urls(message):
    candidates = re.findall(
        r"(?i)\b(?:https?://|www\.)[^\s<>\"']+",
        message,
    )

    results = []
    for candidate in candidates:
        clean_url = candidate.rstrip(".,!?;:)]}")
        parseable = (
            clean_url
            if clean_url.lower().startswith(("http://", "https://"))
            else "https://" + clean_url
        )

        try:
            parsed = urlparse(parseable)
            domain = parsed.hostname or "Unknown domain"
            uses_https = parsed.scheme.lower() == "https"
            shorteners = {
                "bit.ly", "tinyurl.com", "t.co",
                "rb.gy", "cutt.ly", "ow.ly", "is.gd",
            }

            results.append({
                "url": clean_url,
                "domain": domain,
                "https": uses_https,
                "shortened": domain.lower() in shorteners,
            })
        except ValueError:
            results.append({
                "url": clean_url,
                "domain": "Unable to parse",
                "https": False,
                "shortened": False,
            })

    return results


# =========================================================
# SAFE HIGHLIGHTING
# =========================================================
def highlighted_message(message):
    """
    Escape user input before rendering.
    Only matched text is wrapped in our own known CSS spans.
    """
    all_matches = []

    for rule in RISK_RULES.values():
        for pattern in rule["patterns"]:
            try:
                for match in re.finditer(
                    pattern, message, flags=re.IGNORECASE
                ):
                    all_matches.append((
                        match.start(),
                        match.end(),
                        rule["color"],
                    ))
            except re.error:
                continue

    # Prioritize longer matches at the same starting position.
    all_matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    selected = []
    cursor = 0

    for start, end, color in all_matches:
        if start < cursor:
            continue
        selected.append((start, end, color))
        cursor = end

    chunks = []
    cursor = 0

    for start, end, color in selected:
        chunks.append(html.escape(message[cursor:start]))
        safe_text = html.escape(message[start:end])
        chunks.append(
            f'<span class="hl-{color}">{safe_text}</span>'
        )
        cursor = end

    chunks.append(html.escape(message[cursor:]))
    return "".join(chunks)


# =========================================================
# PLOTLY CHARTS
# =========================================================
def make_risk_gauge(score):
    level, color = risk_level(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "suffix": "/100",
            "font": {"size": 34, "color": "#F1F5FF"},
        },
        title={
            "text": f"<b>{level}</b><br><span style='font-size:11px'>HEURISTIC RISK INDICATORS</span>",
            "font": {"size": 15, "color": color},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#7184A8",
                "tickfont": {"color": "#9AA9C5"},
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#101629",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "#153A39"},
                {"range": [20, 40], "color": "#183F5B"},
                {"range": [40, 60], "color": "#564523"},
                {"range": [60, 80], "color": "#653B2C"},
                {"range": [80, 100], "color": "#64283E"},
            ],
            "threshold": {
                "line": {"color": "#FFFFFF", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=25, r=25, t=65, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Space Grotesk"},
    )
    return fig


def make_indicator_chart(findings):
    if not findings:
        return None

    labels = [item["name"] for item in findings]
    values = [item["points"] for item in findings]
    colors = {
        "red": "#FF6482",
        "orange": "#FFC857",
        "purple": "#B58AFF",
        "pink": "#FF70C5",
    }

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.66,
        sort=False,
        marker={
            "colors": [colors.get(item["color"], "#54A8FF") for item in findings],
            "line": {"color": "#101629", "width": 3},
        },
        textinfo="percent",
        hovertemplate="%{label}<br>Indicator weight: %{value}<extra></extra>",
    ))

    fig.update_layout(
        height=290,
        showlegend=True,
        legend={
            "font": {"color": "#DDE7FA", "size": 10},
            "orientation": "h",
            "y": -0.12,
        },
        margin=dict(l=5, r=5, t=15, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Space Grotesk", "color": "#DDE7FA"},
        annotations=[{
            "text": f"<b>{len(findings)}</b><br>signals",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 20, "color": "#F1F5FF"},
        }],
    )
    return fig


# =========================================================
# HEADER
# =========================================================
def render_hero():
    st.markdown(
        """
        <div class="cyber-hero">
          <div class="hero-content">
            <div class="eyebrow">◈ CYBER DEFENSE / MESSAGE INTELLIGENCE</div>
            <div class="hero-title">PhishGuard Intelligence</div>
            <div class="hero-subtitle">
              Detect suspicious messages. Inspect risk signals.
              Understand what your machine-learning model predicts.
            </div>
          </div>
          <div class="radar"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SIDEBAR / NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 0 20px;">
          <div style="font-size:32px;">🛡️</div>
          <div style="font-size:22px;font-weight:700;color:#00F5D4;">
            PHISHGUARD
          </div>
          <div style="font-size:10px;letter-spacing:2px;color:#9AA9C5;">
            INTELLIGENCE CENTER
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Message Analyzer",
            "Scan History",
            "Evaluation Dashboard",
            "About PhishGuard",
        ],
    )

    st.divider()
    st.markdown("#### SYSTEM")
    st.markdown("🟢 Application ready")
    st.caption("Local ML inference · No live threat feed")
    st.caption("Model: TF-IDF + Logistic Regression")


# =========================================================
# INITIALIZE SESSION HISTORY
# =========================================================
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if "latest_scan" not in st.session_state:
    st.session_state.latest_scan = None


# =========================================================
# OVERVIEW
# =========================================================
if page == "Overview":
    render_hero()

    history = st.session_state.scan_history
    total = len(history)
    spam_count = sum(item["prediction"] == "spam" for item in history)
    flagged_count = sum(item["risk_score"] > 0 for item in history)

    st.markdown('<div class="section-label">◈ SESSION OVERVIEW</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Messages analyzed", total)
    c2.metric("Model spam predictions", spam_count)
    c3.metric("Messages with rule matches", flagged_count)

    st.markdown("### Your security workspace")
    a, b = st.columns(2)

    with a:
        st.markdown(
            """
            <div class="glass-card">
              <div style="font-size:28px">🔬</div>
              <h3 style="color:#00F5D4">Message forensics</h3>
              <p style="color:#9AA9C5">
                Analyze a message, inspect matched warning signs,
                and explore its URLs.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Message Analyzer", use_container_width=True):
            st.session_state.go_to_analyzer = True
            st.rerun()

    with b:
        st.markdown(
            """
            <div class="glass-card">
              <div style="font-size:28px">📊</div>
              <h3 style="color:#B58AFF">Model evaluation</h3>
              <p style="color:#9AA9C5">
                Review precision, recall, F1-score, and the confusion matrix.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Evaluation Dashboard", use_container_width=True):
            st.session_state.go_to_dashboard = True
            st.rerun()

    if st.session_state.get("go_to_analyzer"):
        st.info("Choose Message Analyzer in the sidebar to begin.")
        st.session_state.go_to_analyzer = False

    if st.session_state.get("go_to_dashboard"):
        st.info("Choose Evaluation Dashboard in the sidebar.")
        st.session_state.go_to_dashboard = False


# =========================================================
# MESSAGE ANALYZER
# =========================================================
elif page == "Message Analyzer":
    render_hero()

    st.markdown("## Message Threat Analyzer")
    st.caption(
        "Paste a message to run the local classifier and rule-based checks."
    )

    examples = {
        "Suspicious example": (
            "URGENT! Your bank account will be suspended within 24 hours. "
            "Verify your account immediately at https://example.com "
            "and claim your reward."
        ),
        "Prize message": (
            "Congratulations! You have won a free gift. "
            "Claim your prize today."
        ),
        "Ordinary example": (
            "Hi, are we still meeting at 5 PM today? "
            "Please let me know when you get a chance."
        ),
    }

    example_choice = st.selectbox(
        "Quick test examples",
        ["Choose an example..."] + list(examples.keys()),
    )

    default_message = (
        examples[example_choice]
        if example_choice in examples
        else ""
    )

    message = st.text_area(
        "MESSAGE INPUT",
        value=default_message,
        height=160,
        placeholder="Paste an SMS, email, or suspicious message here...",
        help="Avoid including real passwords, OTPs, or private account details.",
    )

    left_button, right_button = st.columns([3, 1])

    with left_button:
        analyze = st.button(
            "⚡ ANALYZE MESSAGE",
            type="primary",
            use_container_width=True,
        )

    with right_button:
        clear = st.button("Clear input", use_container_width=True)

    if clear:
        st.rerun()

    if analyze:
        if not message.strip():
            st.warning("Enter a message before analyzing.")
        else:
            with st.spinner("Inspecting text and calculating indicators..."):
                prediction = str(model.predict([message])[0]).lower()
                probability = spam_probability(message)
                findings = detect_risks(message)

                # Preserve the existing heuristic rule weights.
                raw_score = sum(item["points"] for item in findings)
                score = min(raw_score, 10) * 10
                level, level_color = risk_level(score)

                urls = extract_urls(message)

                result = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": prediction,
                    "probability": probability,
                    "risk_score": score,
                    "risk_level": level,
                    "indicators": [item["name"] for item in findings],
                    "urls_found": len(urls),
                }

                st.session_state.latest_scan = {
                    **result,
                    "message": message,
                    "findings": findings,
                    "urls": urls,
                }

                # Store metadata only, not the raw message.
                st.session_state.scan_history.insert(0, result)
                st.session_state.scan_history = (
                    st.session_state.scan_history[:100]
                )

            st.success("Analysis completed.")

    scan = st.session_state.latest_scan

    if scan:
        st.divider()
        st.markdown('<div class="section-label">◈ ANALYSIS RESULTS</div>',
                    unsafe_allow_html=True)

        top1, top2 = st.columns([1, 1])

        with top1:
            st.markdown("### 🤖 ML Classification")
            if scan["prediction"] == "spam":
                st.error("🚨 Model classified this message as SPAM")
            elif scan["prediction"] == "ham":
                st.success("✓ Model classified this message as LEGITIMATE")
            else:
                st.info(f"Model output: {scan['prediction']}")

            if scan["probability"] is not None:
                st.metric("Estimated spam probability",
                          f"{scan['probability']:.1%}")
                st.caption(
                    "Model estimate, not a guarantee of maliciousness or safety."
                )

        with top2:
            st.markdown("### 🎯 Rule-Based Risk Gauge")
            st.plotly_chart(
                make_risk_gauge(scan["risk_score"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                "Heuristic indicator score: matched rule weights scaled "
                "to 0–100. It is not a phishing probability."
            )

        st.divider()
        st.markdown("### 🧬 Message Forensics")
        st.caption(
            "Matched phrases are highlighted by category. "
            "A highlighted phrase is a warning signal, not proof of fraud."
        )

        st.markdown(
            f'<div class="message-box">{highlighted_message(scan["message"])}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:13px;">
              <span class="risk-chip chip-red">■ Link / sensitive info</span>
              <span class="risk-chip chip-orange">■ Urgency / payment</span>
              <span class="risk-chip chip-purple">■ Possible impersonation</span>
              <span class="risk-chip chip-pink">■ Prize / reward</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("### 🚩 Detected Risk Indicators")

        if scan["findings"]:
            indicator_cols = st.columns(2)

            for index, item in enumerate(scan["findings"]):
                color_class = {
                    "red": "chip-red",
                    "orange": "chip-orange",
                    "purple": "chip-purple",
                    "pink": "chip-pink",
                }.get(item["color"], "chip-cyan")

                with indicator_cols[index % 2]:
                    st.markdown(
                        f"""
                        <div class="glass-card">
                          <span class="risk-chip {color_class}">
                            +{item['points']} WEIGHT
                          </span>
                          <h4>{html.escape(item['name'])}</h4>
                          <p style="color:#9AA9C5;line-height:1.6">
                            {html.escape(item['description'])}
                          </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.success("No configured warning patterns matched.")
            st.caption(
                "New or sophisticated scams may not match these rules."
            )

        chart_col, detail_col = st.columns([1, 1])

        with chart_col:
            st.markdown("### 📡 Indicator Breakdown")
            chart = make_indicator_chart(scan["findings"])
            if chart is not None:
                st.plotly_chart(
                    chart,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.info("No indicator categories to chart for this message.")

        with detail_col:
            st.markdown("### 🔗 URL Inspection")
            if scan["urls"]:
                for url_info in scan["urls"]:
                    with st.container(border=True):
                        st.markdown(f"**{html.escape(url_info['domain'])}**")
                        st.caption(url_info["url"])

                        u1, u2 = st.columns(2)
                        u1.write(
                            "HTTPS: " + ("Yes" if url_info["https"] else "No")
                        )
                        u2.write(
                            "Shortener pattern: "
                            + ("Yes" if url_info["shortened"] else "No")
                        )

                        st.caption(
                            "This is basic URL inspection only. "
                            "No reputation lookup was performed."
                        )
            else:
                st.info("No URL patterns detected in this message.")

        st.divider()
        st.markdown("### 🧭 Recommended Safety Actions")

        st.markdown(
            """
            - Do not click links in unexpected or suspicious messages.
            - Never share an OTP, PIN, password, or CVV in response to a message.
            - Verify account alerts through an official app or a website you type yourself.
            - Confirm unexpected money requests through a separate trusted channel.
            """
        )

        st.caption(
            "Prototype limitation: the classifier was trained on the SMS Spam "
            "Collection. It has not been validated as a dedicated phishing detector."
        )


# =========================================================
# SCAN HISTORY
# =========================================================
elif page == "Scan History":
    render_hero()
    st.markdown("## Session Scan History")
    st.caption(
        "History is kept in the current Streamlit session. "
        "Raw message text is not stored in the history table."
    )

    history = st.session_state.scan_history

    if history:
        df = pd.DataFrame(history)

        filter_choice = st.selectbox(
            "Filter by ML prediction",
            ["All", "spam", "ham"],
        )

        if filter_choice != "All":
            df = df[df["prediction"] == filter_choice]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download history as CSV",
            data=csv_data,
            file_name="phishguard_scan_history.csv",
            mime="text/csv",
        )

        if st.button("🗑️ Clear session history"):
            st.session_state.scan_history = []
            st.session_state.latest_scan = None
            st.rerun()

    else:
        st.markdown(
            """
            <div class="glass-card">
              <div style="font-size:32px">📂</div>
              <h3 style="color:#54A8FF">No scans yet</h3>
              <p style="color:#9AA9C5">
                Analyze a message to populate your session history.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# EXISTING EVALUATION DASHBOARD
# =========================================================
elif page == "Evaluation Dashboard":
    render_hero()
    show_dashboard()


# =========================================================
# ABOUT
# =========================================================
elif page == "About PhishGuard":
    render_hero()

    st.markdown("## About this project")

    st.markdown(
        """
        <div class="glass-card">
          <h3 style="color:#00F5D4">🧠 Machine Learning</h3>
          <p style="color:#AAB9D3">
            TF-IDF converts message text into numerical features, and
            Logistic Regression predicts the learned SMS spam class.
          </p>
        </div>
        <div class="glass-card">
          <h3 style="color:#B58AFF">🔍 Rule-Based Analysis</h3>
          <p style="color:#AAB9D3">
            Pattern matching detects selected warning signs such as
            urgency, links, credential requests, and prize claims.
          </p>
        </div>
        <div class="glass-card">
          <h3 style="color:#FFC857">⚠️ Limitations</h3>
          <p style="color:#AAB9D3">
            This prototype does not use a live threat feed or a URL
            reputation service. Its SMS spam model is not a validated
            phishing detection system. A warning or lack of warnings
            does not prove whether a message is malicious.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="footer">PHISHGUARD · AI/ML SECURITY PROTOTYPE</div>',
        unsafe_allow_html=True,
    )


# Footer for all pages
if page != "About PhishGuard":
    st.markdown(
        '<div class="footer">PHISHGUARD INTELLIGENCE · LOCAL ML ANALYSIS · PROTOTYPE</div>',
        unsafe_allow_html=True,
    )
