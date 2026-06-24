import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

packages = [
    "streamlit",
    "xgboost",
    "scikit-learn",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "folium",
    "streamlit-folium",
    "openpyxl",
    "scipy",
]

for pkg in packages:
    try:
        install(pkg)
    except Exception as e:
        print(f"Could not install {pkg}: {e}")


# ============================================================
# DYNAMIC BEAT OPTIMISATION — STREAMLIT APP
# NCR West Pilot | GCPL GT India
# Models: M1 Productivity Classifier | M2 Cadence Optimiser | M3 Route Optimiser
# ===========================================================
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, accuracy_score, precision_score, recall_score,
    average_precision_score, log_loss, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import xgboost as xgb
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import folium
from streamlit_folium import st_folium
import math

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Beat Optimisation | NCR West",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Color palette ────────────────────────────────────────────
COLORS = {
    "dark":   "#0D1B2A", "navy":  "#1B2A4A", "teal":  "#107A6E",
    "green":  "#1A7A45", "red":   "#C0392B", "amber": "#D4860B",
    "purple": "#6A3FA0", "slate": "#2C3E50",
}
DSR_COLORS = {
    "SM01":"#107A6E","SM02":"#1A7A45","SM04":"#D4860B",
    "SM05":"#C0392B","SM06":"#1B2A4A","SM07":"#6A3FA0","SM09":"#C49A1A"
}

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0D1B2A;}
[data-testid="stSidebar"] *{color:#E8ECF0 !important;}
.metric-card{background:#F5F7FA;border-left:4px solid #107A6E;
             border-radius:6px;padding:14px 18px;margin:6px 0;}
.metric-val{font-size:28px;font-weight:700;color:#0D1B2A;}
.metric-lbl{font-size:12px;color:#6B7280;margin-top:2px;}
.section-hdr{background:#0D1B2A;color:white;padding:10px 16px;
             border-radius:6px;font-weight:700;font-size:15px;margin:12px 0 8px;}
.insight{background:#E0F5F3;border-left:4px solid #107A6E;
         padding:10px 14px;border-radius:4px;font-size:13px;margin:8px 0;}
.warn{background:#FFF0CC;border-left:4px solid #D4860B;
      padding:10px 14px;border-radius:4px;font-size:13px;margin:8px 0;}
.stTabs [data-baseweb="tab"]{font-weight:600;font-size:13px;}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════
PROD_TH = 250
TRAIN_END  = "2026-02-28"
VAL_END    = "2026-04-30"
TEST_END   = "2026-05-31"
NCR_LAT    = (28.0, 29.5)
NCR_LON    = (76.5, 78.0)
THRESHOLD  = 0.53

# ════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading and engineering features...")
def load_and_engineer(file):
    df = pd.read_csv(file, parse_dates=["BillingDate","Outlet_Creation_Date"])
    df.columns = df.columns.str.strip()

    # ── Base flags ──
    df["Was_Scheduled"]           = df["Visit_Source"].isin(["Scheduled_Billed","Scheduled_NoBilling"]).astype(int)
    df["Is_OffBeat_Visit"]        = (df["Visit_Source"]=="OffBeat_Billed").astype(int)
    df["Was_Scheduled_NotBilled"] = (df["Visit_Source"]=="Scheduled_NoBilling").astype(int)
    df["IsProductive"]            = (df["Total_GSV"].fillna(0) > PROD_TH).astype(int)
    df["Has_Null_DSR"]            = df["SalesmanCode"].isna().astype(int)

    # ── Time features ──
    df["BillingMonth"] = df["BillingDate"].dt.to_period("M").dt.to_timestamp()
    df["WeekOfMonth"]  = df["BillingDate"].apply(lambda d: min(4,(d.day-1)//7+1))
    df["Month"]        = df["BillingDate"].dt.month
    df["Year"]         = df["BillingDate"].dt.year
    df["DayOfWeek"]    = df["BillingDate"].dt.dayofweek
    df["IsEndOfMonth"] = (df["BillingDate"].dt.day >= 25).astype(int)
    FESTIVAL_MONTHS    = [10,11,3,4]
    df["IsFestivalMonth"] = df["Month"].isin(FESTIVAL_MONTHS).astype(int)
    df["Outlet_Age_Months"] = ((df["BillingDate"]-df["Outlet_Creation_Date"]).dt.days/30).clip(lower=0)

    # ── Clean lat/long ──
    df["Lat_clean"] = np.where(
        df["Latitude"].between(*NCR_LAT) & df["Longitude"].between(*NCR_LON),
        df["Latitude"], np.nan)
    df["Lon_clean"] = np.where(
        df["Latitude"].between(*NCR_LAT) & df["Longitude"].between(*NCR_LON),
        df["Longitude"], np.nan)

    # ── Category encode ──
    le = LabelEncoder()
    df["Category_enc"] = le.fit_transform(df["RetailerCategory"].fillna("Unknown"))

    # ── Scheduled only for feature computation ──
    df = df.sort_values(["UniqueRetailerCode","SalesmanCode","BillingDate"]).reset_index(drop=True)
    sched = df[df["Was_Scheduled"]==1].copy()

    # ── IPI ──
    sched["_sm"] = sched["SalesmanCode"].fillna("__NULL__")
    sched["schedule_ipi"] = sched.groupby(["UniqueRetailerCode","_sm"])["BillingDate"].diff().dt.days
    df = df.merge(
        sched[["UniqueRetailerCode","SalesmanCode","BillingDate","schedule_ipi"]]
              .drop_duplicates(["UniqueRetailerCode","SalesmanCode","BillingDate"]),
        on=["UniqueRetailerCode","SalesmanCode","BillingDate"], how="left")

    # ── Lag features (on scheduled) ──
    sched2 = df[df["Was_Scheduled"]==1].copy()
    sched2["_sm2"] = sched2["SalesmanCode"].fillna("__NULL__")
    for lag in [1,2]:
        sched2[f"gsv_lag{lag}"]        = sched2.groupby(["UniqueRetailerCode","_sm2"])["Total_GSV"].shift(lag).fillna(0)
        sched2[f"productive_lag{lag}"] = sched2.groupby(["UniqueRetailerCode","_sm2"])["IsProductive"].shift(lag).fillna(0)
    df = df.merge(
        sched2[["UniqueRetailerCode","SalesmanCode","BillingDate",
                "gsv_lag1","gsv_lag2","productive_lag1","productive_lag2"]]
              .drop_duplicates(["UniqueRetailerCode","SalesmanCode","BillingDate"]),
        on=["UniqueRetailerCode","SalesmanCode","BillingDate"], how="left")

    # ── Rolling 3-month productive rate per outlet-DSR ──
    sched3 = df[df["Was_Scheduled"]==1].copy()
    sched3 = sched3.sort_values(["UniqueRetailerCode","SalesmanCode","BillingDate"])
    sched3["_sm3"] = sched3["SalesmanCode"].fillna("__NULL__")
    sched3["rolling_3m_prod"] = (
        sched3.groupby(["UniqueRetailerCode","_sm3"])["IsProductive"]
              .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean()))
    df = df.merge(
        sched3[["UniqueRetailerCode","SalesmanCode","BillingDate","rolling_3m_prod"]]
               .drop_duplicates(["UniqueRetailerCode","SalesmanCode","BillingDate"]),
        on=["UniqueRetailerCode","SalesmanCode","BillingDate"], how="left")

    # ── Historical outlet productive rate ──
    hist_pr = (df[df["Was_Scheduled"]==1]
               .groupby(["UniqueRetailerCode","SalesmanCode"])["IsProductive"]
               .mean().rename("productive_rate").reset_index())
    df = df.merge(hist_pr, on=["UniqueRetailerCode","SalesmanCode"], how="left")

    # ── Consecutive skips ──
    sched4 = df[df["Was_Scheduled"]==1].copy()
    sched4["_sm4"] = sched4["SalesmanCode"].fillna("__NULL__")
    def consec_skips(x):
        out = []
        cnt = 0
        for v in x:
            out.append(cnt)
            cnt = cnt+1 if v==1 else 0
        return out
    sched4 = sched4.sort_values(["UniqueRetailerCode","_sm4","BillingDate"])
    sched4["consec_skips"] = sched4.groupby(["UniqueRetailerCode","_sm4"])["Was_Scheduled_NotBilled"].transform(consec_skips)
    df = df.merge(
        sched4[["UniqueRetailerCode","SalesmanCode","BillingDate","consec_skips"]]
               .drop_duplicates(["UniqueRetailerCode","SalesmanCode","BillingDate"]),
        on=["UniqueRetailerCode","SalesmanCode","BillingDate"], how="left")

    # ── DSR-level rolling metrics ──
    dsr_roll = (df[df["Was_Scheduled"]==1]
                .groupby(["SalesmanCode","BillingMonth"])
                .agg(dsr_prod_rate=("IsProductive","mean"),
                     dsr_notbilled_rate=("Was_Scheduled_NotBilled","mean"))
                .reset_index())
    df = df.merge(dsr_roll, on=["SalesmanCode","BillingMonth"], how="left")

    # ── VPM ──
    vpm = (df[df["Was_Scheduled"]==1]
           .groupby(["UniqueRetailerCode","SalesmanCode"])
           .agg(visits=("BillingDate","size"), months=("BillingMonth","nunique"))
           .reset_index())
    vpm["vpm"] = vpm["visits"]/vpm["months"].clip(lower=1)
    df = df.merge(vpm[["UniqueRetailerCode","SalesmanCode","vpm"]],
                  on=["UniqueRetailerCode","SalesmanCode"], how="left")

    # ── Off-beat propensity ──
    ob_gsv = (df[df["Is_OffBeat_Visit"]==1]
              .groupby("UniqueRetailerCode")["Total_GSV"].sum().rename("offbeat_gsv").reset_index())
    on_gsv = (df[df["Visit_Source"]=="Scheduled_Billed"]
              .groupby("UniqueRetailerCode")["Total_GSV"].sum().rename("onbeat_gsv").reset_index())
    ob_prop = ob_gsv.merge(on_gsv, on="UniqueRetailerCode", how="outer").fillna(0)
    ob_prop["offbeat_propensity"] = ob_prop["offbeat_gsv"]/(ob_prop["offbeat_gsv"]+ob_prop["onbeat_gsv"]).clip(lower=1)
    df = df.merge(ob_prop[["UniqueRetailerCode","offbeat_propensity"]], on="UniqueRetailerCode", how="left")

    df = df.fillna({
        "schedule_ipi":7.0, "gsv_lag1":0.0, "gsv_lag2":0.0,
        "productive_lag1":0.0, "productive_lag2":0.0,
        "rolling_3m_prod":0.35, "productive_rate":0.35,
        "consec_skips":0.0, "dsr_prod_rate":0.35,
        "dsr_notbilled_rate":0.63, "vpm":3.0, "offbeat_propensity":0.0
    })
    return df

# ── Model features list ──────────────────────────────────────
FEATURES = [
    "WeekOfMonth","Month","IsEndOfMonth","IsFestivalMonth","DayOfWeek",
    "schedule_ipi","gsv_lag1","gsv_lag2","productive_lag1","productive_lag2",
    "rolling_3m_prod","productive_rate","consec_skips",
    "Outlet_Age_Months","Category_enc","vpm","offbeat_propensity",
    "dsr_prod_rate","dsr_notbilled_rate"
]
FEATURE_LABELS = {
    "WeekOfMonth":"Week of Month (W1-W4)",
    "Month":"Month of Year",
    "IsEndOfMonth":"Is End of Month (day>=25)",
    "IsFestivalMonth":"Is Festival Month",
    "DayOfWeek":"Day of Week",
    "schedule_ipi":"Schedule IPI (days since last PJP visit)",
    "gsv_lag1":"Last Visit GSV (Rs)",
    "gsv_lag2":"Visit Before Last — GSV (Rs)",
    "productive_lag1":"Last Visit — Was Productive?",
    "productive_lag2":"Visit Before Last — Was Productive?",
    "rolling_3m_prod":"Rolling 3-Month Productivity Rate",
    "productive_rate":"Historical Conversion Rate of Outlet",
    "consec_skips":"Consecutive Skips Before This Visit",
    "Outlet_Age_Months":"Outlet Age (Months from Registration)",
    "Category_enc":"Retailer Category (Encoded)",
    "vpm":"Avg Visits per Month (Frequency)",
    "offbeat_propensity":"Off-Beat Revenue Share (Historical)",
    "dsr_prod_rate":"DSR Monthly Productivity Rate",
    "dsr_notbilled_rate":"DSR Monthly NotBilled Rate"
}

# ── Train M1 ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training M1 XGBoost classifier...")
def train_m1(df_hash):
    df = st.session_state["df"]
    sched = df[df["Was_Scheduled"]==1].copy()
    sched = sched.dropna(subset=["SalesmanCode"])

    train = sched[sched["BillingDate"]<=TRAIN_END]
    val   = sched[(sched["BillingDate"]>TRAIN_END)&(sched["BillingDate"]<=VAL_END)]
    test  = sched[(sched["BillingDate"]>VAL_END)&(sched["BillingDate"]<=TEST_END)]

    X_tr, y_tr = train[FEATURES].fillna(0), train["IsProductive"]
    X_va, y_va = val[FEATURES].fillna(0),   val["IsProductive"]
    X_te, y_te = test[FEATURES].fillna(0),  test["IsProductive"]

    pos_w = (y_tr==0).sum()/(y_tr==1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=pos_w, use_label_encoder=False,
        eval_metric="logloss", subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              verbose=False)

    # Baseline LR
    lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    lr.fit(X_tr, y_tr)

    return model, lr, X_te, y_te, test, X_tr, y_tr, X_va, y_va, val, train

def compute_metrics(model, X, y, threshold=THRESHOLD):
    prob = model.predict_proba(X.fillna(0))[:,1]
    pred = (prob >= threshold).astype(int)
    cm   = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    pr_auc = average_precision_score(y, prob)
    fpr_vals = np.linspace(0,1,100)
    return {
        "prob": prob, "pred": pred,
        "accuracy":  round(accuracy_score(y,pred)*100,2),
        "precision": round(precision_score(y,pred,zero_division=0)*100,2),
        "recall":    round(recall_score(y,pred,zero_division=0)*100,2),
        "f1":        round(f1_score(y,pred,zero_division=0)*100,2),
        "auc_roc":   round(roc_auc_score(y,prob)*100,2),
        "auc_pr":    round(pr_auc*100,2),
        "log_loss":  round(log_loss(y,prob),4),
        "tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn),
    }

def haversine(lat1,lon1,lat2,lon2):
    R=6371
    lat1,lon1,lat2,lon2=map(math.radians,[lat1,lon1,lat2,lon2])
    dlat=lat2-lat1; dlon=lon2-lon1
    a=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R*2*math.asin(math.sqrt(a))

def two_opt(points):
    n=len(points)
    if n<4: return list(range(n))
    route=list(range(n)); improved=True
    while improved:
        improved=False
        for i in range(1,n-2):
            for j in range(i+1,n):
                if j-i==1: continue
                new_route=route[:i]+route[i:j][::-1]+route[j:]
                d_old=sum(haversine(points[route[k]][0],points[route[k]][1],
                                    points[route[(k+1)%n]][0],points[route[(k+1)%n]][1])
                          for k in range(n))
                d_new=sum(haversine(points[new_route[k]][0],points[new_route[k]][1],
                                    points[new_route[(k+1)%n]][0],points[new_route[(k+1)%n]][1])
                          for k in range(n))
                if d_new<d_old:
                    route=new_route; improved=True
    return route

def to_excel(df_dict):
    buf=BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for sheet,df in df_dict.items():
            df.to_excel(w, sheet_name=sheet[:31], index=False)
    return buf.getvalue()

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Beat Optimisation")
    st.markdown("**NCR West Pilot | GCPL GT**")
    st.markdown("---")
    uploaded = st.file_uploader("Upload Input CSV/XLSX", type=["csv","xlsx"])
    st.markdown("---")
    st.markdown("**Productivity Threshold**")
    prod_th_ui = st.number_input("GSV > Rs", value=250, step=50)
    st.markdown("**M1 Decision Threshold**")
    threshold_ui = st.slider("Probability threshold", 0.1, 0.9, THRESHOLD, 0.01)
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("",["📊 Overview","🎯 M1: Productivity","📅 M2: Cadence",
                         "🗺️ M3: Route","📈 Backtesting"])

# ── Load data ────────────────────────────────────────────────
if uploaded:
    if "df" not in st.session_state or st.session_state.get("fname")!=uploaded.name:
        with st.spinner("Loading data..."):
            df = load_and_engineer(uploaded)
            st.session_state["df"] = df
            st.session_state["fname"] = uploaded.name
else:
    st.warning("⬅️  Please upload the input CSV/XLSX file to begin.")
    st.stop()

df = st.session_state["df"]
sched = df[df["Was_Scheduled"]==1].copy()

# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page=="📊 Overview":
    st.markdown('<div class="section-hdr">📊 DYNAMIC BEAT OPTIMISATION — NCR WEST OVERVIEW</div>',
                unsafe_allow_html=True)

    total_sched = int(sched.shape[0])
    total_prod  = int(sched["IsProductive"].sum())
    prod_rate   = round(sched["IsProductive"].mean()*100,1)
    notbilled   = round(sched["Was_Scheduled_NotBilled"].mean()*100,1)
    ob_gsv      = df[df["Is_OffBeat_Visit"]==1]["Total_GSV"].sum()
    on_gsv      = df[df["Visit_Source"]=="Scheduled_Billed"]["Total_GSV"].sum()
    ob_share    = round(ob_gsv/(ob_gsv+on_gsv)*100,1)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,label,val,unit in zip(
        [c1,c2,c3,c4,c5],
        ["Total Scheduled Visits","Productive Calls","Productivity Rate","NotBilled Rate","Off-Beat GSV Share"],
        [f"{total_sched:,}",f"{total_prod:,}",f"{prod_rate}%",f"{notbilled}%",f"{ob_share}%"],
        ["GSV>Rs250, excl. off-beat","GSV>Rs250","Prod/Scheduled","Skipped/Scheduled","Off-beat/(on+off)"]
    ):
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                     f'<div class="metric-lbl">{label}<br><span style="color:#9CA3AF;font-size:10px">{unit}</span></div></div>',
                     unsafe_allow_html=True)

    st.markdown("---")
    col1,col2 = st.columns(2)

    with col1:
        st.markdown("**Monthly Productivity Rate Trend**")
        mo = (sched.groupby("BillingMonth")
                   .agg(Scheduled=("IsProductive","count"), Productive=("IsProductive","sum"))
                   .reset_index())
        mo["Prod_Rate"] = mo["Productive"]/mo["Scheduled"]*100
        mo["YM"] = mo["BillingMonth"].dt.strftime("%b-%y")
        fig,ax = plt.subplots(figsize=(9,3.5))
        ax.plot(mo["YM"],mo["Prod_Rate"],color=COLORS["teal"],marker="o",linewidth=2,markersize=4)
        ax.axhline(mo["Prod_Rate"].mean(),color=COLORS["amber"],linestyle="--",linewidth=1,label="Average")
        ax.fill_between(mo["YM"],mo["Prod_Rate"],alpha=0.1,color=COLORS["teal"])
        ax.set_ylabel("Productivity Rate %"); ax.set_xlabel("")
        plt.xticks(rotation=45,ha="right",fontsize=8)
        ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Visit Outcome Breakdown**")
        vo = sched["Visit_Source"].value_counts()
        prod_count  = int(sched["IsProductive"].sum())
        unprod      = int(sched["Was_Visited_Unproductive"].sum()) if "Was_Visited_Unproductive" in sched else 0
        notbill_cnt = int(sched["Was_Scheduled_NotBilled"].sum())
        sizes  = [prod_count, unprod, notbill_cnt]
        labels = [f"Productive\n{prod_count:,}",f"Unproductive\n{unprod:,}",f"Not Billed\n{notbill_cnt:,}"]
        colors = [COLORS["green"], COLORS["amber"], COLORS["red"]]
        fig2,ax2=plt.subplots(figsize=(5,3.5))
        wedges,texts,autotexts=ax2.pie(sizes,labels=labels,colors=colors,
                                        autopct="%1.1f%%",startangle=90,
                                        wedgeprops=dict(width=0.6))
        for t in autotexts: t.set_fontsize(8)
        ax2.set_title("Scheduled Visit Outcomes",fontsize=10,fontweight="bold")
        fig2.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    col3,col4 = st.columns(2)

    with col3:
        st.markdown("**DSR Performance Summary**")
        dsr_7 = sched[sched["SalesmanCode"].notna()]
        dsr_s = (dsr_7.groupby(["SalesmanCode","SalesmanName"])
                       .agg(Scheduled=("IsProductive","count"),
                            Productive=("IsProductive","sum"),
                            NotBilled=("Was_Scheduled_NotBilled","sum"))
                       .reset_index())
        dsr_s["Prod_Rate_%"] = (dsr_s["Productive"]/dsr_s["Scheduled"]*100).round(1)
        dsr_s["NotBilled_%"] = (dsr_s["NotBilled"]/dsr_s["Scheduled"]*100).round(1)
        ob_dsr = (df[df["Is_OffBeat_Visit"]==1]
                  .groupby("SalesmanCode")["Total_GSV"].sum()
                  .rename("OffBeat_GSV").reset_index())
        dsr_s = dsr_s.merge(ob_dsr,on="SalesmanCode",how="left").fillna({"OffBeat_GSV":0})
        dsr_s["OffBeat_GSV"] = dsr_s["OffBeat_GSV"].round(0).astype(int)
        dsr_s = dsr_s.sort_values("Prod_Rate_%",ascending=False)
        st.dataframe(dsr_s[["SalesmanCode","SalesmanName","Scheduled","Productive",
                              "Prod_Rate_%","NotBilled_%","OffBeat_GSV"]],
                     hide_index=True, use_container_width=True)

    with col4:
        st.markdown("**Category Productivity**")
        cat_s = (sched.groupby("RetailerCategory")
                       .agg(Scheduled=("IsProductive","count"), Productive=("IsProductive","sum"))
                       .reset_index())
        cat_s["Prod_Rate_%"] = (cat_s["Productive"]/cat_s["Scheduled"]*100).round(1)
        cat_s = cat_s.sort_values("Scheduled",ascending=False)
        fig3,ax3=plt.subplots(figsize=(6,3.5))
        bars=ax3.barh(cat_s["RetailerCategory"],cat_s["Prod_Rate_%"],
                      color=COLORS["teal"],alpha=0.85)
        ax3.set_xlabel("Productivity Rate %"); ax3.set_title("Productivity Rate by Category",fontsize=10)
        ax3.axvline(prod_rate,color=COLORS["amber"],linestyle="--",linewidth=1,label=f"Overall {prod_rate}%")
        ax3.legend(fontsize=8); ax3.grid(axis="x",alpha=0.3)
        fig3.tight_layout(); st.pyplot(fig3); plt.close()

    st.markdown('<div class="insight">🔑 <b>Key Insight:</b> '
                f'{notbilled}% of all scheduled visits result in no billing — this is the primary lever. '
                f'Off-beat GSV ({ob_share}%) signals where demand exists but the PJP schedule does not reach.</div>',
                unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — M1: PRODUCTIVITY CLASSIFIER
# ════════════════════════════════════════════════════════════
elif page=="🎯 M1: Productivity":
    st.markdown('<div class="section-hdr">🎯 M1: PRODUCTIVITY CLASSIFIER — Will this visit be productive?</div>',
                unsafe_allow_html=True)
    st.markdown(f"**Binary classification | Target: GSV > Rs{PROD_TH} | Algorithm: XGBoost**")

    # Train
    model_res = train_m1(id(df))
    model, lr, X_te, y_te, test_df, X_tr, y_tr, X_va, y_va, val_df, train_df = model_res

    tabs = st.tabs(["📋 Model Metrics","🔲 Confusion Matrix","📊 Threshold Simulation",
                    "⭐ Feature Importance","🎯 Scored Visits","📉 Model Decomposition"])

    # ── Tab 1: Model Metrics ──────────────────────────────────
    with tabs[0]:
        st.markdown("**Split:** Train ≤ Feb-26 | Val: Mar-Apr-26 | Test: May-26 (holdout)")
        m_te = compute_metrics(model, X_te, y_te, threshold_ui)
        m_va = compute_metrics(model, X_va, y_va, threshold_ui)
        m_tr = compute_metrics(model, X_tr, y_tr, threshold_ui)
        m_lr = compute_metrics(lr,    X_te, y_te, threshold_ui)

        metrics_data = {
            "Metric":["Accuracy","Precision","Recall","F1-Score","AUC-ROC","AUC-PR","Log Loss"],
            "Train":  [m_tr["accuracy"],m_tr["precision"],m_tr["recall"],m_tr["f1"],m_tr["auc_roc"],m_tr["auc_pr"],m_tr["log_loss"]],
            "Validation":[m_va["accuracy"],m_va["precision"],m_va["recall"],m_va["f1"],m_va["auc_roc"],m_va["auc_pr"],m_va["log_loss"]],
            "Test (May-26)":[m_te["accuracy"],m_te["precision"],m_te["recall"],m_te["f1"],m_te["auc_roc"],m_te["auc_pr"],m_te["log_loss"]],
            "Baseline LR":[m_lr["accuracy"],m_lr["precision"],m_lr["recall"],m_lr["f1"],m_lr["auc_roc"],m_lr["auc_pr"],m_lr["log_loss"]],
            "Target":["70-85%",">65%",">60%",">0.62",">0.72",">0.55","<0.55"],
            "Definition":[
                "Overall % predictions correct","Of predicted productive: % actually productive",
                "Of actually productive: % correctly flagged","Harmonic mean of Precision & Recall",
                "Ranking quality across all thresholds","PR curve area — better for imbalanced data",
                "Penalises confident wrong predictions"
            ]
        }
        met_df = pd.DataFrame(metrics_data)

        def highlight_metric(row):
            colors = []
            for col in row.index:
                if col in ["Metric","Target","Definition","Baseline LR"]:
                    colors.append("")
                elif col=="Test (May-26)":
                    colors.append("background-color:#E0F5F3;font-weight:bold")
                else:
                    colors.append("")
            return colors

        st.dataframe(met_df.style.apply(highlight_metric,axis=1), hide_index=True, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        for col,label,val,color in zip([c1,c2,c3,c4],
            ["F1-Score","AUC-ROC","AUC-PR","Precision"],
            [m_te["f1"],m_te["auc_roc"],m_te["auc_pr"],m_te["precision"]],
            [COLORS["teal"],COLORS["green"],COLORS["purple"],COLORS["amber"]]):
            col.markdown(f'<div class="metric-card" style="border-left-color:{color}">'
                         f'<div class="metric-val" style="color:{color}">{val}%</div>'
                         f'<div class="metric-lbl">{label} (Test)</div></div>',
                         unsafe_allow_html=True)

        excel_bytes = to_excel({"Model_Metrics":met_df})
        st.download_button("⬇️ Download Metrics", excel_bytes,
                           "M1_Model_Metrics.xlsx", "application/vnd.ms-excel")

    # ── Tab 2: Confusion Matrix ───────────────────────────────
    with tabs[1]:
        m = compute_metrics(model, X_te, y_te, threshold_ui)
        tp,fp,fn,tn = m["tp"],m["fp"],m["fn"],m["tn"]

        col1,col2 = st.columns([1,1])
        with col1:
            st.markdown("**Confusion Matrix — Test Set (May-26)**")
            fig,ax=plt.subplots(figsize=(5,4))
            cm_arr=np.array([[tp,fn],[fp,tn]])
            labels_arr=np.array([[f"TRUE POSITIVE\n{tp:,}\nModel said GO\nOutlet ordered",
                                   f"FALSE NEGATIVE\n{fn:,}\nModel said SKIP\nWould have ordered"],
                                  [f"FALSE POSITIVE\n{fp:,}\nModel said GO\nDid NOT order",
                                   f"TRUE NEGATIVE\n{tn:,}\nModel said SKIP\nCorrectly skipped"]])
            colors_cm=[[COLORS["green"],"#FDEDEC"],["#FFF0CC",COLORS["teal"]]]
            for i in range(2):
                for j in range(2):
                    ax.add_patch(plt.Rectangle((j,1-i),1,1,facecolor=colors_cm[i][j],edgecolor="white",linewidth=3))
                    ax.text(j+0.5,1-i+0.5,labels_arr[i][j],ha="center",va="center",fontsize=8,
                            color="white" if (i==0 and j==0) or (i==1 and j==1) else "#1F2937",
                            fontweight="bold" if i==j else "normal")
            ax.set_xlim(0,2); ax.set_ylim(0,2)
            ax.set_xticks([0.5,1.5]); ax.set_yticks([0.5,1.5])
            ax.set_xticklabels(["Predicted:\nPRODUCTIVE","Predicted:\nNOT PRODUCTIVE"])
            ax.set_yticklabels(["Actual:\nNOT PRODUCTIVE","Actual:\nPRODUCTIVE"])
            ax.set_title(f"Threshold: {threshold_ui:.2f}",fontsize=10)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("**Business Translation**")
            avg_prod_gsv = test_df[test_df["IsProductive"]==1]["Total_GSV"].mean()
            rev_captured = tp * avg_prod_gsv
            rev_missed   = fn * avg_prod_gsv
            wasted_trips = fp

            biz_data = {
                "Outcome":["✅ Smart Visits (TP)","❌ Missed Orders (FN)",
                           "⚠️ Wasted Trips (FP)","✅ Smart Skips (TN)"],
                "Count":[f"{tp:,}",f"{fn:,}",f"{fp:,}",f"{tn:,}"],
                "Business Meaning":[
                    f"Model sent DSR → outlet ordered | Est. GSV: Rs{rev_captured:,.0f}",
                    f"Model skipped → outlet would have ordered | Est. lost GSV: Rs{rev_missed:,.0f}",
                    "Model sent DSR → outlet did NOT order | Wasted DSR effort",
                    "Model skipped → outlet would NOT have ordered | DSR time saved"
                ]
            }
            st.dataframe(pd.DataFrame(biz_data),hide_index=True,use_container_width=True)

            st.markdown(f'<div class="insight">💰 Est. GSV captured by model-recommended visits: '
                        f'<b>Rs{rev_captured:,.0f}</b><br>'
                        f'💸 Est. GSV missed (false negatives): <b>Rs{rev_missed:,.0f}</b><br>'
                        f'⚡ Precision at threshold {threshold_ui:.2f}: <b>{m["precision"]}%</b> | '
                        f'Recall: <b>{m["recall"]}%</b></div>', unsafe_allow_html=True)

        cm_df = pd.DataFrame({
            "Metric":["True Positive (TP)","False Negative (FN)","False Positive (FP)","True Negative (TN)",
                      "Precision","Recall","F1-Score"],
            "Value":[tp,fn,fp,tn,f'{m["precision"]}%',f'{m["recall"]}%',f'{m["f1"]}%'],
            "Business Meaning":["Model GO → Outlet ordered","Model SKIP → Outlet would have ordered",
                                  "Model GO → Outlet did NOT order","Model SKIP → Correctly skipped",
                                  "Of all GO predictions, % that were correct",
                                  "Of all actual productive, % correctly flagged",
                                  "Balanced measure of Precision + Recall"]
        })
        excel_bytes = to_excel({"Confusion_Matrix":cm_df})
        st.download_button("⬇️ Download Confusion Matrix", excel_bytes,
                           "M1_Confusion_Matrix.xlsx","application/vnd.ms-excel")

    # ── Tab 3: Threshold Simulation ───────────────────────────
    with tabs[2]:
        st.markdown("**How do Precision, Recall, F1, and Revenue change as we move the decision threshold?**")
        prob_te = model.predict_proba(X_te)[:,1]
        thresholds = np.arange(0.05,0.96,0.01)
        rows=[]
        avg_gsv = test_df[test_df["IsProductive"]==1]["Total_GSV"].mean()
        for th in thresholds:
            pred=(prob_te>=th).astype(int)
            cm2=confusion_matrix(y_te,pred)
            tn2,fp2,fn2,tp2=cm2.ravel()
            prec=precision_score(y_te,pred,zero_division=0)
            rec=recall_score(y_te,pred,zero_division=0)
            f1s=f1_score(y_te,pred,zero_division=0)
            rows.append({
                "Threshold":round(th,2),"Precision":round(prec*100,2),
                "Recall":round(rec*100,2),"F1":round(f1s*100,2),
                "Accuracy":round(accuracy_score(y_te,pred)*100,2),
                "TP":int(tp2),"FP":int(fp2),"FN":int(fn2),"TN":int(tn2),
                "Rev_Captured":round(tp2*avg_gsv,0),
                "Rev_Missed":round(fn2*avg_gsv,0),
                "Pct_Prod_Caught":round(tp2/(tp2+fn2)*100,2) if (tp2+fn2)>0 else 0,
                "Pct_Wasted_Avoided":round(tn2/(tn2+fp2)*100,2) if (tn2+fp2)>0 else 0,
            })
        th_df=pd.DataFrame(rows)

        fig4,axes=plt.subplots(1,2,figsize=(12,4))
        axes[0].plot(th_df["Threshold"],th_df["Precision"],label="Precision",color=COLORS["teal"],linewidth=2)
        axes[0].plot(th_df["Threshold"],th_df["Recall"],label="Recall",color=COLORS["green"],linewidth=2)
        axes[0].plot(th_df["Threshold"],th_df["F1"],label="F1",color=COLORS["amber"],linewidth=2)
        axes[0].axvline(threshold_ui,color=COLORS["red"],linestyle="--",linewidth=1.5,label=f"Selected: {threshold_ui:.2f}")
        axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("%"); axes[0].set_title("Precision / Recall / F1 vs Threshold")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        axes[1].plot(th_df["Threshold"],th_df["Rev_Captured"]/1e5,label="Revenue Captured (Rs L)",color=COLORS["green"],linewidth=2)
        axes[1].plot(th_df["Threshold"],th_df["Rev_Missed"]/1e5,label="Revenue Missed (Rs L)",color=COLORS["red"],linewidth=2)
        axes[1].axvline(threshold_ui,color=COLORS["amber"],linestyle="--",linewidth=1.5)
        axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Rs Lakhs"); axes[1].set_title("Revenue Impact vs Threshold")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        fig4.tight_layout(); st.pyplot(fig4); plt.close()

        selected_row = th_df[th_df["Threshold"]==round(threshold_ui,2)]
        if not selected_row.empty:
            sr = selected_row.iloc[0]
            st.markdown(f'<div class="insight">At threshold <b>{threshold_ui:.2f}</b>: '
                        f'Precision=<b>{sr.Precision}%</b> | Recall=<b>{sr.Recall}%</b> | '
                        f'F1=<b>{sr.F1}%</b> | Revenue Captured=<b>Rs{sr.Rev_Captured:,.0f}</b> | '
                        f'Revenue Missed=<b>Rs{sr.Rev_Missed:,.0f}</b></div>', unsafe_allow_html=True)

        st.dataframe(th_df.rename(columns={
            "Rev_Captured":"Revenue Captured (Rs)","Rev_Missed":"Revenue Missed (Rs)",
            "Pct_Prod_Caught":"% Productive Caught","Pct_Wasted_Avoided":"% Wasted Avoided"}),
            hide_index=True, use_container_width=True)
        excel_bytes = to_excel({"Threshold_Simulation":th_df})
        st.download_button("⬇️ Download Threshold Table", excel_bytes,
                           "M1_Threshold_Simulation.xlsx","application/vnd.ms-excel")

    # ── Tab 4: Feature Importance ─────────────────────────────
    with tabs[3]:
        st.markdown("**Feature Importance — XGBoost (Gain-based)**")
        fi = pd.DataFrame({
            "Feature_Technical": FEATURES,
            "Feature_Business":  [FEATURE_LABELS[f] for f in FEATURES],
            "Importance":        model.feature_importances_
        }).sort_values("Importance",ascending=False).reset_index(drop=True)
        fi["Importance_%"] = (fi["Importance"]/fi["Importance"].sum()*100).round(2)
        fi["Rank"] = range(1,len(fi)+1)
        fi["Category"] = fi["Feature_Technical"].map({
            "WeekOfMonth":"Timing","Month":"Timing","IsEndOfMonth":"Timing",
            "IsFestivalMonth":"Timing","DayOfWeek":"Timing",
            "schedule_ipi":"Visit History","gsv_lag1":"Visit History","gsv_lag2":"Visit History",
            "productive_lag1":"Visit History","productive_lag2":"Visit History",
            "rolling_3m_prod":"Visit History","consec_skips":"Visit History",
            "productive_rate":"Outlet Profile","Outlet_Age_Months":"Outlet Profile",
            "Category_enc":"Outlet Profile","vpm":"Outlet Profile","offbeat_propensity":"Outlet Profile",
            "dsr_prod_rate":"DSR Profile","dsr_notbilled_rate":"DSR Profile"
        })
        cat_colors_fi = {"Timing":COLORS["teal"],"Visit History":COLORS["green"],
                         "Outlet Profile":COLORS["amber"],"DSR Profile":COLORS["purple"]}

        col1,col2 = st.columns([2,1])
        with col1:
            fig5,ax5=plt.subplots(figsize=(8,6))
            colors_fi=[cat_colors_fi.get(c,COLORS["slate"]) for c in fi["Category"]]
            bars=ax5.barh(fi["Feature_Business"],fi["Importance_%"],color=colors_fi,alpha=0.85)
            ax5.set_xlabel("Importance %"); ax5.set_title("Feature Importance (XGBoost Gain)",fontsize=10)
            ax5.invert_yaxis()
            patches=[mpatches.Patch(color=v,label=k) for k,v in cat_colors_fi.items()]
            ax5.legend(handles=patches,fontsize=8,loc="lower right")
            ax5.grid(axis="x",alpha=0.3)
            fig5.tight_layout(); st.pyplot(fig5); plt.close()
        with col2:
            st.dataframe(fi[["Rank","Feature_Business","Category","Importance_%"]].rename(
                columns={"Feature_Business":"Feature","Importance_%":"Importance %"}),
                hide_index=True, use_container_width=True)

        excel_bytes = to_excel({"Feature_Importance":fi[["Rank","Feature_Business","Category",
                                                           "Feature_Technical","Importance_%"]]})
        st.download_button("⬇️ Download Feature Importance", excel_bytes,
                           "M1_Feature_Importance.xlsx","application/vnd.ms-excel")

    # ── Tab 5: Scored Visits ──────────────────────────────────
    with tabs[4]:
        st.markdown("**All Scored Test Visits — May 2026**")
        prob_te = model.predict_proba(X_te)[:,1]
        pred_te = (prob_te>=threshold_ui).astype(int)
        scored = test_df[["BillingDate","SalesmanCode","SalesmanName","UniqueRetailerCode",
                           "RetailerURCname","RetailerCategory","Total_GSV","IsProductive"]].copy()
        scored["Model_Probability"] = prob_te.round(4)
        scored["Model_Predicts"]    = pred_te
        scored["Priority_Tier"]     = pd.cut(prob_te,bins=[0,0.4,0.65,1.0],
                                              labels=["Low (<0.4)","Medium (0.4-0.65)","High (>0.65)"])
        outcomes = []
        for ap,mp in zip(scored["IsProductive"],pred_te):
            if ap==1 and mp==1: outcomes.append("True Positive")
            elif ap==1 and mp==0: outcomes.append("False Negative")
            elif ap==0 and mp==1: outcomes.append("False Positive")
            else: outcomes.append("True Negative")
        scored["Outcome"] = outcomes
        scored["Correct"] = (scored["IsProductive"]==scored["Model_Predicts"]).astype(int)

        col1,col2,col3 = st.columns(3)
        col1.selectbox("Filter DSR", ["All"]+sorted(scored["SalesmanCode"].dropna().unique().tolist()),
                       key="scored_dsr")
        col2.selectbox("Filter Category", ["All"]+sorted(scored["RetailerCategory"].dropna().unique().tolist()),
                       key="scored_cat")
        col3.selectbox("Filter Priority", ["All","High (>0.65)","Medium (0.4-0.65)","Low (<0.4)"],
                       key="scored_tier")

        fs = scored.copy()
        if st.session_state.scored_dsr!="All": fs=fs[fs["SalesmanCode"]==st.session_state.scored_dsr]
        if st.session_state.scored_cat!="All": fs=fs[fs["RetailerCategory"]==st.session_state.scored_cat]
        if st.session_state.scored_tier!="All": fs=fs[fs["Priority_Tier"]==st.session_state.scored_tier]

        st.markdown(f"Showing **{len(fs):,}** visits")
        st.dataframe(fs.sort_values("Model_Probability",ascending=False).head(500),
                     hide_index=True,use_container_width=True)

        excel_bytes = to_excel({"Scored_Test_Visits":scored,
                                 "Priority_Summary":scored.groupby("Priority_Tier").agg(
                                     Count=("IsProductive","size"),
                                     Actual_Prod_Rate=("IsProductive","mean"),
                                     Avg_GSV=("Total_GSV","mean")
                                 ).reset_index()})
        st.download_button("⬇️ Download All Scored Visits", excel_bytes,
                           "M1_Scored_Visits.xlsx","application/vnd.ms-excel")

    # ── Tab 6: Model Decomposition ────────────────────────────
    with tabs[5]:
        st.markdown("**Model Decomposition — How productivity varies across key dimensions**")
        prob_all = model.predict_proba(sched[sched["SalesmanCode"].notna()][FEATURES].fillna(0))[:,1]
        sched_scored = sched[sched["SalesmanCode"].notna()].copy()
        sched_scored["Pred_Prob"] = prob_all

        col1,col2 = st.columns(2)
        with col1:
            st.markdown("*Predicted probability by Week of Month*")
            wk_decomp = sched_scored.groupby("WeekOfMonth")["Pred_Prob"].mean().reset_index()
            wk_decomp["WeekLabel"] = wk_decomp["WeekOfMonth"].map(
                {1:"W1",2:"W2",3:"W3",4:"W4"})
            fig6,ax6=plt.subplots(figsize=(5,3))
            ax6.bar(wk_decomp["WeekLabel"],wk_decomp["Pred_Prob"]*100,color=COLORS["teal"],alpha=0.85)
            ax6.set_ylabel("Avg Predicted Probability %")
            ax6.set_title("Predicted Productivity by Week",fontsize=10); ax6.grid(axis="y",alpha=0.3)
            fig6.tight_layout(); st.pyplot(fig6); plt.close()

        with col2:
            st.markdown("*Predicted probability by Outlet Age Bucket*")
            sched_scored["Age_Bucket"]=pd.cut(sched_scored["Outlet_Age_Months"],
                bins=[-1,3,6,12,18,24,9999],labels=["0-3m","3-6m","6-12m","12-18m","18-24m","24m+"])
            age_decomp=sched_scored.groupby("Age_Bucket",observed=True)["Pred_Prob"].mean().reset_index()
            fig7,ax7=plt.subplots(figsize=(5,3))
            ax7.bar(age_decomp["Age_Bucket"].astype(str),age_decomp["Pred_Prob"]*100,
                    color=COLORS["green"],alpha=0.85)
            ax7.set_ylabel("Avg Predicted Probability %")
            ax7.set_title("Predicted Productivity by Outlet Age",fontsize=10); ax7.grid(axis="y",alpha=0.3)
            fig7.tight_layout(); st.pyplot(fig7); plt.close()

        st.markdown("*Predicted probability by DSR*")
        dsr_decomp=sched_scored.groupby("SalesmanCode").agg(
            Avg_Pred_Prob=("Pred_Prob","mean"),
            Actual_Prod_Rate=("IsProductive","mean"),
            Visit_Count=("IsProductive","size")).reset_index()
        dsr_decomp["Avg_Pred_Prob_%"]=(dsr_decomp["Avg_Pred_Prob"]*100).round(2)
        dsr_decomp["Actual_Prod_Rate_%"]=(dsr_decomp["Actual_Prod_Rate"]*100).round(2)
        dsr_decomp["Model_vs_Actual_Gap"]=\
            (dsr_decomp["Avg_Pred_Prob_%"]-dsr_decomp["Actual_Prod_Rate_%"]).round(2)
        st.dataframe(dsr_decomp[["SalesmanCode","Visit_Count","Actual_Prod_Rate_%",
                                   "Avg_Pred_Prob_%","Model_vs_Actual_Gap"]],
                     hide_index=True,use_container_width=True)

        excel_bytes = to_excel({
            "Week_Decomp":wk_decomp, "Age_Decomp":age_decomp, "DSR_Decomp":dsr_decomp,
            "All_Features_Input":sched[sched["SalesmanCode"].notna()][
                ["BillingDate","SalesmanCode","UniqueRetailerCode","RetailerCategory"]+FEATURES+["IsProductive"]]
        })
        st.download_button("⬇️ Download Model Decomposition + Input Data", excel_bytes,
                           "M1_Model_Decomposition.xlsx","application/vnd.ms-excel")

# ════════════════════════════════════════════════════════════
# PAGE 3 — M2: CADENCE OPTIMISER
# ════════════════════════════════════════════════════════════
elif page=="📅 M2: Cadence":
    st.markdown('<div class="section-hdr">📅 M2: CADENCE & RHYTHM OPTIMISER — Right frequency and week per outlet</div>',
                unsafe_allow_html=True)

    tabs2 = st.tabs(["📊 IPI Analysis","📅 Anchor Week","🔄 VPM Recommendation",
                      "♻️ Capacity Rebalancing","⬇️ Downloads"])

    # ── Tab 1: IPI Analysis ───────────────────────────────────
    with tabs2[0]:
        st.markdown("**Inter-Purchase Interval (IPI) — Schedule gap vs Productivity Rate**")
        sched_ipi = sched[sched["schedule_ipi"].notna() & sched["SalesmanCode"].notna()].copy()
        sched_ipi["IPI_Bucket"] = pd.cut(sched_ipi["schedule_ipi"],
            bins=[-1,3,7,14,21,30,60,9999],
            labels=["0-3d","4-7d","8-14d","15-21d","22-30d","31-60d","60d+"])
        ipi_an = sched_ipi.groupby("IPI_Bucket",observed=True).agg(
            Visits=("IsProductive","size"),
            Productive=("IsProductive","sum"),
            Avg_IPI=("schedule_ipi","mean")).reset_index()
        ipi_an["Prod_Rate_%"] = (ipi_an["Productive"]/ipi_an["Visits"]*100).round(1)
        ipi_an["Avg_IPI"]     = ipi_an["Avg_IPI"].round(1)
        ipi_an["Pct_Visits"]  = (ipi_an["Visits"]/ipi_an["Visits"].sum()*100).round(1)

        col1,col2 = st.columns(2)
        with col1:
            fig,ax=plt.subplots(figsize=(6,3.5))
            colors_ipi=[COLORS["red"] if str(b)=="0-3d" else
                         COLORS["teal"] if str(b)=="4-7d" else
                         COLORS["green"] for b in ipi_an["IPI_Bucket"]]
            ax.bar(ipi_an["IPI_Bucket"].astype(str),ipi_an["Prod_Rate_%"],
                   color=colors_ipi,alpha=0.85)
            ax.set_xlabel("Schedule IPI Bucket"); ax.set_ylabel("Productivity Rate %")
            ax.set_title("Productivity Rate by Schedule Gap (IPI)",fontsize=10)
            ax.grid(axis="y",alpha=0.3)
            fig.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            st.dataframe(ipi_an[["IPI_Bucket","Visits","Pct_Visits_%","Prod_Rate_%","Avg_IPI"]
                                  if "Pct_Visits_%" in ipi_an else
                                 ["IPI_Bucket","Visits","Pct_Visits","Prod_Rate_%","Avg_IPI"]],
                         hide_index=True,use_container_width=True)

        st.markdown('<div class="insight">Key finding: Longer IPI → Higher productivity. '
                    '8-14d (fortnightly) consistently outperforms 4-7d (weekly) cadence across all categories.</div>',
                    unsafe_allow_html=True)

        # IPI by category
        st.markdown("**IPI × Category Productivity Cross-tab**")
        ipi_cat = sched_ipi.groupby(["RetailerCategory","IPI_Bucket"],observed=True)["IsProductive"].mean().unstack()
        ipi_cat = (ipi_cat*100).round(1)
        fig2,ax2=plt.subplots(figsize=(10,4))
        sns.heatmap(ipi_cat,annot=True,fmt=".1f",cmap="YlGn",ax=ax2,
                    linewidths=0.5,cbar_kws={"label":"Productivity %"})
        ax2.set_title("Productivity Rate (%) by Category × IPI Bucket",fontsize=10)
        fig2.tight_layout(); st.pyplot(fig2); plt.close()

    # ── Tab 2: Anchor Week ────────────────────────────────────
    with tabs2[1]:
        st.markdown("**Week-Share CV Analysis — Clockwork outlet detection**")
        onbeat_billed = df[df["Visit_Source"]=="Scheduled_Billed"].copy()
        weekly = (onbeat_billed.groupby(["UniqueRetailerCode","SalesmanCode","BillingMonth","WeekOfMonth"])
                               ["Total_GSV"].sum().reset_index())
        mo_tot = (onbeat_billed.groupby(["UniqueRetailerCode","SalesmanCode","BillingMonth"])
                               ["Total_GSV"].sum().reset_index().rename(columns={"Total_GSV":"mt"}))
        ws = weekly.merge(mo_tot,on=["UniqueRetailerCode","SalesmanCode","BillingMonth"])
        ws = ws[ws["mt"]>0]; ws["week_share"]=ws["Total_GSV"]/ws["mt"]
        ws_cv = (ws.groupby(["UniqueRetailerCode","SalesmanCode","WeekOfMonth"])
                   ["week_share"].agg(["mean","std","count"]).reset_index())
        ws_cv = ws_cv[ws_cv["count"]>=3]
        ws_cv["cv"] = (ws_cv["std"]/ws_cv["mean"].clip(lower=0.05)).clip(upper=5)
        ws_cv["is_clockwork"] = (ws_cv["cv"]<0.3).astype(int)

        # Assign anchor week
        anchor = (ws_cv.sort_values("mean",ascending=False)
                       .groupby(["UniqueRetailerCode","SalesmanCode"])
                       .first().reset_index()
                       [["UniqueRetailerCode","SalesmanCode","WeekOfMonth","cv","is_clockwork"]]
                       .rename(columns={"WeekOfMonth":"Anchor_Week","cv":"Week_Share_CV"}))
        anchor["Anchor_Type"] = anchor["is_clockwork"].map({1:"Clockwork (CV<0.3)",0:"Default (W2 assigned)"})
        anchor["Anchor_Week"] = anchor.apply(
            lambda r: r["Anchor_Week"] if r["is_clockwork"]==1 else 2, axis=1)
        anchor["Week_Label"] = anchor["Anchor_Week"].map({1:"W1",2:"W2",3:"W3",4:"W4"})

        col1,col2 = st.columns(2)
        with col1:
            cw_pct = (anchor["is_clockwork"].mean()*100).round(1)
            st.metric("Clockwork Outlets",f"{anchor['is_clockwork'].sum():,}",
                      f"{cw_pct}% of all outlet-DSR pairs")
            st.metric("Total Outlet-DSR Pairs",f"{len(anchor):,}")

            ws_summ = (ws_cv.groupby("WeekOfMonth").agg(
                Avg_CV=("cv","mean"), Pct_Clockwork=("is_clockwork","mean"),
                Total_Combos=("cv","size")).reset_index())
            ws_summ["Pct_Clockwork_%"]=(ws_summ["Pct_Clockwork"]*100).round(1)
            ws_summ["WeekLabel"]=ws_summ["WeekOfMonth"].map({1:"W1",2:"W2",3:"W3",4:"W4"})
            st.dataframe(ws_summ[["WeekLabel","Total_Combos","Avg_CV","Pct_Clockwork_%"]],
                         hide_index=True,use_container_width=True)

        with col2:
            fig3,ax3=plt.subplots(figsize=(5,3.5))
            aw_counts=anchor.groupby(["Week_Label","Anchor_Type"])["UniqueRetailerCode"].count().reset_index()
            colors_aw=[COLORS["teal"],COLORS["amber"]]
            for i,atype in enumerate(["Clockwork (CV<0.3)","Default (W2 assigned)"]):
                sub=aw_counts[aw_counts["Anchor_Type"]==atype]
                ax3.bar(sub["Week_Label"],sub["UniqueRetailerCode"],
                        label=atype,color=colors_aw[i],alpha=0.85)
            ax3.set_title("Anchor Week Assignment Distribution",fontsize=10)
            ax3.set_xlabel("Week"); ax3.set_ylabel("Outlet Count")
            ax3.legend(fontsize=8); ax3.grid(axis="y",alpha=0.3)
            fig3.tight_layout(); st.pyplot(fig3); plt.close()

        st.dataframe(anchor.sort_values(["SalesmanCode","Week_Share_CV"]).head(200),
                     hide_index=True,use_container_width=True)
        st.session_state["anchor"] = anchor

    # ── Tab 3: VPM Recommendation ─────────────────────────────
    with tabs2[2]:
        st.markdown("**Recommended Visits per Month per Outlet-DSR Pair**")
        vpm_data = (sched[sched["SalesmanCode"].notna()]
                    .groupby(["UniqueRetailerCode","RetailerURCname","SalesmanCode","SalesmanName","RetailerCategory"])
                    .agg(Total_Visits=("IsProductive","count"),
                         Productive=("IsProductive","sum"),
                         NotBilled=("Was_Scheduled_NotBilled","sum"),
                         Months_Active=("BillingMonth","nunique"),
                         Avg_GSV_Billed=("Total_GSV","mean"))
                    .reset_index())
        vpm_data["Actual_VPM"]     = (vpm_data["Total_Visits"]/vpm_data["Months_Active"].clip(lower=1)).round(1)
        vpm_data["Prod_Rate"]      = (vpm_data["Productive"]/vpm_data["Total_Visits"]).round(4)
        vpm_data["NotBilled_Rate"] = (vpm_data["NotBilled"]/vpm_data["Total_Visits"]).round(4)
        vpm_data["Avg_GSV_Billed"] = vpm_data["Avg_GSV_Billed"].round(0)

        # Off-beat flag
        ob_outlets = (df[df["Is_OffBeat_Visit"]==1]
                      .groupby("UniqueRetailerCode")["Total_GSV"]
                      .agg(["sum","count"]).reset_index()
                      .rename(columns={"sum":"OffBeat_GSV","count":"OffBeat_Txns"}))
        vpm_data = vpm_data.merge(ob_outlets,on="UniqueRetailerCode",how="left").fillna(
            {"OffBeat_GSV":0,"OffBeat_Txns":0})

        def recommend_vpm(row):
            p=row["Prod_Rate"]; v=row["Actual_VPM"]; nb=row["NotBilled_Rate"]
            ob=row["OffBeat_Txns"]; gsv=row["Avg_GSV_Billed"]
            if p<0.15 and nb>0.80 and ob==0: return 0,"Remove from beat — chronic skip, no off-beat demand"
            if p<0.25 and v>=3:
                if ob>=3: return 2,"Reduce frequency — low on-beat productivity but off-beat demand exists"
                return 1,"Consider remove or reduce to 1/mo — very low conversion"
            if gsv>2000 and p>0.45: return min(v+1,4),"High value outlet — increase or maintain high frequency"
            if 0.25<=p<0.35 and v>=3: return 2,"Over-visited — reduce to fortnightly cadence"
            if p>=0.40 and v<=2: return 3,"Under-visited — increase frequency, strong conversion"
            if ob>=3 and v<=1: return 2,"Add to beat — substantial off-beat demand"
            return round(v),"Maintain current cadence"

        recs = vpm_data.apply(lambda r: pd.Series(recommend_vpm(r),index=["Rec_VPM","Rationale"]),axis=1)
        vpm_data = pd.concat([vpm_data,recs],axis=1)
        vpm_data["VPM_Change"] = vpm_data["Rec_VPM"]-vpm_data["Actual_VPM"].round(0)
        vpm_data["Action"] = vpm_data["VPM_Change"].apply(
            lambda x: "🔴 Remove" if x==-vpm_data["Actual_VPM"].max()
                      else ("🔺 Increase" if x>0 else ("🔻 Reduce" if x<0 else "✅ Maintain")))

        summ_col,_ = st.columns([2,1])
        with summ_col:
            action_summ=vpm_data["Action"].value_counts().reset_index()
            st.dataframe(action_summ,hide_index=True,use_container_width=True)

        dsr_filter = st.selectbox("Filter by DSR",["All"]+sorted(vpm_data["SalesmanCode"].unique().tolist()))
        fv = vpm_data if dsr_filter=="All" else vpm_data[vpm_data["SalesmanCode"]==dsr_filter]

        st.dataframe(fv[["SalesmanCode","UniqueRetailerCode","RetailerURCname","RetailerCategory",
                          "Actual_VPM","Prod_Rate","NotBilled_Rate","OffBeat_Txns",
                          "Rec_VPM","VPM_Change","Action","Rationale"]].sort_values("VPM_Change"),
                     hide_index=True,use_container_width=True)
        st.session_state["vpm_data"] = vpm_data

    # ── Tab 4: Capacity Rebalancing ───────────────────────────
    with tabs2[3]:
        st.markdown("**DSR Capacity Rebalancing — Freed slots → High-potential outlets**")
        if "vpm_data" not in st.session_state:
            st.warning("Run VPM Recommendation tab first."); st.stop()
        vd = st.session_state["vpm_data"]
        freed = (vd[vd["VPM_Change"]<0]
                 .groupby("SalesmanCode")
                 .apply(lambda x: (-x["VPM_Change"]*x["Months_Active"]).sum())
                 .reset_index().rename(columns={0:"Freed_Slots"}))
        total_visits = (vd.groupby("SalesmanCode")["Total_Visits"].sum().reset_index())
        freed = freed.merge(total_visits,on="SalesmanCode")
        freed["Freed_%"] = (freed["Freed_Slots"]/freed["Total_Visits"]*100).round(1)
        freed["Reallocation_Target"] = "High-probability off-beat + 6-12m age outlets"

        st.dataframe(freed,hide_index=True,use_container_width=True)

        add_candidates = vd[(vd["Rec_VPM"]>0) & (vd["Actual_VPM"]<1) & (vd["OffBeat_Txns"]>=2)]
        st.markdown(f"**{len(add_candidates)} outlets recommended for PJP addition** "
                    f"(off-beat active but under-scheduled)")
        st.dataframe(add_candidates[["SalesmanCode","UniqueRetailerCode","RetailerURCname",
                                      "RetailerCategory","Actual_VPM","OffBeat_Txns",
                                      "Rec_VPM","Rationale"]].head(50),
                     hide_index=True,use_container_width=True)

    # ── Tab 5: Downloads ──────────────────────────────────────
    with tabs2[4]:
        if "vpm_data" in st.session_state and "anchor" in st.session_state:
            vd2 = st.session_state["vpm_data"]
            anc = st.session_state["anchor"]
            cadence_out = vd2.merge(anc[["UniqueRetailerCode","SalesmanCode","Anchor_Week",
                                          "Week_Label","Week_Share_CV","Anchor_Type"]],
                                    on=["UniqueRetailerCode","SalesmanCode"],how="left")
            cadence_out["Week_Label"] = cadence_out["Week_Label"].fillna("W2")
            excel_bytes = to_excel({
                "VPM_Recommendations": cadence_out,
                "IPI_Analysis": sched[sched["schedule_ipi"].notna()].groupby(
                    pd.cut(sched[sched["schedule_ipi"].notna()]["schedule_ipi"],
                           bins=[-1,3,7,14,21,30,60,9999],
                           labels=["0-3d","4-7d","8-14d","15-21d","22-30d","31-60d","60d+"])
                ).agg(Visits=("IsProductive","size"),Prod_Rate=("IsProductive","mean")).reset_index()
            })
            st.download_button("⬇️ Download M2 Cadence Recommendations", excel_bytes,
                               "M2_Cadence_Recommendations.xlsx","application/vnd.ms-excel")
        else:
            st.info("Run the IPI and VPM tabs first to generate recommendations.")

# ════════════════════════════════════════════════════════════
# PAGE 4 — M3: ROUTE OPTIMISER
# ════════════════════════════════════════════════════════════
elif page=="🗺️ M3: Route":
    st.markdown('<div class="section-hdr">🗺️ M3: ROUTE & BEAT OPTIMISATION — Geo-clustering + 2-Opt Sequencing</div>',
                unsafe_allow_html=True)

    tabs3 = st.tabs(["⭐ Outlet Scoring","🗺️ Geo-Clustering","📍 Beat Sequencing",
                     "📋 Beat Plan","⬇️ Downloads"])

    # Outlets with valid coords
    outlets = (df[df["Lat_clean"].notna() & df["SalesmanCode"].notna()]
               .drop_duplicates("UniqueRetailerCode")
               [["UniqueRetailerCode","RetailerURCname","RetailerCategory",
                 "SalesmanCode","Lat_clean","Lon_clean"]].copy())
    hist_pr2 = (sched[sched["SalesmanCode"].notna()]
                .groupby("UniqueRetailerCode")
                .agg(prod_rate=("IsProductive","mean"),
                     total_visits=("IsProductive","count"))
                .reset_index())
    ob_score = (df[df["Is_OffBeat_Visit"]==1]
                .groupby("UniqueRetailerCode")["Total_GSV"]
                .agg(["sum","count"]).reset_index()
                .rename(columns={"sum":"ob_gsv","count":"ob_txns"}))
    age_score = (df.groupby("UniqueRetailerCode")["Outlet_Age_Months"].mean().reset_index())
    outlets = (outlets.merge(hist_pr2,on="UniqueRetailerCode",how="left")
                      .merge(ob_score,on="UniqueRetailerCode",how="left")
                      .merge(age_score,on="UniqueRetailerCode",how="left")
                      .fillna({"prod_rate":0.35,"total_visits":0,"ob_txns":0,"ob_gsv":0,"Outlet_Age_Months":24}))

    # Priority score
    outlets["age_bonus"] = outlets["Outlet_Age_Months"].apply(
        lambda a: 10 if 6<=a<=12 else (5 if 3<=a<=18 else 0))
    outlets["ob_score"] = (outlets["ob_txns"].clip(upper=10)/10*20)
    outlets["priority_score"] = (outlets["prod_rate"]*40 + outlets["ob_score"] +
                                  outlets["age_bonus"] + 30).clip(0,100).round(1)
    outlets["eligible"] = (
        ~((outlets["prod_rate"]<0.10) & (outlets["total_visits"]>6) & (outlets["ob_txns"]==0))
    ).astype(int)

    # ── Tab 1: Outlet Scoring ─────────────────────────────────
    with tabs3[0]:
        st.markdown("**Outlet Priority Score (0-100) — basis for beat inclusion**")
        st.markdown("Score = Historical Prod Rate (40%) + Off-Beat Activity (20%) + Outlet Age Bonus (10%) + Base (30%)")
        col1,col2,col3 = st.columns(3)
        col1.metric("Total Outlets (valid coords)",f"{len(outlets):,}")
        col2.metric("Eligible for Beat",f"{outlets['eligible'].sum():,}")
        col3.metric("Flagged for Review",f"{(outlets['eligible']==0).sum():,}")

        fig,ax=plt.subplots(figsize=(8,3))
        ax.hist(outlets["priority_score"],bins=30,color=COLORS["teal"],alpha=0.8,edgecolor="white")
        ax.set_xlabel("Priority Score"); ax.set_ylabel("Outlet Count")
        ax.set_title("Distribution of Outlet Priority Scores",fontsize=10)
        ax.grid(axis="y",alpha=0.3); fig.tight_layout(); st.pyplot(fig); plt.close()

        dsr_f = st.selectbox("Filter by DSR",["All"]+sorted(outlets["SalesmanCode"].dropna().unique().tolist()),
                              key="m3_dsr")
        fo = outlets if dsr_f=="All" else outlets[outlets["SalesmanCode"]==dsr_f]
        st.dataframe(fo[["SalesmanCode","UniqueRetailerCode","RetailerURCname","RetailerCategory",
                          "prod_rate","ob_txns","age_bonus","priority_score","eligible"]]
                     .sort_values("priority_score",ascending=False),
                     hide_index=True,use_container_width=True)
        st.session_state["outlets_scored"] = outlets

    # ── Tab 2: Geo-Clustering ─────────────────────────────────
    with tabs3[1]:
        st.markdown("**K-Means Geo-Clustering — Beat formation from outlet lat/long**")
        elig = outlets[outlets["eligible"]==1].copy()
        n_clusters = st.slider("Number of beat clusters (k)",min_value=5,max_value=50,value=21)
        if st.button("Run Geo-Clustering"):
            coords = elig[["Lat_clean","Lon_clean"]].values
            best_k = n_clusters; best_score=-1
            for k in range(max(3,n_clusters-3), min(n_clusters+4,len(elig)//5)):
                km=KMeans(n_clusters=k,random_state=42,n_init=10)
                labs=km.fit_predict(coords)
                try:
                    sc=silhouette_score(coords,labs)
                    if sc>best_score: best_score=sc; best_k=k
                except: pass
            km_final=KMeans(n_clusters=best_k,random_state=42,n_init=10)
            elig["Beat_Cluster"]=km_final.fit_predict(coords)
            st.success(f"Optimal k={best_k} | Silhouette Score={best_score:.3f}")

            # DBSCAN outliers
            db=DBSCAN(eps=0.005,min_samples=3).fit(coords)
            elig["Is_Outlier"]=(db.labels_==-1).astype(int)
            n_outliers=elig["Is_Outlier"].sum()

            col1,col2,col3=st.columns(3)
            col1.metric("Optimal Clusters",best_k)
            col2.metric("Silhouette Score",f"{best_score:.3f}")
            col3.metric("Geographic Outliers (DBSCAN)",n_outliers)

            # Map
            center_lat=elig["Lat_clean"].mean(); center_lon=elig["Lon_clean"].mean()
            m=folium.Map(location=[center_lat,center_lon],zoom_start=13)
            palette=["#107A6E","#1A7A45","#D4860B","#C0392B","#6A3FA0","#1B2A4A",
                     "#C49A1A","#2C3E50","#E74C3C","#3498DB","#9B59B6","#1ABC9C",
                     "#F39C12","#27AE60","#E67E22","#2980B9","#8E44AD","#16A085",
                     "#D35400","#7F8C8D","#2ECC71"]
            for _,row in elig.iterrows():
                col=palette[int(row["Beat_Cluster"])%len(palette)]
                if row["Is_Outlier"]==1: col="#FF0000"
                folium.CircleMarker(
                    location=[row["Lat_clean"],row["Lon_clean"]],radius=5,
                    color=col,fill=True,fill_opacity=0.8,
                    popup=f"{row['RetailerURCname']} | {row['SalesmanCode']} | Score:{row['priority_score']}"
                ).add_to(m)
            st_folium(m,height=450,width=700)
            st.session_state["elig_clustered"]=elig
            st.session_state["n_clusters"]=best_k
        else:
            st.info("Set the number of clusters and click 'Run Geo-Clustering'.")

    # ── Tab 3: Beat Sequencing ────────────────────────────────
    with tabs3[2]:
        st.markdown("**Within-Beat Sequencing — Nearest Neighbour + 2-Opt**")
        if "elig_clustered" not in st.session_state:
            st.warning("Run Geo-Clustering first."); st.stop()
        elig_c = st.session_state["elig_clustered"]

        # Distributor depot (approx centroid of all outlets)
        depot_lat = elig_c["Lat_clean"].mean()
        depot_lon = elig_c["Lon_clean"].mean()
        cluster_id = st.selectbox("Select Beat Cluster to Sequence",
                                   sorted(elig_c["Beat_Cluster"].unique()))
        cluster_outlets = elig_c[elig_c["Beat_Cluster"]==cluster_id].copy()
        st.markdown(f"Outlets in this cluster: **{len(cluster_outlets)}**")

        if len(cluster_outlets)>=2:
            points = [[depot_lat,depot_lon]] + cluster_outlets[["Lat_clean","Lon_clean"]].values.tolist()
            route_idx = two_opt(points)
            route_outlets = [cluster_outlets.iloc[i-1] for i in route_idx if i>0]

            seq_data=[]
            total_dist=0
            prev_lat,prev_lon=depot_lat,depot_lon
            for i,row in enumerate(route_outlets,1):
                dist=haversine(prev_lat,prev_lon,row["Lat_clean"],row["Lon_clean"])
                total_dist+=dist
                seq_data.append({
                    "Sequence":i,"Outlet":row["RetailerURCname"],
                    "Category":row["RetailerCategory"],"DSR":row["SalesmanCode"],
                    "Priority_Score":row["priority_score"],
                    "Lat":row["Lat_clean"],"Lon":row["Lon_clean"],
                    "Dist_from_Prev_km":round(dist,2),
                    "Cumulative_km":round(total_dist,2)
                })
                prev_lat,prev_lon=row["Lat_clean"],row["Lon_clean"]

            seq_df=pd.DataFrame(seq_data)
            col1,col2=st.columns(2)
            col1.metric("Total Route Distance",f"{total_dist:.1f} km")
            col2.metric("Avg Distance per Stop",f"{total_dist/len(seq_data):.2f} km")
            st.dataframe(seq_df,hide_index=True,use_container_width=True)

            # Map sequence
            m2=folium.Map(location=[depot_lat,depot_lon],zoom_start=14)
            folium.Marker([depot_lat,depot_lon],popup="DEPOT",
                          icon=folium.Icon(color="red",icon="home")).add_to(m2)
            route_coords=[[depot_lat,depot_lon]]
            for i,row in seq_df.iterrows():
                folium.CircleMarker(
                    [row["Lat"],row["Lon"]],radius=8,
                    color=COLORS["teal"],fill=True,fill_opacity=0.9,
                    popup=f"{row['Sequence']}. {row['Outlet']}"
                ).add_to(m2)
                folium.Marker([row["Lat"],row["Lon"]],
                    icon=folium.DivIcon(html=f'<div style="font-size:10px;font-weight:bold;color:#0D1B2A">{row["Sequence"]}</div>')
                ).add_to(m2)
                route_coords.append([row["Lat"],row["Lon"]])
            route_coords.append([depot_lat,depot_lon])
            folium.PolyLine(route_coords,color=COLORS["teal"],weight=2,opacity=0.7).add_to(m2)
            st_folium(m2,height=400,width=700)
            st.session_state["seq_df"]=seq_df

    # ── Tab 4: Beat Plan ──────────────────────────────────────
    with tabs3[3]:
        st.markdown("**Full Revised Beat Plan — All Clusters**")
        if "elig_clustered" not in st.session_state:
            st.warning("Run Geo-Clustering first."); st.stop()
        elig_c = st.session_state["elig_clustered"]

        if st.button("Generate Full Beat Plan (All Clusters)"):
            all_seq=[]
            for cid in sorted(elig_c["Beat_Cluster"].unique()):
                co=elig_c[elig_c["Beat_Cluster"]==cid].copy()
                if len(co)<2:
                    for _,row in co.iterrows():
                        all_seq.append({"Beat_Cluster":cid,"Sequence":1,
                            "DSR":row["SalesmanCode"],"Outlet_ID":row["UniqueRetailerCode"],
                            "Outlet":row["RetailerURCname"],"Category":row["RetailerCategory"],
                            "Priority_Score":row["priority_score"],"Is_Outlier":row["Is_Outlier"],
                            "Lat":row["Lat_clean"],"Lon":row["Lon_clean"],"Route_km":0})
                    continue
                points=[[depot_lat,depot_lon]]+co[["Lat_clean","Lon_clean"]].values.tolist()
                ridx=two_opt(points)
                prev_lat2,prev_lon2=depot_lat,depot_lon; dist2=0
                seq_rows=[co.iloc[i-1] for i in ridx if i>0]
                for seq_i,row in enumerate(seq_rows,1):
                    d=haversine(prev_lat2,prev_lon2,row["Lat_clean"],row["Lon_clean"])
                    dist2+=d
                    all_seq.append({"Beat_Cluster":cid,"Sequence":seq_i,
                        "DSR":row["SalesmanCode"],"Outlet_ID":row["UniqueRetailerCode"],
                        "Outlet":row["RetailerURCname"],"Category":row["RetailerCategory"],
                        "Priority_Score":row["priority_score"],"Is_Outlier":row.get("Is_Outlier",0),
                        "Lat":row["Lat_clean"],"Lon":row["Lon_clean"],"Route_km":round(d,2)})
                    prev_lat2,prev_lon2=row["Lat_clean"],row["Lon_clean"]

            beat_plan=pd.DataFrame(all_seq)
            cluster_summary=(beat_plan.groupby("Beat_Cluster")
                                      .agg(Outlets=("Outlet","count"),
                                           Dominant_DSR=("DSR",lambda x:x.value_counts().index[0]),
                                           Avg_Priority=("Priority_Score","mean"),
                                           Total_km=("Route_km","sum"))
                                      .reset_index())
            st.dataframe(cluster_summary,hide_index=True,use_container_width=True)
            st.markdown(f"**Total outlets in revised beat plan: {len(beat_plan):,}**")
            st.dataframe(beat_plan.head(200),hide_index=True,use_container_width=True)
            st.session_state["beat_plan"]=beat_plan
            st.success("Beat plan generated! Go to Downloads tab to export.")

    # ── Tab 5: Downloads ──────────────────────────────────────
    with tabs3[4]:
        dl_sheets={"Outlet_Scores":outlets}
        if "beat_plan" in st.session_state:
            dl_sheets["Beat_Plan"]=st.session_state["beat_plan"]
        if "seq_df" in st.session_state:
            dl_sheets["Sample_Sequence"]=st.session_state["seq_df"]
        excel_bytes=to_excel(dl_sheets)
        st.download_button("⬇️ Download M3 Outputs", excel_bytes,
                           "M3_Route_Beat_Plan.xlsx","application/vnd.ms-excel")

# ════════════════════════════════════════════════════════════
# PAGE 5 — BACKTESTING
# ════════════════════════════════════════════════════════════
elif page=="📈 Backtesting":
    st.markdown('<div class="section-hdr">📈 BACKTESTING — Would the models have worked on historical data?</div>',
                unsafe_allow_html=True)

    tabs4 = st.tabs(["🔄 M1: Walk-Forward","💰 M1: Business Impact",
                     "📅 M2: Cadence Validation","🗺️ M3: Coverage Validation"])

    sched_bt = sched[sched["SalesmanCode"].notna()].copy()

    # ── Tab 1: M1 Walk-Forward ────────────────────────────────
    with tabs4[0]:
        st.markdown("**Walk-Forward Backtesting — Train on rolling window, predict next month**")
        st.markdown("For each month from Jan-25 to May-26, train on all data before that month and predict it.")

        months = sorted(sched_bt["BillingMonth"].unique())
        min_train = 3
        results=[]

        if st.button("Run Walk-Forward Backtest (may take 1-2 min)"):
            prog=st.progress(0)
            for i,test_month in enumerate(months[min_train:]):
                train_mask = sched_bt["BillingMonth"]<test_month
                test_mask  = sched_bt["BillingMonth"]==test_month
                X_tr_wf = sched_bt[train_mask][FEATURES].fillna(0)
                y_tr_wf = sched_bt[train_mask]["IsProductive"]
                X_te_wf = sched_bt[test_mask][FEATURES].fillna(0)
                y_te_wf = sched_bt[test_mask]["IsProductive"]
                if len(X_te_wf)<50 or y_te_wf.nunique()<2: continue
                pw=max(1,(y_tr_wf==0).sum()/(y_tr_wf==1).sum() if (y_tr_wf==1).sum()>0 else 1)
                wf_model=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,
                    scale_pos_weight=pw,use_label_encoder=False,eval_metric="logloss",
                    random_state=42,n_jobs=-1,verbosity=0)
                wf_model.fit(X_tr_wf,y_tr_wf,verbose=False)
                prob_wf=wf_model.predict_proba(X_te_wf)[:,1]
                pred_wf=(prob_wf>=threshold_ui).astype(int)
                cm_wf=confusion_matrix(y_te_wf,pred_wf)
                tn_w,fp_w,fn_w,tp_w=cm_wf.ravel() if cm_wf.size==4 else (0,0,0,0)
                results.append({
                    "Month":pd.Timestamp(test_month).strftime("%b-%y"),
                    "Train_Months":i+min_train,
                    "Test_Visits":len(y_te_wf),
                    "F1":round(f1_score(y_te_wf,pred_wf,zero_division=0)*100,1),
                    "AUC_ROC":round(roc_auc_score(y_te_wf,prob_wf)*100,1) if y_te_wf.nunique()>1 else 0,
                    "Precision":round(precision_score(y_te_wf,pred_wf,zero_division=0)*100,1),
                    "Recall":round(recall_score(y_te_wf,pred_wf,zero_division=0)*100,1),
                    "Actual_Prod_Rate":round(y_te_wf.mean()*100,1),
                    "TP":int(tp_w),"FP":int(fp_w),"FN":int(fn_w),"TN":int(tn_w)
                })
                prog.progress((i+1)/(len(months)-min_train))
            st.session_state["wf_results"]=pd.DataFrame(results)
            st.success("Walk-forward backtest complete!")

        if "wf_results" in st.session_state:
            wf=st.session_state["wf_results"]
            col1,col2,col3=st.columns(3)
            col1.metric("Avg F1 (walk-forward)",f"{wf['F1'].mean():.1f}%",
                        f"±{wf['F1'].std():.1f}%")
            col2.metric("Avg AUC-ROC",f"{wf['AUC_ROC'].mean():.1f}%")
            col3.metric("Months Tested",len(wf))

            fig,axes=plt.subplots(2,1,figsize=(10,6))
            axes[0].plot(wf["Month"],wf["F1"],marker="o",color=COLORS["teal"],linewidth=2,label="F1")
            axes[0].plot(wf["Month"],wf["AUC_ROC"],marker="s",color=COLORS["green"],linewidth=2,label="AUC-ROC")
            axes[0].axhline(62,color=COLORS["amber"],linestyle="--",linewidth=1,label="F1 Target (62%)")
            axes[0].set_ylabel("%"); axes[0].set_title("Walk-Forward F1 & AUC-ROC by Month")
            axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
            plt.setp(axes[0].xaxis.get_majorticklabels(),rotation=45,ha="right",fontsize=8)
            axes[1].plot(wf["Month"],wf["Precision"],marker="o",color=COLORS["purple"],linewidth=2,label="Precision")
            axes[1].plot(wf["Month"],wf["Recall"],marker="s",color=COLORS["red"],linewidth=2,label="Recall")
            axes[1].set_ylabel("%"); axes[1].set_title("Walk-Forward Precision & Recall by Month")
            axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
            plt.setp(axes[1].xaxis.get_majorticklabels(),rotation=45,ha="right",fontsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(wf,hide_index=True,use_container_width=True)

            excel_bytes=to_excel({"Walk_Forward_Results":wf})
            st.download_button("⬇️ Download Walk-Forward Results",excel_bytes,
                               "M1_WalkForward_Backtest.xlsx","application/vnd.ms-excel")

    # ── Tab 2: M1 Business Impact ─────────────────────────────
    with tabs4[1]:
        st.markdown("**Business Impact Backtest — If M1 had been live from Jan-26, what GSV would we have captured?**")
        model_res_bt = train_m1(id(df))
        model_bt = model_res_bt[0]

        backtest_period = sched_bt[sched_bt["BillingDate"]>"2025-12-31"].copy()
        if len(backtest_period)>0:
            prob_bt = model_bt.predict_proba(backtest_period[FEATURES].fillna(0))[:,1]
            backtest_period["Model_Prob"] = prob_bt
            backtest_period["Model_Pred"] = (prob_bt>=threshold_ui).astype(int)
            backtest_period["Outcome"] = backtest_period.apply(
                lambda r: "TP" if r["IsProductive"]==1 and r["Model_Pred"]==1
                          else "FN" if r["IsProductive"]==1 and r["Model_Pred"]==0
                          else "FP" if r["IsProductive"]==0 and r["Model_Pred"]==1
                          else "TN", axis=1)

            mo_biz = backtest_period.groupby("BillingMonth").agg(
                Scheduled_Visits=("IsProductive","count"),
                Actual_Productive=("IsProductive","sum"),
                Model_TP=("Outcome",lambda x:(x=="TP").sum()),
                Model_FN=("Outcome",lambda x:(x=="FN").sum()),
                Model_FP=("Outcome",lambda x:(x=="FP").sum()),
                Avg_GSV=("Total_GSV","mean")
            ).reset_index()
            mo_biz["YM"]=mo_biz["BillingMonth"].dt.strftime("%b-%y")
            mo_biz["Actual_Prod_Rate_%"]=(mo_biz["Actual_Productive"]/mo_biz["Scheduled_Visits"]*100).round(1)
            mo_biz["Model_Precision_%"]=(mo_biz["Model_TP"]/(mo_biz["Model_TP"]+mo_biz["Model_FP"]).clip(lower=1)*100).round(1)
            mo_biz["GSV_Captured_Rs"]=(mo_biz["Model_TP"]*mo_biz["Avg_GSV"].fillna(0)).round(0)
            mo_biz["GSV_Missed_Rs"]  =(mo_biz["Model_FN"]*mo_biz["Avg_GSV"].fillna(0)).round(0)
            mo_biz["Wasted_Trips"]   = mo_biz["Model_FP"]

            tot_cap = mo_biz["GSV_Captured_Rs"].sum()
            tot_mis = mo_biz["GSV_Missed_Rs"].sum()
            tot_was = mo_biz["Wasted_Trips"].sum()

            c1,c2,c3=st.columns(3)
            c1.metric("Est. GSV Captured (Jan-May 26)",f"Rs{tot_cap/1e5:.1f}L")
            c2.metric("Est. GSV Missed (False Negatives)",f"Rs{tot_mis/1e5:.1f}L")
            c3.metric("Wasted Trips (False Positives)",f"{int(tot_was):,}")

            fig,ax=plt.subplots(figsize=(10,4))
            x=range(len(mo_biz))
            ax.bar([i-0.2 for i in x],mo_biz["GSV_Captured_Rs"]/1e3,0.4,
                   label="GSV Captured (Rs 000)",color=COLORS["green"],alpha=0.85)
            ax.bar([i+0.2 for i in x],mo_biz["GSV_Missed_Rs"]/1e3,0.4,
                   label="GSV Missed (Rs 000)",color=COLORS["red"],alpha=0.85)
            ax.set_xticks(list(x)); ax.set_xticklabels(mo_biz["YM"],rotation=45,ha="right")
            ax.set_ylabel("Rs 000"); ax.set_title("GSV Captured vs Missed by Month (Jan-May 26)")
            ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(mo_biz[["YM","Scheduled_Visits","Actual_Productive","Model_TP","Model_FN",
                                   "Model_FP","GSV_Captured_Rs","GSV_Missed_Rs","Wasted_Trips",
                                   "Actual_Prod_Rate_%","Model_Precision_%"]],
                         hide_index=True,use_container_width=True)

            excel_bytes=to_excel({"Business_Impact_Backtest":mo_biz,
                                   "All_Scored_Jan_May26":backtest_period[
                                       ["BillingDate","SalesmanCode","UniqueRetailerCode",
                                        "RetailerCategory","Total_GSV","IsProductive",
                                        "Model_Prob","Model_Pred","Outcome"]]})
            st.download_button("⬇️ Download Business Impact", excel_bytes,
                               "M1_Business_Impact_Backtest.xlsx","application/vnd.ms-excel")

    # ── Tab 3: M2 Cadence Validation ─────────────────────────
    with tabs4[2]:
        st.markdown("**Cadence Validation — Do M2-aligned outlets actually perform better?**")
        sched_bt2 = sched_bt.copy()
        sched_bt2["IPI_Bucket"]=pd.cut(sched_bt2["schedule_ipi"].fillna(7),
            bins=[-1,3,7,14,21,30,60,9999],labels=["0-3d","4-7d","8-14d","15-21d","22-30d","31-60d","60d+"])

        st.markdown("**Productivity rate by IPI bucket — does recommended cadence (8-14d) outperform default (4-7d)?**")
        ipi_val=(sched_bt2.groupby(["IPI_Bucket","BillingMonth"],observed=True)
                          .agg(Visits=("IsProductive","size"),Prod=("IsProductive","sum"))
                          .reset_index())
        ipi_val["Prod_Rate"]=(ipi_val["Prod"]/ipi_val["Visits"]*100).round(1)
        ipi_val["YM"]=ipi_val["BillingMonth"].dt.strftime("%b-%y")
        ipi_pivot=ipi_val.pivot_table(index="YM",columns="IPI_Bucket",values="Prod_Rate",aggfunc="mean")
        fig,ax=plt.subplots(figsize=(12,4))
        for col in ["4-7d","8-14d","15-21d"]:
            if col in ipi_pivot.columns:
                ax.plot(ipi_pivot.index,ipi_pivot[col],marker="o",linewidth=2,label=col)
        ax.set_ylabel("Productivity Rate %"); ax.set_title("Prod Rate by IPI Bucket Over Time")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.xticks(rotation=45,ha="right",fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**Anchor week validation — do clockwork outlets convert better in their anchor week?**")
        wk_val=(sched_bt2.groupby("WeekOfMonth")
                         .agg(Visits=("IsProductive","size"),Prod=("IsProductive","sum"))
                         .reset_index())
        wk_val["Prod_Rate"]=(wk_val["Prod"]/wk_val["Visits"]*100).round(1)
        wk_val["Week"]=wk_val["WeekOfMonth"].map({1:"W1",2:"W2",3:"W3",4:"W4"})
        col1,col2=st.columns(2)
        with col1:
            fig2,ax2=plt.subplots(figsize=(5,3))
            ax2.bar(wk_val["Week"],wk_val["Prod_Rate"],color=COLORS["teal"],alpha=0.85)
            ax2.set_ylabel("Productivity Rate %"); ax2.set_title("Prod Rate by Week of Month")
            ax2.grid(axis="y",alpha=0.3); fig2.tight_layout(); st.pyplot(fig2); plt.close()
        with col2:
            st.dataframe(wk_val[["Week","Visits","Prod_Rate"]],hide_index=True,use_container_width=True)
            st.markdown('<div class="insight">W2 consistently outperforms W4 — validates anchor week assignment logic.</div>',
                        unsafe_allow_html=True)

    # ── Tab 4: M3 Coverage Validation ────────────────────────
    with tabs4[3]:
        st.markdown("**M3 Coverage Validation — Would M3 outlet selection have captured actual GSV?**")
        if "outlets_scored" not in st.session_state:
            st.warning("Run M3 Outlet Scoring first."); st.stop()
        scored_outlets=st.session_state["outlets_scored"]
        elig_outlets=scored_outlets[scored_outlets["eligible"]==1]["UniqueRetailerCode"].tolist()
        removed_outlets=scored_outlets[scored_outlets["eligible"]==0]["UniqueRetailerCode"].tolist()

        recent=sched_bt[sched_bt["BillingDate"]>"2025-09-30"].copy()
        total_gsv=recent["Total_GSV"].fillna(0).sum()
        elig_gsv=recent[recent["UniqueRetailerCode"].isin(elig_outlets)]["Total_GSV"].fillna(0).sum()
        removed_gsv=recent[recent["UniqueRetailerCode"].isin(removed_outlets)]["Total_GSV"].fillna(0).sum()

        c1,c2,c3=st.columns(3)
        c1.metric("Total GSV (Oct25-May26)",f"Rs{total_gsv/1e5:.1f}L")
        c2.metric("GSV from Eligible Outlets",f"Rs{elig_gsv/1e5:.1f}L",
                  f"{elig_gsv/max(total_gsv,1)*100:.1f}% of total")
        c3.metric("GSV from Flagged-for-Removal",f"Rs{removed_gsv/1e5:.1f}L",
                  f"{removed_gsv/max(total_gsv,1)*100:.1f}% of total")

        st.markdown("**Off-beat outlets that generated revenue after Oct-25 (should be added to beat):**")
        ob_val=(df[df["Is_OffBeat_Visit"]==1]
                  [df["BillingDate"]>"2025-09-30"]
                  .groupby("UniqueRetailerCode")
                  .agg(OffBeat_Txns=("Total_GSV","count"),OffBeat_GSV=("Total_GSV","sum"))
                  .reset_index()
                  .merge(scored_outlets[["UniqueRetailerCode","RetailerURCname","RetailerCategory",
                                          "SalesmanCode","eligible"]],
                         on="UniqueRetailerCode",how="left"))
        ob_not_sched=ob_val[ob_val["eligible"]==1].sort_values("OffBeat_GSV",ascending=False)
        st.markdown(f"**{len(ob_not_sched)} off-beat active outlets** generating Rs{ob_not_sched['OffBeat_GSV'].sum()/1e5:.1f}L "
                    f"off-beat but not fully scheduled — M3 recommends adding to beat plan")
        st.dataframe(ob_not_sched.head(30),hide_index=True,use_container_width=True)

        excel_bytes=to_excel({
            "M3_Coverage_Validation":ob_val,
            "Removed_Outlet_GSV":recent[recent["UniqueRetailerCode"].isin(removed_outlets)]
                                        .groupby("UniqueRetailerCode")
                                        .agg(Total_GSV=("Total_GSV","sum"),Visits=("IsProductive","size"))
                                        .reset_index()
        })
        st.download_button("⬇️ Download M3 Validation",excel_bytes,
                           "M3_Coverage_Validation.xlsx","application/vnd.ms-excel")