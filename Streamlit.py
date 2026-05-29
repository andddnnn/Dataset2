"""
BajetIn — Dashboard Interaktif
Jalankan: streamlit run Streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── sklearn optional (untuk confusion matrix) ──────────────────────────────
try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BajetIn - Dashboard Interaktif",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ─────────────────────────────────────────────────────────────────
def fmt_rupiah(x, pos=None):
    if x >= 1_000_000:
        return f"Rp {x/1_000_000:.1f}M"
    elif x >= 1_000:
        return f"Rp {x/1_000:.0f}K"
    return f"Rp {x:.0f}"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"

STATUS_COLORS = {"Rentan": "#C44E52", "Cukup": "#FFC107", "Sehat": "#55A868"}
BUDGET_COLORS = {"safe": "#55A868", "warning": "#FFC107", "over": "#C44E52"}

# ── Sidebar — upload data ───────────────────────────────────────────────────
st.sidebar.title("📂 Unggah Dataset")
st.sidebar.markdown(
    "Unggah fail yang diperlukan."
)

DATASETS = {
    "users":                    "users.csv",
    "transactions":             "transactions.csv",
    "budgets":                  "budgets.csv",
    "financial_assessments":    "financial_assessments.csv",
    "chat_messages":            "chat_messages.csv",
    "literacy_articles":        "literacy_articles.csv",
    "user_monthly_summary":     "user_monthly_summary.csv",
    "category_spending_summary":"category_spending_summary.csv",
    "nlp_training_dataset":     "nlp_training_dataset.csv",
}

DATE_COLS = {
    "users":                 ["createdAt"],
    "transactions":          ["transactionDate", "createdAt"],
    "financial_assessments": ["createdAt"],
    "chat_messages":         ["createdAt"],
}

uploaded = {}
for key, fname in DATASETS.items():
    f = st.sidebar.file_uploader(fname, type="csv", key=key)
    if f is not None:
        parse = DATE_COLS.get(key, [])
        try:
            uploaded[key] = pd.read_csv(f, parse_dates=parse)
        except Exception as e:
            st.sidebar.error(f"Gagal membaca {fname}: {e}")

# ── Helper to check data availability ──────────────────────────────────────
def need(*keys):
    missing = [k for k in keys if k not in uploaded]
    if missing:
        st.info(f"⬆ Upload terlebih dahulu: **{', '.join(missing)}**")
        return False
    return True

# ── MAIN ────────────────────────────────────────────────────────────────────
st.title("BajetIn — Dashboard Interaktif")
st.caption("Dashboard interaktif data pendukung pembuatan proyek BajetIn.")

# ── Overview tabel ──────────────────────────────────────────────────────────
if uploaded:
    rows = [(k, f"{len(v):,}", v.shape[1]) for k, v in uploaded.items()]
    st.subheader("Dataset yang Diupload")
    st.dataframe(
        pd.DataFrame(rows, columns=["Dataset", "Rows", "Cols"]),
        use_container_width=True, hide_index=True,
    )
    st.divider()
else:
    st.warning("Belum ada dataset yang diupload. Silakan upload file CSV di sidebar kiri.")
    st.stop()

# Navigasi tab utama
tabs = st.tabs([
    "1 · Users",
    "2 · Transactions",
    "3 · Budgets",
    "4 · Assessments",
    "5 · Chat",
    "6 · Articles",
    "7 · Monthly Summary",
    "8 · Cat. Spending",
    "9 · NLP",
    "10 · Cross-Dataset",
])

# ═══════════════════════════════════════════════════════════════════════════
# 1. USERS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("1. Users")
    if not need("users"):
        st.stop()
    users = uploaded["users"]

    with st.expander("Gambaran Umum", expanded=False):
        st.dataframe(users.head(), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Users", f"{len(users):,}")
        c2.metric("Kolom", users.shape[1])
        c3.metric("Duplikat ID", users["id"].duplicated().sum() if "id" in users.columns else "N/A")
        st.write("**Missing values:**")
        st.dataframe(users.isnull().sum().rename("missing").reset_index(), use_container_width=True)

    # 1.2 Role
    if "role" in users.columns:
        st.subheader("1.2 Distribusi Role")
        role_counts = users["role"].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(role_counts.index, role_counts.values, color=["#4C72B0","#DD8452"])
        axes[0].set_title("Jumlah User per Role"); axes[0].set_ylabel("Jumlah")
        for i, v in enumerate(role_counts.values):
            axes[0].text(i, v + 0.5, str(v), ha="center", fontsize=11)
        axes[1].pie(role_counts.values, labels=role_counts.index, autopct="%1.1f%%",
                    startangle=90, colors=["#4C72B0","#DD8452"])
        axes[1].set_title("Proporsi Role")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 1.3 Income
    if "monthlyIncome" in users.columns:
        st.subheader("1.3 Distribusi Monthly Income")
        col_bins = st.slider("Jumlah bins histogram", 5, 60, 25, key="inc_bins")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(users["monthlyIncome"], bins=col_bins, color="#4C72B0", edgecolor="white")
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_rupiah))
        axes[0].set_title("Distribusi Monthly Income"); axes[0].set_xlabel("Monthly Income")
        axes[0].set_ylabel("Jumlah User")
        if "role" in users.columns:
            roles = users["role"].unique()
            data_by_role = [users[users["role"] == r]["monthlyIncome"].values for r in roles]
            axes[1].boxplot(data_by_role, labels=roles, patch_artist=True,
                            boxprops=dict(facecolor="#AEC6CF"))
            axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_rupiah))
            axes[1].set_title("Monthly Income per Role"); axes[1].set_ylabel("Monthly Income")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.write(users["monthlyIncome"].describe().apply(lambda x: f"Rp {x:,.0f}"))

    # 1.4 Registrasi
    if "createdAt" in users.columns:
        st.subheader("1.4 Tren Registrasi User")
        users["reg_month"] = users["createdAt"].dt.to_period("M")
        reg_trend = users.groupby("reg_month").size()
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(reg_trend.index.astype(str), reg_trend.values,
                marker="o", linewidth=2, color="#4C72B0")
        ax.fill_between(reg_trend.index.astype(str), reg_trend.values, alpha=0.15, color="#4C72B0")
        ax.set_title("Tren Registrasi User per Bulan"); ax.set_xlabel("Bulan")
        ax.set_ylabel("Jumlah Registrasi"); plt.xticks(rotation=45)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# 2. TRANSACTIONS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("2. Transactions")
    if not need("transactions"):
        st.stop()
    tx = uploaded["transactions"]

    with st.expander("Gambaran Umum", expanded=False):
        st.dataframe(tx.head(), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Total Transaksi", f"{len(tx):,}")
        c2.metric("Kolom", tx.shape[1])
        st.write("**Missing values:**")
        st.dataframe(tx.isnull().sum().rename("missing").reset_index(), use_container_width=True)

    # Filter sidebar mini
    st.subheader("Filter")
    col_f1, col_f2 = st.columns(2)
    if "type" in tx.columns:
        type_opts = ["Semua"] + sorted(tx["type"].unique().tolist())
        sel_type = col_f1.selectbox("Type", type_opts, key="tx_type")
    else:
        sel_type = "Semua"
    if "category" in tx.columns:
        cat_opts = ["Semua"] + sorted(tx["category"].dropna().unique().tolist())
        sel_cat = col_f2.selectbox("Kategori", cat_opts, key="tx_cat")
    else:
        sel_cat = "Semua"

    tx_f = tx.copy()
    if sel_type != "Semua":
        tx_f = tx_f[tx_f["type"] == sel_type]
    if sel_cat != "Semua":
        tx_f = tx_f[tx_f["category"] == sel_cat]
    st.caption(f"Menampilkan {len(tx_f):,} dari {len(tx):,} transaksi")

    # 2.2 Type & Source
    if "type" in tx_f.columns and "source" in tx_f.columns:
        st.subheader("2.2 Distribusi Type & Source")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        type_counts = tx_f["type"].value_counts()
        axes[0].bar(type_counts.index, type_counts.values, color=["#4C72B0","#DD8452"])
        axes[0].set_title("Income vs Expense"); axes[0].set_ylabel("Jumlah Transaksi")
        for i, v in enumerate(type_counts.values):
            axes[0].text(i, v + 50, f"{v:,}", ha="center")
        source_counts = tx_f["source"].value_counts()
        axes[1].bar(source_counts.index, source_counts.values, color=["#55A868","#C44E52"])
        axes[1].set_title("Source Input Transaksi"); axes[1].set_ylabel("Jumlah Transaksi")
        for i, v in enumerate(source_counts.values):
            axes[1].text(i, v + 50, f"{v:,}", ha="center")
        plt.suptitle("Distribusi Type & Source", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 2.3 Amount
    if "amount" in tx_f.columns and "type" in tx_f.columns:
        st.subheader("2.3 Distribusi Amount")
        bins_amt = st.slider("Jumlah bins", 5, 80, 40, key="amt_bins")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, t, color in zip(axes, ["expense","income"], ["#DD8452","#4C72B0"]):
            data = tx_f[tx_f["type"] == t]["amount"] if "type" in tx_f.columns else tx_f["amount"]
            ax.hist(data, bins=bins_amt, color=color, edgecolor="white")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_rupiah))
            ax.set_title(f"Distribusi Amount — {t.capitalize()}")
        plt.suptitle("Distribusi Amount per Type", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 2.4 Per Kategori
    if "category" in tx_f.columns and "amount" in tx_f.columns and "type" in tx_f.columns:
        st.subheader("2.4 Pengeluaran per Kategori")
        cat_total = (tx_f[tx_f["type"] == "expense"]
                     .groupby("category")["amount"].sum().sort_values(ascending=True))
        cat_count = (tx_f[tx_f["type"] == "expense"]
                     .groupby("category").size().sort_values(ascending=True))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        cat_total.plot(kind="barh", ax=axes[0], color="#4C72B0")
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_rupiah))
        axes[0].set_title("Total Pengeluaran per Kategori")
        cat_count.plot(kind="barh", ax=axes[1], color="#DD8452")
        axes[1].set_title("Jumlah Transaksi per Kategori")
        plt.suptitle("Expense per Kategori", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 2.5 Tren bulanan
    if "transactionDate" in tx_f.columns and "type" in tx_f.columns and "amount" in tx_f.columns:
        st.subheader("2.5 Tren Transaksi per Bulan")
        tx_f["month_period"] = tx_f["transactionDate"].dt.to_period("M")
        monthly_tx = tx_f.groupby(["month_period","type"])["amount"].sum().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 4))
        monthly_tx.plot(ax=ax, marker="o", linewidth=2)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_rupiah))
        ax.set_title("Total Income vs Expense per Bulan")
        plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 2.6 Confidence NLP
    if "source" in tx_f.columns and "confidence" in tx_f.columns:
        st.subheader("2.6 Distribusi Confidence (NLP)")
        nlp_tx = tx_f[tx_f["source"] == "nlp"]
        if len(nlp_tx) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(nlp_tx["confidence"].dropna(), bins=30, color="#55A868", edgecolor="white")
            axes[0].set_title("Distribusi Confidence Score (NLP)")
            conf_cat = nlp_tx.groupby("category")["confidence"].mean().sort_values()
            conf_cat.plot(kind="barh", ax=axes[1], color="#55A868")
            mean_conf = nlp_tx["confidence"].mean()
            axes[1].axvline(mean_conf, color="red", linestyle="--",
                            label=f"Overall mean: {mean_conf:.2f}")
            axes[1].legend(); axes[1].set_title("Rata-rata Confidence per Kategori")
            plt.suptitle("Confidence Score Transaksi NLP", fontsize=13, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 2.7 Per user
    if "userId" in tx_f.columns:
        st.subheader("2.7 Jumlah Transaksi per User")
        tx_per_user = tx_f.groupby("userId").size()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(tx_per_user, bins=30, color="#4C72B0", edgecolor="white")
        ax.axvline(tx_per_user.mean(), color="red", linestyle="--",
                   label=f"Mean: {tx_per_user.mean():.0f}")
        ax.set_title("Distribusi Jumlah Transaksi per User")
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# 3. BUDGETS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("3. Budgets")
    if not need("budgets"):
        st.stop()
    budgets = uploaded["budgets"]

    with st.expander("Gambaran Umum", expanded=False):
        st.dataframe(budgets.head(), use_container_width=True)
        st.dataframe(budgets.isnull().sum().rename("missing").reset_index(), use_container_width=True)

    # 3.2 Status
    if "budgetStatus" in budgets.columns:
        st.subheader("3.2 Distribusi Budget Status")
        status_counts = budgets["budgetStatus"].value_counts()
        bar_colors = [BUDGET_COLORS.get(s, "#999") for s in status_counts.index]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(status_counts.index, status_counts.values, color=bar_colors)
        axes[0].set_title("Distribusi Budget Status"); axes[0].set_ylabel("Jumlah")
        for i, v in enumerate(status_counts.values):
            axes[0].text(i, v + 10, f"{v:,}", ha="center")
        axes[1].pie(status_counts.values, labels=status_counts.index,
                    autopct="%1.1f%%", colors=bar_colors, startangle=90)
        axes[1].set_title("Proporsi Budget Status")
        plt.suptitle("Budget Status Overview", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 3.3 Per kategori
    if "category" in budgets.columns and "budgetStatus" in budgets.columns:
        st.subheader("3.3 Budget Status per Kategori")
        pivot = (budgets.groupby(["category","budgetStatus"]).size().unstack(fill_value=0))
        pivot = pivot.reindex(columns=["safe","warning","over"], fill_value=0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(12, 5))
        pivot_pct.plot(kind="bar", ax=ax, stacked=True,
                       color=["#55A868","#FFC107","#C44E52"], edgecolor="white")
        ax.set_title("Proporsi Budget Status per Kategori")
        ax.legend(title="Status", bbox_to_anchor=(1.01, 1))
        plt.xticks(rotation=30, ha="right"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 3.4 Usage %
    if "usagePercent" in budgets.columns:
        st.subheader("3.4 Distribusi Usage Percent")
        clip_val = st.slider("Clip upper (%)", 100, 300, 200, key="usage_clip")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(budgets["usagePercent"].clip(upper=clip_val), bins=40,
                     color="#4C72B0", edgecolor="white")
        axes[0].axvline(80, color="orange", linestyle="--", label="Warning (80%)")
        axes[0].axvline(100, color="red", linestyle="--", label="Over (100%)")
        axes[0].set_title(f"Distribusi Usage % (clipped {clip_val}%)")
        axes[0].legend()
        if "category" in budgets.columns:
            usage_cat = budgets.groupby("category")["usagePercent"].mean().sort_values()
            usage_cat.plot(kind="barh", ax=axes[1], color="#4C72B0")
            axes[1].axvline(80, color="orange", linestyle="--", label="Warning")
            axes[1].axvline(100, color="red", linestyle="--", label="Over")
            axes[1].legend(); axes[1].set_title("Rata-rata Usage % per Kategori")
        plt.suptitle("Budget Usage Percent", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 3.5 Limit vs Spent
    if all(c in budgets.columns for c in ["category","limitAmount","spentAmount"]):
        st.subheader("3.5 Limit vs Spent per Kategori")
        cat_budget = budgets.groupby("category")[["limitAmount","spentAmount"]].mean()
        x = range(len(cat_budget)); width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([i - width/2 for i in x], cat_budget["limitAmount"], width, label="Limit", color="#4C72B0")
        ax.bar([i + width/2 for i in x], cat_budget["spentAmount"], width, label="Spent", color="#DD8452")
        ax.set_xticks(list(x)); ax.set_xticklabels(cat_budget.index, rotation=30, ha="right")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_rupiah))
        ax.set_title("Rata-rata Limit vs Spent per Kategori"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# 4. FINANCIAL ASSESSMENTS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("4. Financial Assessments")
    if not need("financial_assessments"):
        st.stop()
    assess = uploaded["financial_assessments"]

    with st.expander("Gambaran Umum", expanded=False):
        st.dataframe(assess.head(), use_container_width=True)
        if "score" in assess.columns:
            st.write(assess[["score"]].describe().T)

    # 4.2 Score & Status
    if "score" in assess.columns and "status" in assess.columns:
        st.subheader("4.2 Distribusi Score & Status")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(assess["score"], bins=25, color="#4C72B0", edgecolor="white")
        axes[0].axvline(40, color="red", linestyle="--", label="Rentan/Cukup (40)")
        axes[0].axvline(70, color="orange", linestyle="--", label="Cukup/Sehat (70)")
        axes[0].set_title("Distribusi Health Score"); axes[0].legend()
        status_counts = assess["status"].value_counts()
        bar_colors = [STATUS_COLORS.get(s, "#999") for s in status_counts.index]
        axes[1].bar(status_counts.index, status_counts.values, color=bar_colors)
        axes[1].set_title("Jumlah Assessments per Status")
        for i, v in enumerate(status_counts.values):
            axes[1].text(i, v + 2, f"{v:,}", ha="center")
        plt.suptitle("Distribusi Score & Status", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 4.3 Tren Score
    if "createdAt" in assess.columns and "score" in assess.columns:
        st.subheader("4.3 Tren Score per Bulan")
        assess["month_period"] = assess["createdAt"].dt.to_period("M")
        score_trend = assess.groupby("month_period")["score"].agg(["mean","median"])
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(score_trend.index.astype(str), score_trend["mean"],
                marker="o", label="Mean Score", linewidth=2, color="#4C72B0")
        ax.plot(score_trend.index.astype(str), score_trend["median"],
                marker="s", label="Median Score", linewidth=2, linestyle="--", color="#DD8452")
        ax.axhline(40, color="red", linestyle=":", alpha=0.6, label="Batas Rentan")
        ax.axhline(70, color="orange", linestyle=":", alpha=0.6, label="Batas Sehat")
        ax.set_title("Tren Health Score per Bulan"); ax.legend()
        plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 4.4 Per user
    if "userId" in assess.columns:
        st.subheader("4.4 Distribusi Assessment per User")
        assess_per_user = assess.groupby("userId").size()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(assess_per_user, bins=range(1, assess_per_user.max() + 2),
                color="#4C72B0", edgecolor="white", align="left")
        ax.set_title("Distribusi Jumlah Assessment per User")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# 5. CHAT MESSAGES
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("5. Chat Messages")
    if not need("chat_messages"):
        st.stop()
    chat = uploaded["chat_messages"]

    with st.expander("Gambaran Umum", expanded=False):
        st.dataframe(chat.head(), use_container_width=True)
        if "role" in chat.columns:
            st.write("Distribusi role:", chat["role"].value_counts().to_dict())

    # 5.2 Pesan terpopuler
    if "role" in chat.columns and "message" in chat.columns:
        st.subheader("5.2 Pesan Terpopuler")
        top_n = st.slider("Top N pesan", 5, 20, 10, key="top_msg")
        top_user_msg = (chat[chat["role"] == "user"]["message"]
                        .value_counts().head(top_n))
        fig, ax = plt.subplots(figsize=(11, max(4, top_n * 0.4)))
        top_user_msg.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title(f"Top {top_n} Pertanyaan User ke AI Coach")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 5.3 Aktivitas per bulan
    if "createdAt" in chat.columns and "role" in chat.columns:
        st.subheader("5.3 Aktivitas Chat per Bulan")
        chat["month_period"] = chat["createdAt"].dt.to_period("M")
        chat_trend = chat.groupby(["month_period","role"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(11, 4))
        chat_trend.plot(kind="bar", ax=ax, color=["#4C72B0","#DD8452"], edgecolor="white")
        ax.set_title("Jumlah Pesan per Bulan (User vs Assistant)")
        ax.legend(title="Role"); plt.xticks(rotation=45)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 5.4 Session per user
    if "role" in chat.columns and "userId" in chat.columns:
        st.subheader("5.4 Session per User")
        sessions_per_user = chat[chat["role"] == "user"].groupby("userId").size()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(sessions_per_user, bins=20, color="#55A868", edgecolor="white")
        ax.axvline(sessions_per_user.mean(), color="red", linestyle="--",
                   label=f"Mean: {sessions_per_user.mean():.1f}")
        ax.set_title("Distribusi Jumlah Session Chat per User"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# 6. LITERACY ARTICLES
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("6. Literacy Articles")
    if not need("literacy_articles"):
        st.stop()
    articles = uploaded["literacy_articles"]

    with st.expander("Gambaran Umum", expanded=False):
        st.dataframe(articles.head(), use_container_width=True)
        st.write("Missing values:", articles.isnull().sum().to_dict())

    if "category" in articles.columns and "status" in articles.columns:
        st.subheader("6.2 Distribusi Artikel per Kategori & Status")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        articles["category"].value_counts().sort_values().plot(kind="barh", ax=axes[0], color="#4C72B0")
        axes[0].set_title("Jumlah Artikel per Kategori")
        status_counts = articles["status"].value_counts()
        axes[1].pie(status_counts.values, labels=status_counts.index,
                    autopct="%1.1f%%", colors=["#55A868","#AEC6CF"], startangle=90)
        axes[1].set_title("Status Artikel")
        plt.suptitle("Distribusi Literacy Articles", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.subheader("6.3 Coverage: Kategori Transaksi vs Artikel")
        if "transactions" in uploaded:
            tx2 = uploaded["transactions"]
            tx_cats = set(tx2["category"].dropna().unique()) if "category" in tx2.columns else set()
            art_cats = set(articles[articles["status"] == "published"]["category"].unique())
            missing_cats = tx_cats - art_cats
            if missing_cats:
                st.warning(f"Kategori transaksi **tanpa** artikel published: {', '.join(sorted(missing_cats))}")
            else:
                st.success("Semua kategori transaksi sudah ter-cover oleh artikel published.")
        else:
            st.info("Upload transactions.csv untuk melihat coverage.")

# ═══════════════════════════════════════════════════════════════════════════
# 7. USER MONTHLY SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("7. User Monthly Summary")
    if not need("user_monthly_summary"):
        st.stop()
    monthly = uploaded["user_monthly_summary"]

    with st.expander("Gambaran Umum", expanded=False):
        st.dataframe(monthly.head(), use_container_width=True)
        num_cols = ["totalIncome","totalExpense","netCashflow","expenseIncomeRatio"]
        existing = [c for c in num_cols if c in monthly.columns]
        if existing:
            st.write(monthly[existing].describe().T)

    # 7.2 Net Cashflow
    if "netCashflow" in monthly.columns:
        st.subheader("7.2 Distribusi Net Cashflow")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(monthly["netCashflow"], bins=40, color="#4C72B0", edgecolor="white")
        axes[0].axvline(0, color="red", linestyle="--", label="Break even")
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_rupiah))
        axes[0].set_title("Distribusi Net Cashflow"); axes[0].legend()
        pct_pos = (monthly["netCashflow"] > 0).mean() * 100
        axes[1].pie([pct_pos, 100 - pct_pos],
                    labels=["Positif (surplus)","Negatif (defisit)"],
                    autopct="%1.1f%%", colors=["#55A868","#C44E52"], startangle=90)
        axes[1].set_title("Proporsi User dengan Cashflow Positif")
        plt.suptitle("Net Cashflow Overview", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 7.3 Tren Income & Expense
    if all(c in monthly.columns for c in ["year","month","totalIncome","totalExpense","netCashflow"]):
        st.subheader("7.3 Tren Income & Expense per Bulan")
        trend = monthly.groupby(["year","month"])[["totalIncome","totalExpense","netCashflow"]].sum()
        trend.index = [f"{y}-{m:02d}" for y, m in trend.index]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(trend.index, trend["totalIncome"], marker="o", label="Total Income",
                linewidth=2, color="#4C72B0")
        ax.plot(trend.index, trend["totalExpense"], marker="o", label="Total Expense",
                linewidth=2, color="#DD8452")
        ax.bar(trend.index, trend["netCashflow"], alpha=0.25,
               color=["#55A868" if v >= 0 else "#C44E52" for v in trend["netCashflow"]],
               label="Net Cashflow")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_rupiah))
        ax.set_title("Tren Income, Expense & Net Cashflow"); ax.legend()
        plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 7.4 Expense-Income Ratio
    if "expenseIncomeRatio" in monthly.columns:
        st.subheader("7.4 Expense-Income Ratio")
        clip_r = st.slider("Clip upper ratio", 1.0, 5.0, 2.0, 0.1, key="ei_clip")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ratio_clean = monthly["expenseIncomeRatio"].dropna().clip(upper=clip_r)
        axes[0].hist(ratio_clean, bins=40, color="#DD8452", edgecolor="white")
        axes[0].axvline(1.0, color="red", linestyle="--", label="Break even")
        axes[0].axvline(0.8, color="orange", linestyle="--", label="Target maks (0.8)")
        axes[0].legend(); axes[0].set_title("Distribusi Expense-Income Ratio")
        if all(c in monthly.columns for c in ["year","month"]):
            ratio_monthly = monthly.groupby(["year","month"])["expenseIncomeRatio"].mean()
            ratio_monthly.index = [f"{y}-{m:02d}" for y, m in ratio_monthly.index]
            axes[1].plot(ratio_monthly.index, ratio_monthly.values, marker="o",
                         linewidth=2, color="#DD8452")
            axes[1].axhline(1.0, color="red", linestyle="--", alpha=0.7)
            axes[1].axhline(0.8, color="orange", linestyle="--", alpha=0.7)
            axes[1].set_title("Rata-rata E/I Ratio per Bulan")
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        plt.suptitle("Expense-Income Ratio", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 7.6 Health Score vs Net Cashflow
    if all(c in monthly.columns for c in ["healthScore","netCashflow","healthStatus"]):
        st.subheader("7.6 Health Score vs Net Cashflow")
        fig, ax = plt.subplots(figsize=(9, 5))
        for status, grp in monthly.groupby("healthStatus"):
            ax.scatter(grp["healthScore"], grp["netCashflow"],
                       alpha=0.35, s=15, label=status,
                       color=STATUS_COLORS.get(status, "gray"))
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_rupiah))
        ax.set_title("Health Score vs Net Cashflow"); ax.legend(title="Status")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# 8. CATEGORY SPENDING SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("8. Category Spending Summary")
    if not need("category_spending_summary"):
        st.stop()
    cat_summary = uploaded["category_spending_summary"]

    with st.expander("Gambaran Umum", expanded=False):
        st.dataframe(cat_summary.head(), use_container_width=True)
        num_cs = ["expenseAmount","transactionCount","limitAmount","usagePercent"]
        existing_cs = [c for c in num_cs if c in cat_summary.columns]
        if existing_cs:
            st.write(cat_summary[existing_cs].describe().T)

    # 8.2 Avg Spending
    if "category" in cat_summary.columns and "expenseAmount" in cat_summary.columns:
        st.subheader("8.2 Rata-rata Spending per Kategori")
        avg_spending = cat_summary.groupby("category")["expenseAmount"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        avg_spending.plot(kind="barh", ax=ax, color="#4C72B0")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_rupiah))
        ax.set_title("Rata-rata Monthly Spending per Kategori")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 8.3 Heatmap
    if all(c in cat_summary.columns for c in ["month","category","expenseAmount"]):
        st.subheader("8.3 Heatmap Spending per Bulan")
        heatmap_data = cat_summary.groupby(["month","category"])["expenseAmount"].mean().unstack()
        fig, ax = plt.subplots(figsize=(13, 5))
        sns.heatmap(heatmap_data.T, fmt=".0f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "Avg Expense (IDR)"})
        ax.set_title("Heatmap Rata-rata Spending per Kategori per Bulan")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 8.4 Usage % boxplot
    if "category" in cat_summary.columns and "usagePercent" in cat_summary.columns:
        st.subheader("8.4 Distribusi Usage % per Kategori")
        fig, ax = plt.subplots(figsize=(12, 5))
        cat_summary.boxplot(column="usagePercent", by="category", ax=ax,
                            patch_artist=True,
                            boxprops=dict(facecolor="#AEC6CF", color="#4C72B0"),
                            medianprops=dict(color="red"))
        ax.axhline(80, color="orange", linestyle="--", label="Warning 80%")
        ax.axhline(100, color="red", linestyle="--", label="Over 100%")
        ax.legend(); plt.suptitle(""); plt.xticks(rotation=30, ha="right")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# 9. NLP TRAINING DATASET
# ═══════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.header("9. NLP Training Dataset")
    if not need("nlp_training_dataset"):
        st.stop()
    nlp = uploaded["nlp_training_dataset"]

    with st.expander("Gambaran Umum", expanded=False):
        st.dataframe(nlp.head(), use_container_width=True)
        st.write("Missing values:", nlp.isnull().sum().to_dict())

    # 9.2 Akurasi
    metric_cols = {
        "Category Accuracy": "nlpCategoryCorrect",
        "Type Accuracy":     "nlpTypeCorrect",
        "Overall Correct":   "isCorrect",
        "Needs Correction":  "correctionNeeded",
    }
    existing_m = {k: v for k, v in metric_cols.items() if v in nlp.columns}
    if existing_m:
        st.subheader("9.2 Akurasi Model NLP")
        metrics = {k: nlp[v].mean() for k, v in existing_m.items()}
        c_cols = st.columns(len(metrics))
        for col, (k, v) in zip(c_cols, metrics.items()):
            col.metric(k, f"{v:.1%}")
        fig, ax = plt.subplots(figsize=(9, 4))
        colors_m = ["#55A868","#55A868","#4C72B0","#C44E52"]
        ax.barh(list(metrics.keys()), list(metrics.values()),
                color=colors_m[:len(metrics)])
        ax.set_xlim(0, 1.1)
        for i, (k, v) in enumerate(metrics.items()):
            ax.text(v + 0.01, i, f"{v:.1%}", va="center")
        ax.set_title("Akurasi Model NLP — Overview")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 9.3 Akurasi per kategori
    if "actualCategory" in nlp.columns and "nlpCategoryCorrect" in nlp.columns:
        st.subheader("9.3 Akurasi per Kategori")
        acc_by_cat = nlp.groupby("actualCategory")["nlpCategoryCorrect"].mean().sort_values()
        colors_bar = ["#C44E52" if v < 0.8 else "#55A868" for v in acc_by_cat.values]
        fig, ax = plt.subplots(figsize=(10, 5))
        acc_by_cat.plot(kind="barh", ax=ax, color=colors_bar)
        ax.axvline(0.8, color="orange", linestyle="--", label="Target 80%")
        ax.set_title("Akurasi Deteksi Kategori per Kategori Aktual"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 9.4 Confusion Matrix
    if (SKLEARN_OK and
            "actualCategory" in nlp.columns and
            "detectedCategory" in nlp.columns):
        st.subheader("9.4 Confusion Matrix Kategori")
        cats = sorted(nlp["actualCategory"].unique())
        cm = confusion_matrix(nlp["actualCategory"], nlp["detectedCategory"], labels=cats)
        fig, ax = plt.subplots(figsize=(11, 9))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cats)
        disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
        ax.set_title("Confusion Matrix — Deteksi Kategori NLP")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    elif not SKLEARN_OK:
        st.info("Install scikit-learn untuk menampilkan confusion matrix: `pip install scikit-learn`")

    # 9.5 Confidence vs Akurasi
    if "confidence" in nlp.columns and "nlpCategoryCorrect" in nlp.columns:
        st.subheader("9.5 Confidence vs Akurasi")
        nlp["conf_bucket"] = pd.cut(
            nlp["confidence"],
            bins=[0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            labels=["<0.70","0.70-0.75","0.75-0.80","0.80-0.85","0.85-0.90","0.90-0.95","0.95-1.0"]
        )
        acc_by_conf = nlp.groupby("conf_bucket")["nlpCategoryCorrect"].mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        acc_by_conf.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
        ax.axhline(0.8, color="orange", linestyle="--", label="Target 80%")
        ax.set_title("Akurasi per Bucket Confidence Score"); ax.set_ylim(0, 1.1); ax.legend()
        for i, v in enumerate(acc_by_conf.values):
            ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
        plt.xticks(rotation=30); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 9.6 Label Supervised
    label_cols = ["isNeed","isWant","isRecurring","isImpulse","isOverBudgetRisk"]
    existing_labels = [c for c in label_cols if c in nlp.columns]
    if existing_labels:
        st.subheader("9.6 Distribusi Label Supervised")
        label_rates = nlp[existing_labels].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        label_rates.plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title("Proporsi True per Label Supervised")
        for i, v in enumerate(label_rates.values):
            ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
        ax.set_xlim(0, 1.1); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# 10. CROSS-DATASET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[9]:
    st.header("10. Cross-Dataset Analysis")

    # 10.1 Health Score vs Budget Compliance
    if need("financial_assessments", "budgets"):
        assess2 = uploaded["financial_assessments"]
        budgets2 = uploaded["budgets"]
        if all(c in assess2.columns for c in ["userId","score","status","createdAt"]) and \
           all(c in budgets2.columns for c in ["userId","budgetStatus"]):
            st.subheader("10.1 Health Score vs Budget Compliance")
            latest_assess = (assess2.sort_values("createdAt")
                             .groupby("userId")[["score","status"]].last().reset_index())
            budget_compliance = (budgets2.groupby("userId")["budgetStatus"]
                                 .apply(lambda x: (x == "over").mean())
                                 .reset_index(name="over_rate"))
            merged = latest_assess.merge(budget_compliance, on="userId")
            fig, ax = plt.subplots(figsize=(9, 5))
            for status, grp in merged.groupby("status"):
                ax.scatter(grp["score"], grp["over_rate"],
                           alpha=0.5, s=20, label=status,
                           color=STATUS_COLORS.get(status, "gray"))
            ax.set_title("Health Score vs Budget Over Rate per User")
            ax.set_xlabel("Health Score"); ax.set_ylabel("Proporsi Budget Over (0–1)")
            ax.legend(title="Status"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            corr = merged[["score","over_rate"]].corr().iloc[0, 1]
            st.metric("Korelasi Health Score vs Over Rate", f"{corr:.3f}")

    # 10.2 Spending vs Artikel
    if need("transactions", "literacy_articles"):
        tx3 = uploaded["transactions"]
        art3 = uploaded["literacy_articles"]
        if "type" in tx3.columns and "category" in tx3.columns and "amount" in tx3.columns and \
           "status" in art3.columns and "category" in art3.columns:
            st.subheader("10.2 Spending vs Artikel yang Tersedia")
            top_cats = (tx3[tx3["type"] == "expense"]
                        .groupby("category")["amount"].sum().sort_values(ascending=False))
            art_published = art3[art3["status"] == "published"]["category"].value_counts()
            compare = pd.DataFrame({
                "total_spending": top_cats,
                "artikel_tersedia": art_published
            }).fillna(0).astype({"artikel_tersedia": int})
            compare["spending_rank"] = compare["total_spending"].rank(ascending=False).astype(int)
            compare = compare.sort_values("spending_rank")
            x = range(len(compare))
            fig, ax1 = plt.subplots(figsize=(11, 5))
            ax1.bar(x, compare["total_spending"], color="#4C72B0", label="Total Spending")
            ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_rupiah))
            ax1.set_ylabel("Total Spending (IDR)")
            ax2 = ax1.twinx()
            ax2.plot(x, compare["artikel_tersedia"], marker="o", color="#C44E52",
                     linewidth=2, label="Jumlah Artikel")
            ax2.set_ylabel("Jumlah Artikel Published")
            ax1.set_xticks(list(x)); ax1.set_xticklabels(compare.index, rotation=30, ha="right")
            ax1.set_title("Total Spending vs Ketersediaan Artikel per Kategori")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 10.3 Income vs Health Score
    if need("users", "financial_assessments"):
        users2 = uploaded["users"]
        assess3 = uploaded["financial_assessments"]
        if all(c in users2.columns for c in ["id","monthlyIncome"]) and \
           all(c in assess3.columns for c in ["userId","score","status","createdAt"]):
            st.subheader("10.3 Income Level vs Health Score")
            latest_a = (assess3.sort_values("createdAt")
                        .groupby("userId")[["score","status"]].last().reset_index())
            user_income = users2[["id","monthlyIncome"]].rename(columns={"id":"userId"})
            merged_inc = latest_a.merge(user_income, on="userId")
            merged_inc["income_quartile"] = pd.qcut(
                merged_inc["monthlyIncome"], q=4,
                labels=["Q1 (Rendah)","Q2","Q3","Q4 (Tinggi)"]
            )
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            merged_inc.boxplot(column="score", by="income_quartile", ax=axes[0],
                               patch_artist=True, boxprops=dict(facecolor="#AEC6CF"))
            axes[0].set_title("Health Score per Income Quartile")
            plt.sca(axes[0]); plt.xticks(rotation=20)
            axes[1].scatter(merged_inc["monthlyIncome"], merged_inc["score"],
                            alpha=0.3, s=15, color="#4C72B0")
            axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_rupiah))
            axes[1].set_title("Monthly Income vs Health Score")
            plt.suptitle("Income vs Health Score", fontsize=13, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            corr_inc = merged_inc[["monthlyIncome","score"]].corr().iloc[0, 1]
            st.metric("Korelasi Income vs Health Score", f"{corr_inc:.3f}")

st.divider()
st.caption("BajetIn - Dashboard Interaktif · dibuat dengan Streamlit")