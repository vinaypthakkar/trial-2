"""
ABC Ltd - Customer Retention & Revenue Predictor
A point-and-click tool for managers. No coding needed.

Run locally:   streamlit run app.py
Needs:         abc_ltd_customer_model.sav in the same folder
"""

import io
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Some macOS numpy builds emit harmless matmul warnings.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = Path(__file__).parent
MODEL_PATH = HERE / "abc_ltd_customer_model.sav"
SAMPLE_PATH = HERE / "data" / "sample_sales_ledger.csv"

HIGH_RISK = 0.70
MEDIUM_RISK = 0.40

LABELS = {
    "recency_days": "Days since last order",
    "orders_12m": "Orders in last 12 months",
    "spend_12m": "Spend in last 12 months (£)",
    "tenure_days": "Days since first order",
    "unique_products_12m": "Different products bought (12 months)",
    "return_rate": "Returns as a share of spend",
}

st.set_page_config(
    page_title="ABC Ltd · Customer Retention Predictor",
    page_icon="📉",
    layout="wide",
)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


bundle = load_bundle()
FEATURES = bundle["features"]
HORIZON = bundle["horizon_days"]
LOOKBACK = bundle["lookback_days"]
KEY_ACCOUNT_SPEND = bundle["key_account_spend"]


def money(x):
    return f"£{x:,.0f}"


def risk_band(churn_risk):
    if churn_risk >= HIGH_RISK:
        return "High"
    if churn_risk >= MEDIUM_RISK:
        return "Medium"
    return "Low"


def suggested_action(band, is_key_account):
    if band == "High":
        return "Call this week: key account at risk" if is_key_account else "Send a win-back offer"
    if band == "Medium":
        return "Account manager check-in" if is_key_account else "Add to nurture email campaign"
    return "Business as usual"


def clip_to_training(features):
    """Cap inputs at the range the models were trained on."""
    return features[FEATURES].astype(float).clip(
        lower=pd.Series(bundle["feature_min"]),
        upper=pd.Series(bundle["feature_max"]),
        axis=1,
    )


def score(features):
    """Run both models on a table of customer features."""
    X = clip_to_training(features)
    p_buy = bundle["logistic_model"].predict_proba(X)[:, 1]
    spend = np.clip(bundle["linear_model"].predict(X), 0, None)

    out = features.copy()
    out["churn_risk"] = 1 - p_buy
    out["risk_band"] = out["churn_risk"].map(risk_band)
    out["expected_spend_90d"] = spend
    # What this customer normally brings in per 90 days, weighted by the chance we lose it
    out["revenue_at_risk"] = out["churn_risk"] * out["spend_12m"] * HORIZON / 365
    out["key_account"] = out["spend_12m"] >= KEY_ACCOUNT_SPEND
    out["action"] = [suggested_action(b, k) for b, k in zip(out["risk_band"], out["key_account"])]
    return out


def reasons(row):
    """Plain-English 'why' for one customer, from the logistic regression weights.
    `row` is a one-row DataFrame."""
    pipe = bundle["logistic_model"]
    scaler = pipe.named_steps["scale"]
    coefs = pipe.named_steps["model"].coef_[0]
    medians = bundle["feature_medians"]

    x = np.log1p(clip_to_training(row).iloc[0].to_numpy())
    typical = np.log1p(np.array([medians[f] for f in FEATURES], dtype=float))
    push = coefs * (x - typical) / scaler.scale_  # + means more likely to buy again

    lines = []
    for i in np.argsort(-np.abs(push))[:3]:
        if abs(push[i]) < 0.05:
            continue
        f = FEATURES[i]
        you, med = row[f].iloc[0], medians[f]
        if f == "return_rate":
            you_s, med_s = f"{you:.0%}", f"{med:.0%}"
        elif f == "spend_12m":
            you_s, med_s = money(you), money(med)
        else:
            you_s, med_s = f"{you:,.0f}", f"{med:,.0f}"
        effect = "lowers" if push[i] > 0 else "raises"
        lines.append(f"**{LABELS[f]}** is {you_s} (typical customer: {med_s}), which **{effect}** churn risk.")
    return lines


# --------------------------------------------------------------------------
# Sales ledger -> customer features (same logic as train_model.py)
# --------------------------------------------------------------------------
def guess(columns, candidates):
    norm = {c: c.lower().replace("_", " ").strip() for c in columns}
    for cand in candidates:
        for col, n in norm.items():
            if n == cand:
                return col
    for cand in candidates:
        for col, n in norm.items():
            if cand in n:
                return col
    return None


def to_number(s):
    if s.dtype == object:
        s = s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


@st.cache_data(show_spinner="Reading your file ...")
def read_file(data, name):
    if name.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data), low_memory=False)


@st.cache_data(show_spinner="Building customer profiles from your sales ledger ...")
def ledger_to_features(ledger, as_of):
    as_of = pd.Timestamp(as_of)
    ledger = ledger[ledger["date"] < as_of]
    recent = ledger[ledger["date"] >= as_of - pd.Timedelta(days=LOOKBACK)]
    is_return = recent["amount"] < 0

    sales = recent[~is_return]
    g = sales.groupby("customer")
    feats = pd.DataFrame({
        "recency_days": (as_of - g["date"].max()).dt.days,
        "orders_12m": g["invoice"].nunique(),
        "spend_12m": g["amount"].sum(),
        "unique_products_12m": g["product"].nunique(),
    })
    first = ledger[ledger["amount"] > 0].groupby("customer")["date"].min()
    feats["tenure_days"] = (as_of - first.reindex(feats.index)).dt.days
    returned = recent[is_return].groupby("customer")["amount"].sum().abs()
    feats["return_rate"] = (returned.reindex(feats.index).fillna(0) / feats["spend_12m"]).clip(0, 1)

    lapsed = ledger["customer"].nunique() - len(feats)
    feats = feats[feats["spend_12m"] > 0]
    feats.index.name = "Customer"
    return feats[FEATURES], lapsed


def column_mapper(raw):
    """Let the user confirm which column is which. Returns a clean ledger or None."""
    cols = list(raw.columns)
    none = "— not in my file —"

    def pick(where, label, candidates):
        g = guess(cols, candidates)
        return where.selectbox(label, cols, index=cols.index(g) if g else 0)

    qty_guess = guess(cols, ["quantity", "qty", "units"])
    price_guess = guess(cols, ["price", "unit price", "unitprice", "rate"])
    amt_guess = guess(cols, ["amount", "line total", "sales", "revenue", "value", "net"])

    with st.expander("Step 2 · Check that we matched your columns correctly", expanded=False):
        c1, c2, c3 = st.columns(3)
        cust = pick(c1, "Customer ID", ["customer id", "customerid", "customer", "client", "account"])
        date = pick(c2, "Invoice / order date", ["invoicedate", "invoice date", "order date", "date"])
        inv = pick(c3, "Invoice / order number", ["invoice", "invoiceno", "invoice no", "order id", "order no", "bill no", "receipt"])

        c4, c5, c6 = st.columns(3)
        prod = pick(c4, "Product code", ["stockcode", "stock code", "sku", "product code", "item code", "product", "item"])
        default_mode = 0 if (qty_guess and price_guess) or not amt_guess else 1
        mode = c5.radio("How is the sale value stored?", ["Quantity × Price", "One amount column"], index=default_mode)
        dayfirst = c6.checkbox("Dates are day-first (31/12/2024)", value=False)

        if mode == "Quantity × Price":
            c7, c8 = st.columns(2)
            qty = pick(c7, "Quantity", ["quantity", "qty", "units"])
            price = pick(c8, "Unit price", ["price", "unit price", "unitprice", "rate"])
        else:
            amt = st.selectbox("Line amount (returns should be negative)", [none] + cols,
                               index=(cols.index(amt_guess) + 1) if amt_guess else 0)

    ledger = pd.DataFrame({
        "customer": raw[cust],
        "date": pd.to_datetime(raw[date], errors="coerce", dayfirst=dayfirst),
        "invoice": raw[inv].astype(str),
        "product": raw[prod].astype(str),
    })
    if mode == "Quantity × Price":
        p = to_number(raw[price])
        ledger["amount"] = to_number(raw[qty]) * p
        ledger = ledger[p > 0]
    else:
        if amt == none:
            st.error("Please pick the column that holds the sale amount.")
            return None
        ledger["amount"] = to_number(raw[amt])

    if pd.api.types.is_float_dtype(ledger["customer"]):
        ledger["customer"] = ledger["customer"].astype("Int64")
    ledger = ledger.dropna(subset=["customer", "date", "amount"])
    ledger = ledger[ledger["amount"] != 0].drop_duplicates()
    ledger["customer"] = ledger["customer"].astype(str)

    if ledger.empty:
        st.error("We couldn't read any valid sales rows. Please check the column matching in Step 2.")
        return None
    return ledger


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.title("ABC Ltd · Customer Retention Predictor")
st.markdown(
    f"Find out **which customers are likely to stop buying** in the next {HORIZON} days, "
    "**how much they are likely to spend**, and **where to focus your retention effort**."
)

tab_list, tab_one, tab_about = st.tabs([
    "📋 Score my customer list",
    "👤 Check one customer",
    "ℹ️ How it works",
])

# ---------------- Tab 1: whole customer base ----------------
with tab_list:
    st.subheader("Step 1 · Load your sales ledger")
    st.caption(
        "Export your invoice or POS lines as CSV or Excel: one row per item sold, with customer ID, "
        "date, invoice number, product and value. Returns should have a negative quantity or amount."
    )
    options = ["Upload my sales file"]
    if SAMPLE_PATH.exists():
        options.append("Use ABC Ltd sample data (400 customers)")
    source = st.radio("Data source", options, horizontal=True, label_visibility="collapsed")

    raw = None
    if source.startswith("Upload"):
        up = st.file_uploader("Sales ledger (CSV or Excel)", type=["csv", "xlsx"])
        if up is not None:
            raw = read_file(up.getvalue(), up.name)
    else:
        raw = read_file(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name)

    if raw is not None:
        ledger = column_mapper(raw)
        if ledger is not None:
            last_day = ledger["date"].max().normalize() + pd.Timedelta(days=1)
            as_of = st.date_input(
                "Predict from this date (default: the day after your latest sale)",
                value=last_day.date(),
            )
            feats, lapsed = ledger_to_features(ledger, as_of)

            if feats.empty:
                st.warning("No customers bought anything in the 12 months before this date.")
            else:
                res = score(feats)
                high = res[res["risk_band"] == "High"]

                st.subheader("Step 3 · Results")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Active customers scored", f"{len(res):,}")
                k2.metric("High churn risk", f"{len(high):,} ({len(high) / len(res):.0%})")
                k3.metric(f"Revenue at risk (next {HORIZON} days)", money(res["revenue_at_risk"].sum()))
                k4.metric(f"Forecast revenue (next {HORIZON} days)", money(res["expected_spend_90d"].sum()))
                if lapsed > 0:
                    st.caption(
                        f"{lapsed:,} customers in your file have not bought anything in the last 12 months. "
                        "They are treated as already lost and are not scored."
                    )

                st.markdown("#### What would a retention campaign be worth?")
                c1, c2, c3 = st.columns(3)
                target = c1.selectbox("Who to contact", ["High-risk customers", "High + medium-risk customers", "High-risk key accounts only"])
                save_rate = c2.slider("Share of at-risk revenue you expect to save", 0, 100, 25, format="%d%%") / 100
                cost = c3.number_input("Cost per customer contacted (£)", 0.0, 10_000.0, 10.0, step=5.0)

                if target.startswith("High +"):
                    tgt = res[res["risk_band"] != "Low"]
                elif target.endswith("only"):
                    tgt = res[(res["risk_band"] == "High") & res["key_account"]]
                else:
                    tgt = high
                saved = save_rate * tgt["revenue_at_risk"].sum()
                spend = cost * len(tgt)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Customers to contact", f"{len(tgt):,}")
                m2.metric("Revenue protected", money(saved))
                m3.metric("Campaign cost", money(spend))
                m4.metric("Net benefit", money(saved - spend),
                          f"ROI {((saved - spend) / spend):.0%}" if spend > 0 else None)

                st.markdown("#### Priority call list")
                bands = st.multiselect("Show risk levels", ["High", "Medium", "Low"], default=["High", "Medium"])
                view = (
                    res[res["risk_band"].isin(bands)]
                    .sort_values("revenue_at_risk", ascending=False)
                    .reset_index()
                )
                view["churn_pct"] = (view["churn_risk"] * 100).round(1)
                view = view[[
                    "Customer", "risk_band", "churn_pct", "action", "revenue_at_risk",
                    "expected_spend_90d", "spend_12m", "recency_days", "orders_12m",
                ]]
                st.dataframe(
                    view,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "risk_band": "Risk",
                        "churn_pct": st.column_config.ProgressColumn("Churn risk", format="%.0f%%", min_value=0, max_value=100),
                        "action": "Suggested action",
                        "revenue_at_risk": st.column_config.NumberColumn(f"Revenue at risk ({HORIZON}d)", format="£%.0f"),
                        "expected_spend_90d": st.column_config.NumberColumn(f"Forecast spend ({HORIZON}d)", format="£%.0f"),
                        "spend_12m": st.column_config.NumberColumn("Spend last 12m", format="£%.0f"),
                        "recency_days": st.column_config.NumberColumn("Days since last order"),
                        "orders_12m": st.column_config.NumberColumn("Orders last 12m"),
                    },
                )
                st.download_button(
                    "⬇️ Download this list (CSV)",
                    view.to_csv(index=False).encode(),
                    file_name="abc_ltd_retention_priority_list.csv",
                    mime="text/csv",
                )

# ---------------- Tab 2: single customer ----------------
with tab_one:
    st.subheader("Check one customer")
    st.caption("Fill in what you know about the customer from your sales records, then press **Predict**.")
    med = bundle["feature_medians"]
    with st.form("one_customer"):
        c1, c2, c3 = st.columns(3)
        recency = c1.number_input(LABELS["recency_days"], 0, 365, int(med["recency_days"]),
                                  help="Only customers who bought in the last 12 months can be scored.")
        orders = c2.number_input(LABELS["orders_12m"], 1, 500, int(med["orders_12m"]))
        spend_12m = c3.number_input(LABELS["spend_12m"], 1.0, 1_000_000.0, float(round(med["spend_12m"])), step=100.0)
        c4, c5, c6 = st.columns(3)
        tenure = c4.number_input(LABELS["tenure_days"], 0, 20_000, int(med["tenure_days"]),
                                 help="How long ago this customer first bought from you.")
        products = c5.number_input(LABELS["unique_products_12m"], 1, 5_000, int(med["unique_products_12m"]))
        returns_pct = c6.number_input("Returns as % of spend", 0.0, 100.0, 0.0, step=1.0)
        go = st.form_submit_button("Predict", type="primary")

    if go:
        if tenure < recency:
            st.info("Days since first order can't be less than days since last order. We've set them equal.")
            tenure = recency
        row = pd.DataFrame([{
            "recency_days": recency,
            "orders_12m": orders,
            "spend_12m": spend_12m,
            "tenure_days": tenure,
            "unique_products_12m": products,
            "return_rate": returns_pct / 100,
        }])
        r = score(row).iloc[0]

        colour = {"High": "red", "Medium": "orange", "Low": "green"}[r["risk_band"]]
        st.markdown(f"### Churn risk: :{colour}[{r['risk_band']}] ({r['churn_risk']:.0%})")
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Chance of buying again (next {HORIZON} days)", f"{1 - r['churn_risk']:.0%}")
        k2.metric(f"Forecast spend (next {HORIZON} days)", money(r["expected_spend_90d"]))
        k3.metric(f"Revenue at risk (next {HORIZON} days)", money(r["revenue_at_risk"]))
        st.success(f"**Suggested action:** {r['action']}" + ("  ·  This is a key account (top 20% by spend)." if r["key_account"] else ""))

        why = reasons(row)
        if why:
            st.markdown("**Why?**")
            for line in why:
                st.markdown(f"- {line}")
        if tenure > bundle["feature_max"]["tenure_days"]:
            st.caption("Customers older than ~21 months are all treated as long-standing customers.")

# ---------------- Tab 3: about ----------------
with tab_about:
    lm, rm = bundle["logistic_metrics"], bundle["linear_metrics"]
    st.subheader("What this tool does")
    st.markdown(
        f"""
It uses two models built from **ABC Ltd's own sales ledger** (about 800,000 invoice lines, 5,900 customers, Dec 2009 to Dec 2011).

| Question | Model | What you get |
|---|---|---|
| Will this customer buy again in the next {HORIZON} days? | **Logistic regression** | Churn risk (%) and a High / Medium / Low band |
| How much will this customer spend in the next {HORIZON} days? | **Linear regression** | Forecast spend in £ |

Each customer is described by six facts that any sales ledger holds: days since last order, number of orders,
spend, how long they've been a customer, how many different products they buy, and how much they return.

**Revenue at risk** = churn risk × the customer's normal {HORIZON}-day spend. Use it to rank who to call first.
"""
    )

    st.subheader("How accurate is it?")
    st.markdown(
        f"""
Both models were **trained on 2010 data and then tested on 2011 data they had never seen**, the same way you'd use them in real life.

- **Churn model:** correct for **{lm['Accuracy']:.0%}** of customers, against {lm['Baseline accuracy (always guess majority)']:.0%} if you just assumed everyone would churn.
  When it flags a customer as likely to churn, it is right **{lm['Churn precision']:.0%}** of the time, and it catches **{lm['Churn recall']:.0%}** of the customers who actually leave.
  AUC = {lm['AUC']:.2f}.
- **Spend model:** explains **{rm['R2']:.0%}** of the variation in customer spend (R² = {rm['R2']:.2f}).
  Its total revenue forecast was {money(rm['Total predicted spend (GBP)'])} against an actual {money(rm['Total actual spend (GBP)'])}
  ({rm['Total predicted spend (GBP)'] / rm['Total actual spend (GBP)']:.0%}).
  The typical error per customer is {money(rm['Median abs error (GBP)'])}.
"""
    )

    st.subheader("What drives churn?")
    st.caption("From the logistic regression weights, strongest factor first.")
    coefs = pd.Series(bundle["logistic_coefficients"], dtype=float)
    drivers = pd.DataFrame({
        "Factor": [LABELS[f] for f in coefs.index],
        "When this goes up, the customer is ...": np.where(coefs > 0, "more likely to stay", "more likely to churn"),
        "Strength": (coefs.abs() / coefs.abs().max() * 100).round(0).to_numpy(),
    }).sort_values("Strength", ascending=False)
    st.dataframe(
        drivers,
        hide_index=True,
        width="stretch",
        column_config={"Strength": st.column_config.ProgressColumn("Strength", format="%.0f", min_value=0, max_value=100)},
    )
    st.markdown(
        "The number of **orders in the last 12 months** is by far the strongest signal of loyalty. "
        "**Long gaps since the last order**, **high returns** and a long history with little recent activity all point towards churn."
    )

    st.subheader("Spend model: £ impact of each factor")
    lin = pd.DataFrame({
        "Factor": [LABELS[f] for f in FEATURES],
        f"Change in {HORIZON}-day spend for +1 unit": [f"£{bundle['linear_coefficients'][f]:,.2f}" for f in FEATURES],
    })
    st.dataframe(lin, hide_index=True, width="stretch")
    st.caption(
        f"Starting point (intercept): £{bundle['linear_intercept']:,.2f}. Each figure is the effect of that factor with the others held constant. "
        "For example, every extra £1 spent in the last 12 months adds about "
        f"£{bundle['linear_coefficients']['spend_12m']:.2f} to next-quarter spend, and each extra day since the last order takes away "
        f"£{abs(bundle['linear_coefficients']['recency_days']):.2f}. "
        "Returns are measured as a share of spend (0 to 1), so the returns figure is the effect of going from 0% to 100% returns."
    )

    st.subheader("Limits to keep in mind")
    st.markdown(
        """
- Only customers who bought in the last 12 months are scored. Anyone quieter than that is treated as already lost.
- ABC Ltd's business is seasonal (Q4 is peak), so forecasts made in September run higher than forecasts made in March.
- Figures outside the range the models were trained on are capped at that range. For example, customers older than about 21 months are all treated as long-standing.
- The model shows who is at risk, not why they are leaving. Use it to decide who to call, then ask them.
"""
    )
