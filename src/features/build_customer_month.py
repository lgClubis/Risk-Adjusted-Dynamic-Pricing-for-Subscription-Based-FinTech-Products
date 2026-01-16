from __future__ import annotations

from pathlib import Path
import pandas as pd

PQ_DIR = Path("data/processed/raw_parquet")
OUT_PATH = Path("data/processed/customer_month.parquet")


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(
        f"None of the candidate columns exist. Tried: {candidates}. "
        f"Available: {sorted(df.columns)}"
    )


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def to_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()


def add_month(df: pd.DataFrame, date_col: str, out_col: str = "month") -> pd.DataFrame:
    df = df.copy()
    df[out_col] = to_month(df[date_col])
    return df


def ensure_account_id(
    df: pd.DataFrame,
    subs: pd.DataFrame,
    *,
    df_name: str,
    df_date_col: str | None = None
) -> tuple[pd.DataFrame, str]:
    """
    Ensure df has an account_id column.
    If df has account_id -> use it.
    Else if df has subscription_id -> map subscription_id -> account_id via subscriptions.
    Returns (df_with_account_id, account_id_col_name).
    """
    # Determine subscription mapping columns in subs
    subs_sub_id = pick_col(subs, ["subscription_id"])
    subs_acc_id = pick_col(subs, ["account_id", "customer_id", "user_id", "account"])

    if "account_id" in df.columns:
        return df, "account_id"
    if subs_acc_id in df.columns:
        return df, subs_acc_id

    if "subscription_id" in df.columns:
        out = df.merge(
            subs[[subs_sub_id, subs_acc_id]].drop_duplicates(),
            left_on="subscription_id",
            right_on=subs_sub_id,
            how="left",
        )
        if out[subs_acc_id].isna().all():
            raise ValueError(
                f"[{df_name}] subscription_id -> account_id mapping failed: "
                f"all mapped account_id are NaN. Check join keys."
            )
        return out, subs_acc_id

    raise KeyError(
        f"[{df_name}] Neither account_id nor subscription_id found. "
        f"Columns: {sorted(df.columns)}"
    )


def expand_subscriptions_to_months(subs: pd.DataFrame) -> pd.DataFrame:
    """
    Expand subscriptions into account_id x month rows.
    Uses start/end if present; if end missing, expands only the start month (conservative).
    """
    subs = subs.copy()

    acc_id = pick_col(subs, ["account_id", "customer_id", "user_id", "account"])
    sub_id = pick_col(subs, ["subscription_id"])
    start_col = pick_col(subs, ["start_date", "subscription_start", "started_at", "begin_date", "created_at"])

    end_col = first_existing(subs, ["end_date", "subscription_end", "ended_at", "cancelled_at", "canceled_at", "expires_at"])

    plan_col = first_existing(subs, ["plan", "plan_name", "subscription_plan", "tier"])
    price_col = first_existing(subs, ["price", "monthly_price", "amount", "mrr", "plan_price"])

    subs[start_col] = pd.to_datetime(subs[start_col], errors="coerce")
    if end_col:
        subs[end_col] = pd.to_datetime(subs[end_col], errors="coerce")
        subs["_end_for_expand"] = subs[end_col].fillna(subs[start_col])
    else:
        subs["_end_for_expand"] = subs[start_col]

    def expand_row(row) -> pd.DataFrame:
        start = row[start_col]
        end = row["_end_for_expand"]
        if pd.isna(start) or pd.isna(end):
            return pd.DataFrame(columns=[acc_id, "month"])
        start_m = start.to_period("M").to_timestamp()
        end_m = end.to_period("M").to_timestamp()
        months = pd.date_range(start=start_m, end=end_m, freq="MS")

        out = pd.DataFrame({acc_id: row[acc_id], "month": months})
        out["subscription_id"] = row[sub_id]
        if plan_col:
            out["plan"] = row[plan_col]
        if price_col:
            out["price"] = row[price_col]
        return out

    expanded = pd.concat([expand_row(r) for _, r in subs.iterrows()], ignore_index=True)
    expanded = expanded.drop_duplicates(subset=[acc_id, "month"], keep="last")
    expanded = expanded.rename(columns={acc_id: "account_id"})
    expanded["is_subscribed"] = 1
    return expanded


def main() -> None:
    # Load
    accounts = pd.read_parquet(PQ_DIR / "ravenstack_accounts.parquet")
    churn_events = pd.read_parquet(PQ_DIR / "ravenstack_churn_events.parquet")
    usage = pd.read_parquet(PQ_DIR / "ravenstack_feature_usage.parquet")
    subs = pd.read_parquet(PQ_DIR / "ravenstack_subscriptions.parquet")
    tickets = pd.read_parquet(PQ_DIR / "ravenstack_support_tickets.parquet")

    # Expand subscriptions to account-month rows (also gives plan/price if present)
    subs_month = expand_subscriptions_to_months(subs)

    # --- Ensure account_id exists everywhere (map from subscription_id if needed)
    churn_date_col = pick_col(churn_events, ["churn_date", "date", "event_date", "timestamp", "created_at"])
    usage_date_col = pick_col(usage, ["usage_date", "date", "event_date", "used_at", "timestamp", "created_at"])
    tickets_date_col = pick_col(tickets, ["submitted_at", "created_at", "date", "opened_at", "timestamp"])


    churn_events_m, churn_acc_col = ensure_account_id(churn_events, subs, df_name="churn_events")
    usage_m, usage_acc_col = ensure_account_id(usage, subs, df_name="feature_usage")
    tickets_m, tickets_acc_col = ensure_account_id(tickets, subs, df_name="support_tickets")

    # Add month columns
    churn_events_m = add_month(churn_events_m, churn_date_col)
    usage_m = add_month(usage_m, usage_date_col)
    tickets_m = add_month(tickets_m, tickets_date_col)

    # Skeleton months: union of subscription-months + activity months + churn months
    skeleton = pd.concat(
        [
            subs_month[["account_id", "month"]],
            usage_m[[usage_acc_col, "month"]].rename(columns={usage_acc_col: "account_id"}),
            tickets_m[[tickets_acc_col, "month"]].rename(columns={tickets_acc_col: "account_id"}),
            churn_events_m[[churn_acc_col, "month"]].rename(columns={churn_acc_col: "account_id"}),
        ],
        ignore_index=True,
    ).dropna().drop_duplicates()

    # --- Aggregate usage numeric cols
    usage_exclude = {usage_acc_col, "month", usage_date_col, "usage_id", "subscription_id", "feature_name"}
    usage_num_cols = [c for c in usage_m.columns if c not in usage_exclude and pd.api.types.is_numeric_dtype(usage_m[c])]

    if usage_num_cols:
        usage_agg = usage_m.groupby([usage_acc_col, "month"], as_index=False)[usage_num_cols].sum()
        usage_agg["usage_events"] = usage_m.groupby([usage_acc_col, "month"]).size().values
    else:
        usage_agg = usage_m.groupby([usage_acc_col, "month"], as_index=False).size().rename(columns={"size": "usage_events"})

    usage_agg = usage_agg.rename(columns={usage_acc_col: "account_id"})

    # --- Aggregate tickets
    ticket_agg = tickets_m.groupby([tickets_acc_col, "month"], as_index=False).size().rename(columns={"size": "ticket_count"})
    ticket_agg = ticket_agg.rename(columns={tickets_acc_col: "account_id"})

    # Optional numeric ticket metrics (means)
    ticket_exclude = {tickets_acc_col, "month", tickets_date_col, "ticket_id", "subscription_id"}
    ticket_num_cols = [c for c in tickets_m.columns if c not in ticket_exclude and pd.api.types.is_numeric_dtype(tickets_m[c])]
    if ticket_num_cols:
        extra = tickets_m.groupby([tickets_acc_col, "month"], as_index=False)[ticket_num_cols].mean()
        extra = extra.rename(columns={tickets_acc_col: "account_id"})
        ticket_agg = ticket_agg.merge(extra, on=["account_id", "month"], how="left")

    # --- Churn flags
    churn_flag = (
        churn_events_m.groupby([churn_acc_col, "month"], as_index=False)
        .size()
        .rename(columns={"size": "churn_events"})
    )
    churn_flag["churn_in_month"] = 1
    churn_flag = churn_flag[[churn_acc_col, "month", "churn_in_month"]].rename(columns={churn_acc_col: "account_id"})

    # --- Assemble final panel
    panel = skeleton.merge(subs_month.drop_duplicates(["account_id", "month"]), on=["account_id", "month"], how="left")
    panel = panel.merge(usage_agg, on=["account_id", "month"], how="left")
    panel = panel.merge(ticket_agg, on=["account_id", "month"], how="left")
    panel = panel.merge(churn_flag, on=["account_id", "month"], how="left")

    # Fill core flags
    panel["is_subscribed"] = panel.get("is_subscribed", 0).fillna(0).astype(int)
    panel["churn_in_month"] = panel["churn_in_month"].fillna(0).astype(int)

    # churn_next_month label
    panel = panel.sort_values(["account_id", "month"])
    panel["churn_next_month"] = panel.groupby("account_id")["churn_in_month"].shift(-1).fillna(0).astype(int)

    # Fill numeric NaNs with 0 for engineered metrics
    for c in panel.columns:
        if c in {"account_id", "month", "plan"}:
            continue
        if pd.api.types.is_numeric_dtype(panel[c]):
            panel[c] = panel[c].fillna(0)

    # Tenure
    first_month = panel.groupby("account_id")["month"].transform("min")
    panel["tenure_months"] = ((panel["month"].dt.to_period("M") - first_month.dt.to_period("M")).apply(lambda x: x.n)).astype(int)

    # Write parquet
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)

    print(f"Wrote: {OUT_PATH} | rows={len(panel):,} cols={panel.shape[1]}")
    print("Columns:", list(panel.columns))


if __name__ == "__main__":
    main()
