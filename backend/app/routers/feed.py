"""
SAHELI Backend — Live Alert Feed
Detects real risk-level transitions across the historical scored dataset
(each row already carries a model prediction) and exposes them as a
chronological event feed — no fabricated events, purely derived from data.
"""
from fastapi import APIRouter, Query, Depends
import pandas as pd
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import get_scored_df
from routers.auth import get_current_user

router = APIRouter(prefix="/api", tags=["feed"])

SEVERITY = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


@router.get("/feed/events")
def get_feed_events(limit: int = Query(50, le=300), user: dict = Depends(get_current_user)):
    df = get_scored_df()
    df = df[df["country"] == user["country"]][["date", "district", "country", "zone", "predicted_risk"]].copy()
    df = df.sort_values(["district", "date"])
    df["prev_risk"] = df.groupby("district")["predicted_risk"].shift(1)
    transitions = df[df["prev_risk"].notna() & (df["predicted_risk"] != df["prev_risk"])].copy()
    transitions["direction"] = transitions.apply(
        lambda r: "worsening" if SEVERITY[r["predicted_risk"]] > SEVERITY[r["prev_risk"]] else "improving",
        axis=1,
    )
    transitions = transitions.sort_values("date", ascending=False).head(limit)

    events = [
        {
            "date": row["date"].isoformat(),
            "district": row["district"],
            "country": row["country"],
            "zone": row["zone"],
            "from_risk": row["prev_risk"],
            "to_risk": row["predicted_risk"],
            "direction": row["direction"],
        }
        for _, row in transitions.iterrows()
    ]
    return {"count": len(events), "events": events}
