"""
SAHELI Backend — Database (SQLite)

Deliberately lightweight: a single SQLite file, stdlib sqlite3, no ORM.
This is a real persistence layer (not a mock), sized appropriately for an
MVP — sufficient for genuine authentication and per-country data scoping
without the operational overhead of a separate database server.
"""
import sqlite3
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saheli.db")

SUPPORTED_COUNTRIES = ["Niger", "Mali", "Burkina Faso", "Chad", "Mauritania", "Senegal"]


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                country TEXT NOT NULL,
                organization TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # Safe, additive migration for installs created before these columns
        # existed — SQLite errors if the column is already there, so each
        # is wrapped individually.
        for col_def in ["photo_base64 TEXT", "bio TEXT"]:
            try:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists

        conn.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS crop_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                district TEXT,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_user(email: str, password_hash: str, full_name: str, country: str, organization: str | None) -> dict:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, full_name, country, organization) VALUES (?, ?, ?, ?, ?)",
            (email.lower().strip(), password_hash, full_name, country, organization),
        )
        conn.commit()
        return get_user_by_id(cur.lastrowid)


def get_user_by_email(email: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def update_user_profile(user_id: int, full_name: str | None = None, organization: str | None = None,
                         bio: str | None = None, photo_base64: str | None = None) -> dict:
    fields, values = [], []
    if full_name is not None:
        fields.append("full_name = ?"); values.append(full_name)
    if organization is not None:
        fields.append("organization = ?"); values.append(organization)
    if bio is not None:
        fields.append("bio = ?"); values.append(bio)
    if photo_base64 is not None:
        fields.append("photo_base64 = ?"); values.append(photo_base64)
    if not fields:
        return get_user_by_id(user_id)
    with get_db() as conn:
        conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", (*values, user_id))
        conn.commit()
    return get_user_by_id(user_id)


def log_activity(user_id: int, action_type: str):
    """Real activity logging — starts recording from the moment this
    feature ships. Earlier usage was never tracked, so history begins
    now, honestly, rather than being backfilled with invented data."""
    with get_db() as conn:
        conn.execute("INSERT INTO activity_log (user_id, action_type) VALUES (?, ?)", (user_id, action_type))
        conn.commit()


def get_activity_counts(user_id: int, days: int = 365) -> dict:
    """Real daily activity counts for this user, for the contribution-
    style heatmap. Returns {'YYYY-MM-DD': count, ...} for days with
    at least one logged action — days with zero are simply absent."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT date(created_at) as day, COUNT(*) as cnt
               FROM activity_log WHERE user_id = ? AND created_at >= date('now', ?)
               GROUP BY day ORDER BY day""",
            (user_id, f"-{days} days"),
        ).fetchall()
        return {row["day"]: row["cnt"] for row in rows}


def get_activity_total(user_id: int) -> int:
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM activity_log WHERE user_id = ?", (user_id,)).fetchone()
        return row["cnt"] if row else 0


def log_crop_scan(user_id: int, district: str | None, predicted_class: str, confidence: float):
    """Real logging of every Corn Scanner result, with an optional
    district tag. This is the honest link between the CV scanner and a
    district's context: a qualitative field report log, surfaced
    alongside the quantitative risk model, not silently fused into its
    score — there is no real data connecting detected disease severity
    to IPC-scale food security risk, so this does not pretend there is."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO crop_scans (user_id, district, predicted_class, confidence) VALUES (?, ?, ?, ?)",
            (user_id, district, predicted_class, confidence),
        )
        conn.commit()


def get_recent_crop_scans(district: str, limit: int = 10) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT predicted_class, confidence, created_at FROM crop_scans
               WHERE district = ? ORDER BY created_at DESC LIMIT ?""",
            (district, limit),
        ).fetchall()
        return [dict(r) for r in rows]
