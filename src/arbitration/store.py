"""SQLite verdict store and critic behavior analytics."""

from __future__ import annotations

import json
import sqlite3
import time


class VerdictStore:
    def __init__(self, path: str = "arbitrations.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS arbitrations (
                   arbitration_id TEXT PRIMARY KEY, timestamp REAL,
                   overall_score INTEGER, confidence REAL,
                   confirmed INTEGER, dismissed INTEGER,
                   disagreements INTEGER, short_circuited INTEGER,
                   payload TEXT)""")
        self.conn.commit()

    def save(self, arbitration) -> None:
        verdict = arbitration.verdict
        self.conn.execute(
            "INSERT OR REPLACE INTO arbitrations VALUES (?,?,?,?,?,?,?,?,?)",
            (arbitration.arbitration_id, time.time(), verdict.overall_score,
             verdict.confidence, len(verdict.confirmed_issues),
             len(verdict.dismissed_flags), len(arbitration.disagreements),
             int(arbitration.short_circuited),
             arbitration.model_dump_json()))
        self.conn.commit()

    def load(self, arbitration_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT payload FROM arbitrations WHERE arbitration_id = ?",
            (arbitration_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def analytics(self) -> dict:
        rows = self.conn.execute(
            "SELECT payload FROM arbitrations").fetchall()
        issues_by_critic: dict[str, int] = {}
        overruled_by_critic: dict[str, int] = {}
        total = len(rows)
        agreements = 0
        for (payload,) in rows:
            arb = json.loads(payload)
            for critique in arb["critiques"]:
                issues_by_critic.setdefault(critique["dimension"], 0)
                issues_by_critic[critique["dimension"]] += len(critique["issues"])
            for flag in arb["verdict"]["dismissed_flags"]:
                overruled_by_critic.setdefault(flag["dimension"], 0)
                overruled_by_critic[flag["dimension"]] += 1
            if not arb["disagreements"]:
                agreements += 1
        return {
            "total_arbitrations": total,
            "issues_found_by_critic": issues_by_critic,
            "critic_overruled_count": overruled_by_critic,
            "full_agreement_rate": round(agreements / total, 3) if total else 0,
            "average_score": round(sum(
                json.loads(p)["verdict"]["overall_score"]
                for (p,) in rows) / total, 2) if total else 0,
        }
