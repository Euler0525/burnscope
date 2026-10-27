import json
import sqlite3
import time

import psutil

from .model import data_dir, now


class Store:
    def __init__(self, path=None):
        self.path = path or data_dir() / "history.sqlite3"
        self.db = sqlite3.connect(self.path, timeout=5)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, report TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS packets(run TEXT, elapsed REAL, payload TEXT);
            CREATE INDEX IF NOT EXISTS packets_run ON packets(run);
        """)

    def save(self, report):
        self.db.execute("INSERT OR REPLACE INTO runs VALUES (?, ?)",
                        (report["id"], json.dumps(report, ensure_ascii=False, allow_nan=False)))
        self.db.commit()

    def append(self, run, elapsed, packet):
        self.db.execute("INSERT INTO packets VALUES (?, ?, ?)",
                        (run, elapsed, json.dumps(packet, ensure_ascii=False, allow_nan=False)))
        self.db.commit()

    def history(self):
        return [json.loads(row[0]) for row in self.db.execute("SELECT report FROM runs ORDER BY rowid DESC")]

    def get(self, run):
        row = self.db.execute("SELECT report FROM runs WHERE id=?", (run,)).fetchone()
        if row is None:
            raise ValueError("找不到测试记录")
        return json.loads(row[0])

    def packets(self, run):
        return [{"elapsed": elapsed, **json.loads(payload)} for elapsed, payload in self.db.execute(
            "SELECT elapsed, payload FROM packets WHERE run=? ORDER BY rowid", (run,))]

    def recover(self):
        for report in self.history():
            if report["state"] != "running":
                continue
            try:
                alive = abs(psutil.Process(report["pid"]).create_time() - report["process_created"]) < 1
            except psutil.Error:
                alive = False
            if not alive:
                report.update(state="interrupted", exit_code=3, finished=now())
                report["findings"] = ["上次测试意外中断，已有数据保留；无法据此判断中断原因。"]
                self.save(report)

    def close(self):
        self.db.close()


def new_report(plan):
    process = psutil.Process()
    return {"id": f"{time.time_ns():x}", "started": now(), "finished": None, "state": "running",
            "exit_code": 0, "pid": process.pid, "process_created": process.create_time(),
            "version": "0.2.0", "plan": plan.plain(), "workers": {}, "findings": [],
            "cpu_temperature_monitoring": False}
