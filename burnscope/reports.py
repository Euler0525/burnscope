import csv
import html
import json
from pathlib import Path


LABELS = {"utilization": "利用率", "reported_clock": "系统报告频率", "clock": "核心频率",
          "temperature": "GPU 温度", "power": "功耗", "memory_used": "已用内存 / 显存",
          "memory_available": "可用内存", "memory_percent": "内存使用率", "memory_free": "空闲显存"}
STATES = {"completed": "已完成", "cancelled": "已停止", "protected_stop": "保护停止",
          "failed": "测试异常", "unavailable": "条件不满足", "interrupted": "意外中断", "running": "运行中"}
MODES = {"cpu": "CPU", "gpu": "GPU", "combined": "CPU + GPU", "ram": "内存", "vram": "显存"}


def export(store, run, path):
    report, packets = store.get(run), store.packets(run)
    path = Path(path)
    if path.suffix == ".json":
        path.write_text(json.dumps({"report": report, "packets": packets}, ensure_ascii=False, indent=2), encoding="utf-8")
    elif path.suffix == ".csv":
        with path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["秒", "设备", "指标", "数值", "单位", "来源", "状态", "说明"])
            for packet in packets:
                for sample in packet.get("samples", []):
                    writer.writerow([round(packet["elapsed"], 2), sample["device"],
                        LABELS.get(sample["metric"], sample["metric"]), sample["value"], sample["unit"],
                        sample["source"], sample["status"], sample["reason"]])
    else:
        def esc(value):
            return html.escape(str(value))
        events = [f'<li>{p["elapsed"]:.1f} 秒 · {esc(p["message"])}</li>'
                  for p in packets if p["type"] == "event"]
        statistics = {}
        for packet in packets:
            for sample in packet.get("samples", []):
                if sample["status"] == "ok" and sample["device"] in {"cpu", "gpu", "system"}:
                    key = (sample["device"], sample["metric"], sample["unit"])
                    if sample["metric"] in LABELS:
                        statistics.setdefault(key, []).append(sample["value"])
        rows = "".join(f"<tr><td>{esc(device)}</td><td>{esc(LABELS[metric])}</td>"
            f"<td>{values[-1]:.1f}</td><td>{sum(values) / len(values):.1f}</td><td>{max(values):.1f} {esc(unit)}</td></tr>"
            for (device, metric, unit), values in statistics.items())
        findings = "".join(f"<li>{esc(text)}</li>" for text in report["findings"])
        path.write_text(f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>BurnScope 测试记录</title><style>
body{{font:16px/1.7 system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 24px;color:#203345}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}
h1,h2{{color:#087e76}}small{{color:#607080}}</style>
<h1>BurnScope 测试记录</h1><p>{esc(MODES[report['plan']['mode']])} · {esc(STATES[report['state']])}</p>
<small>{esc(report['started'])} · 记录 {esc(run)}</small><h2>结果</h2><ul>{findings}</ul>
<h2>主要指标</h2><table><tr><th>设备</th><th>指标</th><th>最后值</th><th>均值</th><th>峰值</th></tr>{rows}</table>
<h2>事件</h2><ul>{''.join(events)}</ul><p>仅覆盖成功分配并校验的内存区域；本程序不读取 CPU 温度。</p>
</html>""", encoding="utf-8")
    return path
