from collections import defaultdict, deque
from pathlib import Path
import math
import sys
import time

from PySide6 import QtCore, QtGui, QtWidgets as W
import pyqtgraph as pg

from .model import Plan, data_dir
from .processes import Child, probe_gpus
from .reports import MODES, STATES, export
from .storage import Store


STYLE = """
QWidget{font-size:14px;color:#23394c} QMainWindow{background:#f4f7fa}
QLabel#title{font-size:28px;font-weight:700} QLabel#muted{color:#6c7d8d}
QLabel#value{font-size:29px;font-weight:700;color:#16384a}
QFrame#card{background:white;border:1px solid #e0e7ed;border-radius:12px}
QPushButton{padding:10px 20px;border:1px solid #cbd8e1;background:white;border-radius:7px}
QPushButton:hover{background:#e9f0f5} QPushButton:disabled{color:#96a5b0;background:#edf1f4}
QPushButton#start{background:#087e76;color:white;border:0;font-weight:700}
QPushButton#stop{background:#fff0ee;border-color:#e7b4ac;color:#ad3333}
QComboBox,QDoubleSpinBox{background:white;border:1px solid #cbd8e1;border-radius:6px;padding:6px;min-height:23px}
QProgressBar{height:7px;border:0;border-radius:3px;background:#dce6ed}
QProgressBar::chunk{background:#0d9488;border-radius:3px}
QPlainTextEdit{background:white;border:1px solid #dce5ed;border-radius:6px}
"""


def text(value, style=None):
    widget = W.QLabel(value)
    if style:
        widget.setObjectName(style)
    return widget


class Scan(QtCore.QThread):
    result = QtCore.Signal(object)

    def run(self):
        try:
            self.result.emit(probe_gpus())
        except Exception as exc:
            self.result.emit({"devices": [], "issues": [str(exc)]})


class Window(W.QMainWindow):
    def __init__(self, demo=False):
        super().__init__()
        self.demo, self.session, self.monitor = demo, None, None
        self.latest_id, self.started, self.exit_seen = None, time.monotonic(), None
        self.points, self.curves = defaultdict(lambda: deque(maxlen=600)), {}
        self.setWindowTitle("BurnScope · 简洁烤机" + (" · 演示" if demo else ""))
        self.resize(1060, 740)
        self.setMinimumSize(880, 640)
        self._layout()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(150)
        store = Store()
        try:
            store.recover()
            history = store.history()
            self.latest_id = history[0]["id"] if history else None
            self.report_button.setEnabled(self.latest_id is not None)
        finally:
            store.close()
        if demo:
            self.status.setText("演示数据 · 未运行测试")
            self.gpu.addItem("演示 GPU", {"name": "演示 GPU"})
            self.scan_button.setEnabled(False)
            for second in range(-80, 1):
                self.demo_sample(second)
        else:
            self.scan()

    def _layout(self):
        root = W.QWidget()
        box = W.QVBoxLayout(root)
        box.setContentsMargins(28, 24, 28, 22)
        box.setSpacing(16)
        title = W.QHBoxLayout()
        title.addWidget(text("BurnScope", "title"))
        title.addWidget(text("简洁烤机 · Python", "muted"))
        title.addStretch()
        self.status = text("正在检测设备", "muted")
        title.addWidget(self.status)
        box.addLayout(title)
        controls = W.QHBoxLayout()
        self.mode = W.QComboBox()
        for key, name in MODES.items():
            self.mode.addItem(name + "测试", key)
        self.duration = W.QDoubleSpinBox()
        self.duration.setRange(.1, 60)
        self.duration.setValue(5)
        self.duration.setSuffix(" 分钟")
        self.load = W.QComboBox()
        for value in (100, 75, 50, 25):
            self.load.addItem(f"{value}%", value / 100)
        for caption, widget in [("测试", self.mode), ("时长", self.duration), ("负载", self.load)]:
            controls.addWidget(text(caption, "muted"))
            controls.addWidget(widget)
        controls.addStretch()
        self.start_button = W.QPushButton("开始测试", objectName="start")
        self.start_button.clicked.connect(self.start_run)
        self.start_button.setEnabled(False)
        self.stop_button = W.QPushButton("停止", objectName="stop")
        self.stop_button.clicked.connect(self.stop_run)
        self.stop_button.setEnabled(False)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        box.addLayout(controls)
        gpu_row = W.QHBoxLayout()
        gpu_row.addWidget(text("显卡", "muted"))
        self.gpu = W.QComboBox()
        self.gpu.setMinimumWidth(330)
        self.gpu.currentIndexChanged.connect(self.restart_monitor)
        gpu_row.addWidget(self.gpu)
        self.scan_button = W.QPushButton("重新检测")
        self.scan_button.clicked.connect(self.scan)
        gpu_row.addWidget(self.scan_button)
        gpu_row.addStretch()
        self.report_button = W.QPushButton("查看上次报告")
        self.report_button.clicked.connect(self.show_report)
        self.report_button.setEnabled(False)
        gpu_row.addWidget(self.report_button)
        box.addLayout(gpu_row)
        cards = W.QHBoxLayout()
        self.cards = {}
        for key, title, detail in [(("cpu", "utilization"), "CPU 利用率", "全部逻辑核"),
                                   (("cpu", "reported_clock"), "CPU 报告频率", "可能为标称值"),
                                   (("gpu", "utilization"), "GPU 利用率", "驱动支持时显示"),
                                   (("gpu", "temperature"), "GPU 温度", "驱动支持时显示")]:
            frame = W.QFrame(objectName="card")
            column = W.QVBoxLayout(frame)
            column.setContentsMargins(17, 15, 17, 15)
            column.addWidget(text(title, "muted"))
            value = text("—", "value")
            column.addWidget(value)
            column.addWidget(text(detail, "muted"))
            self.cards[key] = value
            cards.addWidget(frame)
        box.addLayout(cards)
        self.progress = W.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        box.addWidget(self.progress)
        charts = W.QHBoxLayout()
        self.plots = {}
        for metric, title, unit in [("utilization", "利用率", "%"), ("clock", "频率", "MHz")]:
            plot = pg.PlotWidget(background="w")
            plot.setTitle(title, color="#23394c", size="12pt")
            plot.setLabel("left", unit)
            plot.setLabel("bottom", "时间", units="s")
            plot.showGrid(x=True, y=True, alpha=.12)
            plot.addLegend(labelTextColor="#506c80")
            if metric == "utilization":
                plot.setYRange(0, 105)
            self.plots[metric] = plot
            charts.addWidget(plot)
        box.addLayout(charts, 1)
        self.summary = text("内存 —   ·   GPU 频率 —   ·   GPU 功耗 —", "muted")
        box.addWidget(self.summary)
        self.note = text("CPU 不读取温度，不提供 CPU 温度保护；测试默认使用全部可用逻辑核。", "muted")
        self.note.setWordWrap(True)
        box.addWidget(self.note)
        self.details = W.QPushButton("展开运行记录")
        self.details.setCheckable(True)
        box.addWidget(self.details)
        self.log = W.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(140)
        self.log.hide()
        self.details.toggled.connect(self.log.setVisible)
        self.details.toggled.connect(lambda shown: self.details.setText("收起运行记录" if shown else "展开运行记录"))
        box.addWidget(self.log)
        self.setCentralWidget(root)
        self.readings = {}

    def scan(self):
        if self.session or self.demo:
            return
        self.start_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.status.setText("正在检测 GPU · CPU 无需驱动")
        self.scanner = Scan(self)
        self.scanner.result.connect(self.scanned)
        self.scanner.start()

    def scanned(self, result):
        self.gpu.blockSignals(True)
        self.gpu.clear()
        # Prefer a discrete card; preserve exact discovery identity in item data.
        for device in sorted(result["devices"], key=lambda d: d["shared_memory"]):
            self.gpu.addItem(device["name"], device)
        if not result["devices"]:
            self.gpu.addItem("GPU 不可用 · CPU / 内存测试可用", None)
        self.gpu.blockSignals(False)
        self.log.appendPlainText("\n".join(result["issues"]) or "设备检测完成")
        self.start_button.setEnabled(True)
        self.scan_button.setEnabled(True)
        self.restart_monitor()

    def restart_monitor(self, *_):
        if self.demo or self.session:
            return
        if self.monitor:
            self.monitor.close()
            self.monitor.pipe.close()
        self.monitor = Child("monitor", {"gpu": self.gpu.currentData()})
        self.status.setText("就绪")
        self.clear_data()

    def clear_data(self):
        self.started = time.monotonic()
        self.points.clear()
        self.curves.clear()
        self.readings.clear()
        for card in self.cards.values():
            card.setText("—")
        for plot in self.plots.values():
            plot.clear()

    def start_run(self):
        if self.session or self.demo:
            return
        try:
            plan = Plan(mode=self.mode.currentData(), duration=self.duration.value() * 60,
                        load=self.load.currentData(), gpu=self.gpu.currentData()).validate()
            if self.monitor:
                self.monitor.close()
                self.monitor.pipe.close()
                self.monitor = None
            self.clear_data()
            self.session = Child("controller", plan.plain())
            self.exit_seen = None
            self.status.setText("正在准备")
            self.set_running(True)
        except (ValueError, OSError) as exc:
            W.QMessageBox.information(self, "BurnScope", str(exc))

    def set_running(self, running):
        for control in (self.mode, self.duration, self.load, self.gpu, self.scan_button):
            control.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def stop_run(self):
        if self.session:
            self.session.stop.set()
            self.stop_button.setEnabled(False)
            self.status.setText("正在停止并保存")

    def poll(self):
        if self.monitor:
            self.monitor.ping()
            for packet in self.monitor.drain():
                if packet["type"] == "samples":
                    self.samples(packet["samples"])
            if not self.monitor.alive():
                self.monitor.close()
                self.monitor.pipe.close()
                self.monitor = None
                self.status.setText("监控已停止，可重新检测")
        if not self.session:
            return
        self.session.ping()
        finished = None
        for packet in self.session.drain():
            if packet["type"] == "samples":
                self.samples(packet["samples"], packet["elapsed"])
            elif packet["type"] == "event":
                self.log.appendPlainText(packet["message"])
            elif packet["type"] == "progress":
                self.progress.setValue(int(1000 * min(1, packet["seconds"] / packet["duration"])))
                remaining = max(0, packet["duration"] - packet["seconds"])
                self.status.setText(f"测试中 · 剩余 {remaining:.0f} 秒")
            elif packet["type"] == "finished":
                finished = packet["report"]
            elif packet["type"] == "error":
                self.log.appendPlainText(packet["message"])
        if finished:
            self.latest_id = finished["id"]
            self.status.setText(STATES[finished["state"]])
            self.log.appendPlainText("\n".join(finished["findings"]))
            self.report_button.setEnabled(True)
            self.progress.setValue(1000 if finished["state"] == "completed" else self.progress.value())
            self.end_run()
        elif not self.session.alive():
            if self.exit_seen and time.monotonic() - self.exit_seen > .5:
                self.status.setText("测试中断 · 数据已自动保存")
                self.end_run()
            elif self.exit_seen is None:
                self.exit_seen = time.monotonic()

    def end_run(self):
        self.session.close()
        self.session.pipe.close()
        self.session = None
        self.set_running(False)

    def samples(self, samples, elapsed=None):
        elapsed = time.monotonic() - self.started if elapsed is None else elapsed
        for sample in samples:
            key = sample["device"], sample["metric"]
            self.readings[key] = sample
            if key in self.cards:
                self.cards[key].setText(f'{sample["value"]:.1f} {sample["unit"]}' if sample["status"] == "ok" else "—")
                self.cards[key].setToolTip(sample["source"] + " · " + sample["reason"])
            metric = "clock" if sample["metric"] == "reported_clock" else sample["metric"]
            if sample["device"] not in {"cpu", "gpu"} or metric not in self.plots or sample["status"] != "ok":
                continue
            plot_key = sample["device"], metric
            self.points[plot_key].append((elapsed, sample["value"]))
            if plot_key not in self.curves:
                cpu = sample["device"] == "cpu"
                name = "CPU 报告频率" if cpu and metric == "clock" else "CPU" if cpu else "GPU"
                self.curves[plot_key] = self.plots[metric].plot(
                    pen=pg.mkPen("#0d9488" if cpu else "#367cd0", width=2), name=name)
            points = self.points[plot_key]
            self.curves[plot_key].setData([p[0] for p in points], [p[1] for p in points])

        def value(device, name):
            sample = self.readings.get((device, name), {})
            return f'{sample["value"]:.1f} {sample["unit"]}' if sample.get("status") == "ok" else "—"
        self.summary.setText(f'内存 {value("system", "memory_percent")}   ·   '
            f'GPU 频率 {value("gpu", "clock")}   ·   GPU 功耗 {value("gpu", "power")}')

    def show_report(self):
        if not self.latest_id:
            return
        store = Store()
        try:
            path = export(store, self.latest_id, data_dir() / f"report-{self.latest_id}.html")
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
        finally:
            store.close()

    def demo_sample(self, second):
        from .monitor import metric
        self.samples([
            metric("cpu", "utilization", 98 + math.sin(second / 8), "%", "演示"),
            metric("cpu", "reported_clock", 4200 + 35 * math.sin(second / 11), "MHz", "演示"),
            metric("gpu", "utilization", 97 + math.sin(second / 9), "%", "演示"),
            metric("gpu", "temperature", 65 + math.sin(second / 12), "℃", "演示"),
            metric("gpu", "clock", 2300 + 25 * math.sin(second / 10), "MHz", "演示"),
            metric("gpu", "power", 105 + math.sin(second / 7), "W", "演示"),
            metric("system", "memory_percent", 46, "%", "演示")], second)

    def closeEvent(self, event):
        self.timer.stop()
        if self.session:
            self.session.stop.set()
            deadline = time.monotonic() + 1.8
            while self.session.alive() and time.monotonic() < deadline:
                self.session.ping()
                self.session.drain()
                time.sleep(.05)
            self.session.close()
            self.session.pipe.close()
        if self.monitor:
            self.monitor.close()
            self.monitor.pipe.close()
        if hasattr(self, "scanner") and self.scanner.isRunning():
            self.scanner.wait(17000)
        event.accept()


def main(demo=False, preview=None):
    app = W.QApplication.instance() or W.QApplication(sys.argv[:1])
    # Resolve fonts through Qt on every platform, including headless previews.
    for folder in QtCore.QStandardPaths.standardLocations(QtCore.QStandardPaths.FontsLocation):
        for name in ("msyh.ttc", "msyhbd.ttc", "segoeui.ttf", "NotoSansCJK-Regular.ttc", "DejaVuSans.ttf"):
            font_file = Path(folder) / name
            if font_file.exists():
                QtGui.QFontDatabase.addApplicationFont(str(font_file))
    font = QtGui.QFont()
    font.setFamilies(["Microsoft YaHei", "Noto Sans CJK SC", "sans-serif"])
    app.setFont(font)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    pg.setConfigOptions(antialias=True)
    window = Window(demo)
    window.show()
    if demo and preview:
        def capture():
            path = Path(preview)
            path.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(path))
            window.close()
            app.quit()
        QtCore.QTimer.singleShot(400, capture)
    return app.exec()
