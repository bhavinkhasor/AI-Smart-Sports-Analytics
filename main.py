
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import threading
import queue
import time

import cv2
from PIL import Image, ImageTk

APP_NAME = "AI Smart Sports Analytics System"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolo11n.pt"

BG = "#0B1220"
PANEL = "#172334"
DARK = "#060B13"
ACCENT = "#35B8F2"
GREEN = "#17A673"
TEXT = "#F0F5FA"
MUTED = "#9BAEC4"


class SportsAnalyticsApp:

    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1400x850")
        self.root.minsize(1050, 650)
        self.root.configure(bg=BG)

        self.video_path = None
        self.cap = None
        self.preview_job = None
        self.photo = None

        self.events = queue.Queue()
        self.processing = False
        self.closing = False
        self.progress_fraction = 0

        self.mode = tk.StringVar(value="Detection")
        self.status = tk.StringVar(value="System ready")
        self.video_info = tk.StringVar(value="No video selected")

        self.cards = {}

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.poll_events)

    # ---------------- UI ----------------

    def build_ui(self):
        header = tk.Frame(self.root, bg=PANEL, height=75)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="AI SMART SPORTS ANALYTICS",
            bg=PANEL,
            fg=ACCENT,
            font=("Segoe UI", 22, "bold")
        ).pack(anchor="w", padx=22, pady=(8, 0))

        tk.Label(
            header,
            text="Tennis Cricket | Computer Vision | Match Intelligence",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10)
        ).pack(anchor="w", padx=25)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=14)

        # Left: video and analysis underneath
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        # Right: controls and statistics
        right = tk.Frame(body, bg=PANEL, width=280)
        right.pack(side="right", fill="y", padx=(14, 0))
        right.pack_propagate(False)

        # VIDEO PANEL
        video_panel = tk.Frame(left, bg=PANEL)
        video_panel.pack(fill="both", expand=True)

        tk.Label(
            video_panel,
            text="MATCH VIDEO",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", padx=14, pady=(12, 6))

        self.video_label = tk.Label(
            video_panel,
            text="Select a match video to begin",
            bg=DARK,
            fg=MUTED,
            font=("Segoe UI", 16)
        )
        self.video_label.pack(
            fill="both", expand=True, padx=14, pady=6
        )

        tk.Label(
            video_panel,
            textvariable=self.video_info,
            bg=PANEL,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 9)
        ).pack(fill="x", padx=14, pady=(4, 12))

        # ANALYSIS BELOW VIDEO
        analysis_panel = tk.Frame(left, bg=PANEL)
        analysis_panel.pack(fill="x", pady=(12, 0))

        tk.Label(
            analysis_panel,
            text="VIDEO ANALYSIS RESULTS",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", padx=14, pady=(12, 6))

        self.results = tk.Text(
            analysis_panel,
            height=9,
            bg=DARK,
            fg=TEXT,
            font=("Consolas", 10),
            relief="flat",
            wrap="word",
            padx=12,
            pady=10
        )
        self.results.pack(fill="x", padx=14, pady=(0, 8))
        self.results.insert(
            "1.0",
            "SYSTEM READY\n\n"
            "1. Select a match video.\n"
            "2. Choose Detection or Player Tracking.\n"
            "3. Run AI Analysis.\n\n"
            "The processed video and analysis report "
            "will appear on this page."
        )
        self.results.configure(state="disabled")

        self.progress = tk.Canvas(
            analysis_panel,
            height=8,
            bg="#29384B",
            highlightthickness=0
        )
        self.progress.pack(fill="x", padx=14)

        self.progress_rect = self.progress.create_rectangle(
            0, 0, 0, 8, fill=GREEN, width=0
        )
        self.progress.bind("<Configure>", self.draw_progress)

        tk.Label(
            analysis_panel,
            textvariable=self.status,
            bg=PANEL,
            fg=ACCENT,
            anchor="w",
            font=("Segoe UI", 10)
        ).pack(fill="x", padx=14, pady=10)

        # RIGHT SIDE CONTROLS
        tk.Label(
            right,
            text="MATCH CONTROL",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 15, "bold")
        ).pack(anchor="w", padx=16, pady=(20, 18))

        self.button(
            right, "Select Match Video",
            self.select_video
        ).pack(fill="x", padx=14, pady=5)

        tk.Label(
            right,
            text="ANALYSIS MODE",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=15, pady=(20, 7))

        tk.OptionMenu(
            right, self.mode,
            "Detection", "Player Tracking"
        ).pack(fill="x", padx=14)

        self.run_button = self.button(
            right, "Run AI Analysis",
            self.start_analysis, GREEN
        )
        self.run_button.pack(fill="x", padx=14, pady=(15, 5))

        self.button(
            right, "Open Processed Video",
            self.open_processed
        ).pack(fill="x", padx=14, pady=5)

        tk.Label(
            right,
            text="MATCH STATISTICS",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", padx=15, pady=(28, 12))

        self.add_stat(right, "Frames processed", "0")
        self.add_stat(right, "Person detections", "0")
        self.add_stat(right, "Ball detections", "0")
        self.add_stat(right, "Tracked IDs", "0")

        tk.Label(
            right,
            text="TECHNOLOGY\n\n"
                 "Python + Tkinter\n"
                 "OpenCV\n"
                 "Ultralytics YOLO\n"
                 "ByteTrack",
            bg=PANEL,
            fg=MUTED,
            justify="left",
            font=("Segoe UI", 10)
        ).pack(anchor="w", padx=16, pady=(25, 10))

    def button(self, parent, text, command, color="#1769AA"):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="white",
            activebackground="#2789C9",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=10
        )

    def add_stat(self, parent, title, value):
        box = tk.Frame(parent, bg="#101A2A", padx=12, pady=10)
        box.pack(fill="x", padx=14, pady=4)

        tk.Label(
            box, text=title,
            bg="#101A2A", fg=MUTED,
            font=("Segoe UI", 9)
        ).pack(anchor="w")

        label = tk.Label(
            box, text=value,
            bg="#101A2A", fg=TEXT,
            font=("Segoe UI", 17, "bold")
        )
        label.pack(anchor="w", pady=(3, 0))
        self.cards[title] = label

    def set_stat(self, title, value):
        if title in self.cards:
            self.cards[title].configure(text=str(value))

    def show_results(self, text):
        self.results.configure(state="normal")
        self.results.delete("1.0", "end")
        self.results.insert("1.0", text)
        self.results.configure(state="disabled")

    def draw_progress(self, event=None):
        width = self.progress.winfo_width()
        self.progress.coords(
            self.progress_rect,
            0, 0, width * self.progress_fraction, 8
        )

    def set_progress(self, percentage):
        self.progress_fraction = max(
            0, min(100, percentage)
        ) / 100
        self.draw_progress()

    # ---------------- VIDEO ----------------

    def stop_preview(self):
        if self.preview_job is not None:
            try:
                self.root.after_cancel(self.preview_job)
            except tk.TclError:
                pass
            self.preview_job = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def select_video(self):
        if self.processing:
            self.status.set("Wait for the current analysis to finish.")
            return

        path = filedialog.askopenfilename(
            title="Select Match Video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv"),
                ("All files", "*.*")
            ]
        )

        if path:
            self.open_video(path)

    def open_video(self, path):
        self.stop_preview()

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.stop_preview()
            self.status.set("Could not open the selected video.")
            return False

        self.video_path = Path(path)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.video_info.set(
            f"{self.video_path.name} | "
            f"{width}x{height} | {fps:.2f} FPS"
        )

        self.status.set("Video loaded")
        self.preview_frame()
        return True

    def preview_frame(self):
        if self.cap is None or self.closing:
            return

        ok, frame = self.cap.read()

        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()

        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            max_w = max(320, self.video_label.winfo_width() - 20)
            max_h = max(200, self.video_label.winfo_height() - 20)
            image.thumbnail((max_w, max_h))

            self.photo = ImageTk.PhotoImage(image)
            self.video_label.configure(image=self.photo, text="")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = max(15, min(int(1000 / fps), 100)) if fps > 0 else 40

        self.preview_job = self.root.after(
            delay, self.preview_frame
        )

    # ---------------- ANALYSIS ----------------

    def start_analysis(self):
        if self.processing:
            return

        if not self.video_path or not self.video_path.exists():
            self.status.set("Select a valid match video first.")
            self.show_results("ERROR\nPlease select a match video.")
            return

        mode = self.mode.get()
        input_path = self.video_path

        filename = (
            "tracked_match.mp4"
            if mode == "Player Tracking"
            else "detected_match.mp4"
        )
        output_path = OUTPUT_DIR / filename

        self.processing = True
        self.run_button.configure(state="disabled")
        self.stop_preview()
        self.set_progress(0)
        self.status.set("Starting AI analysis...")

        self.show_results(
            "ANALYSIS IN PROGRESS\n\n"
            f"Mode: {mode}\n"
            f"Input: {input_path.name}\n"
            "Loading YOLO model and processing frames..."
        )

        threading.Thread(
            target=self.process_video,
            args=(input_path, output_path, mode),
            daemon=True
        ).start()

    def process_video(self, input_path, output_path, mode):
        cap = None
        writer = None

        try:
            from ultralytics import YOLO

            model = YOLO(MODEL_NAME)
            cap = cv2.VideoCapture(str(input_path))

            if not cap.isOpened():
                raise RuntimeError("Could not open input video.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if width <= 0 or height <= 0:
                raise RuntimeError("Invalid video dimensions.")

            if fps <= 0:
                fps = 25

            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (width, height)
            )

            if not writer.isOpened():
                raise RuntimeError("Could not create output video.")

            frame_no = 0
            people = 0
            balls = 0
            ids_seen = set()
            start = time.time()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if mode == "Player Tracking":
                    result = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[0],
                        conf=0.25,
                        verbose=False
                    )[0]

                    if result.boxes is not None:
                        people += len(result.boxes)

                        if result.boxes.id is not None:
                            ids_seen.update(
                                result.boxes.id.int().cpu().tolist()
                            )
                else:
                    result = model.predict(
                        frame,
                        classes=[0, 32],
                        conf=0.25,
                        verbose=False
                    )[0]

                    if result.boxes is not None:
                        for cls in result.boxes.cls.cpu().tolist():
                            if int(cls) == 0:
                                people += 1
                            elif int(cls) == 32:
                                balls += 1

                writer.write(result.plot())
                frame_no += 1

                percent = frame_no / total * 100 if total > 0 else 0
                self.events.put((
                    "progress", percent, frame_no, total
                ))

            if frame_no == 0:
                raise RuntimeError("No frames were read.")

            self.events.put((
                "done", {
                    "mode": mode,
                    "frames": frame_no,
                    "people": people,
                    "balls": balls,
                    "ids": sorted(ids_seen),
                    "seconds": time.time() - start,
                    "output": str(output_path.resolve())
                }
            ))

        except Exception as exc:
            self.events.put(("error", str(exc)))

        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()

    # ---------------- RESULTS ----------------

    def poll_events(self):
        if self.closing:
            return

        try:
            while True:
                event = self.events.get_nowait()

                if event[0] == "progress":
                    _, percent, current, total = event
                    self.set_progress(percent)
                    self.status.set(
                        f"Processing frames: {current}/{total} "
                        f"({percent:.1f}%)"
                    )

                elif event[0] == "done":
                    report = event[1]
                    self.processing = False
                    self.run_button.configure(state="normal")
                    self.set_progress(100)

                    self.set_stat("Frames processed", report["frames"])
                    self.set_stat("Person detections", report["people"])
                    self.set_stat("Ball detections", report["balls"])
                    self.set_stat("Tracked IDs", len(report["ids"]))

                    if report["mode"] == "Player Tracking":
                        details = (
                            "PLAYER TRACKING REPORT\n"
                            "------------------------------------\n"
                            f"Frames processed: {report['frames']}\n"
                            f"Person detections across frames: {report['people']}\n"
                            f"Observed track IDs: {len(report['ids'])}\n"
                            f"Track IDs: {report['ids']}\n"
                        )
                    else:
                        details = (
                            "OBJECT DETECTION REPORT\n"
                            "------------------------------------\n"
                            f"Frames processed: {report['frames']}\n"
                            f"Person detections across frames: {report['people']}\n"
                            f"Ball detections across frames: {report['balls']}\n"
                        )

                    details += (
                        f"Processing time: {report['seconds']:.1f} seconds\n\n"
                        f"Output video:\n{report['output']}\n\n"
                        "The processed video is now shown above."
                    )

                    self.show_results(details)
                    self.status.set("Analysis completed")

                    if Path(report["output"]).exists():
                        self.open_video(report["output"])

                elif event[0] == "error":
                    self.processing = False
                    self.run_button.configure(state="normal")
                    self.status.set("Analysis failed")
                    self.show_results(
                        "ANALYSIS ERROR\n"
                        "------------------------------------\n"
                        + event[1]
                    )

        except queue.Empty:
            pass

        self.root.after(100, self.poll_events)

    def open_processed(self):
        if self.processing:
            self.status.set("Wait until analysis finishes.")
            return

        path = filedialog.askopenfilename(
            title="Open Processed Video",
            initialdir=str(OUTPUT_DIR),
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")]
        )

        if path:
            self.open_video(path)

    def close(self):
        self.closing = True
        self.stop_preview()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SportsAnalyticsApp(root)
    root.mainloop()
