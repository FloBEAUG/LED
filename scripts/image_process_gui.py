#!/usr/bin/env python3
"""
LED-ICCV23 GUI - RAW Image Denoiser
A graphical user interface for the LED-ICCV23 RAW denoising pipeline.

Usage:
    python scripts/image_process_gui.py

Dependencies:
    - tkinter (built-in)
    - tkinterdnd2 (optional, for drag-and-drop)
    - led package (must be installed: pip install -e . --no-deps)
    - LED-ICCV23 conda environment activated
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Add the project root to path so we can import led
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from led.utils.options import yaml_load


class LEDDenoiserGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LED-ICCV23 - RAW Denoiser")
        self.root.geometry("900x700")
        self.root.minsize(700, 500)

        # Variables
        self.pretrained_network_var = tk.StringVar()
        self.data_path_var = tk.StringVar()
        self.save_path_var = tk.StringVar()
        self.network_options_var = tk.StringVar(value="options/base/network_g/unet.yaml")
        self.ratio_var = tk.DoubleVar(value=1.0)
        self.bps_var = tk.IntVar(value=16)
        self.led_var = tk.BooleanVar(value=False)
        self.cpu_var = tk.BooleanVar(value=False)
        self.no_denoise_var = tk.BooleanVar(value=False)
        self.margin_var = tk.IntVar(value=32)

        self.build_ui()

    def build_ui(self):
        # Main container with scrollbar
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create scrollable frame
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.scrollable_frame.columnconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(
            self.scrollable_frame,
            text="LED-ICCV23 RAW Denoiser",
            font=("TkDefaultFont", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Pretrained network
        ttk.Label(self.scrollable_frame, text="Pretrained Network:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.pretrained_entry = ttk.Entry(
            self.scrollable_frame, textvariable=self.pretrained_network_var, width=60
        )
        self.pretrained_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(
            self.scrollable_frame, text="Browse...",
            command=self.browse_pretrained
        ).grid(row=1, column=2, padx=5, pady=5)

        # Data path
        ttk.Label(self.scrollable_frame, text="Input Folder (RAW images):").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.data_entry = ttk.Entry(
            self.scrollable_frame, textvariable=self.data_path_var, width=60
        )
        self.data_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(
            self.scrollable_frame, text="Browse...",
            command=self.browse_data_path
        ).grid(row=2, column=2, padx=5, pady=5)

        # Save path
        ttk.Label(self.scrollable_frame, text="Output Folder:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.save_entry = ttk.Entry(
            self.scrollable_frame, textvariable=self.save_path_var, width=60
        )
        self.save_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(
            self.scrollable_frame, text="Browse...",
            command=self.browse_save_path
        ).grid(row=3, column=2, padx=5, pady=5)

        # Network options
        ttk.Label(self.scrollable_frame, text="Network Options YAML:").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        self.network_entry = ttk.Entry(
            self.scrollable_frame, textvariable=self.network_options_var, width=60
        )
        self.network_entry.grid(row=4, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(
            self.scrollable_frame, text="Browse...",
            command=self.browse_network_options
        ).grid(row=4, column=2, padx=5, pady=5)

        # Ratio / DGain
        ttk.Label(self.scrollable_frame, text="Ratio / DGain:").grid(
            row=5, column=0, sticky=tk.W, pady=5
        )
        ratio_frame = ttk.Frame(self.scrollable_frame)
        ratio_frame.grid(row=5, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Spinbox(
            ratio_frame, from_=0.01, to=10000.0, increment=0.1,
            textvariable=self.ratio_var, width=15
        ).pack(side=tk.LEFT)
        ttk.Label(ratio_frame, text="(additional digital gain)").pack(side=tk.LEFT, padx=5)

        # BPS
        ttk.Label(self.scrollable_frame, text="Bit Depth (BPS):").grid(
            row=6, column=0, sticky=tk.W, pady=5
        )
        ttk.Spinbox(
            self.scrollable_frame, from_=8, to=16, increment=1,
            textvariable=self.bps_var, width=15
        ).grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)

        # Options checkboxes
        options_frame = ttk.LabelFrame(self.scrollable_frame, text="Options", padding="10")
        options_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        options_frame.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            options_frame, text="Force CPU (disable GPU)",
            variable=self.cpu_var
        ).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Checkbutton(
            options_frame, text="No Denoise (passthrough only)",
            variable=self.no_denoise_var
        ).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Checkbutton(
            options_frame, text="LED Fine-tuned Checkpoint",
            variable=self.led_var
        ).grid(row=0, column=2, sticky=tk.W, padx=5)

        # Margin
        ttk.Label(options_frame, text="Tile Margin (GPU only):").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        ttk.Spinbox(
            options_frame, from_=0, to=256, increment=1,
            textvariable=self.margin_var, width=15
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Run button
        self.run_button = ttk.Button(
            self.scrollable_frame, text="Start Processing",
            command=self.run_processing
        )
        self.run_button.grid(row=8, column=0, columnspan=3, pady=20)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.scrollable_frame, variable=self.progress_var,
            maximum=100, length=400
        )
        self.progress_bar.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            self.scrollable_frame, textvariable=self.status_var
        )
        self.status_label.grid(row=10, column=0, columnspan=3, pady=5)

        # Log area
        ttk.Label(self.scrollable_frame, text="Log:").grid(
            row=11, column=0, sticky=tk.W, pady=(10, 5)
        )
        self.log_text = tk.Text(
            self.scrollable_frame, height=10, width=80,
            wrap=tk.WORD
        )
        self.log_text.grid(row=12, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.scrollable_frame.rowconfigure(12, weight=1)

        # Scrollbar for log
        log_scroll = ttk.Scrollbar(
            self.scrollable_frame, orient="vertical",
            command=self.log_text.yview
        )
        log_scroll.grid(row=12, column=3, sticky=(tk.N, tk.S), pady=5)
        self.log_text.configure(yscrollcommand=log_scroll.set)

    def browse_pretrained(self):
        filename = filedialog.askopenfilename(
            title="Select Pretrained Network",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            self.pretrained_network_var.set(filename)

    def browse_data_path(self):
        directory = filedialog.askdirectory(title="Select Folder with RAW Images")
        if directory:
            self.data_path_var.set(directory)

    def browse_save_path(self):
        directory = filedialog.askdirectory(title="Select Output Folder")
        if directory:
            self.save_path_var.set(directory)

    def browse_network_options(self):
        filename = filedialog.askopenfilename(
            title="Select Network Options YAML",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            self.network_options_var.set(filename)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def run_processing(self):
        # Validate inputs
        if not self.pretrained_network_var.get():
            messagebox.showerror("Error", "Please select a pretrained network file.")
            return
        if not self.data_path_var.get():
            messagebox.showerror("Error", "Please select a data folder.")
            return
        if not self.save_path_var.get():
            messagebox.showerror("Error", "Please select a save folder.")
            return

        # Disable run button during processing
        self.run_button.configure(state="disabled")
        self.log_text.delete(1.0, tk.END)
        self.progress_var.set(0)

        # Run in separate thread to keep GUI responsive
        thread = threading.Thread(target=self.run_processing_thread, daemon=True)
        thread.start()

    def run_processing_thread(self):
        import subprocess
        import sys

        # Build command
        cmd = [
            sys.executable,
            "scripts/image_process.py",
            "-p", self.pretrained_network_var.get(),
            "--data_path", self.data_path_var.get(),
            "--save_path", self.save_path_var.get(),
            "--network_options", self.network_options_var.get(),
            "--ratio", str(self.ratio_var.get()),
            "--bps", str(self.bps_var.get()),
        ]

        if args.led:
            cmd.append("--led")
        if args.cpu:
            cmd.append("--cpu")
        if args.no_denoise:
            cmd.append("--no-denoise")

        cmd.extend(["--ratio", str(self.ratio_var.get())])
        cmd.extend(["--bps", str(self.bps_var.get())])
        cmd.extend(["--margin", str(self.margin_var.get())])

        self.log(f"Running: {' '.join(cmd)}\n")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            for line in process.stdout:
                line = line.strip()
                if line:
                    self.log(line)
                    # Parse progress from tqdm output
                    if "it/s" in line:
                        try:
                            # Parse percentage from tqdm output like: "50%|████     | 5/10 [00:10<00:10,  2.0s/it]"
                            import re
                            match = re.search(r'(\d+)%', line)
                            if match:
                                pct = float(match.group(1))
                                self.progress_var.set(pct)
                        except Exception:
                            pass

            process.wait()
            if process.returncode == 0:
                self.log("Processing completed successfully!")
                self.status_var.set("Done")
                self.root.after(0, lambda: messagebox.showinfo("Success", "Processing completed successfully!"))
            else:
                self.log(f"Process exited with code {process.returncode}")
                self.status_var.set("Error")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Process exited with code {process.returncode}"))
        except Exception as e:
            self.log(f"Error: {e}")
            self.status_var.set("Error")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.run_button.configure(state="normal"))
            self.root.after(0, lambda: self.progress_var.set(100))


def main():
    root = tk.Tk()
    app = LEDDenoiseGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
