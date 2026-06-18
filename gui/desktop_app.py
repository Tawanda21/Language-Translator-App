"""Desktop GUI for the Language Translator App.

This is a real desktop application built with Tkinter. It provides:
- a translation workspace for entering source text,
- language selection for English, German, and Afrikaans,
- a dataset browser powered by the streaming loader in data/load_datasets.py,
- a pluggable translation backend that can be replaced with a TensorFlow model later.

The current backend is deterministic and lightweight so the app runs immediately
without requiring a trained model file.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk


LANGUAGE_LABELS = {
    "en": "English",
    "de": "German",
    "af": "Afrikaans",
}

LANGUAGE_PAIRS = [
    ("en-de", "English ↔ German"),
    ("en-af", "English ↔ Afrikaans"),
    ("de-en", "German ↔ English"),
    ("af-en", "Afrikaans ↔ English"),
]


@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    backend_name: str


class DemoTranslationBackend:
    """Small fallback backend so the GUI is usable before a trained model exists."""

    name = "Demo backend"

    def __init__(self) -> None:
        self._phrase_table = {
            ("en", "de"): {
                "hello": "hallo",
                "good morning": "guten morgen",
                "how are you": "wie geht es dir",
                "thank you": "danke",
            },
            ("en", "af"): {
                "hello": "hallo",
                "good morning": "goeie more",
                "how are you": "hoe gaan dit",
                "thank you": "dankie",
            },
            ("de", "en"): {
                "hallo": "hello",
                "guten morgen": "good morning",
                "wie geht es dir": "how are you",
                "danke": "thank you",
            },
            ("af", "en"): {
                "hallo": "hello",
                "goeie more": "good morning",
                "hoe gaan dit": "how are you",
                "dankie": "thank you",
            },
        }

    def translate(self, text: str, source_language: str, target_language: str) -> TranslationResult:
        normalized = text.strip().lower()
        translated = self._phrase_table.get((source_language, target_language), {}).get(normalized)
        if translated is None:
            translated = f"[{target_language}] {text.strip()}"
        return TranslationResult(
            source_text=text.strip(),
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            backend_name=self.name,
        )


class DesktopTranslatorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Language Translator App")
        self.geometry("1180x760")
        self.minsize(1000, 680)

        self.backend = DemoTranslationBackend()
        self.dataset_queue: queue.Queue = queue.Queue()
        self.dataset_rows: list[dict[str, str]] = []
        self.dataset_thread: threading.Thread | None = None

        self.source_language = tk.StringVar(value="en")
        self.target_language = tk.StringVar(value="de")
        self.dataset_pair = tk.StringVar(value="en-de")
        self.dataset_split = tk.StringVar(value="train")
        self.max_examples = tk.StringVar(value="200")

        self._build_style()
        self._build_layout()
        self._seed_examples()
        self.after(150, self._drain_dataset_queue)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#101828")
        style.configure("Panel.TFrame", background="#182235")
        style.configure("Title.TLabel", background="#101828", foreground="#F9FAFB", font=("Segoe UI", 22, "bold"))
        style.configure("Body.TLabel", background="#101828", foreground="#C7D2FE", font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background="#182235", foreground="#F9FAFB", font=("Segoe UI", 12, "bold"))
        style.configure("PanelBody.TLabel", background="#182235", foreground="#D1D5DB", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))
        style.map("Accent.TButton", foreground=[("active", "#FFFFFF")], background=[("active", "#2563EB")])

    def _build_layout(self) -> None:
        container = ttk.Frame(self, style="App.TFrame", padding=16)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, style="App.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Language Translator App", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Desktop translator + dataset browser for English, German, and Afrikaans.",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(6, 12))

        content = ttk.Panedwindow(container, orient=tk.HORIZONTAL)
        content.pack(fill="both", expand=True)

        left = ttk.Frame(content, style="Panel.TFrame", padding=14)
        right = ttk.Frame(content, style="Panel.TFrame", padding=14)
        content.add(left, weight=2)
        content.add(right, weight=1)

        self._build_translation_panel(left)
        self._build_dataset_panel(right)

    def _build_translation_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Translate text", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Enter a sentence, choose a language pair, and click Translate.",
            style="PanelBody.TLabel",
        ).pack(anchor="w", pady=(4, 10))

        form = ttk.Frame(parent, style="Panel.TFrame")
        form.pack(fill="x", pady=(0, 10))
        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Source language", style="PanelBody.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(form, text="Target language", style="PanelBody.TLabel").grid(row=0, column=1, sticky="w", padx=(10, 0))

        source_box = ttk.Combobox(
            form,
            textvariable=self.source_language,
            values=["en", "de", "af"],
            state="readonly",
            width=18,
        )
        target_box = ttk.Combobox(
            form,
            textvariable=self.target_language,
            values=["en", "de", "af"],
            state="readonly",
            width=18,
        )
        source_box.grid(row=1, column=0, sticky="we", pady=(4, 10))
        target_box.grid(row=1, column=1, sticky="we", padx=(10, 0), pady=(4, 10))

        source_box.bind("<<ComboboxSelected>>", lambda _event: self._sync_pair_from_languages())
        target_box.bind("<<ComboboxSelected>>", lambda _event: self._sync_pair_from_languages())

        ttk.Label(parent, text="Input text", style="PanelBody.TLabel").pack(anchor="w")
        self.input_text = tk.Text(parent, height=8, wrap="word", font=("Segoe UI", 11))
        self.input_text.pack(fill="both", expand=False, pady=(4, 10))
        self.input_text.insert("1.0", "Hello world")

        action_row = ttk.Frame(parent, style="Panel.TFrame")
        action_row.pack(fill="x")

        ttk.Button(action_row, text="Translate", style="Accent.TButton", command=self._translate_text).pack(side="left")
        ttk.Button(action_row, text="Use selected dataset example", command=self._load_selected_example).pack(side="left", padx=8)
        ttk.Button(action_row, text="Clear", command=self._clear_translation).pack(side="left")

        ttk.Label(parent, text="Translated output", style="PanelBody.TLabel").pack(anchor="w", pady=(14, 4))
        self.output_text = tk.Text(parent, height=10, wrap="word", font=("Segoe UI", 11), state="disabled")
        self.output_text.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(parent, textvariable=self.status_var, style="PanelBody.TLabel").pack(anchor="w", pady=(8, 0))

    def _build_dataset_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Dataset browser", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Stream examples from English-German and English-Afrikaans datasets directly from the Hugging Face Hub.",
            style="PanelBody.TLabel",
            wraplength=340,
        ).pack(anchor="w", pady=(4, 10))

        controls = ttk.Frame(parent, style="Panel.TFrame")
        controls.pack(fill="x", pady=(0, 10))

        ttk.Label(controls, text="Pair", style="PanelBody.TLabel").grid(row=0, column=0, sticky="w")
        pair_box = ttk.Combobox(
            controls,
            textvariable=self.dataset_pair,
            values=[value for value, _label in LANGUAGE_PAIRS],
            state="readonly",
            width=18,
        )
        pair_box.grid(row=1, column=0, sticky="we", pady=(4, 8))
        ttk.Label(controls, text="Split", style="PanelBody.TLabel").grid(row=2, column=0, sticky="w")
        split_box = ttk.Combobox(
            controls,
            textvariable=self.dataset_split,
            values=["train", "validation", "test"],
            state="readonly",
            width=18,
        )
        split_box.grid(row=3, column=0, sticky="we", pady=(4, 8))
        ttk.Label(controls, text="Max examples", style="PanelBody.TLabel").grid(row=4, column=0, sticky="w")
        max_entry = ttk.Entry(controls, textvariable=self.max_examples, width=20)
        max_entry.grid(row=5, column=0, sticky="we", pady=(4, 8))

        buttons = ttk.Frame(parent, style="Panel.TFrame")
        buttons.pack(fill="x", pady=(0, 8))
        ttk.Button(buttons, text="Load stream", style="Accent.TButton", command=self._load_dataset_stream).pack(side="left")
        ttk.Button(buttons, text="Refresh list", command=self._refresh_dataset_list).pack(side="left", padx=8)

        self.dataset_status_var = tk.StringVar(value="No dataset loaded yet.")
        ttk.Label(parent, textvariable=self.dataset_status_var, style="PanelBody.TLabel", wraplength=340).pack(anchor="w", pady=(0, 8))

        list_frame = ttk.Frame(parent, style="Panel.TFrame")
        list_frame.pack(fill="both", expand=True)

        self.dataset_list = tk.Listbox(list_frame, height=18, font=("Segoe UI", 10))
        self.dataset_list.pack(side="left", fill="both", expand=True)
        self.dataset_list.bind("<<ListboxSelect>>", lambda _event: self._preview_dataset_example())
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.dataset_list.yview)
        scrollbar.pack(side="right", fill="y")
        self.dataset_list.config(yscrollcommand=scrollbar.set)

    def _sync_pair_from_languages(self) -> None:
        pair = f"{self.source_language.get()}-{self.target_language.get()}"
        if pair in {value for value, _label in LANGUAGE_PAIRS}:
            self.dataset_pair.set(pair)

    def _translate_text(self) -> None:
        source_text = self.input_text.get("1.0", "end").strip()
        if not source_text:
            messagebox.showinfo("Translate", "Please enter some text first.")
            return

        result = self.backend.translate(source_text, self.source_language.get(), self.target_language.get())
        self._set_output(
            f"Backend: {result.backend_name}\n"
            f"{LANGUAGE_LABELS.get(result.source_language, result.source_language)} -> {LANGUAGE_LABELS.get(result.target_language, result.target_language)}\n\n"
            f"{result.translated_text}"
        )
        self.status_var.set("Translation generated with the demo backend.")

    def _clear_translation(self) -> None:
        self.input_text.delete("1.0", "end")
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.configure(state="disabled")
        self.status_var.set("Cleared.")

    def _set_output(self, text: str) -> None:
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text)
        self.output_text.configure(state="disabled")

    def _seed_examples(self) -> None:
        self.dataset_rows = [
            {"pair": "en-de", "source": "Hello world", "target": "Hallo Welt"},
            {"pair": "en-af", "source": "Good morning", "target": "Goeie more"},
            {"pair": "de-en", "source": "Guten Morgen", "target": "Good morning"},
            {"pair": "af-en", "source": "Dankie", "target": "Thank you"},
        ]
        self._refresh_dataset_list()

    def _refresh_dataset_list(self) -> None:
        self.dataset_list.delete(0, tk.END)
        for row in self.dataset_rows:
            self.dataset_list.insert(tk.END, f"[{row['pair']}] {row['source']} → {row['target']}")

    def _preview_dataset_example(self) -> None:
        selection = self.dataset_list.curselection()
        if not selection:
            return
        row = self.dataset_rows[selection[0]]
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", row["source"])
        src_lang, tgt_lang = row["pair"].split("-")
        self.source_language.set(src_lang)
        self.target_language.set(tgt_lang)
        self._sync_pair_from_languages()
        self.status_var.set(f"Loaded example from {row['pair']}.")

    def _load_selected_example(self) -> None:
        self._preview_dataset_example()
        self._translate_text()

    def _load_dataset_stream(self) -> None:
        if self.dataset_thread and self.dataset_thread.is_alive():
            messagebox.showinfo("Load stream", "A dataset stream is already loading.")
            return

        try:
            max_examples = int(self.max_examples.get()) if self.max_examples.get().strip() else None
        except ValueError:
            messagebox.showerror("Load stream", "Max examples must be a whole number.")
            return

        pair = self.dataset_pair.get().strip()
        split = self.dataset_split.get().strip()
        self.dataset_status_var.set(f"Streaming {pair} ({split})...")
        self.dataset_thread = threading.Thread(
            target=self._stream_dataset_worker,
            args=(pair, split, max_examples),
            daemon=True,
        )
        self.dataset_thread.start()

    def _stream_dataset_worker(self, pair: str, split: str, max_examples: int | None) -> None:
        try:
            rows: list[dict[str, str]] = []
            stream_langpair = self._load_stream_langpair()
            for src, tgt in stream_langpair(pair, split, max_examples):
                rows.append({"pair": pair, "source": src, "target": tgt})
                if len(rows) >= 1000:
                    break
            self.dataset_queue.put(("ok", pair, split, rows))
        except Exception as exc:  # pragma: no cover - surfaced in GUI
            self.dataset_queue.put(("error", pair, split, str(exc)))

    def _load_stream_langpair(self):
        try:
            from data.load_datasets import stream_langpair
        except Exception as exc:  # pragma: no cover - surfaced in GUI
            raise RuntimeError(
                "Dataset streaming requires the `datasets` package and a reachable Hugging Face Hub dataset."
            ) from exc
        return stream_langpair

    def _drain_dataset_queue(self) -> None:
        try:
            while True:
                message = self.dataset_queue.get_nowait()
                kind = message[0]
                if kind == "ok":
                    _kind, pair, split, rows = message
                    self.dataset_rows = rows or self.dataset_rows
                    self._refresh_dataset_list()
                    self.dataset_status_var.set(f"Loaded {len(rows)} streamed examples for {pair} ({split}).")
                else:
                    _kind, pair, split, error_text = message
                    self.dataset_status_var.set(f"Failed to stream {pair} ({split}): {error_text}")
        except queue.Empty:
            pass
        self.after(150, self._drain_dataset_queue)


def main() -> None:
    app = DesktopTranslatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
