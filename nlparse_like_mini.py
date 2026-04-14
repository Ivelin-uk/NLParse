#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLParse-like educational prototype for macOS/Windows/Linux.
Purpose:
- define morphology categories
- define lexicon
- define grammar rules
- parse simple Bulgarian sentences in the style of the exercises
- visualize syntactic and semantic trees
- export results to text files

This is not a full clone of NLParse.
It is a lightweight study tool that follows the exercise logic:
morphological categories -> productions -> frames -> lexicon -> parse.
"""

from __future__ import annotations

import json
import re
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk, messagebox, filedialog
from typing import Dict, List, Optional, Tuple


# =========================================================
# Data model
# =========================================================

@dataclass
class TreeNode:
    label: str
    children: List["TreeNode"] = field(default_factory=list)

    def pretty(self, level: int = 0) -> str:
        indent = "  " * level
        out = [f"{indent}{self.label}"]
        for child in self.children:
            out.append(child.pretty(level + 1))
        return "\n".join(out)


@dataclass
class KnowledgeBase:
    categories: List[str] = field(default_factory=lambda: [
        "глагол",
        "съществително",
        "прилагателно",
        "наречие",
        "предлог",
        "собствено име",
    ])
    lexicon: Dict[str, str] = field(default_factory=dict)
    grammar_rules: List[str] = field(default_factory=lambda: [
        "S -> Група* ГрГл Група*",
        "Група -> ГрСщ",
        "Група -> ГрПрил",
        "Група -> ГрНар",
        "Група -> ГрПр",
        "ГрГл -> глагол",
        "ГрСщ -> прилагателно* съществително",
        "ГрСщ -> собствено име",
        "ГрПрил -> прилагателно",
        "ГрНар -> наречие",
        "ГрПр -> предлог ГрСщ",
    ])
    frames: List[str] = field(default_factory=lambda: [
        "Обекти",
        "Действия",
        "Свойства",
        "Отношения",
        "собствено име",
    ])


# =========================================================
# Parsing engine
# =========================================================

class SimpleBulgarianParser:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.strip().lower()
        text = re.sub(r"[^\w\sа-я\-]", "", text, flags=re.IGNORECASE)
        return [t for t in text.split() if t]

    def tag_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        tagged = []
        for tok in tokens:
            if tok not in self.kb.lexicon:
                raise ValueError(f"Думата '{tok}' липсва в речника.")
            tagged.append((tok, self.kb.lexicon[tok]))
        return tagged

    def parse_sentence(self, text: str) -> Tuple[TreeNode, TreeNode]:
        tokens = self.tokenize(text)
        tagged = self.tag_tokens(tokens)

        syntax = self._build_syntax_tree(tagged)
        semantic = self._build_semantic_tree(tagged)
        return syntax, semantic

    def _build_syntax_tree(self, tagged: List[Tuple[str, str]]) -> TreeNode:
        # Educational heuristic parser that follows the exercise structure.
        # S -> left groups + verb group + right groups
        i = 0
        left_groups: List[TreeNode] = []
        right_groups: List[TreeNode] = []

        # collect groups until first verb
        while i < len(tagged) and tagged[i][1] != "глагол":
            group, i = self._consume_group(tagged, i)
            left_groups.append(group)

        if i >= len(tagged) or tagged[i][1] != "глагол":
            raise ValueError("Не е намерен глаголна група в изречението.")

        verb_word = tagged[i][0]
        verb_group = TreeNode("ГрГл", [TreeNode(f"глагол → {verb_word}")])
        i += 1

        while i < len(tagged):
            group, i = self._consume_group(tagged, i)
            right_groups.append(group)

        return TreeNode("S", left_groups + [verb_group] + right_groups)

    def _consume_group(self, tagged: List[Tuple[str, str]], i: int) -> Tuple[TreeNode, int]:
        if i >= len(tagged):
            raise ValueError("Неочакван край при разбор на група.")

        word, cat = tagged[i]

        if cat == "наречие":
            return TreeNode("Група", [TreeNode("ГрНар", [TreeNode(f"наречие → {word}")])]), i + 1

        if cat == "предлог":
            j = i + 1
            noun_children: List[TreeNode] = []

            while j < len(tagged) and tagged[j][1] == "прилагателно":
                noun_children.append(TreeNode(f"прилагателно → {tagged[j][0]}"))
                j += 1

            if j < len(tagged) and tagged[j][1] in ("съществително", "собствено име"):
                noun_children.append(TreeNode(f"{tagged[j][1]} → {tagged[j][0]}"))
                j += 1
            else:
                raise ValueError(f"След предлог '{word}' се очаква съществителна група.")

            noun_group = TreeNode("ГрСщ", noun_children)
            prep_group = TreeNode("ГрПр", [TreeNode(f"предлог → {word}"), noun_group])
            return TreeNode("Група", [prep_group]), j

        if cat in ("прилагателно", "съществително", "собствено име"):
            j = i
            noun_children: List[TreeNode] = []

            while j < len(tagged) and tagged[j][1] == "прилагателно":
                noun_children.append(TreeNode(f"прилагателно → {tagged[j][0]}"))
                j += 1

            if j < len(tagged) and tagged[j][1] in ("съществително", "собствено име"):
                noun_children.append(TreeNode(f"{tagged[j][1]} → {tagged[j][0]}"))
                j += 1
                return TreeNode("Група", [TreeNode("ГрСщ", noun_children)]), j

        raise ValueError(f"Не мога да построя група, започваща с '{word}' ({cat}).")

    def _build_semantic_tree(self, tagged: List[Tuple[str, str]]) -> TreeNode:
        objects: List[TreeNode] = []
        actions: List[TreeNode] = []
        properties: List[TreeNode] = []
        relations: List[TreeNode] = []

        i = 0
        pending_adjectives: List[str] = []

        while i < len(tagged):
            word, cat = tagged[i]

            if cat == "прилагателно":
                pending_adjectives.append(word)
                i += 1
                continue

            if cat == "съществително":
                obj_children = [TreeNode(f"име → {word}")]
                for adj in pending_adjectives:
                    obj_children.append(TreeNode(f"свойство → {adj}"))
                    properties.append(TreeNode(adj))
                pending_adjectives.clear()
                objects.append(TreeNode("Обект", obj_children))
                i += 1
                continue

            if cat == "собствено име":
                objects.append(TreeNode("Обект", [TreeNode("тип → собствено име"), TreeNode(f"име → {word}")]))
                i += 1
                continue

            if cat == "глагол":
                actions.append(TreeNode("Действие", [TreeNode(f"име → {word}")]))
                i += 1
                continue

            if cat == "наречие":
                relations.append(TreeNode("Обстоятелство", [TreeNode(f"име → {word}")]))
                i += 1
                continue

            if cat == "предлог":
                if i + 1 >= len(tagged):
                    raise ValueError(f"Предлогът '{word}' е без допълнение.")
                # simplified semantic function of preposition
                role = self._guess_preposition_role(word)
                j = i + 1
                attrs = []
                while j < len(tagged) and tagged[j][1] == "прилагателно":
                    attrs.append(tagged[j][0])
                    j += 1
                if j < len(tagged) and tagged[j][1] in ("съществително", "собствено име"):
                    target = tagged[j][0]
                    rel_children = [TreeNode(f"слот → {target}")]
                    for a in attrs:
                        rel_children.append(TreeNode(f"свойство → {a}"))
                    relations.append(TreeNode(role, rel_children))
                    i = j + 1
                    continue
                else:
                    raise ValueError(f"След предлог '{word}' липсва съществителна група.")

            i += 1

        return TreeNode("СемантичноДърво", [
            TreeNode("Действия", actions),
            TreeNode("Обекти", objects),
            TreeNode("Свойства", properties),
            TreeNode("Отношения", relations),
        ])

    @staticmethod
    def _guess_preposition_role(prep: str) -> str:
        mapping = {
            "в": "Време/Място",
            "на": "Място/Притежание",
            "от": "Източник/Продължителност",
            "през": "Време",
            "над": "ПространственоОтношение",
            "под": "ПространственоОтношение",
            "до": "Посока/Граница",
            "към": "Посока",
        }
        return mapping.get(prep, "Отношение")


# =========================================================
# GUI
# =========================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mini NLParse за macOS")
        self.geometry("1280x820")
        self.kb = KnowledgeBase()
        self._load_demo_data()
        self.parser = SimpleBulgarianParser(self.kb)
        self._build_ui()

    def _load_demo_data(self) -> None:
        # Demo words from exercises 2, 3, 4 and coursework-like examples
        demo = {
            "вали": "глагол",
            "валя": "глагол",
            "удари": "глагол",
            "лае": "глагол",
            "лежи": "глагол",
            "чете": "глагол",
            "носи": "глагол",
            "спи": "глагол",
            "изследва": "глагол",

            "дъжд": "съществително",
            "сняг": "съществително",
            "градушка": "съществително",
            "околности": "съществително",
            "куче": "съществително",
            "двора": "съществително",
            "понеделник": "съществително",
            "векове": "съществително",
            "земята": "съществително",
            "година": "съществително",
            "грохотът": "съществително",
            "водата": "съществително",
            "котка": "съществително",
            "перваза": "съществително",
            "книга": "съществително",
            "библиотеката": "съществително",
            "вятър": "съществително",
            "листата": "съществително",
            "пътеката": "съществително",
            "терасата": "съществително",

            "силен": "прилагателно",
            "пухкав": "прилагателно",
            "пухкавата": "прилагателно",
            "едра": "прилагателно",
            "близките": "прилагателно",
            "кафявото": "прилагателно",
            "пороен": "прилагателно",
            "плодородна": "прилагателно",
            "интересна": "прилагателно",
            "силният": "прилагателно",

            "днес": "наречие",
            "навън": "наречие",
            "вчера": "наречие",
            "силно": "наречие",
            "надалеч": "наречие",

            "в": "предлог",
            "на": "предлог",
            "от": "предлог",
            "през": "предлог",
            "над": "предлог",

            "лора": "собствено име",
            "мария": "собствено име",
            "софия": "собствено име",
            "петър": "собствено име",
            "мартин": "собствено име",
            "канада": "собствено име",
        }
        self.kb.lexicon.update(demo)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")

        # Parse tab
        parse_tab = ttk.Frame(notebook)
        lexicon_tab = ttk.Frame(notebook)
        grammar_tab = ttk.Frame(notebook)
        frames_tab = ttk.Frame(notebook)

        notebook.add(parse_tab, text="Парсинг")
        notebook.add(lexicon_tab, text="Речник")
        notebook.add(grammar_tab, text="Продукции")
        notebook.add(frames_tab, text="Фреймове")

        self._build_parse_tab(parse_tab)
        self._build_lexicon_tab(lexicon_tab)
        self._build_grammar_tab(grammar_tab)
        self._build_frames_tab(frames_tab)

    def _build_parse_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        top = ttk.Frame(parent)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Изречение:").grid(row=0, column=0, sticky="w")
        self.sentence_var = tk.StringVar(value="В сряда Мария чете интересна книга в библиотеката.")
        sentence_entry = ttk.Entry(top, textvariable=self.sentence_var)
        sentence_entry.grid(row=0, column=1, sticky="ew", padx=8)

        ttk.Button(top, text="Анализирай", command=self.analyze_sentence).grid(row=0, column=2, padx=6)
        ttk.Button(top, text="Експорт", command=self.export_analysis).grid(row=0, column=3, padx=6)

        samples = ttk.Frame(parent)
        samples.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10)
        ttk.Label(samples, text="Примери:").pack(side="left")

        example_sentences = [
            "Вали силен пухкав сняг.",
            "Кафявото куче лае силно на двора.",
            "На перваза лежи пухкавата котка Лора.",
            "В сряда Мария чете интересна книга в библиотеката.",
            "Силният вятър носи листата над пътеката.",
        ]
        for s in example_sentences:
            ttk.Button(samples, text=s, command=lambda x=s: self.sentence_var.set(x)).pack(side="left", padx=4, pady=4)

        syntax_frame = ttk.LabelFrame(parent, text="Синтактично дърво")
        syntax_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        syntax_frame.rowconfigure(0, weight=1)
        syntax_frame.columnconfigure(0, weight=1)

        self.syntax_text = tk.Text(syntax_frame, wrap="word", font=("Menlo", 12))
        self.syntax_text.grid(row=0, column=0, sticky="nsew")
        ttk.Scrollbar(syntax_frame, orient="vertical", command=self.syntax_text.yview).grid(row=0, column=1, sticky="ns")
        self.syntax_text.configure(yscrollcommand=lambda *args: None)

        semantic_frame = ttk.LabelFrame(parent, text="Семантично дърво")
        semantic_frame.grid(row=2, column=1, sticky="nsew", padx=10, pady=10)
        semantic_frame.rowconfigure(0, weight=1)
        semantic_frame.columnconfigure(0, weight=1)

        self.semantic_text = tk.Text(semantic_frame, wrap="word", font=("Menlo", 12))
        self.semantic_text.grid(row=0, column=0, sticky="nsew")

    def _build_lexicon_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.Frame(parent)
        controls.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.word_var = tk.StringVar()
        self.cat_var = tk.StringVar(value=self.kb.categories[0])

        ttk.Label(controls, text="Дума:").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.word_var, width=25).grid(row=0, column=1, padx=6)

        ttk.Label(controls, text="Категория:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(controls, textvariable=self.cat_var, values=self.kb.categories, width=20, state="readonly").grid(row=0, column=3, padx=6)

        ttk.Button(controls, text="Добави/Обнови", command=self.add_word).grid(row=0, column=4, padx=6)
        ttk.Button(controls, text="Запази JSON", command=self.save_kb).grid(row=0, column=5, padx=6)
        ttk.Button(controls, text="Зареди JSON", command=self.load_kb).grid(row=0, column=6, padx=6)

        self.lexicon_text = tk.Text(parent, wrap="word", font=("Menlo", 12))
        self.lexicon_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.refresh_lexicon_view()

    def _build_grammar_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.Frame(parent)
        controls.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.rule_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.rule_var, width=70).grid(row=0, column=0, padx=6)
        ttk.Button(controls, text="Добави правило", command=self.add_rule).grid(row=0, column=1, padx=6)

        self.grammar_text = tk.Text(parent, wrap="word", font=("Menlo", 12))
        self.grammar_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.refresh_grammar_view()

    def _build_frames_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.Frame(parent)
        controls.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.frame_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.frame_var, width=50).grid(row=0, column=0, padx=6)
        ttk.Button(controls, text="Добави фрейм", command=self.add_frame).grid(row=0, column=1, padx=6)

        self.frames_text = tk.Text(parent, wrap="word", font=("Menlo", 12))
        self.frames_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.refresh_frames_view()

    # --------------------------
    # Actions
    # --------------------------

    def analyze_sentence(self) -> None:
        sentence = self.sentence_var.get().strip()
        if not sentence:
            messagebox.showwarning("Липсва текст", "Въведи изречение.")
            return

        try:
            self.parser = SimpleBulgarianParser(self.kb)
            syntax, semantic = self.parser.parse_sentence(sentence)
        except Exception as e:
            messagebox.showerror("Грешка при анализ", str(e))
            return

        self.syntax_text.delete("1.0", tk.END)
        self.syntax_text.insert(tk.END, syntax.pretty())

        self.semantic_text.delete("1.0", tk.END)
        self.semantic_text.insert(tk.END, semantic.pretty())

    def export_analysis(self) -> None:
        sentence = self.sentence_var.get().strip()
        syntax = self.syntax_text.get("1.0", tk.END).strip()
        semantic = self.semantic_text.get("1.0", tk.END).strip()

        if not syntax or not semantic:
            messagebox.showwarning("Няма анализ", "Първо анализирай изречението.")
            return

        path = filedialog.asksaveasfilename(
            title="Запази анализ",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if not path:
            return

        content = []
        content.append("ИЗРЕЧЕНИЕ")
        content.append(sentence)
        content.append("")
        content.append("СИНТАКТИЧНО ДЪРВО")
        content.append(syntax)
        content.append("")
        content.append("СЕМАНТИЧНО ДЪРВО")
        content.append(semantic)
        Path(path).write_text("\n".join(content), encoding="utf-8")
        messagebox.showinfo("Готово", f"Анализът е записан в:\n{path}")

    def add_word(self) -> None:
        word = self.word_var.get().strip().lower()
        cat = self.cat_var.get().strip()
        if not word or not cat:
            return
        self.kb.lexicon[word] = cat
        self.word_var.set("")
        self.refresh_lexicon_view()

    def add_rule(self) -> None:
        rule = self.rule_var.get().strip()
        if not rule:
            return
        self.kb.grammar_rules.append(rule)
        self.rule_var.set("")
        self.refresh_grammar_view()

    def add_frame(self) -> None:
        frame = self.frame_var.get().strip()
        if not frame:
            return
        self.kb.frames.append(frame)
        self.frame_var.set("")
        self.refresh_frames_view()

    def refresh_lexicon_view(self) -> None:
        self.lexicon_text.delete("1.0", tk.END)
        self.lexicon_text.insert(tk.END, "МОРФОЛОГИЧНИ КАТЕГОРИИ:\n")
        for c in self.kb.categories:
            self.lexicon_text.insert(tk.END, f"- {c}\n")

        self.lexicon_text.insert(tk.END, "\nРЕЧНИК:\n")
        for word in sorted(self.kb.lexicon.keys()):
            self.lexicon_text.insert(tk.END, f"{word} → {self.kb.lexicon[word]}\n")

    def refresh_grammar_view(self) -> None:
        self.grammar_text.delete("1.0", tk.END)
        for rule in self.kb.grammar_rules:
            self.grammar_text.insert(tk.END, rule + "\n")

    def refresh_frames_view(self) -> None:
        self.frames_text.delete("1.0", tk.END)
        for frame in self.kb.frames:
            self.frames_text.insert(tk.END, f"- {frame}\n")

    def save_kb(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Запази база знания",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return

        data = {
            "categories": self.kb.categories,
            "lexicon": self.kb.lexicon,
            "grammar_rules": self.kb.grammar_rules,
            "frames": self.kb.frames,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("Готово", f"Базата знания е запазена в:\n{path}")

    def load_kb(self) -> None:
        path = filedialog.askopenfilename(
            title="Зареди база знания",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.kb.categories = data.get("categories", self.kb.categories)
        self.kb.lexicon = data.get("lexicon", self.kb.lexicon)
        self.kb.grammar_rules = data.get("grammar_rules", self.kb.grammar_rules)
        self.kb.frames = data.get("frames", self.kb.frames)

        self.refresh_lexicon_view()
        self.refresh_grammar_view()
        self.refresh_frames_view()
        messagebox.showinfo("Готово", f"Базата знания е заредена от:\n{path}")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
