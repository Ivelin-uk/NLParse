#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Mini NLParse-like parser for Bulgarian.
This version is more reliable for coursework-style sentences.
"""

from __future__ import annotations

import json
import re
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk, messagebox, filedialog
from typing import Dict, List, Optional, Tuple


@dataclass
class TreeNode:
    label: str
    children: List["TreeNode"] = field(default_factory=list)

    def pretty(self, level: int = 0) -> str:
        indent = "  " * level
        lines = [f"{indent}{self.label}"]
        for child in self.children:
            lines.append(child.pretty(level + 1))
        return "\n".join(lines)


@dataclass
class KnowledgeBase:
    categories: List[str] = field(default_factory=lambda: [
        "глагол",
        "съществително",
        "прилагателно",
        "наречие",
        "предлог",
        "собствено име",
        "съюз",
        "частица",
    ])
    lexicon: Dict[str, str] = field(default_factory=dict)
    grammar_rules: List[str] = field(default_factory=lambda: [
        "S -> Група* ГрГл Група*",
        "Група -> ГрСщ",
        "Група -> ГрПрил",
        "Група -> ГрНар",
        "Група -> ГрПр",
        "ГрГл -> Глагол",
        "ГрСщ -> Прилагателно* Съществително",
        "ГрСщ -> собствено име",
        "ГрПрил -> Прилагателно",
        "ГрНар -> Наречие",
        "ГрПр -> Предлог ГрСщ",
    ])
    frames: List[str] = field(default_factory=lambda: [
        "Обекти",
        "Действия",
        "Свойства",
        "Отношения",
        "собствено име",
    ])


class SimpleBulgarianParser:
    PREPOSITIONS = {
        "в", "на", "от", "за", "с", "със", "по", "до", "към", "през",
        "под", "над", "без", "между", "сред", "след", "при", "около",
        "покрай", "срещу", "чрез", "относно", "освен"
    }
    CONJUNCTIONS = {
        "и", "или", "но", "а", "ни", "нито", "обаче", "ала", "ама",
        "че", "защото", "щом", "ако", "макар", "дали", "докато"
    }
    PARTICLES = {"не", "да", "ли", "ще", "нека", "май", "дори", "даже", "нали"}
    COMMON_ADVERBS = {
        "днес", "вчера", "утре", "навън", "вътре", "горе", "долу",
        "много", "малко", "бързо", "тихо", "тук", "там", "сега",
        "после", "отново", "вече", "още", "винаги", "никога",
        "веднага", "надалеч", "наблизо", "отвън", "рано", "късно",
        "силно", "весело"
    }
    TEMPORAL_NOUNS = {
        "понеделник", "вторник", "сряда", "четвъртък", "петък",
        "събота", "неделя", "сутринта", "вечерта", "обед", "нощта",
        "пролетта", "лятото", "есента", "зимата", "годината",
        "месеца", "деня", "седмицата", "векове"
    }
    LOCATION_NOUNS = {
        "библиотеката", "терасата", "двора", "парка", "градината",
        "реката", "пътеката", "стола", "софия", "канада"
    }
    VERB_LIKE = {
        "вали", "валя", "удари", "лае", "лежи", "чете", "носи", "спи",
        "изследва", "играе", "расте", "люлее", "пише", "е"
    }

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.strip().lower()
        text = re.sub(r"[^\w\sа-я\-]", "", text, flags=re.IGNORECASE)
        return [t for t in text.split() if t]

    @classmethod
    def guess_category(cls, word: str) -> str:
        w = word.lower()

        if w in cls.PREPOSITIONS:
            return "предлог"
        if w in cls.CONJUNCTIONS:
            return "съюз"
        if w in cls.PARTICLES:
            return "частица"
        if w in cls.COMMON_ADVERBS:
            return "наречие"
        if w in cls.VERB_LIKE:
            return "глагол"

        if word[:1].isupper():
            return "собствено име"

        if w.endswith(("ият", "ия", "ска", "ско", "ски", "ова", "ево", "ен", "на", "но")) and len(w) >= 5:
            return "прилагателно"

        if w.endswith(("ът", "ят", "ата", "ето", "ото", "ите", "ове", "ци", "ища", "ета", "та")):
            return "съществително"

        if w.endswith(("ам", "ям", "ем", "им", "ах", "ях", "ех", "их", "ава", "ява", "еше", "и", "е")) and len(w) >= 4:
            if not w.endswith(("ние", "тие", "ие")):
                return "глагол"

        if w.endswith("о") and len(w) >= 4:
            return "наречие"

        return "съществително"

    def tag_tokens(self, tokens: List[str], unknown_handler: Optional[callable] = None) -> List[Tuple[str, str]]:
        tagged = []
        for tok in tokens:
            if tok not in self.kb.lexicon:
                guessed = self.guess_category(tok)
                if unknown_handler is not None:
                    cat = unknown_handler(tok, guessed)
                    if cat is None:
                        raise ValueError("Анализът е прекъснат от потребителя.")
                else:
                    cat = guessed
                self.kb.lexicon[tok] = cat
            tagged.append((tok, self.kb.lexicon[tok]))
        return tagged

    def parse_sentence(self, text: str, unknown_handler: Optional[callable] = None) -> Tuple[TreeNode, TreeNode]:
        tokens = self.tokenize(text)
        tagged = self.tag_tokens(tokens, unknown_handler=unknown_handler)
        return self._build_syntax_tree(tagged), self._build_semantic_tree(tagged)

    def _build_syntax_tree(self, tagged: List[Tuple[str, str]]) -> TreeNode:
        if not tagged:
            return TreeNode("S")

        i = 0
        left_groups = []
        right_groups = []

        while i < len(tagged) and tagged[i][1] not in ("глагол", "частица"):
            group, i = self._consume_group(tagged, i)
            left_groups.append(group)

        verb_children = []
        while i < len(tagged) and tagged[i][1] == "частица":
            verb_children.append(TreeNode("Частица", [TreeNode(tagged[i][0])]))
            i += 1

        verb_group = None
        if i < len(tagged) and tagged[i][1] == "глагол":
            verb_children.append(TreeNode("Глагол", [TreeNode(tagged[i][0])]))
            verb_group = TreeNode("ГрГл", verb_children)
            i += 1

        while i < len(tagged):
            group, i = self._consume_group(tagged, i)
            right_groups.append(group)

        children = left_groups + right_groups
        if verb_group is not None:
            children = left_groups + [verb_group] + right_groups
        return TreeNode("S", children)

    def _consume_group(self, tagged: List[Tuple[str, str]], i: int) -> Tuple[TreeNode, int]:
        word, cat = tagged[i]

        if cat == "наречие":
            return TreeNode("Група", [TreeNode("ГрНар", [TreeNode("Наречие", [TreeNode(word)])])]), i + 1

        if cat == "предлог":
            return self._consume_prepositional_group(tagged, i)

        if cat == "прилагателно":
            j = i
            while j < len(tagged) and tagged[j][1] == "прилагателно":
                j += 1
            if j < len(tagged) and tagged[j][1] in ("съществително", "собствено име"):
                return self._consume_noun_group(tagged, i)
            return TreeNode("Група", [TreeNode("ГрПрил", [TreeNode("Прилагателно", [TreeNode(word)])])]), i + 1

        if cat in ("съществително", "собствено име"):
            return self._consume_noun_group(tagged, i)

        if cat == "съюз":
            return TreeNode("Група", [TreeNode("Съюз", [TreeNode(word)])]), i + 1

        if cat == "частица":
            return TreeNode("Група", [TreeNode("Частица", [TreeNode(word)])]), i + 1

        return TreeNode("Група", [TreeNode(f"{cat} → {word}")]), i + 1

    def _consume_noun_group(self, tagged: List[Tuple[str, str]], i: int) -> Tuple[TreeNode, int]:
        j = i
        children = []

        while j < len(tagged) and tagged[j][1] == "прилагателно":
            children.append(TreeNode("Прилагателно", [TreeNode(tagged[j][0])]))
            j += 1

        if j < len(tagged) and tagged[j][1] == "собствено име":
            children.append(TreeNode("СобственоИме", [TreeNode(tagged[j][0])]))
            return TreeNode("Група", [TreeNode("ГрСщ", children)]), j + 1

        if j < len(tagged) and tagged[j][1] == "съществително":
            children.append(TreeNode("Съществително", [TreeNode(tagged[j][0])]))
            return TreeNode("Група", [TreeNode("ГрСщ", children)]), j + 1

        if children:
            return TreeNode("Група", [TreeNode("ГрПрил", children)]), j

        return TreeNode("Група", [TreeNode(f"{tagged[i][1]} → {tagged[i][0]}")]), i + 1

    def _consume_prepositional_group(self, tagged: List[Tuple[str, str]], i: int) -> Tuple[TreeNode, int]:
        prep = tagged[i][0]
        j = i + 1
        while j < len(tagged) and tagged[j][1] == "прилагателно":
            j += 1
        if j >= len(tagged) or tagged[j][1] not in ("съществително", "собствено име"):
            return TreeNode("Група", [TreeNode("ГрПр", [TreeNode("Предлог", [TreeNode(prep)])])]), i + 1
        noun_group, k = self._consume_noun_group(tagged, i + 1)
        return TreeNode("Група", [
            TreeNode("ГрПр", [
                TreeNode("Предлог", [TreeNode(prep)]),
                noun_group.children[0]
            ])
        ]), k

    def _build_semantic_tree(self, tagged: List[Tuple[str, str]]) -> TreeNode:
        verb_index = self._find_main_verb_index(tagged)
        actions = []
        subjects = []
        objects = []
        properties = []
        relations = []

        if verb_index is not None:
            actions.append(TreeNode("Действие", [TreeNode(f"име → {tagged[verb_index][0]}")]))
        else:
            verb_index = len(tagged)

        subj_nodes, subj_props, subj_rel = self._extract_roles(tagged[:verb_index], is_subject_zone=True)
        obj_nodes, obj_props, obj_rel = self._extract_roles(tagged[verb_index + 1:], is_subject_zone=False)

        subjects.extend(subj_nodes)
        objects.extend(obj_nodes)
        properties.extend(subj_props + obj_props)
        relations.extend(subj_rel + obj_rel)

        return TreeNode("СемантичноДърво", [
            TreeNode("Извършител", subjects),
            TreeNode("Действия", actions),
            TreeNode("Обекти", objects),
            TreeNode("Свойства", properties),
            TreeNode("Отношения", relations),
        ])

    def _find_main_verb_index(self, tagged: List[Tuple[str, str]]) -> Optional[int]:
        for i, (_, cat) in enumerate(tagged):
            if cat == "глагол":
                return i
        return None

    def _extract_roles(self, tagged: List[Tuple[str, str]], is_subject_zone: bool) -> Tuple[List[TreeNode], List[TreeNode], List[TreeNode]]:
        nominals = []
        props = []
        rels = []
        i = 0

        while i < len(tagged):
            word, cat = tagged[i]

            if cat == "предлог":
                rel, i = self._extract_relation(tagged, i)
                rels.append(rel)
                continue

            if cat == "наречие":
                rels.append(TreeNode("Начин", [TreeNode(f"слот → {word}")]))
                i += 1
                continue

            if cat in ("прилагателно", "съществително", "собствено име"):
                nominal, new_i, nominal_props = self._extract_nominal(tagged, i)
                if nominal is not None:
                    nominals.append(nominal)
                    props.extend(nominal_props)
                    i = new_i
                    continue

            i += 1

        return nominals, props, rels

    def _extract_nominal(self, tagged: List[Tuple[str, str]], i: int) -> Tuple[Optional[TreeNode], int, List[TreeNode]]:
        j = i
        adjs = []

        while j < len(tagged) and tagged[j][1] == "прилагателно":
            adjs.append(tagged[j][0])
            j += 1

        props = [TreeNode(f"свойство → {adj}") for adj in adjs]

        if j < len(tagged) and tagged[j][1] == "собствено име":
            return TreeNode("Обект", [TreeNode("тип → собствено име"), TreeNode(f"име → {tagged[j][0]}")]), j + 1, props

        if j < len(tagged) and tagged[j][1] == "съществително":
            node = TreeNode("Обект", [TreeNode(f"име → {tagged[j][0]}")])
            for adj in adjs:
                node.children.append(TreeNode(f"свойство → {adj}"))
            return node, j + 1, props

        return None, i, []

    def _extract_relation(self, tagged: List[Tuple[str, str]], i: int) -> Tuple[TreeNode, int]:
        prep = tagged[i][0]
        j = i + 1
        adjs = []

        while j < len(tagged) and tagged[j][1] == "прилагателно":
            adjs.append(tagged[j][0])
            j += 1

        if j < len(tagged) and tagged[j][1] in ("съществително", "собствено име"):
            target = tagged[j][0]
            role = self._guess_preposition_role(prep, target)
            children = [TreeNode(f"слот → {target}")]
            for adj in adjs:
                children.append(TreeNode(f"свойство → {adj}"))
            return TreeNode(role, children), j + 1

        return TreeNode("Отношение", [TreeNode(f"предлог → {prep}")]), i + 1

    def _guess_preposition_role(self, prep: str, target: str) -> str:
        target = target.lower()

        if prep == "в":
            if target in self.TEMPORAL_NOUNS:
                return "Време"
            if target in self.LOCATION_NOUNS:
                return "Място"
            return "Време/Място"

        if prep == "на":
            if target in self.LOCATION_NOUNS:
                return "Място"
            return "Притежание/Място"

        if prep == "от":
            if target in self.TEMPORAL_NOUNS:
                return "Продължителност"
            return "Източник"

        if prep == "през":
            return "Време"
        if prep in {"над", "под"}:
            return "Пространствено отношение"
        if prep == "до":
            return "Посока/Граница"
        if prep == "към":
            return "Посока"
        return "Отношение"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mini NLParse за macOS")
        self.geometry("1280x840")
        self.kb = KnowledgeBase()
        self._load_demo_data()
        self.parser = SimpleBulgarianParser(self.kb)
        self.last_syntax_tree: Optional[TreeNode] = None
        self.last_semantic_tree: Optional[TreeNode] = None
        self._build_ui()

    def _load_demo_data(self) -> None:
        demo = {
            "вали": "глагол", "валя": "глагол", "удари": "глагол", "лае": "глагол",
            "лежи": "глагол", "чете": "глагол", "носи": "глагол", "спи": "глагол",
            "изследва": "глагол", "играе": "глагол", "расте": "глагол", "люлее": "глагол",
            "пише": "глагол", "тича": "глагол", "пее": "глагол", "яде": "глагол",
            "пие": "глагол", "ходи": "глагол", "работи": "глагол", "учи": "глагол",
            "стои": "глагол", "седи": "глагол", "бяга": "глагол", "плува": "глагол",
            "говори": "глагол", "мисли": "глагол", "гледа": "глагол", "слуша": "глагол",
            "обича": "глагол", "живее": "глагол", "вижда": "глагол", "знае": "глагол",

            "дъжд": "съществително", "сняг": "съществително", "градушка": "съществително",
            "околности": "съществително", "куче": "съществително", "двора": "съществително",
            "понеделник": "съществително", "векове": "съществително", "земята": "съществително",
            "година": "съществително", "грохотът": "съществително", "водата": "съществително",
            "котка": "съществително", "перваза": "съществително", "книга": "съществително",
            "библиотеката": "съществително", "вятър": "съществително", "листата": "съществително",
            "пътеката": "съществително", "терасата": "съществително", "дете": "съществително",
            "парка": "съществително", "градината": "съществително", "роза": "съществително",
            "пролетта": "съществително", "сряда": "съществително", "петък": "съществително",
            "домашното": "съществително", "стола": "съществително", "реката": "съществително",
            "клоните": "съществително", "дърветата": "съществително", "къщата": "съществително",
            "момчето": "съществително", "момичето": "съществително", "училището": "съществително",
            "улицата": "съществително", "планината": "съществително", "морето": "съществително",
            "слънцето": "съществително", "небето": "съществително", "птица": "съществително",
            "цвете": "съществително", "маса": "съществително", "прозореца": "съществително",
            "книги": "съществително", "деца": "съществително", "хора": "съществително",

            "силен": "прилагателно", "пухкав": "прилагателно", "пухкавата": "прилагателно",
            "едра": "прилагателно", "близките": "прилагателно", "кафявото": "прилагателно",
            "пороен": "прилагателно", "плодородна": "прилагателно", "интересна": "прилагателно",
            "силният": "прилагателно", "малката": "прилагателно", "студеният": "прилагателно",
            "красива": "прилагателно", "нова": "прилагателно", "голям": "прилагателно",
            "малък": "прилагателно", "стар": "прилагателно", "млад": "прилагателно",
            "висок": "прилагателно", "нисък": "прилагателно", "хубав": "прилагателно",
            "голямата": "прилагателно", "малкото": "прилагателно", "старата": "прилагателно",

            "днес": "наречие", "навън": "наречие", "вчера": "наречие",
            "силно": "наречие", "надалеч": "наречие", "весело": "наречие",
            "тихо": "наречие", "бавно": "наречие",

            "в": "предлог", "на": "предлог", "от": "предлог", "през": "предлог", "над": "предлог",
            "с": "предлог", "със": "предлог", "по": "предлог", "до": "предлог", "към": "предлог",
            "под": "предлог", "без": "предлог", "между": "предлог", "след": "предлог",
            "при": "предлог", "около": "предлог",

            "лора": "собствено име", "мария": "собствено име", "софия": "собствено име",
            "петър": "собствено име", "мартин": "собствено име", "канада": "собствено име",
            "георги": "собствено име", "елица": "собствено име", "иван": "собствено име",
            "николай": "собствено име", "анна": "собствено име", "димитър": "собствено име",
        }
        self.kb.lexicon.update(demo)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")

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
        parent.rowconfigure(1, weight=1)

        top = ttk.Frame(parent)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Изречение:").grid(row=0, column=0, sticky="w")
        self.sentence_var = tk.StringVar(value="В петък Елица пише домашното в библиотеката.")
        ttk.Entry(top, textvariable=self.sentence_var).grid(row=0, column=1, sticky="ew", padx=8)

        ttk.Button(top, text="Анализирай", command=self.analyze_sentence).grid(row=0, column=2, padx=6)
        ttk.Button(top, text="Експорт", command=self.export_analysis).grid(row=0, column=3, padx=6)

        self.syntax_tree = self._create_tree_panel(parent, "Синтактично дърво", 1, 0)
        self.semantic_tree = self._create_tree_panel(parent, "Семантично дърво", 1, 1)

        self.syntax_tree.bind("<<TreeviewSelect>>", self._on_tree_selection)
        self.semantic_tree.bind("<<TreeviewSelect>>", self._on_tree_selection)

        selected_panel = ttk.LabelFrame(parent, text="Избор")
        selected_panel.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        selected_panel.columnconfigure(1, weight=1)
        selected_panel.columnconfigure(3, weight=1)

        self.selected_frame_var = tk.StringVar()
        self.selected_word_var = tk.StringVar()

        ttk.Label(selected_panel, text="Избран фрейм:").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=8)
        ttk.Entry(selected_panel, textvariable=self.selected_frame_var, state="readonly").grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=8)
        ttk.Label(selected_panel, text="Избрана дума:").grid(row=0, column=2, sticky="w", padx=(8, 4), pady=8)
        ttk.Entry(selected_panel, textvariable=self.selected_word_var, state="readonly").grid(row=0, column=3, sticky="ew", padx=(0, 8), pady=8)

    def _create_tree_panel(self, parent: ttk.Frame, title: str, row: int, column: int) -> ttk.Treeview:
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=row, column=column, sticky="nsew", padx=10, pady=10)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        tree = ttk.Treeview(frame, show="tree", selectmode="browse")
        tree.grid(row=0, column=0, sticky="nsew")

        ttk.Scrollbar(frame, orient="vertical", command=tree.yview).grid(row=0, column=1, sticky="ns")
        ttk.Scrollbar(frame, orient="horizontal", command=tree.xview).grid(row=1, column=0, sticky="ew")
        return tree

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

    def _ask_unknown_word(self, word: str, guessed: str) -> Optional[str]:
        dialog = tk.Toplevel(self)
        dialog.title("Непозната дума")
        dialog.geometry("420x200")
        dialog.resizable(False, False)
        dialog.grab_set()

        result = [None]
        ttk.Label(dialog, text=f"Думата '{word}' не е в речника.", font=("Helvetica", 13, "bold")).pack(pady=(14, 4))
        ttk.Label(dialog, text=f"Предложена категория: {guessed}", font=("Helvetica", 12)).pack(pady=(0, 8))
        cat_var = tk.StringVar(value=guessed)
        ttk.Combobox(dialog, textvariable=cat_var, values=self.kb.categories, width=25, state="readonly").pack(pady=4)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=12)

        def accept():
            result[0] = cat_var.get()
            dialog.destroy()

        def cancel():
            dialog.destroy()

        ttk.Button(btn_frame, text="Приеми", command=accept).pack(side="left", padx=8)
        ttk.Button(btn_frame, text="Откажи", command=cancel).pack(side="left", padx=8)

        dialog.wait_window()
        return result[0]

    def analyze_sentence(self) -> None:
        sentence = self.sentence_var.get().strip()
        if not sentence:
            messagebox.showwarning("Липсва текст", "Въведи изречение.")
            return

        try:
            syntax, semantic = self.parser.parse_sentence(sentence, unknown_handler=self._ask_unknown_word)
        except Exception as e:
            messagebox.showerror("Грешка при анализ", str(e))
            return

        self.last_syntax_tree = syntax
        self.last_semantic_tree = semantic
        self._populate_treeview(self.syntax_tree, syntax)
        self._populate_treeview(self.semantic_tree, semantic)
        self.selected_frame_var.set("")
        self.selected_word_var.set("")
        self.refresh_lexicon_view()

    def export_analysis(self) -> None:
        if self.last_syntax_tree is None or self.last_semantic_tree is None:
            messagebox.showwarning("Няма анализ", "Първо анализирай изречението.")
            return

        path = filedialog.asksaveasfilename(
            title="Запази анализ",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if not path:
            return

        content = [
            "ИЗРЕЧЕНИЕ",
            self.sentence_var.get().strip(),
            "",
            "СИНТАКТИЧНО ДЪРВО",
            self.last_syntax_tree.pretty(),
            "",
            "СЕМАНТИЧНО ДЪРВО",
            self.last_semantic_tree.pretty(),
        ]
        Path(path).write_text("\n".join(content), encoding="utf-8")
        messagebox.showinfo("Готово", f"Анализът е записан в:\n{path}")

    def add_word(self) -> None:
        word = self.word_var.get().strip().lower()
        cat = self.cat_var.get().strip()
        if word and cat:
            self.kb.lexicon[word] = cat
            self.word_var.set("")
            self.refresh_lexicon_view()

    def add_rule(self) -> None:
        rule = self.rule_var.get().strip()
        if rule:
            self.kb.grammar_rules.append(rule)
            self.rule_var.set("")
            self.refresh_grammar_view()

    def add_frame(self) -> None:
        frame = self.frame_var.get().strip()
        if frame:
            self.kb.frames.append(frame)
            self.frame_var.set("")
            self.refresh_frames_view()

    def refresh_lexicon_view(self) -> None:
        self.lexicon_text.delete("1.0", tk.END)
        self.lexicon_text.insert(tk.END, "МОРФОЛОГИЧНИ КАТЕГОРИИ:\n")
        for c in self.kb.categories:
            self.lexicon_text.insert(tk.END, f"- {c}\n")
        self.lexicon_text.insert(tk.END, "\nРЕЧНИК:\n")
        for word in sorted(self.kb.lexicon):
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
        path = filedialog.askopenfilename(title="Зареди база знания", filetypes=[("JSON files", "*.json")])
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

    def _populate_treeview(self, tree: ttk.Treeview, root_node: TreeNode) -> None:
        tree.delete(*tree.get_children())
        root_id = tree.insert("", "end", text=root_node.label, open=True)
        self._insert_tree_children(tree, root_id, root_node.children)

    def _insert_tree_children(self, tree: ttk.Treeview, parent_id: str, nodes: List[TreeNode]) -> None:
        for node in nodes:
            item_id = tree.insert(parent_id, "end", text=node.label, open=True)
            self._insert_tree_children(tree, item_id, node.children)

    def _on_tree_selection(self, event: tk.Event) -> None:
        tree = event.widget
        selected = tree.selection()
        if not selected:
            return
        item_id = selected[0]
        label = tree.item(item_id, "text")
        self.selected_word_var.set(self._extract_word_from_label(label))
        self.selected_frame_var.set(self._infer_frame_from_selection(tree, item_id, label))

    @staticmethod
    def _extract_word_from_label(label: str) -> str:
        if "→" in label:
            return label.split("→", 1)[1].strip()
        return ""

    def _infer_frame_from_selection(self, tree: ttk.Treeview, item_id: str, label: str) -> str:
        semantic_frame = self._infer_semantic_frame(tree, item_id)
        if semantic_frame:
            return semantic_frame
        return self._infer_frame_from_label(label)

    def _infer_semantic_frame(self, tree: ttk.Treeview, item_id: str) -> str:
        if tree is not self.semantic_tree:
            return ""
        lineage = []
        current = item_id
        while current:
            lineage.append(current)
            current = tree.parent(current)
        for idx in range(len(lineage) - 1, -1, -1):
            label = tree.item(lineage[idx], "text")
            if label in {"Извършител", "Действия", "Обекти", "Свойства", "Отношения"}:
                return label
        return ""

    @staticmethod
    def _infer_frame_from_label(label: str) -> str:
        left = label.split("→", 1)[0].strip().lower()
        if "глагол" in left or "действие" in left:
            return "Действия"
        if "собствено" in left:
            return "собствено име"
        if "съществително" in left or "обект" in left:
            return "Обекти"
        if "прилагателно" in left or "свойство" in left:
            return "Свойства"
        if "предлог" in left or "отнош" in left or "наречие" in left:
            return "Отношения"
        return ""


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
