"""Simple i18n helpers for MathLang CLI/components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

DEFAULT_LANGUAGE = "ja"

_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "ja": {
        # Lexer / Parser -----------------------------------------------------
        "lexer.unexpected_character": "未対応の文字 '{char}' (行 {line}, 列 {column})",
        "lexer.unterminated_string": "文字列リテラルが閉じられていません (行 {line}, 列 {column})",
        "parser.show_identifier_required": "'show' の後には識別子が必要です",
        "parser.assignment_identifier_required": "代入の先頭には識別子が必要です",
        "parser.assignment_equal_missing": "代入には '=' が必要です",
        "parser.closing_paren_missing": "括弧を閉じる ')' が必要です",
        "parser.unexpected_token": "予期しないトークン '{lexeme}' (行 {line}, 列 {column})",
        "parser.colon_missing": "'{keyword}' には ':' が必要です",
        "parser.step_id_required": "step[...] には識別子または数値IDが必要です",
        "parser.step_bracket_close_missing": "step[...] の ']' が不足しています",
        "parser.step_equal_missing": "step には '=' が必要です",
        "parser.explain_string_required": "explain には文字列リテラルが必要です",
        "parser.problem_required": "プログラムは problem から開始する必要があります",
        "parser.problem_name_identifier": "problem には識別子名を指定してください",
        "parser.step_required": "problem ブロックには少なくとも1つの step が必要です",
        "parser.trailing_tokens": "end の後ろに予期しないトークン '{lexeme}' (行 {line}, 列 {column}) があります",

        # Evaluator ----------------------------------------------------------
        "evaluator.undefined_identifier": "識別子 '{name}' が未定義です",
        "evaluator.show_step": "表示 {identifier} → {value}",
        "evaluator.output": "出力: {value}",
        "evaluator.unsupported_statement": "未対応の文です: {statement}",
        "evaluator.unsupported_expression": "未対応の式です: {expression}",
        "evaluator.division_by_zero": "ゼロ除算が発生しました",
        "evaluator.unsupported_operator": "未対応の演算子です: {operator}",
        "evaluator.core.problem": "[problem] {expression}",
        "evaluator.core.step": "[{label}] {expression} ({status})",
        "evaluator.core.end": "[end] {expression} ({status})",
        "evaluator.core.end_done": "[end] done",
        "evaluator.core.explain": "[explain] {message}",
        "evaluator.core.status.verified": "同値判定: OK",
        "evaluator.core.status.unverified": "同値判定: 未検証",
        "evaluator.core.missing_problem": "problem 宣言の前に step/end は使用できません",
        "evaluator.core.invalid_step": "途中計算が不正です (期待: {expected}, 現在: {actual})",
        "evaluator.core.rule": "rule: {rule}",
        "evaluator.core.no_rule": "同値性を説明するルールが見つかりません: {before} = {after}",
        "evaluator.core.non_equivalent": "同値ではありません: {before} ≠ {after}",

        # Symbolic trace -----------------------------------------------------
        "symbolic.disabled": "[シンボリック無効] {error}",
        "symbolic.result": "シンボリック: {result}",
        "symbolic.explanation": "説明: {explanation}",
        "symbolic.structure": "構造: {structure}",

        # CLI ----------------------------------------------------------------
        "cli.description": "MathLang プログラムをファイルまたはコードから実行します / Run MathLang programs from files or inline code.",
        "cli.script_help": ".mlang ファイルへのパス（--code と排他） / Path to a .mlang file (mutually exclusive with --code).",
        "cli.code_help": "インラインで実行する MathLang コード（引用符推奨） / Inline MathLang snippet to execute (quotes recommended).",
        "cli.symbolic_help": "DSL 実行の代わりに式をシンボリック解析 / Analyze an expression symbolically instead of running a DSL program.",
        "cli.symbolic_trace_help": "DSL 実行時にシンボリック説明を追加 / Emit symbolic explanations during DSL execution.",
        "cli.polynomial_help": "多項式式を多項式評価器で計算 / Evaluate an expression with the polynomial evaluator.",
        "cli.hello_help": "Hello World サンプルで CLI を自己診断 / Run the built-in Hello World sample to verify the CLI.",
        "cli.language_help": "出力言語コード (ja/en) / Output language code (ja/en).",
        "cli.step_label": "ステップ",
        "cli.heading_execution": "== MathLang 実行 ({label}) ==",
        "cli.heading_hello": "== MathLang 実行 (Hello World 自己診断) ==",
        "cli.hello_step": "ステップ 1: greeting を表示 → Hello World",
        "cli.hello_output": "出力: Hello World",
        "cli.heading_symbolic": "== MathLang シンボリック解析 ==",
        "cli.symbolic_input": "入力: {expression}",
        "cli.symbolic_result": "簡約結果: {result}",
        "cli.symbolic_explanation": "説明: {explanation}",
        "cli.symbolic_structure": "構造: {structure}",
        "cli.heading_polynomial": "== MathLang 多項式評価 ==",
        "cli.polynomial_input": "入力: {expression}",
        "cli.inline_label": "インラインコード",
        "cli.script_not_found": "スクリプト '{path}' が見つかりません",
        "cli.missing_input": "スクリプトパスまたは --code を指定してください",
        "cli.symbolic_conflict_script": "--symbolic はスクリプト実行用の引数と同時に利用できません",
        "cli.symbolic_conflict_trace": "--symbolic は --symbolic-trace と同時に利用できません",
        "cli.polynomial_conflict_script": "--polynomial はスクリプト実行用の引数と同時に利用できません",
        "cli.polynomial_conflict_trace": "--polynomial は --symbolic-trace と同時に利用できません",
        "cli.hello_conflict_script": "--hello-world-test はスクリプト実行用の引数と同時に利用できません",
        "cli.hello_conflict_symbolic": "--hello-world-test は --symbolic と同時に利用できません",
        "cli.hello_conflict_polynomial": "--hello-world-test は --polynomial と同時に利用できません",
        "cli.polynomial_parse_error": "[多項式パースエラー] {error}",
        "cli.polynomial_error": "[多項式エラー] {error}",

        "fuzzy.label.exact": "完全一致",
        "fuzzy.label.equivalent": "数学的に同値",
        "fuzzy.label.approx_eq": "近似的に一致",
        "fuzzy.label.analogous": "類似の推論",
        "fuzzy.label.contradict": "矛盾",
        "fuzzy.label.unknown": "不明",
        
        "fuzzy.result": "曖昧判定: {label} (スコア: {score:.2f})",
        "fuzzy.judge.detail": "スコア: {combined:.2f} (式: {expr:.2f}, ルール: {rule:.2f}, テキスト: {text:.2f})",
        "fuzzy.reason.approx_eq": "式は近似していますが、厳密には一致しません。",
        "fuzzy.reason.analogous": "構造は似ていますが、数値や変数が異なります。",

        # Causal Analysis ----------------------------------------------------
        "causal.analysis.header": "== 因果推論による分析 ==",
        "causal.cause_steps": "確認すべきステップ: {steps}",
        "causal.cause_rules": "関連するルール: {rules}",
        "causal.fix_candidate": "修正候補: {node_id}",
    },
    "en": {
        # Lexer / Parser -----------------------------------------------------
        "lexer.unexpected_character": "Unexpected character '{char}' (line {line}, column {column})",
        "lexer.unterminated_string": "Unterminated string literal (line {line}, column {column})",
        "parser.show_identifier_required": "Identifier required after 'show'.",
        "parser.assignment_identifier_required": "Assignment must start with an identifier.",
        "parser.assignment_equal_missing": "Assignment requires '='.",
        "parser.closing_paren_missing": "Missing closing ')'.",
        "parser.unexpected_token": "Unexpected token '{lexeme}' (line {line}, column {column}).",
        "parser.colon_missing": "'{keyword}' requires a ':' separator.",
        "parser.step_id_required": "step[...] requires an identifier or numeric id.",
        "parser.step_bracket_close_missing": "Missing closing ']' in step label.",
        "parser.step_equal_missing": "A step requires an '=' sign.",
        "parser.explain_string_required": "'explain' requires a string literal.",
        "parser.problem_required": "A MathLang program must start with a 'problem' block.",
        "parser.problem_name_identifier": "Provide an identifier after 'problem'.",
        "parser.step_required": "A problem block requires at least one step.",
        "parser.trailing_tokens": "Unexpected token '{lexeme}' after 'end' (line {line}, column {column}).",

        # Evaluator ----------------------------------------------------------
        "evaluator.undefined_identifier": "Identifier '{name}' is not defined",
        "evaluator.show_step": "show {identifier} → {value}",
        "evaluator.output": "Output: {value}",
        "evaluator.unsupported_statement": "Unsupported statement: {statement}",
        "evaluator.unsupported_expression": "Unsupported expression: {expression}",
        "evaluator.division_by_zero": "Division by zero detected",
        "evaluator.unsupported_operator": "Unsupported operator: {operator}",
        "evaluator.core.problem": "[problem] {expression}",
        "evaluator.core.step": "[{label}] {expression} ({status})",
        "evaluator.core.end": "[end] {expression} ({status})",
        "evaluator.core.end_done": "[end] done",
        "evaluator.core.explain": "[explain] {message}",
        "evaluator.core.status.verified": "equivalence: OK",
        "evaluator.core.status.unverified": "equivalence: not checked",
        "evaluator.core.missing_problem": "step/end cannot appear before a problem declaration",
        "evaluator.core.invalid_step": "Step is not equivalent (expected {expected}, got {actual})",
        "evaluator.core.rule": "rule: {rule}",
        "evaluator.core.no_rule": "No knowledge rule matches: {before} = {after}",
        "evaluator.core.non_equivalent": "Not equivalent: {before} ≠ {after}",

        # Symbolic trace -----------------------------------------------------
        "symbolic.disabled": "[Symbolic Disabled] {error}",
        "symbolic.result": "Symbolic: {result}",
        "symbolic.explanation": "Explanation: {explanation}",
        "symbolic.structure": "Structure: {structure}",

        # CLI ----------------------------------------------------------------
        "cli.description": "Run MathLang programs from files or inline code / MathLang プログラムを実行します。",
        "cli.script_help": "Path to a .mlang file (mutually exclusive with --code) / .mlang ファイルへのパス。",
        "cli.code_help": "Inline MathLang snippet to execute (quotes recommended) / インライン MathLang コード。",
        "cli.symbolic_help": "Analyze an expression symbolically instead of executing a DSL program / DSL 実行の代わりに式を解析。",
        "cli.symbolic_trace_help": "Emit symbolic explanations during DSL execution / DSL 実行中にシンボリック説明を追加。",
        "cli.polynomial_help": "Evaluate an expression with the polynomial evaluator / 多項式評価器で式を計算。",
        "cli.hello_help": "Run the built-in Hello World sample / Hello World サンプルを実行。",
        "cli.language_help": "Output language code (ja/en) / 出力言語コード (ja/en)。",
        "cli.step_label": "Step",
        "cli.heading_execution": "== MathLang Execution ({label}) ==",
        "cli.heading_hello": "== MathLang Execution (Hello World self-test) ==",
        "cli.hello_step": "Step 1: show greeting → Hello World",
        "cli.hello_output": "Output: Hello World",
        "cli.heading_symbolic": "== MathLang Symbolic Analysis ==",
        "cli.symbolic_input": "Input: {expression}",
        "cli.symbolic_result": "Simplified: {result}",
        "cli.symbolic_explanation": "Explanation: {explanation}",
        "cli.symbolic_structure": "Structure: {structure}",
        "cli.heading_polynomial": "== MathLang Polynomial Evaluation ==",
        "cli.polynomial_input": "Input: {expression}",
        "cli.inline_label": "inline snippet",
        "cli.script_not_found": "Script '{path}' not found",
        "cli.missing_input": "Provide either a script path or --code snippet",
        "cli.symbolic_conflict_script": "--symbolic cannot be combined with script execution arguments",
        "cli.symbolic_conflict_trace": "--symbolic cannot be combined with --symbolic-trace",
        "cli.polynomial_conflict_script": "--polynomial cannot be combined with script execution arguments",
        "cli.polynomial_conflict_trace": "--polynomial cannot be combined with --symbolic-trace",
        "cli.hello_conflict_script": "--hello-world-test cannot be combined with script execution arguments",
        "cli.hello_conflict_symbolic": "--hello-world-test cannot be combined with --symbolic",
        "cli.hello_conflict_polynomial": "--hello-world-test cannot be combined with --polynomial",
        "cli.polynomial_parse_error": "[Polynomial Parse Error] {error}",
        "cli.polynomial_error": "[Polynomial Error] {error}",

        "fuzzy.label.exact": "Exact Match",
        "fuzzy.label.equivalent": "Mathematically Equivalent",
        "fuzzy.label.approx_eq": "Approximately Equal",
        "fuzzy.label.analogous": "Analogous Reasoning",
        "fuzzy.label.contradict": "Contradiction",
        "fuzzy.label.unknown": "Unknown",

        "fuzzy.result": "Fuzzy Result: {label} (Score: {score:.2f})",
        "fuzzy.judge.detail": "Score: {combined:.2f} (Expr: {expr:.2f}, Rule: {rule:.2f}, Text: {text:.2f})",
        "fuzzy.reason.approx_eq": "The expression is approximate but not strictly exact.",
        "fuzzy.reason.analogous": "The structure is similar, but values or variables differ.",

        # Causal Analysis ----------------------------------------------------
        "causal.analysis.header": "== Causal Analysis ==",
        "causal.cause_steps": "Steps to check: {steps}",
        "causal.cause_rules": "Related rules: {rules}",
        "causal.fix_candidate": "Fix candidate: {node_id}",
    },
}


@dataclass(frozen=True)
class LanguagePack:
    """Lightweight accessor around translation templates."""

    code: str

    def text(self, key: str, **kwargs: object) -> str:
        try:
            template = _TRANSLATIONS[self.code][key]
        except KeyError as exc:  # pragma: no cover - defensive guard.
            raise KeyError(f"Missing translation for key '{key}' in language '{self.code}'") from exc
        return template.format(**kwargs)


_CACHE: Dict[str, LanguagePack] = {code: LanguagePack(code) for code in _TRANSLATIONS}


def get_language_pack(code: str | None = None) -> LanguagePack:
    normalized = (code or DEFAULT_LANGUAGE).lower()
    if normalized not in _CACHE:
        supported = ", ".join(sorted(_CACHE))
        raise ValueError(f"Unsupported language '{code}'. Supported: {supported}")
    return _CACHE[normalized]


def available_languages() -> tuple[str, ...]:
    return tuple(sorted(_CACHE))
