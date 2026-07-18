#!/usr/bin/env python3
"""Normalize Cura's setting tree into SimplyPrint's slicer artifact contract."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


TYPE_MAP = {
    "str": "string",
    "bool": "bool",
    "int": "int",
    "float": "float",
    "enum": "enum",
    "polygon": "points",
    "polygons": "pointsgroups",
    "optional_extruder": "int",
    "extruder": "int",
    "[int]": "ints",
}

MACHINE_CATEGORIES = {"machine_settings", "command_line_settings"}
FILAMENT_CATEGORIES = {"material"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, ensure_ascii=False)
        output.write("\n")


def concrete_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str) or not re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value.strip()
    ):
        return None
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def normalize_default(value: Any, normalized_type: str) -> Any:
    if normalized_type in {"int", "float"}:
        number = concrete_number(value)
        return number if number is not None else value
    if normalized_type == "ints":
        candidate = value
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except json.JSONDecodeError:
                return value
        if isinstance(candidate, list):
            normalized: list[int] = []
            for item in candidate:
                number = concrete_number(item)
                if number is None or not float(number).is_integer():
                    return value
                normalized.append(int(number))
            return normalized
    return value


def iter_settings(
    settings: dict[str, Any],
    *,
    category_key: str,
    category_label: str,
    parents: tuple[str, ...] = (),
) -> Iterator[tuple[str, dict[str, Any], str, str, tuple[str, ...]]]:
    for key, definition in settings.items():
        setting_type = definition.get("type")
        label = str(definition.get("label") or key)
        children = definition.get("children")

        if setting_type != "category":
            yield key, definition, category_key, category_label, parents

        if isinstance(children, dict):
            next_parents = parents if setting_type == "category" else (*parents, label)
            yield from iter_settings(
                children,
                category_key=category_key,
                category_label=category_label,
                parents=next_parents,
            )


def normalize_definition(
    key: str,
    definition: dict[str, Any],
    setting_modes: dict[str, str],
) -> dict[str, Any] | None:
    source_type = definition.get("type")
    normalized_type = TYPE_MAP.get(source_type)
    if normalized_type is None:
        return None

    normalized: dict[str, Any] = {
        "type": normalized_type,
        "label": str(definition.get("label") or key),
        "tooltip": str(definition.get("description") or ""),
        "mode": setting_modes.get(key, "expert"),
    }

    unit = definition.get("unit")
    if isinstance(unit, str) and unit:
        normalized["sidetext"] = unit

    # default_value is already a literal in Cura definitions. Do not fall back
    # to value: that field is commonly a Python expression requiring Cura's
    # frontend resolver and must never be mistaken for a concrete default.
    if "default_value" in definition and isinstance(
        definition["default_value"], (str, bool, int, float, list)
    ):
        normalized["default_value"] = normalize_default(
            definition["default_value"], normalized_type
        )
    elif isinstance(definition.get("value"), (bool, int, float, list)):
        normalized["default_value"] = normalize_default(
            definition["value"], normalized_type
        )

    minimum = concrete_number(definition.get("minimum_value"))
    maximum = concrete_number(definition.get("maximum_value"))
    if minimum is not None:
        normalized["min"] = minimum
    if maximum is not None:
        normalized["max"] = maximum

    if normalized_type == "enum":
        options = definition.get("options")
        if not isinstance(options, dict) or not options:
            return None
        normalized["enum_values"] = list(options)
        normalized["enum_labels"] = [str(label) for label in options.values()]

    if normalized_type == "string" and "\n" in str(definition.get("default_value", "")):
        normalized["multiline"] = "true"

    return normalized


def ui_target(category_key: str) -> str:
    if category_key in MACHINE_CATEGORIES:
        return "machine"
    if category_key in FILAMENT_CATEGORIES:
        return "filament"
    return "process"


def load_setting_modes(resources: Path) -> dict[str, str]:
    """Map Cura's own visibility presets to SimplyPrint's UI levels."""

    presets: list[tuple[int, str, list[str]]] = []
    for path in sorted((resources / "setting_visibility").glob("*.cfg")):
        name = path.stem.lower()
        weight = 999
        keys: list[str] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", ";", "[")):
                continue
            if "=" in line:
                field, value = (part.strip() for part in line.split("=", 1))
                if field == "name":
                    name = value.lower()
                elif field == "weight" and value.isdigit():
                    weight = int(value)
                continue
            keys.append(line)
        presets.append((weight, name, keys))

    modes: dict[str, str] = {}
    for _, name, keys in sorted(presets):
        # SimplyPrint calls Cura's first editable tier "simple". Higher tiers
        # use the same names; unknown/upstream tiers remain expert-visible.
        mode = {"basic": "simple", "simple": "simple", "advanced": "advanced"}.get(
            name, "expert"
        )
        for key in keys:
            modes.setdefault(key, mode)
    return modes


class VisibilityExpressionError(ValueError):
    pass


def _setting_name_from_call(node: ast.AST, function: str) -> str | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.func.id != function or len(node.args) != 1 or node.keywords:
        return None
    argument = node.args[0]
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        return None
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", argument.value):
        return None
    return argument.value


def _emit_js(node: ast.AST, substitutions: dict[str, str] | None = None) -> str:
    substitutions = substitutions or {}

    if isinstance(node, ast.Name):
        return substitutions.get(node.id, node.id)
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, bool, int, float)
    ):
        return json.dumps(node.value, ensure_ascii=False)
    if isinstance(node, (ast.List, ast.Tuple)):
        return (
            "[" + ", ".join(_emit_js(item, substitutions) for item in node.elts) + "]"
        )
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        operator = " && " if isinstance(node.op, ast.And) else " || "
        return (
            "("
            + operator.join(_emit_js(value, substitutions) for value in node.values)
            + ")"
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return f"!({_emit_js(node.operand, substitutions)})"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return f"({_emit_js(node.left, substitutions)} % {_emit_js(node.right, substitutions)})"
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left = _emit_js(node.left, substitutions)
        right_node = node.comparators[0]
        operator = node.ops[0]
        if isinstance(operator, (ast.In, ast.NotIn)):
            if not isinstance(right_node, (ast.List, ast.Tuple)):
                raise VisibilityExpressionError(
                    "membership requires a literal list or tuple"
                )
            contains = f"{_emit_js(right_node, substitutions)}.includes({left})"
            return f"!({contains})" if isinstance(operator, ast.NotIn) else contains
        operators = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Lt: "<",
            ast.LtE: "<=",
        }
        js_operator = next(
            (value for kind, value in operators.items() if isinstance(operator, kind)),
            None,
        )
        if js_operator is None:
            raise VisibilityExpressionError(
                f"unsupported comparison: {type(operator).__name__}"
            )
        return f"({left} {js_operator} {_emit_js(right_node, substitutions)})"
    if isinstance(node, ast.Call):
        resolved = _setting_name_from_call(node, "resolveOrValue")
        if resolved is not None:
            return resolved

        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "max"
            and len(node.args) == 1
        ):
            extruder_setting = _setting_name_from_call(node.args[0], "extruderValues")
            if extruder_setting is not None:
                # The current Cura contract is single-extruder, so the maximum
                # of that one resolved value is the value itself.
                return extruder_setting

        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "all"
            and len(node.args) == 1
        ):
            generator = node.args[0]
            if (
                isinstance(generator, ast.GeneratorExp)
                and len(generator.generators) == 1
            ):
                comprehension = generator.generators[0]
                extruder_setting = _setting_name_from_call(
                    comprehension.iter, "extruderValues"
                )
                if (
                    extruder_setting is not None
                    and isinstance(comprehension.target, ast.Name)
                    and not comprehension.ifs
                    and not comprehension.is_async
                ):
                    return _emit_js(
                        generator.elt,
                        {**substitutions, comprehension.target.id: extruder_setting},
                    )

        raise VisibilityExpressionError("unsupported function call")

    raise VisibilityExpressionError(
        f"unsupported expression node: {type(node).__name__}"
    )


def js_condition(expression: Any) -> str | None:
    if expression is False:
        return "false"
    if expression is True or not isinstance(expression, str) or not expression.strip():
        return None
    try:
        return _emit_js(ast.parse(expression, mode="eval").body)
    except (SyntaxError, VisibilityExpressionError) as error:
        raise VisibilityExpressionError(
            f"could not translate {expression!r}: {error}"
        ) from error


def generate(
    resources: Path, output: Path, version: str, engine_ref: str, resources_ref: str
) -> None:
    definition_sources: list[tuple[Path, dict[str, Any], bool, bool]] = []
    for path in sorted((resources / "definitions").glob("*.def.json")):
        document = load_json(path)
        settings = document.get("settings")
        if isinstance(settings, dict):
            metadata = document.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            # Cura's machine base is intentionally not selectable itself, so
            # it is marked invisible even though it owns the editable setting
            # tree. Hidden supplemental roots (currently the extruder root)
            # contribute runtime inputs without exposing unsafe controls.
            machine_root = metadata.get("type") == "machine"
            runtime_only = (
                metadata.get("visible") is False and not machine_root
            )
            definition_sources.append((path, settings, runtime_only, machine_root))
    if not definition_sources:
        raise SystemExit("Cura resources do not contain root setting definitions")

    # The machine root owns shared definitions when a setting also appears in
    # the extruder root. Extra root files contribute only genuinely new keys.
    definition_sources.sort(key=lambda source: (not source[3], source[0].name))
    setting_modes = load_setting_modes(resources)

    definitions: dict[str, Any] = {}
    panels: dict[str, dict[str, dict[str, list[str]]]] = {
        "machine": defaultdict(lambda: defaultdict(list)),
        "filament": defaultdict(lambda: defaultdict(list)),
        "process": defaultdict(lambda: defaultdict(list)),
    }
    conditions: dict[str, str] = {}
    runtime_only_settings: set[str] = set()

    editable_setting_keys: set[str] = set()
    for _, root_settings, runtime_only_source, _ in definition_sources:
        if runtime_only_source:
            continue
        for category_key, category in root_settings.items():
            if category.get("type") != "category" or not isinstance(
                category.get("children"), dict
            ):
                continue
            category_label = str(category.get("label") or category_key)
            editable_setting_keys.update(
                key
                for key, *_ in iter_settings(
                    category["children"],
                    category_key=category_key,
                    category_label=category_label,
                )
            )

    for _, root_settings, _, _ in definition_sources:
        for category_key, category in root_settings.items():
            if category.get("type") != "category" or not isinstance(
                category.get("children"), dict
            ):
                continue
            category_label = str(category.get("label") or category_key)
            for key, source, source_category, label, parents in iter_settings(
                category["children"],
                category_key=category_key,
                category_label=category_label,
            ):
                if key in definitions:
                    continue
                normalized = normalize_definition(key, source, setting_modes)
                if normalized is None:
                    raise SystemExit(
                        f"unsupported Cura setting type for {key}: {source.get('type')}"
                    )
                runtime_only = key not in editable_setting_keys
                if runtime_only:
                    normalized["runtime_only"] = True
                    runtime_only_settings.add(key)
                definitions[key] = normalized

                if not runtime_only:
                    target = ui_target(source_category)
                    group = parents[0] if parents else "General"
                    panels[target][label][group].append(key)

                condition = js_condition(source.get("enabled"))
                if condition is not None:
                    conditions[key] = condition

    if not definitions:
        raise SystemExit("no Cura settings were normalized")

    write_json(output / "print_config_def.json", definitions)
    for target, categories in panels.items():
        write_json(
            output / f"{target}.json",
            {
                category: {group: keys for group, keys in sorted(groups.items())}
                for category, groups in sorted(categories.items())
            },
        )

    write_json(
        output / "conditional_visibility.json",
        # Every Cura dependency resolves to a generated setting key. The
        # contract's variables/functions lists are only for external context.
        {"conditions": conditions, "functions": [], "variables": []},
    )
    write_json(
        output / "metadata.json",
        {
            "engine_ref": engine_ref,
            "engine_repo": "Ultimaker/CuraEngine",
            "resources_ref": resources_ref,
            "resources_repo": "Ultimaker/Cura",
            "runtime_only_settings_count": len(runtime_only_settings),
            "settings_contract": "cura-resolved-v1",
            "settings_count": len(definitions),
            "source_definition_files": [
                path.name for path, _, _, _ in definition_sources
            ],
            "visibility_condition_count": len(conditions),
            "version": version,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--engine-ref", required=True)
    parser.add_argument("--resources-ref", required=True)
    args = parser.parse_args()
    generate(
        args.resources, args.output, args.version, args.engine_ref, args.resources_ref
    )


if __name__ == "__main__":
    main()
