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

WIRE_TYPE_MAP = {
    "str": "string",
    "bool": "boolean",
    "int": "integer",
    "float": "number",
    "enum": "string",
    "polygon": "array",
    "polygons": "array",
    "optional_extruder": "integer",
    "extruder": "integer",
    "[int]": "array",
}

MACHINE_CATEGORIES = {"machine_settings", "command_line_settings"}
FILAMENT_CATEGORIES = {"material"}

TRANSLATION_NAMESPACE = "cura"
TRANSLATION_SOURCE_LOCALE = "en"
TOOL_REFERENCE_SENTINEL_LABEL = "Not overridden"
TOOL_REFERENCE_SENTINEL_I18N = "cura.editors.tool_reference.not_overridden"


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


def is_statically_disabled(value: Any) -> bool:
    """Return whether Cura declares a definition as unconditionally disabled."""

    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value == 0
    if not isinstance(value, str):
        return False
    try:
        parsed = ast.parse(value.strip(), mode="eval").body
    except SyntaxError:
        return False
    if not isinstance(parsed, ast.Constant):
        return False
    if isinstance(parsed.value, bool):
        return not parsed.value
    return isinstance(parsed.value, (int, float)) and parsed.value == 0


def structured_string_editor(default_value: Any) -> dict[str, Any] | None:
    """Describe JSON-shaped Cura strings without changing their wire encoding."""

    if not isinstance(default_value, str):
        return None
    try:
        parsed = json.loads(default_value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None

    if parsed and all(isinstance(row, list) for row in parsed):
        column_counts = {len(row) for row in parsed}
        values = [value for row in parsed for value in row]
        if (
            len(column_counts) == 1
            and column_counts != {0}
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            )
        ):
            return {
                "kind": "matrix",
                "codec": "json_string",
                "rows": len(parsed),
                "columns": next(iter(column_counts)),
                "value_type": "number",
            }

    return {"kind": "structured", "codec": "json_string", "value_type": "array"}


def editor_metadata(source_type: str, definition: dict[str, Any]) -> dict[str, Any]:
    """Map Cura-native types to generic editor semantics."""

    if source_type in {"optional_extruder", "extruder"}:
        editor: dict[str, Any] = {
            "kind": "tool_reference",
            "index_base": 0,
            "display_index_base": 1,
            "selection": "optional"
            if source_type == "optional_extruder"
            else "required",
        }
        if source_type == "optional_extruder":
            editor["sentinels"] = [
                {
                    "value": -1,
                    "semantic": "inherit",
                    "label": TOOL_REFERENCE_SENTINEL_LABEL,
                    "i18n": TOOL_REFERENCE_SENTINEL_I18N,
                }
            ]
        return editor

    if source_type == "polygon":
        return {"kind": "polygon", "dimensions": 2, "value_type": "number"}
    if source_type == "polygons":
        return {"kind": "polygon_list", "dimensions": 2, "value_type": "number"}
    if source_type == "[int]":
        return {"kind": "list", "value_type": "int"}
    if source_type == "enum":
        return {"kind": "select"}
    if source_type == "bool":
        return {"kind": "toggle"}
    if source_type in {"int", "float"}:
        return {"kind": "number", "value_type": source_type}
    if source_type == "str":
        structured = structured_string_editor(definition.get("default_value"))
        if structured is not None:
            return structured
        return {
            "kind": "text",
            "multiline": "\n" in str(definition.get("default_value", "")),
        }
    return {"kind": "text"}


def definition_i18n(key: str, definition: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "label": f"{TRANSLATION_NAMESPACE}.definitions.{key}.label",
        "tooltip": f"{TRANSLATION_NAMESPACE}.definitions.{key}.description",
    }
    options = definition.get("options")
    if isinstance(options, dict):
        metadata["enum_labels"] = {
            str(value): f"{TRANSLATION_NAMESPACE}.definitions.{key}.options.{value}"
            for value in options
        }
    return metadata


def iter_settings(
    settings: dict[str, Any],
    *,
    category_key: str,
    category_label: str,
    parents: tuple[tuple[str, str], ...] = (),
    ui_enabled: bool = True,
) -> Iterator[
    tuple[
        str,
        dict[str, Any],
        str,
        str,
        tuple[tuple[str, str], ...],
        bool,
    ]
]:
    for key, definition in settings.items():
        setting_type = definition.get("type")
        label = str(definition.get("label") or key)
        children = definition.get("children")
        setting_ui_enabled = ui_enabled and not is_statically_disabled(
            definition.get("enabled")
        )

        if setting_type != "category":
            yield (
                key,
                definition,
                category_key,
                category_label,
                parents,
                setting_ui_enabled,
            )

        if isinstance(children, dict):
            next_parents = (
                parents if setting_type == "category" else (*parents, (key, label))
            )
            yield from iter_settings(
                children,
                category_key=category_key,
                category_label=category_label,
                parents=next_parents,
                ui_enabled=setting_ui_enabled,
            )


def normalize_definition(
    key: str,
    definition: dict[str, Any],
    setting_modes: dict[str, str],
    visibility_tiers: dict[str, str],
) -> dict[str, Any] | None:
    source_type = definition.get("type")
    normalized_type = TYPE_MAP.get(source_type)
    if normalized_type is None:
        return None

    normalized: dict[str, Any] = {
        "type": normalized_type,
        "wire_type": WIRE_TYPE_MAP[str(source_type)],
        "wire_codec": "json",
        "native_type": str(source_type),
        "editor": editor_metadata(str(source_type), definition),
        "label": str(definition.get("label") or key),
        "tooltip": str(definition.get("description") or ""),
        "mode": setting_modes.get(key, "expert"),
        "visibility_tier": visibility_tiers.get(key, "expert"),
        "i18n": definition_i18n(key, definition),
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


def load_visibility_tiers(
    resources: Path,
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    """Preserve Cura tiers while mapping them to SimplyPrint's editable modes."""

    presets: list[tuple[int, str, str, list[str]]] = []
    for path in sorted((resources / "setting_visibility").glob("*.cfg")):
        tier_id = path.stem.lower()
        label = path.stem
        weight = 999
        keys: list[str] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", ";", "[")):
                continue
            if "=" in line:
                field, value = (part.strip() for part in line.split("=", 1))
                if field == "name":
                    label = value
                elif field == "weight" and value.isdigit():
                    weight = int(value)
                continue
            keys.append(line)
        presets.append((weight, tier_id, label, keys))

    modes: dict[str, str] = {}
    setting_tiers: dict[str, str] = {}
    tiers: list[dict[str, Any]] = []
    for weight, tier_id, label, keys in sorted(presets):
        # Cura's first native visibility tier backs the shared detailed Simple
        # mode. Basic is a distinct, engine-independent control surface rather
        # than an alias for Cura's native Basic tier.
        mode = {"basic": "simple", "simple": "simple", "advanced": "advanced"}.get(
            tier_id, "expert"
        )
        for key in keys:
            modes.setdefault(key, mode)
            setting_tiers.setdefault(key, tier_id)
        tiers.append(
            {
                "id": tier_id,
                "label": label,
                "order": weight,
                "ui_mode": mode,
                "setting_keys": keys,
            }
        )
    return modes, setting_tiers, tiers


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
                return (
                    "Math.max(...extruderValues("
                    f"{json.dumps(extruder_setting, ensure_ascii=False)}))"
                )

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
                    item_name = comprehension.target.id
                    predicate = _emit_js(
                        generator.elt,
                        {**substitutions, item_name: item_name},
                    )
                    return (
                        f"extruderValues({json.dumps(extruder_setting, ensure_ascii=False)})"
                        f".every(({item_name}) => {predicate})"
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


def _po_literal(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"invalid PO string literal: {value!r}") from error
    if not isinstance(parsed, str):
        raise ValueError(f"PO literal is not a string: {value!r}")
    return parsed


def load_po(path: Path) -> list[dict[str, str | None]]:
    """Parse the singular gettext entries needed by Cura's resource catalogs."""

    entries: list[dict[str, str | None]] = []
    entry: dict[str, str | None] = {"context": None, "id": "", "translation": ""}
    active_field: str | None = None
    fuzzy = False

    def flush() -> None:
        nonlocal entry, active_field, fuzzy
        if entry["id"] and entry["translation"] and not fuzzy:
            entries.append(entry)
        entry = {"context": None, "id": "", "translation": ""}
        active_field = None
        fuzzy = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("#~"):
            continue
        if line.startswith("#,"):
            fuzzy = fuzzy or any(flag.strip() == "fuzzy" for flag in line.split(","))
            continue
        if line.startswith("#"):
            continue
        if line.startswith("msgctxt "):
            active_field = "context"
            entry[active_field] = _po_literal(line[len("msgctxt ") :])
            continue
        if line.startswith("msgid_plural "):
            active_field = None
            continue
        if line.startswith("msgid "):
            active_field = "id"
            entry[active_field] = _po_literal(line[len("msgid ") :])
            continue
        if line.startswith("msgstr "):
            active_field = "translation"
            entry[active_field] = _po_literal(line[len("msgstr ") :])
            continue
        if line.startswith("msgstr[0] "):
            active_field = "translation"
            entry[active_field] = _po_literal(line[len("msgstr[0] ") :])
            continue
        if line.startswith("msgstr["):
            active_field = None
            continue
        if line.startswith('"') and active_field is not None:
            entry[active_field] = str(entry[active_field] or "") + _po_literal(line)

    flush()
    return entries


def translation_id(context: str) -> str | None:
    if context.endswith(" label"):
        key = context[: -len(" label")]
        return f"{TRANSLATION_NAMESPACE}.definitions.{key}.label"
    if context.endswith(" description"):
        key = context[: -len(" description")]
        return f"{TRANSLATION_NAMESPACE}.definitions.{key}.description"
    if " option " in context:
        key, option = context.split(" option ", 1)
        return f"{TRANSLATION_NAMESPACE}.definitions.{key}.options.{option}"
    return None


def generate_translations(
    resources: Path, output: Path, allowed_ids: set[str]
) -> dict[str, str]:
    locales: dict[str, str] = {}
    i18n_path = resources / "i18n"
    locale_paths = sorted(i18n_path.iterdir()) if i18n_path.is_dir() else []
    for locale_path in locale_paths:
        if not locale_path.is_dir():
            continue

        messages: dict[str, str] = {}
        for filename in ("fdmprinter.def.json.po", "fdmextruder.def.json.po"):
            catalog_path = locale_path / filename
            if not catalog_path.is_file():
                continue
            for entry in load_po(catalog_path):
                context = entry["context"]
                if not isinstance(context, str):
                    continue
                key = translation_id(context)
                translation = entry["translation"]
                if key in allowed_ids and isinstance(translation, str):
                    messages.setdefault(key, translation)

        cura_catalog = locale_path / "cura.po"
        if cura_catalog.is_file():
            for entry in load_po(cura_catalog):
                if (
                    TOOL_REFERENCE_SENTINEL_I18N in allowed_ids
                    and entry["id"] == TOOL_REFERENCE_SENTINEL_LABEL
                    and isinstance(entry["translation"], str)
                ):
                    messages[TOOL_REFERENCE_SENTINEL_I18N] = entry["translation"]
                    break

        if not messages:
            continue
        filename = f"{locale_path.name}.json"
        write_json(
            output / "translations" / filename,
            {
                "schema_version": 1,
                "locale": locale_path.name,
                "source_locale": TRANSLATION_SOURCE_LOCALE,
                "messages": messages,
            },
        )
        locales[locale_path.name] = filename

    write_json(
        output / "translations" / "_index.json",
        {
            "schema_version": 1,
            "source_locale": TRANSLATION_SOURCE_LOCALE,
            "locales": locales,
        },
    )
    return locales


def icon_reference(name: Any) -> str | None:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        return None
    return f"cura:{name}"


def load_icon_registry(resources: Path, icon_names: set[str]) -> dict[str, Any]:
    registry: dict[str, Any] = {}
    for name in sorted(icon_names):
        candidates = [
            resources / "themes" / "cura-light" / "icons" / "default" / f"{name}.svg",
            *sorted((resources / "themes").glob(f"*/icons/default/{name}.svg")),
        ]
        path = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if path is None:
            continue
        reference = icon_reference(name)
        if reference is None:
            continue
        registry[reference] = {
            "media_type": "image/svg+xml",
            "source_name": name,
            "svg": path.read_text(encoding="utf-8"),
        }
    return registry


def generate(resources: Path, output: Path) -> None:
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
            runtime_only = metadata.get("visible") is False and not machine_root
            definition_sources.append((path, settings, runtime_only, machine_root))
    if not definition_sources:
        raise SystemExit("Cura resources do not contain root setting definitions")

    # The machine root owns shared definitions when a setting also appears in
    # the extruder root. Extra root files contribute only genuinely new keys.
    definition_sources.sort(key=lambda source: (not source[3], source[0].name))
    setting_modes, setting_tiers, visibility_tiers = load_visibility_tiers(resources)

    definitions: dict[str, Any] = {}
    panels: dict[str, dict[str, dict[str, list[str]]]] = {
        "machine": defaultdict(lambda: defaultdict(list)),
        "filament": defaultdict(lambda: defaultdict(list)),
        "process": defaultdict(lambda: defaultdict(list)),
    }
    panel_metadata: dict[str, dict[str, dict[str, Any]]] = {
        "machine": {},
        "filament": {},
        "process": {},
    }
    icon_names: set[str] = set()
    conditions: dict[str, str] = {}
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
                for key, *_, ui_enabled in iter_settings(
                    category["children"],
                    category_key=category_key,
                    category_label=category_label,
                    ui_enabled=not is_statically_disabled(category.get("enabled")),
                )
                if ui_enabled
            )

    for _, root_settings, _, _ in definition_sources:
        for category_key, category in root_settings.items():
            if category.get("type") != "category" or not isinstance(
                category.get("children"), dict
            ):
                continue
            category_label = str(category.get("label") or category_key)
            for (
                key,
                source,
                source_category,
                source_category_label,
                parents,
                _,
            ) in iter_settings(
                category["children"],
                category_key=category_key,
                category_label=category_label,
                ui_enabled=not is_statically_disabled(category.get("enabled")),
            ):
                if key in definitions:
                    continue
                normalized = normalize_definition(
                    key, source, setting_modes, setting_tiers
                )
                if normalized is None:
                    raise SystemExit(
                        f"unsupported Cura setting type for {key}: {source.get('type')}"
                    )
                runtime_only = key not in editable_setting_keys
                if runtime_only:
                    normalized["runtime_only"] = True
                definitions[key] = normalized

                if not runtime_only:
                    target = ui_target(source_category)
                    group_id, group_label = (
                        parents[0] if parents else ("general", "General")
                    )
                    panels[target][source_category_label][group_label].append(key)

                    categories = panel_metadata[target]
                    if source_category not in categories:
                        icon_name = category.get("icon")
                        icon = icon_reference(icon_name)
                        if icon is not None:
                            icon_names.add(str(icon_name))
                        categories[source_category] = {
                            "id": source_category,
                            "label": source_category_label,
                            "description": str(category.get("description") or ""),
                            "i18n": {
                                "label": (
                                    f"{TRANSLATION_NAMESPACE}.definitions."
                                    f"{source_category}.label"
                                ),
                                "description": (
                                    f"{TRANSLATION_NAMESPACE}.definitions."
                                    f"{source_category}.description"
                                ),
                            },
                            "icon": icon,
                            "groups": {},
                        }
                    groups = categories[source_category]["groups"]
                    if group_id not in groups:
                        group: dict[str, Any] = {
                            "id": group_id,
                            "label": group_label,
                            "setting_keys": [],
                        }
                        if parents:
                            group["i18n"] = {
                                "label": (
                                    f"{TRANSLATION_NAMESPACE}.definitions."
                                    f"{group_id}.label"
                                )
                            }
                        groups[group_id] = group
                    groups[group_id]["setting_keys"].append(key)

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

    uses_extruder_values = any(
        "extruderValues(" in condition for condition in conditions.values()
    )
    write_json(
        output / "conditional_visibility.json",
        {
            "conditions": conditions,
            "function_definitions": (
                {
                    "extruderValues": {
                        "kind": "setting_values_by_tool",
                        "argument": "setting_key",
                        "scope_template": "extruder.{tool_index}",
                        "order": "tool_index",
                        "missing": "omit",
                        "fallback": "merged_value",
                    }
                }
                if uses_extruder_values
                else {}
            ),
            "functions": ["extruderValues"] if uses_extruder_values else [],
            "variable_defaults": {},
            "variables": [],
        },
    )

    editable_keys = {
        key
        for key, definition in definitions.items()
        if not definition.get("runtime_only")
    }
    tier_order = {tier["id"]: tier["order"] for tier in visibility_tiers}
    fallback_tier_order = max(tier_order.values(), default=999)
    tier_metadata = []
    for tier in visibility_tiers:
        source_setting_keys = [
            key for key in tier["setting_keys"] if key in editable_keys
        ]
        setting_keys = list(source_setting_keys)
        included = set(setting_keys)
        for key, definition in definitions.items():
            if key not in editable_keys or key in included:
                continue
            setting_tier_order = tier_order.get(
                definition["visibility_tier"], fallback_tier_order
            )
            if setting_tier_order <= tier["order"]:
                setting_keys.append(key)
                included.add(key)
        tier_metadata.append(
            {
                **tier,
                "source_setting_keys": source_setting_keys,
                "setting_keys": setting_keys,
            }
        )

    modes_by_id: dict[str, dict[str, Any]] = {
        "basic": {
            "id": "basic",
            "label": "Basic",
            "order": 0,
            "source_tiers": [],
            "surface": "basic_controls",
        }
    }
    mode_labels = {
        "basic": "Basic",
        "simple": "Simple",
        "advanced": "Advanced",
        "expert": "Expert",
    }
    for tier in tier_metadata:
        mode_id = tier["ui_mode"]
        mode = modes_by_id.setdefault(
            mode_id,
            {
                "id": mode_id,
                "label": mode_labels.get(mode_id, str(mode_id).title()),
                "order": tier["order"],
                "source_tiers": [],
            },
        )
        mode["order"] = min(mode["order"], tier["order"])
        mode["source_tiers"].append(tier["id"])

    serialized_panels: dict[str, list[dict[str, Any]]] = {}
    for target, categories in panel_metadata.items():
        serialized_panels[target] = []
        for category in categories.values():
            serialized_panels[target].append(
                {
                    **category,
                    "groups": list(category["groups"].values()),
                }
            )

    translation_ids = {TOOL_REFERENCE_SENTINEL_I18N}
    for definition in definitions.values():
        i18n = definition["i18n"]
        translation_ids.add(i18n["label"])
        translation_ids.add(i18n["tooltip"])
        translation_ids.update(i18n.get("enum_labels", {}).values())
    for categories in serialized_panels.values():
        for category in categories:
            translation_ids.update(category["i18n"].values())
            for group in category["groups"]:
                translation_ids.update(group.get("i18n", {}).values())

    locales = generate_translations(resources, output, translation_ids)
    write_json(
        output / "ui_metadata.json",
        {
            "schema_version": 1,
            "source": {
                "family": "cura",
                "definition_format": 2,
                "source_locale": TRANSLATION_SOURCE_LOCALE,
            },
            "conditional_settings": {"false_behavior": "hide"},
            "ui_modes": sorted(modes_by_id.values(), key=lambda mode: mode["order"]),
            "visibility_tiers": tier_metadata,
            "panels": serialized_panels,
            "icons": load_icon_registry(resources, icon_names),
            "translations": {
                "index": "translations/_index.json",
                "locales": sorted(locales),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.resources, args.output)


if __name__ == "__main__":
    main()
