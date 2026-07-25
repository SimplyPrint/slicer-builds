#!/usr/bin/env python3
"""Generate versioned SimplyPrint translation catalogs from slicer gettext files."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_REGIONS = {
    "ca": "ES",
    "cs": "CZ",
    "da": "DK",
    "de": "DE",
    "en": "US",
    "es": "ES",
    "fi": "FI",
    "fr": "FR",
    "hu": "HU",
    "it": "IT",
    "ja": "JP",
    "ko": "KR",
    "lt": "LT",
    "nl": "NL",
    "pl": "PL",
    "pt": "PT",
    "ru": "RU",
    "sv": "SE",
    "tr": "TR",
    "uk": "UA",
}


def canonical_locale(value: str) -> str:
    parts = [part for part in re.split(r"[-_]+", value.strip()) if part]
    if not parts:
        raise ValueError("locale must not be empty")

    language = parts[0].lower()
    if len(parts) == 1:
        return "zh_CN" if language == "zh" else language

    normalized = [language]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            normalized.append(part.title())
        elif (len(part) == 2 and part.isalpha()) or part.isdigit():
            normalized.append(part.upper())
        else:
            normalized.append(part)

    if len(normalized) == 2 and DEFAULT_REGIONS.get(language) == normalized[1]:
        return language
    return "_".join(normalized)


def po_literal(value: str) -> str:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, str):
        raise ValueError(f"PO literal is not a string: {value!r}")
    return parsed


def load_po(path: Path) -> dict[str, str]:
    messages: dict[str, str] = {}
    entry = {"id": "", "translation": ""}
    active_field: str | None = None
    fuzzy = False

    def flush() -> None:
        nonlocal entry, active_field, fuzzy
        if entry["id"] and entry["translation"] and not fuzzy:
            messages.setdefault(entry["id"], entry["translation"])
        entry = {"id": "", "translation": ""}
        active_field = None
        fuzzy = False

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
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
        if line.startswith("msgid_plural "):
            active_field = None
            continue
        if line.startswith("msgid "):
            active_field = "id"
            entry[active_field] = po_literal(line[len("msgid ") :])
            continue
        if line.startswith("msgstr "):
            active_field = "translation"
            entry[active_field] = po_literal(line[len("msgstr ") :])
            continue
        if line.startswith("msgstr[0] "):
            active_field = "translation"
            entry[active_field] = po_literal(line[len("msgstr[0] ") :])
            continue
        if line.startswith("msgstr["):
            active_field = None
            continue
        if line.startswith('"') and active_field is not None:
            entry[active_field] += po_literal(line)

    flush()
    return messages


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    indent = 2
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if re.search(r'\n    "[^"]+":', existing):
            indent = 4
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )


def gather_ui_strings(value: Any, result: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in {"icon_data", "icon_filename", "icon_svg_data"}:
                result.add(key)
                gather_ui_strings(child, result)
    elif isinstance(value, list):
        for child in value:
            gather_ui_strings(child, result)


def translated_value(translations: dict[str, str], source: Any) -> str | None:
    if not isinstance(source, str) or not source:
        return None
    translated = translations.get(source)
    return translated if isinstance(translated, str) and translated else None


def build_catalog(
    locale: str,
    definitions: dict[str, Any],
    ui_strings: set[str],
    translations: dict[str, str],
) -> dict[str, Any]:
    settings: dict[str, dict[str, Any]] = {}

    for setting_key, definition in definitions.items():
        if not isinstance(definition, dict):
            continue

        legacy: dict[str, Any] = {}
        for field in ("label", "full_label", "category", "tooltip"):
            translated = translated_value(translations, definition.get(field))
            if translated is not None:
                legacy[field] = translated

        enum_values = definition.get("enum_values")
        enum_labels = definition.get("enum_labels")
        if isinstance(enum_values, list) and isinstance(enum_labels, list):
            legacy_enum: list[str] = []
            for _, label in zip(enum_values, enum_labels, strict=False):
                translated = translated_value(translations, label)
                if translated is None:
                    legacy_enum.append(label)
                    continue
                legacy_enum.append(translated)
            if legacy_enum:
                legacy["enum_labels"] = legacy_enum

        if legacy:
            settings[setting_key] = legacy

    ui = {}
    for source in sorted(ui_strings):
        translated = translated_value(translations, source)
        if translated is not None:
            ui[source] = translated

    return {
        "schema_version": 1,
        "locale": locale,
        "source_locale": "en",
        "messages": {},
        "settings": dict(sorted(settings.items())),
        "ui": ui,
    }


def generate(source: Path, output: Path, domain: str) -> None:
    localization = source / "localization" / "i18n"
    if not localization.is_dir():
        raise SystemExit(f"Missing gettext source directory: {localization}")

    definitions_path = output / "print_config_def.json"
    definitions = read_json(definitions_path)
    if not isinstance(definitions, dict):
        raise SystemExit(f"Expected an object in {definitions_path}")

    ui_strings: set[str] = set()
    for filename in ("filament.json", "machine.json", "process.json"):
        path = output / filename
        if path.is_file():
            gather_ui_strings(read_json(path), ui_strings)

    catalogs: dict[str, dict[str, Any]] = {}
    for locale_dir in sorted(localization.iterdir()):
        if not locale_dir.is_dir():
            continue
        candidates = sorted(locale_dir.glob(f"{domain}_*.po"))
        if not candidates:
            continue
        locale = canonical_locale(locale_dir.name)
        if locale == "en":
            continue
        catalog = build_catalog(
            locale,
            definitions,
            ui_strings,
            load_po(candidates[0]),
        )
        if not catalog["messages"] and not catalog["ui"]:
            continue
        if locale in catalogs:
            raise SystemExit(
                f"Multiple upstream gettext directories map to canonical locale {locale}"
            )
        catalogs[locale] = catalog

    translation_dir = output / "translations"
    if translation_dir.is_dir():
        for stale in translation_dir.glob("*.json"):
            stale.unlink()

    locales = {}
    for locale, catalog in sorted(catalogs.items()):
        filename = f"{locale}.json"
        write_json(translation_dir / filename, catalog)
        locales[locale] = filename
    write_json(
        translation_dir / "_index.json",
        {
            "schema_version": 1,
            "source_locale": "en",
            "locales": locales,
        },
    )

    metadata_path = output / "ui_metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    if not isinstance(metadata, dict):
        raise SystemExit(f"Expected an object in {metadata_path}")
    metadata["schema_version"] = metadata.get("schema_version", 1)
    metadata["translations"] = {
        "index": "translations/_index.json",
        "locales": sorted(locales),
    }
    write_json(metadata_path, metadata)

    print(
        f"Generated {len(catalogs)} canonical {domain} translation catalogs "
        f"at {translation_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    args = parser.parse_args()
    generate(
        args.source.resolve(),
        args.output.resolve(),
        args.domain,
    )


if __name__ == "__main__":
    main()
