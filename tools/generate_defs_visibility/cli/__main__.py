import argparse
import json
import re
from pathlib import Path

import deepdiff
import requests

from .map_missing_variables import map_missing_variables
from .parse_conditions import ParseConditionalVisibility, ConditionalVisibility

SUPPORTED_SLICERS = {
    'OrcaSlicer':  {
        'toggle_print_fff_options': (
            "https://raw.githubusercontent.com/SoftFever/OrcaSlicer/{}/src/slic3r/GUI/ConfigManipulation.cpp",
            "ConfigManipulation::toggle_print_fff_options",
            "main",
        )
    },
    'PrusaSlicer': {
        'toggle_print_fff_options': (
            # PrusaSlicer 3 moved the toggles into AbstractConfigManipulation.
            (
                "https://raw.githubusercontent.com/prusa3d/PrusaSlicer/{}/src/slic3r-shared/src/Slic3r/Biz/Preset/AbstractConfigManipulation.cpp",
                "https://raw.githubusercontent.com/prusa3d/PrusaSlicer/{}/src/slic3r/GUI/ConfigManipulation.cpp",
            ),
            "ConfigManipulation::toggle_print_fff_options",
            "main",
        ),
    },
    'BambuStudio': {

        'toggle_print_fff_options': (
            "https://raw.githubusercontent.com/bambulab/BambuStudio/{}/src/slic3r/GUI/ConfigManipulation.cpp",
            "ConfigManipulation::toggle_print_fff_options",
            "master",
        )
    },
    'ElegooSlicer': {
        'toggle_print_fff_options': (
            "https://raw.githubusercontent.com/ELEGOO-3D/ElegooSlicer/{}/src/slic3r/GUI/ConfigManipulation.cpp",
            "ConfigManipulation::toggle_print_fff_options",
            "main",
        )
    },
    'CrealityPrint': {
        'toggle_print_fff_options': (
            "https://raw.githubusercontent.com/CrealityOfficial/CrealityPrint/{}/src/slic3r/GUI/ConfigManipulation.cpp",
            "ConfigManipulation::toggle_print_fff_options",
            "master",
        )
    },
}

# Where each slicer maps its C++ enum members to config strings (s_keys_map_<Enum>).
ENUM_KEY_SOURCES = {
    'OrcaSlicer': ("https://raw.githubusercontent.com/SoftFever/OrcaSlicer/{}/src/libslic3r/PrintConfig.cpp",),
    'PrusaSlicer': (
        "https://raw.githubusercontent.com/prusa3d/PrusaSlicer/{}/src/slic3r-shared/src/Slic3r/Biz/Config/Legacy/PrintConfig.cpp",
        "https://raw.githubusercontent.com/prusa3d/PrusaSlicer/{}/src/libslic3r/PrintConfig.cpp",
    ),
    'BambuStudio': ("https://raw.githubusercontent.com/bambulab/BambuStudio/{}/src/libslic3r/PrintConfig.cpp",),
    'ElegooSlicer': ("https://raw.githubusercontent.com/ELEGOO-3D/ElegooSlicer/{}/src/libslic3r/PrintConfig.cpp",),
    'CrealityPrint': ("https://raw.githubusercontent.com/CrealityOfficial/CrealityPrint/{}/src/libslic3r/PrintConfig.cpp",),
}

# These capabilities are upstream printer-resource metadata, not slicer settings.
# Keep the family policy in the producer so the browser only executes declared,
# portable helpers and unknown/future models retain the conservative false fallback.
NATIVE_FUNCTION_DEFINITIONS = {
    'BambuStudio': {
        'printerSupports': {
            'kind': 'printer_capability',
            'setting_prefix': 'support_',
            'fallback': False,
            'capabilities': {
                'wrapping_detection': [
                    'N7', 'O1C', 'O1C2', 'O1D', 'O1E', 'O1S',
                    'Bambu Lab P2S', 'Bambu Lab H2C', 'Bambu Lab H2D',
                    'Bambu Lab H2D Pro', 'Bambu Lab H2S',
                ],
            },
        },
    },
    'OrcaSlicer': {
        'printerSupports': {
            'kind': 'printer_capability',
            'setting_prefix': 'support_',
            'fallback': False,
            'capabilities': {
                'wrapping_detection': [
                    'N7', 'O1C', 'O1C2', 'O1D', 'O1E',
                    'Bambu Lab P2S', 'Bambu Lab H2C', 'Bambu Lab H2D',
                    'Bambu Lab H2D Pro',
                ],
            },
        },
    },
    'CrealityPrint': {
        'printerMatchesFamily': {
            'kind': 'printer_family',
            'fallback': False,
            'families': {
                'k2_series': {
                    'match': 'prefix',
                    'normalize': 'trim_lowercase',
                    'values': ['creality k2'],
                },
            },
        },
    },
}


# OrcaSlicer's is_separable_infill_pattern (src/libslic3r/PrintConfig.hpp) returns true for these.
ORCA_SEPARABLE_INFILL_PATTERNS = (
    'ipMonotonic', 'ipMonotonicLine', 'ipRectilinear', 'ipAlignedRectilinear', 'ipZigZag',
    'ipCrossZag', 'ipLockedZag', 'ipLine', 'ipGrid', 'ipTriangles', 'ipStars', 'ipCubic',
    'ipQuarterCubic', 'ipHoneycomb', 'ip3DHoneycomb', 'ipLateralHoneycomb', 'ipLateralLattice',
    'ipCrossHatch', 'ipTpmsD', 'ipTpmsFK', 'ipGyroid', 'ipHilbertCurve', 'ipArchimedeanChords',
    'ipOctagramSpiral',
)

# Helpers the toggle function calls with one argument. The browser cannot call them, so
# each call is replaced by the helper's body as one expression before parsing, with
# `{arg}` standing for the argument. Copied from upstream: update them when it changes.
INLINE_HELPERS = {
    'CrealityPrint': {
        # 7.3.0: Field infill sets its own density through interior_coefficient.
        'has_effective_sparse_infill': (
            '{arg}->option<ConfigOptionPercent>("sparse_infill_density")->value > 0'
            ' || ({arg}->opt_enum<InfillPattern>("sparse_infill_pattern") == ipField'
            ' && {arg}->option<ConfigOptionPercent>("interior_coefficient")->value > 0)'
        ),
    },
    'OrcaSlicer': {
        # Both from src/libslic3r/PrintConfig.hpp.
        'is_adaptive_infill_pattern': '{arg} == ipGyroid || {arg} == ipTpmsD || {arg} == ipTpmsFK',
        'is_separable_infill_pattern': ' || '.join(f'{{arg}} == {p}' for p in ORCA_SEPARABLE_INFILL_PATTERNS),
    },
}

# Undeclared helpers the web evaluator implements itself (LEGACY_VISIBILITY_FUNCTIONS
# in web's visibility-evaluator.ts). It rejects a file that calls any other helper
# without a function definition.
BROWSER_FUNCTIONS = {'is_auto', 'is_tree', 'creality::is_k2_series_printer_from_string', 'extruderValues'}


# A one-parameter lambda whose body is a single return, e.g. OrcaSlicer's
# `auto is_centered_fill = [](InfillPattern p) { return p == ipConcentric || ...; };`
LAMBDA = re.compile(
    r'auto\s+(\w+)\s*=\s*\[[^\]]*\]\s*\(\s*[\w:<>\s]+?[\s&*](\w+)\s*\)\s*\{\s*return\s+([^;{}]*);\s*\}\s*;',
)


def lambda_helpers(code: str) -> tuple[str, dict[str, str]]:
    """Remove one-expression lambdas from the code and return them as helpers."""
    helpers = {}

    def collect(match: re.Match) -> str:
        name, param, body = match.groups()
        helpers[name] = re.sub(rf'\b{param}\b', '{arg}', body)
        return ''

    return LAMBDA.sub(collect, code), helpers


def inline_helpers(code: str, helpers: dict[str, str]) -> str:
    """Replace each one-argument call of a helper by its body."""
    for name, expression in helpers.items():
        call = re.compile(rf'(?<![\w.>:]){re.escape(name)}\s*\(')
        out, pos = [], 0
        while match := call.search(code, pos):
            depth, end = 1, match.end()
            while depth:
                depth += {'(': 1, ')': -1}.get(code[end], 0)
                end += 1
            argument = code[match.end():end - 1].strip()
            if not re.fullmatch(r'\w+', argument):
                argument = f'({argument})'
            out += [code[pos:match.start()], '(', expression.replace('{arg}', argument), ')']
            pos = end
        code = ''.join(out) + code[pos:]
    return code


def undefined_functions(cv: ConditionalVisibility) -> list[str]:
    return sorted(set(cv.functions) - set(cv.function_definitions) - BROWSER_FUNCTIONS)


def enum_keys(code: str) -> dict[str, dict[str, list[str]]]:
    """Read every `s_keys_map_<Enum> { { "key", member }, ... }` as {Enum: {member: [keys]}}."""
    maps = {}
    for enum, body in re.findall(r's_keys_map_(\w+)\s*=?\s*\{(.*?)\n\s*\};', code, re.DOTALL):
        members = maps.setdefault(enum, {})
        for key, member in re.findall(r'\{\s*"([^"]*)"\s*,\s*(?:[\w:]+::)?(\w+)\s*\}', body):
            members.setdefault(member, []).append(key)
    return maps


def fetch_enum_keys(urls: tuple[str, ...]) -> dict[str, dict[str, list[str]]]:
    """The enum key maps from the first URL that exists."""
    for url in urls:
        response = requests.get(url)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        return enum_keys(response.text)
    print(f"No enum key maps at {urls}")
    return {}


def extract_function(code: str, function_name: str) -> str | None:
    pattern = re.compile(
        rf'((?:[\w\s:*&<>]+)?{re.escape(function_name)}\s*\(.*?\)\s*{{)',
        re.DOTALL
    )
    match = pattern.search(code)
    if not match:
        return None
    start = match.start()
    brace_count = 0
    for idx in range(start, len(code)):
        if code[idx] == '{':
            brace_count += 1
        elif code[idx] == '}':
            brace_count -= 1
            if brace_count == 0:
                return code[start:idx + 1]
    return None


def fetch_and_extract(
        urls: tuple[str, ...],
        function_name: str,
) -> str | None:
    """Return the function from the first URL that exists."""
    for url in urls:
        response = requests.get(url)
        if response.status_code == 404 and url != urls[-1]:
            continue
        response.raise_for_status()
        return extract_function(response.text, function_name)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract conditional visibility from slic3r based slicers toggle functions"
    )

    parser.add_argument(
        "--slicer", "-s",
        type=str,
        choices=SUPPORTED_SLICERS.keys(),
        required=True,
        help="Slicer to use"
    )

    parser.add_argument(
        "--ref", "-r",
        type=str,
        required=True,
        help="Slicer version (git ref) to use"
    )

    parser.add_argument(
        "--work-dir", "-w",
        type=Path,
        default=Path('.'),
        help="Working directory"
    )

    parser.add_argument(
        "--cache-dir", "-c",
        type=Path,
        default=Path('cache'),
        help="Cache directory relative to work-dir"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use"
    )

    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    cache_dir = (work_dir / args.cache_dir).resolve()
    config_def_path = work_dir / "print_config_def.json"
    config_def = json.load(config_def_path.open('r'))

    cache_dir.mkdir(parents=True, exist_ok=True)

    slicer_settings = SUPPORTED_SLICERS[args.slicer]

    code_urls, func_name, default_ref = slicer_settings['toggle_print_fff_options']
    if isinstance(code_urls, str):
        code_urls = (code_urls,)

    ref = args.ref or default_ref
    code = fetch_and_extract(tuple(url.format(ref) for url in code_urls), func_name)
    code, lambdas = lambda_helpers(code)
    code = inline_helpers(code, {**INLINE_HELPERS.get(args.slicer, {}), **lambdas})

    # Process input.
    p = ParseConditionalVisibility(
        config_def,
        code=code,
        function_definitions=NATIVE_FUNCTION_DEFINITIONS.get(args.slicer),
    )
    cv = p.process()

    cache_file = cache_dir / f"{args.slicer}_intermediate_cv.json"

    if cache_file.exists():
        fc = ConditionalVisibility.model_validate_json(cache_file.read_text())
        diff = deepdiff.DeepDiff(fc, cv, ignore_order=True, ignore_private_variables=True)

        if not diff:
            print(f"Cache file {cache_file} is up to date.")
        else:
            cache_file.write_text(cv.model_dump_json())
    else:
        cache_file.write_text(cv.model_dump_json())

    # Fill out enum values using OpenAI API.
    # and substitute the variables in the code.
    enum_key_urls = tuple(url.format(ref) for url in ENUM_KEY_SOURCES.get(args.slicer, ()))
    map_missing_variables(cv, config_def, fetch_enum_keys(enum_key_urls))

    cv.format_conditions()

    if undefined := undefined_functions(cv):
        raise SystemExit(f"{args.slicer} {ref}: visibility calls helpers the browser cannot evaluate: {undefined}")

    # Sorted, as for-loop settings come out in a different order on each run.
    cv.conditions = dict(sorted(cv.conditions.items()))

    with (work_dir / 'conditional_visibility.json').open('w') as f:
        f.write(cv.model_dump_json(indent=2))

    print(f"Conditional visibility saved to {work_dir / 'conditional_visibility.json'}")


if __name__ == "__main__":
    main()
