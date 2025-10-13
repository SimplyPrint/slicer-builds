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
            "https://raw.githubusercontent.com/prusa3d/PrusaSlicer/{}/src/slic3r/GUI/ConfigManipulation.cpp",
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
}


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
        url: str,
        function_name: str,
) -> str | None:
    response = requests.get(url)
    response.raise_for_status()
    function_code = extract_function(response.text, function_name)
    return function_code


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

    code_url, func_name, default_ref = slicer_settings['toggle_print_fff_options']

    code = fetch_and_extract(code_url.format(args.ref or default_ref), func_name)

    # Process input.
    p = ParseConditionalVisibility(config_def, code=code)
    cv = p.process()

    cache_file = cache_dir / f"{args.slicer}_intermediate_cv.json"

    if cache_file.exists():
        fc = ConditionalVisibility.model_validate_json(cache_file.read_text())
        diff = deepdiff.DeepDiff(fc, cv, ignore_order=True, ignore_private_variables=True)

        if not diff:
            print(f"Cache file {cache_file} is up to date.")
            return

    with cache_file.open('w') as f:
        f.write(cv.model_dump_json())

    # Fill out enum values using OpenAI API.
    # and substitute the variables in the code.
    map_missing_variables(cv, config_def)

    cv.format_conditions()

    with (work_dir / 'conditional_visibility.json').open('w') as f:
        f.write(cv.model_dump_json(indent=2))

    print(f"Conditional visibility saved to {work_dir / 'conditional_visibility.json'}")


if __name__ == "__main__":
    main()
