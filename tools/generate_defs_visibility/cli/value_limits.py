"""Extract the value limits the slicer GUI enforces outside print_config_def.

Bambu-family GUIs reset some values in code instead of declaring a min/max, e.g.

    if (config->opt_float("ironing_spacing") < 0.05) {
        ... new_conf.set_key_value("ironing_spacing", new ConfigOptionFloat(0.1)); ...
    }

The exported definition then says min 0, but a CLI slice with 0 never finishes.
This module finds such rules in ConfigManipulation.cpp and writes them as data
(gui_value_limits.json), so consumers enforce them with their generic range check.

A rule is classified only when all of these hold, otherwise it is listed as unclassified:
  - the condition compares one option's value with a constant (literal or named
    constant resolved from the sources), optionally through abs(), a local variable
    or a per-item loop over the option's values;
  - other `&&` terms are only GUI state (`wxGetApp()...`) or a nil check of that option;
  - the `if` is not nested in another `if` (so it holds unconditionally);
  - its body resets that same option.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests
from tree_sitter import Language, Node, Parser
from tree_sitter_cpp import language

CPP_LANG = Language(language())
PARSER = Parser(CPP_LANG)

OUTPUT_FILE = 'gui_value_limits.json'

NUMERIC_TYPES = {
    'int', 'ints', 'float', 'floats', 'percent', 'percents', 'floatorpercent', 'floatorpercents',
}

# Comparison that triggers the reset -> (bound, exclusive) of the allowed range.
ALLOWED = {
    '<': ('min', False),
    '<=': ('min', True),
    '>': ('max', False),
    '>=': ('max', True),
}
FLIPPED = {'<': '>', '<=': '>=', '>': '<', '>=': '<='}

ABS_FUNCTIONS = {'abs', 'fabs', 'std::abs', 'std::fabs'}
# Calls that return the value (or the option holding it) of the setting named by a string literal.
ACCESSORS = re.compile(r'^(opt_\w+|option)$')
VALUE_FIELDS = {'value', 'values'}

_CONSTANT_DECL = r'(?:\b(?:static|inline|constexpr|const)\s+)+[\w:]+\s+{name}\s*=\s*([^;]+);'
_CONSTANT_DEFINE = r'#\s*define\s+{name}\s+(.+)'


def _text(node: Node | None) -> str:
    return node.text.decode() if node is not None and node.text is not None else ''


def _strip(node: Node) -> Node:
    while node.type in ('parenthesized_expression', 'condition_clause'):
        node = node.named_children[0] if node.type == 'parenthesized_expression' else node.child_by_field_name('value')
    return node


def _walk(node: Node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _string_arg(call: Node) -> str | None:
    args = call.child_by_field_name('arguments')
    if args is None or not args.named_children:
        return None
    first = args.named_children[0]
    if first.type != 'string_literal':
        return None
    content = [c for c in first.named_children if c.type == 'string_content']
    return _text(content[0]) if content else ''


def _callee_name(call: Node) -> str:
    fn = call.child_by_field_name('function')
    if fn is None:
        return ''
    if fn.type == 'field_expression':
        field = fn.child_by_field_name('field')
        if field is not None and field.type == 'template_method':
            field = field.child_by_field_name('name')
        return _text(field)
    if fn.type in ('template_function',):
        return _text(fn.child_by_field_name('name'))
    return _text(fn)


def _line(node: Node) -> int:
    return node.start_point[0] + 1


class Constants:
    """Named numeric constants, resolved from the given source texts on demand."""

    def __init__(self, sources: list[str]):
        self.sources = sources
        self.cache: dict[str, float | None] = {}

    def get(self, name: str) -> float | None:
        name = name.split('::')[-1]
        if name in self.cache:
            return self.cache[name]
        self.cache[name] = None  # Guards against self-reference.
        for source in self.sources:
            for pattern in (_CONSTANT_DECL, _CONSTANT_DEFINE):
                match = re.search(pattern.format(name=re.escape(name)), source)
                if match:
                    value = self.evaluate_text(match.group(1).split('//')[0].strip())
                    if value is not None:
                        self.cache[name] = value
                        return value
        return None

    def evaluate_text(self, expression: str) -> float | None:
        tree = PARSER.parse(f'auto __x = {expression};'.encode())
        decl = next((n for n in _walk(tree.root_node) if n.type == 'init_declarator'), None)
        value = decl.child_by_field_name('value') if decl is not None else None
        return evaluate(value, self, Env()) if value is not None else None


def evaluate(node: Node, constants: Constants, env: Env) -> float | None:
    """Value of a constant expression, or None when it depends on anything else."""
    node = _strip(node)
    if node.type == 'number_literal':
        literal = _text(node).replace("'", '').rstrip('fFlLuU')
        try:
            return float(int(literal, 0)) if re.fullmatch(r'0[xX][0-9a-fA-F]+|\d+', literal) else float(literal)
        except ValueError:
            return None
    if node.type in ('identifier', 'qualified_identifier'):
        if _text(node) in env:
            return None
        return constants.get(_text(node))
    if node.type == 'cast_expression':
        return evaluate(node.child_by_field_name('value'), constants, env)
    if node.type == 'unary_expression':
        value = evaluate(node.child_by_field_name('argument'), constants, env)
        op = _text(node.child_by_field_name('operator'))
        if value is None or op not in ('-', '+'):
            return None
        return -value if op == '-' else value
    if node.type == 'binary_expression':
        left = evaluate(node.child_by_field_name('left'), constants, env)
        right = evaluate(node.child_by_field_name('right'), constants, env)
        op = _text(node.child_by_field_name('operator'))
        if left is None or right is None:
            return None
        if op == '+':
            return left + right
        if op == '-':
            return left - right
        if op == '*':
            return left * right
        if op == '/' and right != 0:
            return left / right
    return None


class Access:
    """An expression that reads one setting's value."""

    def __init__(self, key: str, absolute: bool = False, per_item: bool = False):
        self.key = key
        self.absolute = absolute
        self.per_item = per_item


def access(node: Node, env: Env, seen: frozenset = frozenset()) -> Access | None:
    """The setting `node` reads as a plain value, or None for anything derived from it."""
    node = _strip(node)
    if node.type == 'call_expression':
        name = _callee_name(node)
        if ACCESSORS.match(name):
            key = _string_arg(node)
            return Access(key) if key else None
        if _text(node.child_by_field_name('function')) in ABS_FUNCTIONS:
            args = node.child_by_field_name('arguments').named_children
            inner = access(args[0], env, seen) if len(args) == 1 else None
            return Access(inner.key, True, inner.per_item) if inner else None
        if name == 'get_at':
            # option<...>("key")->get_at(0)
            fn = node.child_by_field_name('function')
            return access(fn.child_by_field_name('argument'), env, seen)
        return None
    if node.type == 'field_expression':
        if _text(node.child_by_field_name('field')) not in VALUE_FIELDS:
            return None
        return access(node.child_by_field_name('argument'), env, seen)
    if node.type == 'subscript_expression':
        inner = access(node.child_by_field_name('argument'), env, seen)
        return Access(inner.key, inner.absolute, True) if inner else None
    if node.type == 'pointer_expression':
        return access(node.child_by_field_name('argument'), env, seen)
    if node.type == 'cast_expression':
        return access(node.child_by_field_name('value'), env, seen)
    if node.type == 'conditional_expression':
        # `config->has("key") ? config->opt_float("key") : <constant>`
        consequence = node.child_by_field_name('consequence')
        condition = _strip(node.child_by_field_name('condition'))
        inner = access(consequence, env, seen) if consequence is not None else None
        if inner and condition.type == 'call_expression' and _callee_name(condition) == 'has' \
                and _string_arg(condition) == inner.key:
            return inner
        return None
    if node.type == 'identifier':
        name = _text(node)
        if name in env and name not in seen:
            inner = access(env[name], env, seen | {name})
            if inner is not None and name in getattr(env, 'items', ()):
                return Access(inner.key, inner.absolute, True)
            return inner
    return None


def _function_name(node: Node) -> str:
    declarator = node.child_by_field_name('declarator')
    while declarator is not None and declarator.type != 'function_declarator':
        declarator = declarator.child_by_field_name('declarator')
    name = _text(declarator.child_by_field_name('declarator')) if declarator is not None else ''
    return name.split('::')[-1]


class Env(dict):
    """Local variable -> initializer; `items` are range-for variables over a vector."""

    def __init__(self):
        super().__init__()
        self.items: set[str] = set()


def _local_variables(function: Node) -> Env:
    env = Env()
    for node in _walk(function):
        if node.type == 'for_range_loop':
            declarator = node.child_by_field_name('declarator')
            while declarator is not None and declarator.type in ('pointer_declarator', 'reference_declarator'):
                declarator = declarator.named_children[-1]
            right = node.child_by_field_name('right')
            if declarator is not None and declarator.type == 'identifier' and right is not None:
                env[_text(declarator)] = right
                env.items.add(_text(declarator))
        if node.type == 'init_declarator':
            declarator = node.child_by_field_name('declarator')
            while declarator is not None and declarator.type in ('pointer_declarator', 'reference_declarator'):
                declarator = declarator.child_by_field_name('declarator') or declarator.named_children[-1]
            value = node.child_by_field_name('value')
            if declarator is not None and declarator.type == 'identifier' and value is not None:
                env[_text(declarator)] = value
    return env


def _conjuncts(node: Node) -> list[Node]:
    node = _strip(node)
    if node.type == 'binary_expression' and _text(node.child_by_field_name('operator')) == '&&':
        return _conjuncts(node.child_by_field_name('left')) + _conjuncts(node.child_by_field_name('right'))
    return [node]


def _ignorable(term: Node, key: str, env: Env) -> bool:
    """GUI state, or a nil check of the setting itself (nil items are never range-checked)."""
    text = _text(term)
    if re.fullmatch(r'wxGetApp\(\)\s*\.\s*initialized\(\)', text):
        return True
    term = _strip(term)
    if term.type == 'unary_expression' and _text(term.child_by_field_name('operator')) == '!':
        call = _strip(term.child_by_field_name('argument'))
        if call.type == 'call_expression' and _callee_name(call) == 'is_nil':
            receiver = call.child_by_field_name('function').child_by_field_name('argument')
            found = access(receiver, env)
            return found is not None and found.key == key
    return False


def _resets(body: Node, key: str, env: Env, constants: Constants, per_item: bool) -> tuple[bool, float | None]:
    """Whether `body` resets `key`, and the constant it resets to when known."""
    for node in _walk(body):
        if node.type == 'call_expression' and _callee_name(node) == 'set_key_value' and _string_arg(node) == key:
            args = node.child_by_field_name('arguments').named_children
            return True, _reset_value(args[1], constants) if len(args) > 1 else None
        if per_item and node.type == 'assignment_expression':
            target = access(node.child_by_field_name('left'), env)
            if target is not None and target.key == key and target.per_item:
                return True, evaluate(node.child_by_field_name('right'), constants, env)
    return False, None


def _reset_value(node: Node, constants: Constants) -> float | None:
    """The constant in `new ConfigOptionFloat(0.1)`, `ConfigOptionFloats({0.5})` or `{10, true}`."""
    for child in _walk(node):
        if child.type in ('argument_list', 'initializer_list') and child.named_children:
            first = child.named_children[0]
            if first.type != 'initializer_list':
                return evaluate(first, constants, Env())
    return None


def _enclosing_if(node: Node, stop: Node | None) -> Node | None:
    parent = node.parent
    while parent is not None and parent != stop:
        if parent.type == 'if_statement':
            return parent
        parent = parent.parent
    return None


def _number(value: float) -> float | int:
    rounded = round(value, 10)
    return int(rounded) if rounded == int(rounded) else rounded


def extract(code: str, constants: Constants) -> tuple[dict[str, dict], list[dict]]:
    """Return (limits by setting, unclassified value-setting calls)."""
    tree = PARSER.parse(code.encode())
    limits: dict[str, dict] = {}
    claimed: set[int] = set()  # writes explained by a classified rule
    writes: list[tuple[Node, str, str]] = []  # (node, setting, function)

    for function in (n for n in _walk(tree.root_node) if n.type == 'function_definition'):
        function_name = _function_name(function)
        if function_name == 'apply':
            continue
        env = _local_variables(function)

        for node in _walk(function):
            if node.type == 'call_expression' and _callee_name(node) == 'set_key_value':
                key = _string_arg(node) or _text(node.child_by_field_name('arguments').named_children[0])
                writes.append((node, key, function_name))
            elif node.type == 'assignment_expression':
                # e.g. `values[i] = 0.1`, or `speed = 0.5` for `double& speed : option->values`.
                # Reassigning a local copy (`layer_height = ...`) does not write the setting.
                left = _strip(node.child_by_field_name('left'))
                target = access(left, env)
                if target is not None and (left.type != 'identifier' or _text(left) in env.items):
                    writes.append((node, target.key, function_name))

        for statement in (n for n in _walk(function) if n.type == 'if_statement'):
            if _enclosing_if(statement, function) is not None:
                continue
            rule = _classify(statement, env, constants, function)
            if rule is None:
                continue
            key, bound, value, exclusive, absolute, per_item, reset = rule
            bounds = [(bound, value, exclusive)]
            if absolute:
                # abs(v) > x resets outside [-x, x]; abs(v) < x is not a range.
                if bound != 'max':
                    continue
                bounds = [('min', -value, exclusive), ('max', value, exclusive)]
            entry = limits.setdefault(key, {'_resets': set(), '_lines': []})
            for name, limit, is_exclusive in bounds:
                _merge(entry, name, _number(limit), is_exclusive)
            entry['_resets'].add(None if reset is None else _number(reset))
            entry['_lines'].append(_line(statement))
            # The per-item variant resets inside the loop and writes the vector back after it.
            scope = function if per_item else statement
            for node, written, _ in writes:
                if written == key and _within(node, scope):
                    claimed.add(node.id)

    for entry in limits.values():
        resets = entry.pop('_resets')
        entry['lines'] = sorted(entry.pop('_lines'))
        if len(resets) == 1 and None not in resets:
            entry['reset'] = next(iter(resets))

    unclassified = []
    for node, key, function_name in writes:
        if node.id in claimed:
            continue
        condition = _enclosing_if(node, None)
        condition_text = _text(condition.child_by_field_name('condition')) if condition is not None else ''
        unclassified.append({
            'key': key,
            'function': function_name,
            'line': _line(node),
            'condition': re.sub(r'\s+', ' ', condition_text).strip(),
        })
    return dict(sorted(limits.items())), unclassified


def _within(node: Node, scope: Node) -> bool:
    return scope.start_byte <= node.start_byte and node.end_byte <= scope.end_byte


def _merge(entry: dict, bound: str, value: float, exclusive: bool) -> None:
    """Keep the stricter of two bounds."""
    current = entry.get(bound)
    current_exclusive = entry.get(f'exclusive_{bound}', False)
    stricter = current is None or (value > current if bound == 'min' else value < current) \
        or (value == current and exclusive and not current_exclusive)
    if stricter:
        entry[bound] = value
        if exclusive:
            entry[f'exclusive_{bound}'] = True
        else:
            entry.pop(f'exclusive_{bound}', None)


def _classify(statement: Node, env: Env, constants: Constants, function: Node):
    condition = statement.child_by_field_name('condition')
    body = statement.child_by_field_name('consequence')
    if condition is None or body is None:
        return None
    terms = _conjuncts(condition)
    comparisons = []
    for term in terms:
        if term.type != 'binary_expression':
            continue
        op = _text(term.child_by_field_name('operator'))
        if op not in ALLOWED:
            continue
        left, right = term.child_by_field_name('left'), term.child_by_field_name('right')
        found = access(left, env)
        bound = evaluate(right, constants, env)
        if found is None or bound is None:
            found, bound, op = access(right, env), evaluate(left, constants, env), FLIPPED[op]
        if found is not None and bound is not None:
            comparisons.append((term, found, op, bound))
    if len(comparisons) != 1:
        return None
    term, found, op, bound = comparisons[0]
    if any(t is not term and not _ignorable(t, found.key, env) for t in terms):
        return None
    if found.per_item and _enclosing_loop(statement, function) is None:
        return None
    resets, reset = _resets(body, found.key, env, constants, found.per_item)
    if not resets:
        return None
    name, exclusive = ALLOWED[op]
    return found.key, name, bound, exclusive, found.absolute, found.per_item, reset


def _enclosing_loop(node: Node, stop: Node) -> Node | None:
    parent = node.parent
    while parent is not None and parent != stop:
        if parent.type in ('for_statement', 'for_range_loop', 'while_statement'):
            return parent
        parent = parent.parent
    return None


def apply_definitions(limits: dict[str, dict], unclassified: list[dict], config_def: dict | None) -> dict[str, dict]:
    """Drop limits for settings the dump does not define as numbers, listing them as unclassified."""
    if config_def is None:
        return limits
    kept = {}
    for key, entry in limits.items():
        definition = config_def.get(key)
        if isinstance(definition, dict) and definition.get('type') in NUMERIC_TYPES:
            kept[key] = entry
        else:
            for line in entry['lines']:
                unclassified.append({'key': key, 'function': '', 'line': line,
                                     'condition': 'limit on a setting that is not a numeric definition'})
    return kept


def fingerprint(item: dict) -> tuple[str, str, str]:
    """Identity of an unclassified call that survives line moves between versions."""
    return item['key'], item['function'], item['condition']


def new_unclassified(current: dict, previous: dict) -> list[dict]:
    known = {fingerprint(item) for item in previous.get('unclassified', [])}
    return [item for item in current.get('unclassified', []) if fingerprint(item) not in known]


def _fetch(url: str) -> str | None:
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text


def _header_urls(source_url: str) -> list[str]:
    root, _, path = source_url.partition('/src/')
    headers = [re.sub(r'\.cpp$', '.hpp', source_url)]
    headers += [f'{root}/src/libslic3r/{name}' for name in ('libslic3r.h', 'PrintConfig.hpp')]
    return [url for url in headers if url != source_url]


def fetch_sources(slicer: str, ref: str) -> tuple[str, str, list[str]]:
    """(source url, ConfigManipulation code, [code, headers...]) for the first source that exists."""
    from .__main__ import SUPPORTED_SLICERS

    if slicer not in SUPPORTED_SLICERS:
        raise SystemExit(f'Unsupported slicer {slicer}; choose from {sorted(SUPPORTED_SLICERS)} or pass --source')
    urls = SUPPORTED_SLICERS[slicer]['toggle_print_fff_options'][0]
    urls = (urls,) if isinstance(urls, str) else urls
    for template in urls:
        url = template.format(ref)
        code = _fetch(url)
        if code is None:
            continue
        headers = [text for header in _header_urls(url) if (text := _fetch(header)) is not None]
        return url, code, [code, *headers]
    raise SystemExit(f'{slicer} {ref}: no ConfigManipulation source at {urls}')


def build(code: str, constant_sources: list[str], config_def: dict | None, source: str) -> dict:
    limits, unclassified = extract(code, Constants(constant_sources))
    limits = apply_definitions(limits, unclassified, config_def)
    return {
        'source': source,
        'limits': limits,
        'unclassified': sorted(unclassified, key=lambda item: (item['line'], item['key'])),
    }


def report_new_unclassified(result: dict, previous_path: Path, strict: bool) -> int:
    if not previous_path.is_file():
        print(f'No previous {OUTPUT_FILE} at {previous_path}; nothing to compare.')
        return 0
    new = new_unclassified(result, json.loads(previous_path.read_text()))
    for item in new:
        # GitHub Actions annotation; plain text elsewhere.
        print(f"::warning title=New GUI value rule::{item['key']} set in {item['function'] or '?'}"
              f" (line {item['line']}) when `{item['condition'] or 'always'}`. Classify it or confirm it is not a value limit.")
    return 1 if new and strict else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--slicer', '-s', required=True)
    parser.add_argument('--ref', '-r', required=True, help='Upstream git ref to read the GUI source at')
    parser.add_argument('--work-dir', '-w', type=Path, default=Path('.'),
                        help='Version folder holding print_config_def.json; the output is written here')
    parser.add_argument('--source', type=Path, help='Read ConfigManipulation.cpp from this file instead of GitHub')
    parser.add_argument('--header', type=Path, action='append', default=[],
                        help='Extra file to resolve named constants from (with --source)')
    parser.add_argument('--list-unclassified', action='store_true',
                        help='Print value-setting calls that are not a simple threshold rule')
    parser.add_argument('--compare-to', type=Path,
                        help=f'Previous {OUTPUT_FILE}: warn about unclassified calls it does not have')
    parser.add_argument('--strict', action='store_true', help='With --compare-to, exit 1 on new unclassified calls')
    parser.add_argument('--dry-run', action='store_true', help='Do not write the output file')
    args = parser.parse_args(argv)

    if args.source is not None:
        code = args.source.read_text()
        sources = [code, *(path.read_text() for path in args.header)]
        source = args.source.name
    else:
        url, code, sources = fetch_sources(args.slicer, args.ref)
        source = 'src/' + url.partition('/src/')[2]

    config_def_path = args.work_dir / 'print_config_def.json'
    config_def = json.loads(config_def_path.read_text()) if config_def_path.is_file() else None
    result = build(code, sources, config_def, source)

    for key, entry in result['limits'].items():
        print(f"{key}: {json.dumps({k: v for k, v in entry.items() if k != 'lines'})}")
    if args.list_unclassified:
        for item in result['unclassified']:
            print(f"unclassified {item['key']} in {item['function']}:{item['line']} when `{item['condition'] or 'always'}`")

    if not args.dry_run:
        (args.work_dir / OUTPUT_FILE).write_text(json.dumps(result, indent=2) + '\n')
        print(f'GUI value limits saved to {args.work_dir / OUTPUT_FILE}')

    return report_new_unclassified(result, args.compare_to, args.strict) if args.compare_to else 0


if __name__ == '__main__':
    sys.exit(main())
