import json
import re

import openai

from .parse_conditions import ConditionalVisibility, BinaryOp, Variable, LitStr


ENUM_VALUE_ALIASES = {
    'btautobrim': 'auto_brim',
    'btear': 'brim_ears',
    'btbrimears': 'brim_ears',
    'btnobrim': 'no_brim',
    'gcfklipper': 'klipper',
    'gcfmarlinfirmware': 'marlin',
    'iprectilinear': 'line',
    'ipstars': 'tri-hexagon',
    'noironing': 'no ironing',
    'rear': 'back',
    'treeorganic': 'organic',
}


def normalize_enum_token(value: str) -> str:
    return re.sub(r'[^a-z0-9]', '', value.lower())


def deterministic_enum_value(variable: str, enum_values: list[str]) -> str | None:
    member = variable.rsplit('::', 1)[-1]
    tokens = [member]
    for prefix in ('bt', 'ds', 'gcf', 'gft', 'ip', 'sms', 'sp', 'wtw'):
        if member.lower().startswith(prefix) and len(member) > len(prefix):
            tokens.append(member[len(prefix):])

    normalized_values = {normalize_enum_token(value): value for value in enum_values}
    for token in tokens:
        alias = ENUM_VALUE_ALIASES.get(normalize_enum_token(token))
        if alias in enum_values:
            return alias
        matched = normalized_values.get(normalize_enum_token(token))
        if matched is not None:
            return matched
    return None


def map_missing_variables(cv: ConditionalVisibility, config_def: dict):
    # Only keep the config_def "enum_values" field and the key, throw away the rest.
    config_def = {
        k: v['enum_values']
        for k, v in config_def.items()
        if 'enum_values' in v
    }

    config_keys_variable_relations = {}

    for v in cv.variables:
        # Find all comparison operations for the variable between
        # a variable and a setting from config_def.
        usages = cv.find_variable_usages(v)
        usages = filter(lambda u: isinstance(u, BinaryOp), usages)
        usages = filter(lambda u: u.op in ('==', '!='), usages)
        usages = filter(lambda u: isinstance(u.left, Variable) and isinstance(u.right, Variable), usages)
        usages = filter(lambda u: u.left.name in config_def or u.right.name in config_def, usages)
        usages = map(lambda u: (u.left.name, u.right.name), usages)
        usages = map(lambda u: (u[1], u[0]) if u[0] == v else (u[0], u[1]), usages)

        for ck, vx in usages:
            if ck not in config_def:
                continue

            assert v == vx

            if ck not in config_keys_variable_relations:
                config_keys_variable_relations[ck] = set()

            config_keys_variable_relations[ck].add(v)

    # Remove all config keys that are not used by any variable.
    mapping_request = [
        {
            'variables':   list(config_keys_variable_relations[k]),
            'enum_values': config_def[k]
        }

        for k, v in config_def.items()
        if k in config_keys_variable_relations
    ]

    if not mapping_request:
        return

    mappings = {}
    unresolved_request = []
    for request in mapping_request:
        unresolved = []
        for variable in request['variables']:
            value = deterministic_enum_value(variable, request['enum_values'])
            if value is None:
                unresolved.append(variable)
            else:
                mappings[variable] = value
        if unresolved:
            unresolved_request.append({**request, 'variables': unresolved})

    if unresolved_request:
        print("Unresolved enum mappings:", json.dumps(unresolved_request))
        print("Calling OpenAI API to map unresolved enum values...")
        try:
            mappings.update(map_enum_variables(unresolved_request))
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            exit(1)

    for var, enum_value in mappings.items():
        cv.subst_var(var, LitStr(value=enum_value))


def map_enum_variables(enum_definitions, model="gpt-4o") -> dict:
    functions = [
        {
            "name":        "store_enum_mapping",
            "description": "Store the mapping between variables and enum values.",
            "parameters":  {
                "type":       "object",
                "properties": {
                    "mapping": {
                        "type":                 "object",
                        "additionalProperties": {
                            "type": "string"
                        }
                    }
                },
                "required":   ["mapping"]
            }
        }
    ]

    messages = [
        {"role":    "system",
         "content": "Your only purpose is to decide what variable correspond to what single enum value and only store the mapping between every variable and its enum value."},
        {
            "role":    "user",
            "content": f"""{json.dumps(enum_definitions, indent=2)}"""
        }
    ]

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call={"name": "store_enum_mapping"},
        temperature=0,
    )

    msg = response.choices[0].message

    if not msg.function_call or msg.function_call.name != "store_enum_mapping" or msg.function_call.arguments is None:
        raise ValueError("No function call in response")

    arguments = json.loads(msg.function_call.arguments)
    mapping = arguments.get("mapping", arguments)

    if not isinstance(mapping, dict):
        raise ValueError("OpenAI enum mapping response must be an object")

    non_string_values = {
        key: value
        for key, value in mapping.items()
        if not isinstance(value, str)
    }
    if non_string_values:
        raise ValueError(f"OpenAI enum mapping contains non-string values: {non_string_values}")

    return mapping
