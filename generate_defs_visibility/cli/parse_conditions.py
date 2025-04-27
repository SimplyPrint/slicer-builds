import functools
from enum import StrEnum
from pathlib import Path
from types import NoneType
from typing import Union, Annotated, cast, Any

from pydantic import BaseModel, Discriminator, Tag
from tree_sitter import Language, Parser, Node, Tree
from tree_sitter_cpp import language

CPP_LANG: Language = Language(language())

# Find toggle_line and toggle_field calls with a constant string literal as the first argument
q_constant_toggle = CPP_LANG.query(fr"""
(call_expression
    (identifier) @func.name
    (#match? @func.name "toggle_(line|field)")
    (argument_list
        (string_literal (string_content) @func.setting)
        (_) @func.condition
    )
)""")


# Find toggle_line and toggle_field call with a variable as the first argument
@functools.cache
def q_ident_toggle(ident):
    return CPP_LANG.query(fr"""
(call_expression
    (identifier) @func.name
    (#match? @func.name "toggle_(line|field)")
    (argument_list
        (identifier) @_ident
        (#match? @_ident "{ident}")
        (_) @func.condition
    )
)""")


# Find the condition of an if statement
q_if_condition = CPP_LANG.query(r"""
(if_statement
    condition: (
        condition_clause
        value: (_) @condition
    )
)""")

# Find for_range_loop with a string literal as the initializer_list
q_for_loop_toggle = CPP_LANG.query(r"""
(for_range_loop
  type: (_)
  declarator: (identifier) @loop.var
  right: (initializer_list
           (string_literal (string_content))+
         ) @loop.settings
  body: (_) @loop.body
)""")

# Find string content in a string literal (used to get values out of initializer_list)
q_string_content = CPP_LANG.query(r"""(string_content) @string_content""")

q_initial_assigment = CPP_LANG.query(r"""
(init_declarator
    declarator: (identifier) @ident
    value: (_) @value
)
""")

q_opt_function_call = CPP_LANG.query(r'''
(call_expression
  function: [
    ;; free function call:   opt_bool("…")
    (identifier) @func.name

    ;; member call via . or ->, with NO template-args
    (field_expression
       field: (field_identifier) @func.name
    )

    ;; member call via . or ->, WITH <T> template-args
    (field_expression
       field: (template_method
                 name:      (field_identifier)    @func.name
                 arguments: (template_argument_list)
              )
    )
  ]
  arguments: (argument_list
               (string_literal (string_content) @func.setting)
             )
)
(#match? @func.name "^opt_[A-Za-z_][A-Za-z0-9_]*$")
''')


class Op(StrEnum):
    AND = '&&'
    OR = '||'
    EQ = '=='
    NOT_EQ = '!='
    LT = '<'
    LTE = '<='
    GT = '>'
    GTE = '>='
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    MOD = '%'
    BIT_AND = '&'
    BIT_OR = '|'
    BIT_XOR = '^'
    BIT_NOT = '~'
    NOT = '!'


# C‐style precedence levels (higher number binds tighter)
_prec = {
    Op.NOT:     14,
    Op.MUL:     13,
    Op.DIV:     13,
    Op.MOD:     13,
    Op.ADD:     12,
    Op.SUB:     12,
    Op.BIT_NOT: 11,
    Op.BIT_AND: 10,
    Op.BIT_XOR: 9,
    Op.BIT_OR:  8,
    Op.LT:      7,
    Op.LTE:     7,
    Op.GT:      7,
    Op.GTE:     7,
    Op.EQ:      6,
    Op.NOT_EQ:  6,
    Op.AND:     5,
    Op.OR:      4,
}


class Variable(BaseModel):
    name: str


class LitInteger(BaseModel):
    value: int


class LitFloat(BaseModel):
    value: float


class LitStr(BaseModel):
    value: str


class LitBool(BaseModel):
    value: bool


def get_expr_discriminator(v: Any) -> str | None:
    if v is None or v == 'null':
        return 'none'

    if isinstance(v, dict):
        # Check what fields exist and what keys they have
        if 'name' in v:
            if 'args' in v:
                return 'func'
            return 'var'
        if 'value' in v:
            if isinstance(v['value'], bool):
                return 'bool'
            if isinstance(v['value'], str):
                return 'str'
            if isinstance(v['value'], int):
                return 'int'
            if isinstance(v['value'], float):
                return 'float'
        if 'op' in v:
            if 'left' in v and 'right' in v:
                return 'binop'
            if 'expr' in v:
                return 'unop'

    if isinstance(v, BaseModel):
        return {
            Variable:   'var',
            LitInteger: 'int',
            LitFloat:   'float',
            LitStr:     'str',
            LitBool:    'bool',
            Function:   'func',
            BinaryOp:   'binop',
            UnaryOp:    'unop',
            NoneType:   'none',
        }.get(type(cast(TExpr | None, v)))

    return None


TExpr = Annotated[
    Union[
        Annotated['Variable', Tag('var')],
        Annotated['LitInteger', Tag('int')],
        Annotated['LitFloat', Tag('float')],
        Annotated['LitStr', Tag('str')],
        Annotated['LitBool', Tag('bool')],
        Annotated['Function', Tag('func')],
        Annotated['BinaryOp', Tag('binop')],
        Annotated['UnaryOp', Tag('unop')],
        Annotated[NoneType, Tag('none')],
    ],
    Discriminator(get_expr_discriminator)
]


class Function(BaseModel):
    name: str
    args: list[TExpr]


class BinaryOp(BaseModel):
    op: Op
    left: TExpr
    right: TExpr


class UnaryOp(BaseModel):
    op: Op
    expr: TExpr


def pretty_print(expr: TExpr, pp: int = 0) -> str:
    p = 0

    match expr:
        case Variable(name=name):
            s = name
        case LitInteger(value=value):
            s = str(value)
        case LitFloat(value=value):
            s = repr(value)
        case LitStr(value=value):
            # escape as in Python
            s = repr(value)
        case LitBool(value=value):
            s = "true" if value else "false"
        case Function(name=name, args=args):
            args = ", ".join(pretty_print(a, 0) for a in args)
            s = f"{name}({args})"
        case BinaryOp(op=op, left=left, right=right):
            p = _prec[op]
            # left is left‐associative: same‐precedence child needs parentheses on the right
            left_s = pretty_print(left, p)
            right_s = pretty_print(right, p + 1)
            s = f"{left_s} {op} {right_s}"
        case UnaryOp(op=op, expr=expr):
            p = _prec[op]
            inner = pretty_print(expr, p)
            s = f"{op}{inner}"
        case None:
            s = "null"
        case _:
            raise TypeError(f"Unknown Expr: {expr}")

    # Add parentheses if this subexpr binds more weakly than its context
    if pp and p < pp and isinstance(expr, (BinaryOp, UnaryOp)):
        return f"({s})"

    return s


class ConditionalVisibility(BaseModel):
    variables: list[str]  # External variables that are used in the conditions
    functions: list[str]  # External functions that are used in the conditions
    conditions: dict[str, TExpr | str | None]  # Mapping of setting name to condition expression

    # prepare for final output.
    def format_conditions(self):
        for k, v in self.conditions.items():
            if isinstance(v, str):
                self.conditions[k] = v
            else:
                self.conditions[k] = pretty_print(v)

    def find_variable_usages(self, v: str) -> list[TExpr]:
        # Create a list of all external variables and where they are used, convert expr to str
        exprs = []

        for expr in self.conditions.values():
            self.find_all_variable_usages_in_expr(v, expr, exprs)

        return exprs

    def find_all_variable_usages_in_expr(self, var: str, expr: TExpr, out: list[TExpr]) -> bool:
        """Find all *direct* usages of a variable in an expression."""
        match expr:
            case Variable(name=name):
                # Mark for the parent that the variable is used
                if name == var:
                    return True

            case Function(args=args):
                # Check all arguments
                for arg in args:
                    if self.find_all_variable_usages_in_expr(var, arg, out):
                        out.append(expr)

            case UnaryOp(expr=expr):
                # Check the expression
                if self.find_all_variable_usages_in_expr(var, expr, out):
                    out.append(expr)

            case BinaryOp(left=left, right=right):
                # Check left and right
                if self.find_all_variable_usages_in_expr(var, left, out):
                    out.append(expr)
                if self.find_all_variable_usages_in_expr(var, right, out):
                    out.append(expr)

        return False

    def subst_var(self, var: str, value: TExpr):
        for k, v in self.conditions.items():
            if isinstance(v, str) or v is None:
                continue

            self.conditions[k] = self.subst_var_single(var, value, v)

        self.variables.remove(var)

    def subst_var_single(self, var: str, value: TExpr, condition: TExpr) -> TExpr:
        match condition:
            case Variable(name=name):
                if name == var:
                    return value
                return condition
            case Function(name=name, args=args):
                # Substitute all arguments
                args = [self.subst_var_single(var, value, arg) for arg in args]
                return Function(name=name, args=args)
            case BinaryOp(op=op, left=left, right=right):
                # Substitute left and right
                left = self.subst_var_single(var, value, left)
                right = self.subst_var_single(var, value, right)
                return BinaryOp(op=op, left=left, right=right)
            case UnaryOp(op=op, expr=expr):
                # Substitute the expression
                expr = self.subst_var_single(var, value, expr)
                return UnaryOp(op=op, expr=expr)
            case _:
                return condition


class ParseConditionalVisibility:
    parser: Parser
    file: Path
    code: str
    tree: Tree
    defs: dict
    conditions: dict[str, TExpr | None]  # Mapping of setting name to condition expression
    internal_variables: dict[str, TExpr]  # Internal mapping and caching of variables
    external_variables: set[str]
    external_functions: set[str]

    _resolved_state: dict[str, bool] = {}

    def __init__(self, defs: dict, path: str | Path | None = None, code: str | None = None):
        # Load C++ language for tree-sitter
        self.parser = Parser()
        self.parser.language = CPP_LANG

        if code is None and path is None:
            raise ValueError("Either code or path must be provided")

        # Read a source file
        self.file = Path(path) if isinstance(path, str) else path
        self.code = code or self.file.read_text(encoding='utf-8')
        self.tree = self.parser.parse(self.code.encode('utf-8'))
        self.defs = defs
        self.conditions = {}
        self.internal_variables = {}
        self.external_variables = set()
        self.external_functions = set()

    def node_to_str(self, node: Node):
        return self.code[node.start_byte:node.end_byte]

    def expand_variable(self, ident: str) -> TExpr:
        # Already is a variable
        if ident in self.defs:
            self._resolved_state[ident] = True
            return  Variable(name=ident)

        if ident in self.internal_variables:
            return self.internal_variables[ident]

        # Last resort, we cannot resolve the variable any deeper.
        return Variable(name=ident)

    def expand_condition(self, node: Node) -> TExpr | None:
        match node.type:
            case 'identifier':
                ident = self.node_to_str(node)
                return self.expand_variable(ident)

            case 'qualified_identifier':
                # TODO: Translate enum values etc.
                ident = self.node_to_str(node)
                return Variable(name=ident)

            case 'call_expression':
                matches = q_opt_function_call.matches(node)
                matches = [match for match in matches if match[0] == 0]

                if matches:
                    setting_key = self.node_to_str(matches[0][1]['func.setting'][0])
                    return self.expand_variable(setting_key)

                func_name = self.node_to_str(node.child_by_field_name('function'))
                args = [[self.expand_condition(arg0) for arg0 in arg.children if
                         arg0 is not None and arg0.type not in ('(', ')')] for arg in
                        node.children if arg.type == 'argument_list']
                args = [x for xs in args for x in xs if x is not None]
                self.external_functions.add(func_name)
                return Function(name=func_name, args=args)

            case 'field_expression':
                matches = q_opt_function_call.matches(node)
                matches = [match for match in matches if match[0] == 0]

                if not matches:
                    print("Field expression that is not a call not supported")
                    return None

                setting_key = self.node_to_str(matches[0][1]['func.setting'][0])

                return self.expand_variable(setting_key)

            case 'binary_expression':
                op = Op(node.child_by_field_name('operator').type)
                left = self.expand_condition(node.child_by_field_name('left'))
                right = self.expand_condition(node.child_by_field_name('right'))
                return BinaryOp(op=op, left=left, right=right)

            case 'unary_expression':
                op = Op(node.child_by_field_name('operator').type)
                expr = self.expand_condition(node.child_by_field_name('argument'))
                return UnaryOp(op=op, expr=expr)

            case 'number_literal':
                literal_content = self.node_to_str(node)

                if '.' in literal_content:
                    return LitFloat(value=float(literal_content.strip('f')))

                return LitInteger(value=int(literal_content))

            case 'parenthesized_expression':
                return self.expand_condition(node.child(1))

            case 'pointer_expression':
                return self.expand_condition(node.child_by_field_name('argument'))

            case 'string_literal':
                value = ''.join([self.node_to_str(arg) for arg in node.children if
                                 arg is not None and arg.type in ['string_content', 'escape_sequence']])

                return LitStr(value=value)

            case _:
                print(f"Unknown node type: {node.type} - {self.node_to_str(node)}")
                return None

    def register_setting_with_conditions(self, setting: Node, *conditions: Node):
        assert setting.type == 'string_content', f"Expected string_content, got {setting.type}"
        setting_key = self.node_to_str(setting)

        for cond in conditions:
            prev = self.conditions.get(setting_key)

            if prev is None:
                self.conditions[setting_key] = self.expand_condition(cond)
            else:
                self.conditions[setting_key] = BinaryOp(op=Op.AND, left=prev, right=self.expand_condition(cond))

    def process(self) -> ConditionalVisibility:
        # process all init assignments first
        for ret, match in q_initial_assigment.matches(self.tree.root_node):
            if ret != 0:
                continue

            ident = match.get('ident')[0]
            value = match.get('value')[0]

            ident = self.node_to_str(ident)
            self.internal_variables[ident] = self.expand_condition(value)

        for ret, match in q_constant_toggle.matches(self.tree.root_node):
            if ret != 0:
                continue

            name = match.get('func.name')[0]
            setting = match.get('func.setting')[0]
            condition = match.get('func.condition')[0]

            p_node = name.parent

            constrained_by_ifs = []

            while p_node is not None:
                if p_node.type == 'function_definition':
                    break

                if p_node.type == 'if_statement':
                    if_condition = q_if_condition.captures(p_node)

                    if if_condition:
                        constrained_by_ifs.extend(if_condition['condition'])

                p_node = p_node.parent
            else:
                continue

            self.register_setting_with_conditions(setting, condition, *constrained_by_ifs)

        for ret, match in q_for_loop_toggle.matches(self.tree.root_node):
            if ret != 0:
                continue

            loop_var = match.get('loop.var')[0]
            loop_settings = match.get('loop.settings')[0]
            loop_body = match.get('loop.body')[0]

            loop_var_ident = self.node_to_str(loop_var)
            body_matches = q_ident_toggle(loop_var_ident).matches(loop_body)
            body_matches = [match for match in body_matches if match[0] == 0]

            if not body_matches:
                continue

            setting_matches = q_string_content.captures(loop_settings)

            for setting in setting_matches['string_content']:
                # if there is multiple here the last one wins.
                self.register_setting_with_conditions(setting, *body_matches[0][1]['func.condition'])

        self.substitute_dict_of_internal_variables(self.conditions)

        return ConditionalVisibility(
            variables=list(self.external_variables),
            functions=list(self.external_functions),
            conditions=self.conditions
        )

    def substitute_dict_of_internal_variables(self, d: dict[str, TExpr]) -> dict[str, TExpr]:
        for k, v in d.items():
            d[k] = self.substitute_internal_variables(v)

    def substitute_internal_variables(self, condition: TExpr) -> TExpr:
        match condition:
            case Variable(name=name):
                if self._resolved_state.get(name, False):
                    # If the variable is already resolved, return it as is.
                    return condition

                # If the variable is not in the internal variables, return it as is.
                if name not in self.internal_variables:
                    self.external_variables.add(name)
                    self._resolved_state[name] = True
                    return Variable(name=name)

                # Otherwise, substitute it with the internal variable.
                return self.internal_variables[name]

            case Function(name=name, args=args):
                # Substitute all arguments
                args = [self.substitute_internal_variables(arg) for arg in args]
                return Function(name=name, args=args)

            case BinaryOp(op=op, left=left, right=right):
                # Substitute left and right
                left = self.substitute_internal_variables(left)
                right = self.substitute_internal_variables(right)
                return BinaryOp(op=op, left=left, right=right)

            case UnaryOp(op=op, expr=expr):
                # Substitute the expression
                expr = self.substitute_internal_variables(expr)
                return UnaryOp(op=op, expr=expr)

            case _:
                return condition
