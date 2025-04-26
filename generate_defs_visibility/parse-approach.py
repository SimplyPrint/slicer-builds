import functools
import json
from collections import deque
from enum import Enum
from pathlib import Path

from pydantic import BaseModel
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


class Expr(BaseModel):
    ...


class Variable(Expr):
    name: str


class InternalProxiedVariable(Expr):
    """Temporary variable for internal use that will be substituted later."""
    name: str


class LitInteger(Expr):
    value: int


class LitFloat(Expr):
    value: float


class LitStr(Expr):
    value: str


class LitBool(Expr):
    value: bool


class Function(Expr):
    name: str
    args: list[Expr]


class Op(Enum):
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


class BinaryOp(Expr):
    op: Op
    left: Expr
    right: Expr


class UnaryOp(Expr):
    op: Op
    expr: Expr


class PrettyPrinter:
    # C‐style precedence levels (higher number binds tighter)
    _prec = {
        Op.NOT.value:     14,
        Op.MUL.value:     13,
        Op.DIV.value:     13,
        Op.MOD.value:     13,
        Op.ADD.value:     12,
        Op.SUB.value:     12,
        Op.BIT_NOT.value: 11,
        Op.BIT_AND.value: 10,
        Op.BIT_XOR.value: 9,
        Op.BIT_OR.value:  8,
        Op.LT.value:      7,
        Op.LTE.value:     7,
        Op.GT.value:      7,
        Op.GTE.value:     7,
        Op.EQ.value:      6,
        Op.NOT_EQ.value:  6,
        Op.AND.value:     5,
        Op.OR.value:      4,
    }

    def pprint(self, expr: Expr) -> str:
        return self._p(expr, parent_prec=0)

    def _p(self, expr: Expr, parent_prec: int) -> str:
        if isinstance(expr, Variable):
            s = expr.name
        elif isinstance(expr, LitInteger):
            s = str(expr.value)
        elif isinstance(expr, LitFloat):
            s = repr(expr.value)
        elif isinstance(expr, LitStr):
            # escape as in Python
            s = repr(expr.value)
        elif isinstance(expr, LitBool):
            s = "true" if expr.value else "false"
        elif isinstance(expr, Function):
            args = ", ".join(self._p(a, 0) for a in expr.args)
            s = f"{expr.name}({args})"
        elif isinstance(expr, UnaryOp):
            op = expr.op.value
            prec = self._prec[op]
            inner = self._p(expr.expr, prec)
            s = f"{op}{inner}"
        elif isinstance(expr, BinaryOp):
            op = expr.op.value
            prec = self._prec[op]
            # left is left‐associative: same‐precedence child needs parentheses on right
            left_s = self._p(expr.left, prec)
            right_s = self._p(expr.right, prec + 1)
            s = f"{left_s} {op} {right_s}"
        elif expr is None:
            s = "null"
        else:
            raise TypeError(f"Unknown Expr: {expr}")

        # add parentheses if this subexpr binds more weakly than its context
        if parent_prec and isinstance(expr, (BinaryOp, UnaryOp)):
            my_prec = prec if 'prec' in locals() else 0
            if my_prec < parent_prec:
                return f"({s})"
        return s


class ConditionalVisibility(BaseModel):
    variables: set[str]
    conditions: dict[str, Expr | None]  # Mapping of setting name to condition expression


class ParseConditionalVisibility:
    parser: Parser
    file: Path
    code: str
    tree: Tree
    defs: dict
    conditions: dict[str, Expr | None]  # Mapping of setting name to condition expression
    internal_variables: dict[str, Expr]  # Internal mapping and caching of variables
    external_variables: set[str]

    def __init__(self, defs: dict, path: str):
        # Load C++ language for tree-sitter
        self.parser = Parser()
        self.parser.language = CPP_LANG

        # Read a source file
        self.file = Path(path)
        self.code = self.file.read_text(encoding='utf-8')
        self.tree = self.parser.parse(self.code.encode('utf-8'))
        self.defs = defs
        self.conditions = {}
        self.internal_variables = {}
        self.external_variables = set()

    def node_to_str(self, node: Node):
        return self.code[node.start_byte:node.end_byte]

    def expand_variable(self, ident: str) -> Expr:
        # Already is a variable
        if ident in self.defs:
            return Variable(name=ident)

        if ident in self.internal_variables:
            return self.internal_variables[ident]

        # Last resort, we cannot resolve the variable any deeper.
        return InternalProxiedVariable(name=ident)

    def expand_condition(self, node: Node) -> Expr | None:
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

    def main(self):
        # process all init assignments first
        for ret, match in q_initial_assigment.matches(self.tree.root_node):
            if ret != 0:
                continue

            ident = match.get('ident')[0]
            value = match.get('value')[0]

            ident = self.node_to_str(ident)
            self.internal_variables[ident] = self.expand_condition(value)

        # Forward resolution of variables
        self.resolve_internal_variables_topo()

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

    def collect_deps(self, expr: Expr, out: set[str]):
        """Walk an Expr to collect all Variable names it refers to."""
        if isinstance(expr, Variable):
            out.add(expr.name)
        elif isinstance(expr, Function):
            for a in expr.args:
                self.collect_deps(a, out)
        elif isinstance(expr, BinaryOp):
            self.collect_deps(expr.left, out)
            self.collect_deps(expr.right, out)
        elif isinstance(expr, UnaryOp):
            self.collect_deps(expr.expr, out)
        # literals and proxies carry no deps

    def resolve_internal_variables_topo(self):
        # 1) Build dependency graph: var → set(vars it depends on)
        graph: dict[str, set[str]] = {}
        for name, expr in self.internal_variables.items():
            deps: set[str] = set()
            self.collect_deps(expr, deps)
            # only keep edges to other internal_variables
            graph[name] = {d for d in deps if d in self.internal_variables}

        # 2) Compute in-degrees
        in_deg = {u: 0 for u in graph}
        for u, vs in graph.items():
            for v in vs:
                in_deg[u] += 1

        # 3) Kahn’s algorithm
        queue = deque(u for u, d in in_deg.items() if d == 0)
        resolved: dict[str, Expr] = {}

        while queue:
            u = queue.popleft()
            # substitute only using already–resolved vars
            # we temporarily override self.internal_variables so substitute_internal uses `resolved`
            old_map = self.internal_variables
            self.internal_variables = resolved
            resolved[u] = self.substitute_internal_variables(old_map[u])
            self.internal_variables = old_map

            # decrement dependents
            for w, deps in graph.items():
                if u in deps:
                    in_deg[w] -= 1
                    if in_deg[w] == 0:
                        queue.append(w)

        # 4) detect cycles (anything still missing)
        for u in graph:
            if u not in resolved:
                # leave it unexpanded (or keep the proxy)
                resolved[u] = self.internal_variables[u]

        # 5) replace
        self.internal_variables = resolved

    def substitute_dict_of_internal_variables(self, d: dict[str, Expr]) -> dict[str, Expr]:
        return {k: self.substitute_internal_variables(v) for k, v in d.items()}

    def substitute_internal_variables(self, condition: Expr) -> Expr:
        match condition:
            case InternalProxiedVariable(name=name):
                # If the variable is not in the internal variables, return it as is.
                if name not in self.internal_variables:
                    self.external_variables.add(name)
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

    def finalize(self) -> ConditionalVisibility:
        self.internal_variables = self.substitute_dict_of_internal_variables(self.internal_variables)
        self.conditions = self.substitute_dict_of_internal_variables(self.conditions)
        return ConditionalVisibility(variables=self.external_variables, conditions=self.conditions)


if __name__ == '__main__':
    parser = ParseConditionalVisibility(json.load(open('./print_config_def.json')),
                                        'cache/toggle_print_fff_options.cpp')
    parser.main()

    cv = parser.finalize()

    print(cv.model_dump_json(serialize_as_any=True))
