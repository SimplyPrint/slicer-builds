import unittest

from cli.value_limits import Constants, apply_definitions, build, extract, new_unclassified

HEADER = '''
static constexpr double EPSILON = 1e-4;
constexpr double IRONING_SPACING_MIN = 0.05;
'''


def limits(body: str, header: str = HEADER) -> tuple[dict, list]:
    code = f'void ConfigManipulation::update(DynamicPrintConfig* config, const bool is_global_config)\n{{\n{body}\n}}\n'
    return extract(code, Constants([code, header]))


class ThresholdRuleTests(unittest.TestCase):
    def test_literal_floor_resets_same_setting(self) -> None:
        found, unclassified = limits('''
            if (config->opt_float("ironing_spacing") < 0.05) {
                DynamicPrintConfig new_conf = *config;
                dialog.ShowModal();
                new_conf.set_key_value("ironing_spacing", new ConfigOptionFloat(0.1));
                apply(config, &new_conf);
            }''')
        self.assertEqual(found, {'ironing_spacing': {'min': 0.05, 'lines': [4], 'reset': 0.1}})
        self.assertEqual(unclassified, [])

    def test_named_constant_from_header(self) -> None:
        found, _ = limits('''
            if (config->opt_float("support_ironing_spacing") < IRONING_SPACING_MIN)
                new_conf.set_key_value("support_ironing_spacing", new ConfigOptionFloat(0.1));''')
        self.assertEqual(found['support_ironing_spacing']['min'], 0.05)

    def test_unresolved_constant_is_unclassified(self) -> None:
        found, unclassified = limits('''
            if (config->opt_float("ironing_spacing") < SOMEWHERE_ELSE)
                new_conf.set_key_value("ironing_spacing", new ConfigOptionFloat(0.1));''')
        self.assertEqual(found, {})
        self.assertEqual([item['key'] for item in unclassified], ['ironing_spacing'])

    def test_local_variable_and_option_value(self) -> None:
        found, _ = limits('''
            auto layer_height = config->opt_float("layer_height");
            if (layer_height < EPSILON)
                new_conf.set_key_value("layer_height", new ConfigOptionFloat(0.2));
            if (config->option<ConfigOptionFloat>("initial_layer_print_height")->value < EPSILON)
                new_conf.set_key_value("initial_layer_print_height", new ConfigOptionFloat(0.2));''')
        self.assertEqual(found['layer_height']['min'], 0.0001)
        self.assertEqual(found['initial_layer_print_height']['min'], 0.0001)

    def test_derived_local_variable_is_not_the_setting(self) -> None:
        found, _ = limits('''
            double half = config->opt_float("ironing_spacing") / 2;
            if (half < 0.05)
                new_conf.set_key_value("ironing_spacing", new ConfigOptionFloat(0.1));''')
        self.assertEqual(found, {})

    def test_has_guarded_read_and_gui_state_term(self) -> None:
        found, _ = limits('''
            float speed = config->has("filament_max_volumetric_speed") ? config->opt_float("filament_max_volumetric_speed", (float) 0.5) : 0.5;
            if (speed < 0.5 && wxGetApp().initialized())
                new_conf.set_key_value("filament_max_volumetric_speed", new ConfigOptionFloats({0.5}));''')
        self.assertEqual(found['filament_max_volumetric_speed']['min'], 0.5)
        self.assertEqual(found['filament_max_volumetric_speed']['reset'], 0.5)

    def test_ceiling_with_constant_arithmetic_and_nested_reset(self) -> None:
        found, _ = limits('''
            if (config->opt_float("layer_height") > 0.6 + EPSILON) {
                if (developer_mode) {
                    if (dialog.ShowModal() == wxID_YES) new_conf.set_key_value("layer_height", new ConfigOptionFloat(0.2));
                } else {
                    new_conf.set_key_value("layer_height", new ConfigOptionFloat(0.2));
                }
            }''')
        self.assertEqual(found['layer_height']['max'], 0.6001)

    def test_abs_is_symmetric_range(self) -> None:
        found, _ = limits('''
            if (abs(config->option<ConfigOptionFloat>("xy_hole_compensation")->value) > 2)
                new_conf.set_key_value("xy_hole_compensation", new ConfigOptionFloat(0));''')
        self.assertEqual({k: v for k, v in found['xy_hole_compensation'].items() if k != 'lines'},
                         {'min': -2, 'max': 2, 'reset': 0})

    def test_operator_maps_to_inclusive_or_exclusive_bound(self) -> None:
        found, _ = limits('''
            if (config->opt_float("a") <= 0.)
                new_conf.set_key_value("a", new ConfigOptionFloat(0.01));
            if (2 <= config->opt_int("b"))
                new_conf.set_key_value("b", new ConfigOptionInt(1));''')
        self.assertEqual(found['a']['min'], 0)
        self.assertTrue(found['a']['exclusive_min'])
        self.assertEqual(found['b']['max'], 2)
        self.assertTrue(found['b']['exclusive_max'])

    def test_conditional_rules_are_unclassified(self) -> None:
        found, unclassified = limits('''
            if (config->opt_bool("interlocking_beam") && config->opt_float("interlocking_beam_width") <= 0.)
                new_conf.set_key_value("interlocking_beam_width", new ConfigOptionFloat(0.01));
            if (is_global_config) {
                if (config->opt_float("skirt_height") > 1)
                    new_conf.set_key_value("skirt_height", new ConfigOptionFloat(1));
            }
            if (config->opt_float("infill_lock_depth") > skin_depth)
                new_conf.set_key_value("infill_lock_depth", new ConfigOptionFloat(skin_depth / 2));
            if (config->opt_float("a") < 0.5)
                new_conf.set_key_value("b", new ConfigOptionFloat(1));''')
        self.assertEqual(found, {})
        self.assertEqual([item['key'] for item in unclassified],
                         ['interlocking_beam_width', 'skirt_height', 'infill_lock_depth', 'b'])
        self.assertEqual(unclassified[1]['condition'], '(config->opt_float("skirt_height") > 1)')

    def test_per_item_loop_with_nil_check(self) -> None:
        found, unclassified = limits('''
            const auto *opt = config->option<ConfigOptionFloatsNullable>("filament_ironing_spacing");
            if (opt == nullptr) return;
            std::vector<double> values = opt->values;
            for (size_t i = 0; i < values.size(); ++i)
                if (!opt->is_nil(i) && values[i] < IRONING_SPACING_MIN) {
                    values[i] = 0.1;
                    reset = true;
                }
            new_conf.set_key_value("filament_ironing_spacing", new ConfigOptionFloatsNullable(values));''')
        self.assertEqual(found['filament_ironing_spacing']['min'], 0.05)
        self.assertEqual(found['filament_ironing_spacing']['reset'], 0.1)
        self.assertEqual(unclassified, [])

    def test_range_for_over_option_values(self) -> None:
        found, unclassified = limits('''
            auto* corrected = new_conf.option<ConfigOptionFloats>("filament_max_volumetric_speed");
            for (double& speed : corrected->values)
                if (speed < 0.5)
                    speed = 0.5;''')
        self.assertEqual(found['filament_max_volumetric_speed']['min'], 0.5)
        self.assertEqual(unclassified, [])

    def test_stricter_bound_wins(self) -> None:
        found, _ = limits('''
            if (config->opt_float("layer_height") < EPSILON)
                new_conf.set_key_value("layer_height", new ConfigOptionFloat(0.2));
            if (config->opt_float("layer_height") < 0.01)
                new_conf.set_key_value("layer_height", new ConfigOptionFloat(0.1));''')
        self.assertEqual(found['layer_height']['min'], 0.01)
        self.assertNotIn('reset', found['layer_height'])


class OutputTests(unittest.TestCase):
    def test_limits_need_a_numeric_definition(self) -> None:
        found = {'a': {'min': 1, 'lines': [3]}, 'b': {'min': 1, 'lines': [4]}, 'c': {'min': 1, 'lines': [5]}}
        unclassified: list[dict] = []
        kept = apply_definitions(found, unclassified, {'a': {'type': 'floats'}, 'b': {'type': 'bool'}})
        self.assertEqual(list(kept), ['a'])
        self.assertEqual([item['key'] for item in unclassified], ['b', 'c'])

    def test_new_unclassified_ignores_moved_lines(self) -> None:
        code_v1 = '''void ConfigManipulation::update(DynamicPrintConfig* config) {
            if (config->opt_bool("spiral_mode")) new_conf.set_key_value("wall_loops", new ConfigOptionInt(1));
        }'''
        code_v2 = '''void ConfigManipulation::update(DynamicPrintConfig* config) {

            if (config->opt_bool("spiral_mode")) new_conf.set_key_value("wall_loops", new ConfigOptionInt(1));
            if (config->opt_float("x") > y) new_conf.set_key_value("x", new ConfigOptionFloat(0));
        }'''
        previous = build(code_v1, [], None, 'a.cpp')
        current = build(code_v2, [], None, 'a.cpp')
        self.assertEqual([item['key'] for item in new_unclassified(current, previous)], ['x'])


if __name__ == '__main__':
    unittest.main()
