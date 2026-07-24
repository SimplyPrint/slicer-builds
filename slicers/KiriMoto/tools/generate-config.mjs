#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

const args = parseArgs(process.argv.slice(2));
const sourceRoot = await findSourceRoot(path.resolve(required(args, 'source')));
const outputDir = path.resolve(required(args, 'output'));
const FILAMENT_KEYS = new Set([
  'firstLayerBedTemp',
  'firstLayerFanSpeed',
  'firstLayerNozzleTemp',
  'outputBedTemp',
  'outputFanLayer',
  'outputFanSpeed',
  'outputFillMult',
  'outputMaxFlowrate',
  'outputMinLayerTime',
  'outputMinSpeed',
  'outputShellMult',
  'outputSparseMult',
  'outputTemp',
]);

// GridSpace installs this helper from its browser bootstrap before loading
// defaults.js. The generator loads that authoritative module directly.
Object.clone ??= structuredClone;
const defaultsModule = await importSource('kiri/app/conf/defaults.js');
const constantsModule = await importSource('kiri/app/consts.js');
globalThis.self = { lang: {} };
await importSource('kiri/app/lang-en.js');

const language = globalThis.self.lang.en ?? globalThis.self.lang['en-us'];
const deviceDefaults = defaultsModule.conf?.defaults?.fdm?.d;
const processDefaults = defaultsModule.conf?.defaults?.fdm?.p;
const lists = constantsModule.consts?.LISTS;

if (!isPlainObject(deviceDefaults) || !isPlainObject(processDefaults) || !isPlainObject(lists)) {
  throw new Error('Kiri:Moto upstream source does not expose FDM defaults and selection lists');
}
if (!isPlainObject(language)) {
  throw new Error('Kiri:Moto upstream English language catalog was not loaded');
}

const processSource = await readSource('kiri/mode/fdm/app/init-menu.js');
const inputSource = await readSource('kiri/app/init/input.js');
const processUi = parseMenu(processSource);
const machineUi = parseMachineMenu(inputSource);
const machineDefaults = flattenMachineDefaults(deviceDefaults);
const declarations = new Map([...machineUi.declarations, ...processUi.declarations]);
const definitions = {};

for (const [key, value] of Object.entries(machineDefaults)) {
  definitions[key] = makeDefinition(key, value, declarations.get(key), 'machine');
}
for (const [key, value] of Object.entries(processDefaults)) {
  definitions[key] = makeDefinition(
    key,
    value,
    declarations.get(key),
    FILAMENT_KEYS.has(key) ? 'filament' : 'process',
  );
}

const machine = buildMachinePanel(machineUi, definitions);
const { filament, process: processPanel } = buildProcessPanels(processUi, definitions);
const conditions = {};

for (const [key, declaration] of declarations) {
  if (!definitions[key]) continue;
  const condition = conditionFor(declaration.visibility);
  if (condition) conditions[key] = condition;
}

const metadata = {
  capabilities: {
    profile_metadata_transport: 'envelope.v1',
    settings_codec: 'kirimoto-json.v1',
  },
  conditional_settings: {
    false_behavior: 'hide',
  },
  generator: {
    defaults: 'src/kiri/app/conf/defaults.js',
    device_ui: 'src/kiri/app/init/input.js',
    process_ui: 'src/kiri/mode/fdm/app/init-menu.js',
    selections: 'src/kiri/app/consts.js',
    translations: 'src/kiri/app/lang-en.js',
  },
};

validateArtifacts({ definitions, machine, filament, process: processPanel });
await fs.mkdir(outputDir, { recursive: true });
await Promise.all([
  writeJson('machine.json', machine),
  writeJson('filament.json', filament),
  writeJson('process.json', processPanel),
  writeJson('print_config_def.json', sortObject(definitions)),
  writeJson('conditional_visibility.json', { conditions: sortObject(conditions) }),
  writeJson('ui_metadata.json', metadata),
]);

process.stdout.write(
  `Generated ${Object.keys(definitions).length} Kiri:Moto FDM settings ` +
    `(${countPanelSettings(machine)} machine, ${countPanelSettings(filament)} filament, ` +
    `${countPanelSettings(processPanel)} process) at ${outputDir}\n`,
);

function parseArgs(values) {
  const parsed = {};
  for (let index = 0; index < values.length; index += 2) {
    const key = values[index];
    if (!key?.startsWith('--') || values[index + 1] == null) {
      throw new Error(`invalid argument list near ${key ?? '<end>'}`);
    }
    parsed[key.slice(2)] = values[index + 1];
  }
  return parsed;
}

function required(value, key) {
  if (!value[key]) throw new Error(`--${key} is required`);
  return value[key];
}

async function findSourceRoot(base) {
  for (const candidate of [path.join(base, 'src'), base]) {
    try {
      await fs.access(path.join(candidate, 'kiri', 'app', 'conf', 'defaults.js'));
      return candidate;
    } catch {
      // Try the next supported upstream checkout layout.
    }
  }
  throw new Error(`${base} is not a GridSpace/grid-apps source checkout`);
}

async function readSource(relative) {
  return fs.readFile(path.join(sourceRoot, relative), 'utf8');
}

async function importSource(relative) {
  const url = pathToFileURL(path.join(sourceRoot, relative));
  url.searchParams.set('simplyprint-config-generator', '1');
  return import(url.href);
}

function parseMenu(source) {
  const declarations = new Map();
  let group = 'General';

  for (const line of source.split(/\r?\n/)) {
    const groupMatch = line.match(/\bnewGroup\((LANG\.[A-Za-z0-9_]+|["'][^"']+["'])/);
    if (groupMatch) {
      group = labelFromToken(groupMatch[1]) || 'General';
      continue;
    }
    const declaration = parseDeclaration(line, group);
    if (declaration) declarations.set(declaration.key, declaration);
  }

  return { declarations };
}

function parseMachineMenu(source) {
  const start = source.indexOf('/** Device Browser / Editor */');
  const end = source.indexOf('/** Preferences Menu */', start);
  if (start < 0 || end < 0) {
    throw new Error('Could not locate Kiri:Moto device editor declarations');
  }

  const aliases = new Map([
    ['deviceOrigin', 'originCenter'],
    ['deviceRound', 'bedRound'],
    ['deviceBelt', 'bedBelt'],
  ]);
  const parsed = parseMenu(source.slice(start, end));
  const declarations = new Map();

  for (const declaration of parsed.declarations.values()) {
    const key = aliases.get(declaration.key) ?? declaration.key;
    if (!Object.hasOwn(flattenMachineDefaults(deviceDefaults), key)) continue;
    declarations.set(key, { ...declaration, key });
  }

  const gcodeKeys = Object.keys(deviceDefaults).filter((key) => key.startsWith('gcode'));
  for (const key of gcodeKeys) {
    if (declarations.has(key)) continue;
    declarations.set(key, {
      key,
      kind: Array.isArray(deviceDefaults[key]) ? 'list' : 'input',
      label: humanize(key),
      tooltip: '',
      group: 'G-code',
      visibility: '',
      selectList: null,
      min: null,
      max: null,
      convert: null,
    });
  }

  return { declarations };
}

function parseDeclaration(line, group) {
  const match = line.match(
    /^\s*([A-Za-z][A-Za-z0-9_]*):\s*new(Input|Boolean|Select)\((.+)$/,
  );
  if (!match) return null;

  const [, key, constructor, body] = match;
  const labelToken = body.match(/^\s*(LANG\.[A-Za-z0-9_]+|["'][^"']*["'])/)?.[1];
  const titleToken = body.match(/\btitle\s*:\s*(LANG\.[A-Za-z0-9_]+|["'][^"']*["'])/)?.[1];
  const bounds = body.match(/\bbound\s*:\s*bound\(\s*(-?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*(-?(?:\d+(?:\.\d+)?|\.\d+))\s*\)/);
  const selectList =
    constructor === 'Select'
      ? body.match(/,\s*["']([A-Za-z0-9_-]+)["']\s*\)\s*,?\s*$/)?.[1] ?? null
      : null;
  const visibility = /\b(?:xshow|show)\s*:/.test(body) ? body : '';

  return {
    key,
    kind: constructor.toLowerCase(),
    label: labelFromToken(labelToken) || humanize(key),
    tooltip: labelFromToken(titleToken),
    group: titleCase(group),
    visibility,
    selectList,
    min: bounds ? Number(bounds[1]) : null,
    max: bounds ? Number(bounds[2]) : null,
    convert: body.includes('convert:toInt')
      ? 'int'
      : body.includes('convert:toFloat')
        ? 'float'
        : null,
  };
}

function flattenMachineDefaults(device) {
  const result = structuredClone(device);
  const firstExtruder = Array.isArray(device.extruders) ? device.extruders[0] : null;
  if (isPlainObject(firstExtruder)) {
    Object.assign(result, structuredClone(firstExtruder));
  }
  return result;
}

function makeDefinition(key, defaultValue, declaration, role) {
  const type = settingType(defaultValue, declaration);
  const runtimeOnly = !declaration || complexRuntimeValue(defaultValue, key);
  const definition = {
    type,
    default_value: structuredClone(defaultValue),
    label: declaration?.label || humanize(key),
    tooltip: declaration?.tooltip || '',
    mode: modeFor(declaration?.group),
    native_type: nativeType(type, defaultValue),
    wire_type: wireType(type, defaultValue),
    wire_codec: 'json',
    editor: editorFor(type, defaultValue),
    visibility_tier: modeFor(declaration?.group),
    runtime_only: runtimeOnly,
    profile_role: role,
  };

  if (declaration?.min != null) definition.min = declaration.min;
  if (declaration?.max != null) definition.max = declaration.max;
  const units = unitFor(key);
  if (units) definition.sidetext = units;

  if (declaration?.selectList) {
    const options = lists[declaration.selectList];
    if (declaration.selectList === 'extruders') {
      definition.type = 'int';
      definition.native_type = 'int';
      definition.wire_type = 'integer';
      definition.editor = {
        kind: 'tool_reference',
        index_base: 0,
        display_index_base: 1,
        selection: 'required',
      };
      return definition;
    }
    if (!Array.isArray(options) || options.length === 0) {
      throw new Error(`${key} references missing Kiri:Moto selection list ${declaration.selectList}`);
    }
    definition.type = 'enum';
    definition.native_type = 'enum';
    definition.wire_type = 'string';
    definition.editor = { kind: 'select' };
    definition.enum_values = options.map((item) =>
      Object.hasOwn(item, 'id') ? item.id : item.name,
    );
    definition.enum_labels = options.map((item) => String(item.name));
  }

  return definition;
}

function settingType(value, declaration) {
  if (declaration?.kind === 'select') return 'enum';
  if (typeof value === 'boolean') return 'bool';
  if (typeof value === 'number') {
    return declaration?.convert === 'int' || (Number.isInteger(value) && declaration?.convert !== 'float')
      ? 'int'
      : 'float';
  }
  if (Array.isArray(value)) {
    if (value.every((item) => typeof item === 'boolean')) return 'bools';
    if (value.length > 0 && value.every((item) => Number.isInteger(item))) return 'ints';
    if (value.length > 0 && value.every((item) => typeof item === 'number')) return 'floats';
    return 'strings';
  }
  return 'string';
}

function nativeType(type, value) {
  if (isPlainObject(value)) return 'object';
  return {
    bool: 'bool',
    bools: 'array',
    int: 'int',
    ints: 'array',
    float: 'float',
    floats: 'array',
    string: 'str',
    strings: 'array',
    enum: 'enum',
  }[type];
}

function wireType(type, value) {
  if (isPlainObject(value)) return 'object';
  if (type.endsWith('s')) return 'array';
  return {
    bool: 'boolean',
    int: 'integer',
    float: 'number',
    string: 'string',
    enum: 'string',
  }[type];
}

function editorFor(type, value) {
  if (isPlainObject(value)) return { kind: 'json' };
  if (type === 'bool') return { kind: 'toggle' };
  if (type === 'enum') return { kind: 'select' };
  if (type === 'int' || type === 'float') return { kind: 'number', value_type: type };
  if (type.endsWith('s')) return { kind: 'list', item_type: type.slice(0, -1) };
  return { kind: 'text', multiline: false };
}

function complexRuntimeValue(value, key) {
  if (isPlainObject(value)) return true;
  if (!Array.isArray(value)) return false;
  return ['extruders', 'profiles', 'ranges'].includes(key);
}

function buildMachinePanel(ui, definitions) {
  const panel = { Machine: {} };
  for (const declaration of ui.declarations.values()) {
    if (!definitions[declaration.key] || definitions[declaration.key].runtime_only) continue;
    pushPanel(panel.Machine, declaration.group, declaration.key);
  }
  return sortPanel(panel);
}

function buildProcessPanels(ui, definitions) {
  const filament = { Material: {} };
  const process = {};

  for (const declaration of ui.declarations.values()) {
    const definition = definitions[declaration.key];
    if (!definition || definition.runtime_only) continue;
    if (FILAMENT_KEYS.has(declaration.key)) {
      pushPanel(filament.Material, declaration.group, declaration.key);
    } else {
      process[declaration.group] ??= {};
      pushPanel(process[declaration.group], 'General', declaration.key);
    }
  }

  return { filament: sortPanel(filament), process: sortPanel(process) };
}

function pushPanel(category, subcategory, key) {
  category[subcategory] ??= [];
  if (!category[subcategory].includes(key)) category[subcategory].push(key);
}

function sortPanel(panel) {
  const result = {};
  for (const category of Object.keys(panel).sort(naturalCompare)) {
    result[category] = {};
    for (const subcategory of Object.keys(panel[category]).sort(naturalCompare)) {
      result[category][subcategory] = [...panel[category][subcategory]];
    }
  }
  return result;
}

function conditionFor(source) {
  if (!source) return null;
  const normalized = source.replace(/\s+/g, '');
  if (normalized.includes('outputRaft.checked')) return 'outputRaft && !bedBelt';
  if (normalized.includes('fdmArcEnabled.checked')) return 'fdmArcEnabled';
  if (normalized.includes('sliceAdaptive.checked')) return 'sliceAdaptive';
  if (normalized.includes('!ui.deviceRound.checked')) return '!bedRound';
  if (normalized.includes('!isTree()')) return '!sliceSupportTree';
  if (normalized.includes('fillIsLinear')) return 'sliceFillType == "linear"';
  if (normalized.includes('hasInfill')) return 'sliceFillType != "none"';
  if (normalized.includes('manualSupport')) return 'sliceSupportType == "manual"';
  if (normalized.includes('notFwRetract')) return '!fwRetract';
  if (normalized.includes('isMultiHead')) return 'false';
  if (normalized.includes('zIntShow')) return 'false';
  if (normalized.includes('isNotBelt')) return '!bedBelt';
  if (normalized.includes('isBelt')) return 'bedBelt';
  if (normalized.includes('isTree')) return 'sliceSupportTree';
  return null;
}

function modeFor(group) {
  const normalized = String(group ?? '').toLowerCase();
  if (normalized.includes('expert')) return 'expert';
  if (
    normalized.includes('quality') ||
    normalized.includes('layer') ||
    normalized.includes('wall') ||
    normalized.includes('fill') ||
    normalized.includes('heat') ||
    normalized.includes('support')
  ) {
    return 'simple';
  }
  return 'advanced';
}

function unitFor(key) {
  if (/Temp$/.test(key)) return '°C';
  if (/Speed$|rate$|Rate$|Seekrate$|Feedrate$|Finishrate$/.test(key)) return 'mm/s';
  if (/Angle$/.test(key)) return '°';
  if (/Density$|Sparse$|Overlap$|Fact$|Mult$/.test(key)) return '';
  if (
    /Height$|Width$|Depth$|Nozzle$|Filament$|Offset[XY]$|Distance$|Spacing$|Lead$|Bump$|Gap$|Grow$|Thick$|Area$/.test(
      key,
    )
  ) {
    return 'mm';
  }
  return '';
}

function labelFromToken(token) {
  if (!token) return '';
  if (token.startsWith('LANG.')) {
    const value = language[token.slice(5)];
    return normalizeLanguageValue(value);
  }
  if (/^["']/.test(token)) return token.slice(1, -1);
  return '';
}

function normalizeLanguageValue(value) {
  if (Array.isArray(value)) return value.join(' ');
  return typeof value === 'string' ? value : '';
}

function humanize(value) {
  return titleCase(value.replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_-]+/g, ' '));
}

function titleCase(value) {
  return String(value)
    .trim()
    .replace(/\b[a-z]/g, (letter) => letter.toUpperCase());
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function sortObject(value) {
  return Object.fromEntries(Object.entries(value).sort(([left], [right]) => naturalCompare(left, right)));
}

function naturalCompare(left, right) {
  return left.localeCompare(right, 'en', { sensitivity: 'base' });
}

function countPanelSettings(panel) {
  let count = 0;
  for (const subcategories of Object.values(panel)) {
    for (const settings of Object.values(subcategories)) count += settings.length;
  }
  return count;
}

function validateArtifacts(artifacts) {
  const panelKeys = [];
  for (const panel of [artifacts.machine, artifacts.filament, artifacts.process]) {
    for (const subcategories of Object.values(panel)) {
      for (const settings of Object.values(subcategories)) panelKeys.push(...settings);
    }
  }

  const duplicateKeys = panelKeys.filter((key, index) => panelKeys.indexOf(key) !== index);
  if (duplicateKeys.length > 0) {
    throw new Error(`Kiri:Moto generated duplicate UI settings: ${[...new Set(duplicateKeys)].join(', ')}`);
  }

  const panelKeySet = new Set(panelKeys);
  const missingUiKeys = Object.entries(artifacts.definitions)
    .filter(([, definition]) => !definition.runtime_only)
    .map(([key]) => key)
    .filter((key) => !panelKeySet.has(key));
  if (missingUiKeys.length > 0) {
    throw new Error(`Kiri:Moto generated settings missing from UI panels: ${missingUiKeys.join(', ')}`);
  }

  for (const key of ['bedBelt', 'extNozzle', 'sliceAngle', 'sliceHeight', 'sliceSupportType']) {
    if (!artifacts.definitions[key]) {
      throw new Error(`Kiri:Moto upstream FDM schema is missing required setting ${key}`);
    }
  }
  for (const [key, definition] of Object.entries(artifacts.definitions)) {
    if (
      definition.type === 'enum' &&
      !definition.enum_values.some((value) => value === definition.default_value)
    ) {
      throw new Error(`${key} has a default outside its Kiri:Moto selection list`);
    }
  }

  for (const [panel, role] of [
    [artifacts.machine, 'machine'],
    [artifacts.filament, 'filament'],
    [artifacts.process, 'process'],
  ]) {
    for (const subcategories of Object.values(panel)) {
      for (const keys of Object.values(subcategories)) {
        for (const key of keys) {
          if (artifacts.definitions[key].profile_role !== role) {
            throw new Error(
              `${key} is in the ${role} panel but generated with role ` +
                `${artifacts.definitions[key].profile_role}`,
            );
          }
        }
      }
    }
  }
}

async function writeJson(filename, value) {
  await fs.writeFile(path.join(outputDir, filename), `${JSON.stringify(value, null, 2)}\n`);
}
