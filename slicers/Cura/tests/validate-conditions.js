#!/usr/bin/env node

const fs = require('node:fs');

const file = process.argv[2];
const definitionsFile = process.argv[3];
if (!file || !definitionsFile) {
  throw new Error(
    'Usage: validate-conditions.js <conditional_visibility.json> <print_config_def.json>'
  );
}

const document = JSON.parse(fs.readFileSync(file, 'utf8'));
const definitions = JSON.parse(fs.readFileSync(definitionsFile, 'utf8'));
const context = Object.fromEntries(
  Object.entries(definitions).map(([key, definition]) => [key, definition.default_value])
);
for (const [key, condition] of Object.entries(document.conditions)) {
  if (/\b(?:and|or|not|in|is|True|False)\b/.test(condition)) {
    throw new Error(`${key}: unnormalized Python token in ${condition}`);
  }
  try {
    const evaluate = new Function(...Object.keys(context), `return (${condition});`);
    const result = evaluate(...Object.values(context));
    if (typeof result !== 'boolean') {
      throw new Error(`returned ${typeof result}, expected boolean`);
    }
  } catch (error) {
    throw new Error(`${key}: invalid/effectless condition ${condition}: ${error.message}`);
  }
}

console.log(`Validated ${Object.keys(document.conditions).length} Cura visibility conditions`);
