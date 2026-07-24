#!/usr/bin/env node

import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import process from 'node:process';

const args = parseArgs(process.argv.slice(2));
const sourceDir = path.resolve(required(args, 'source'));
const outputDir = path.resolve(required(args, 'output'));
const sourceRoot = path.join(sourceDir, 'src');
const nodeSourceRoot = path.join(outputDir, 'source');
const browserSourceRoot = path.join(outputDir, 'browser-source');
const libDir = path.join(outputDir, 'lib', 'kirimoto');
const wasmDir = path.join(outputDir, 'lib', 'wasm');
const browserDir = path.join(outputDir, 'browser');
const binDir = path.join(outputDir, 'bin');
const upstreamPackage = JSON.parse(
  await fs.readFile(path.join(sourceDir, 'package.json'), 'utf8'),
);
const sourceRequire = createRequire(path.join(sourceDir, 'package.json'));
const { build } = sourceRequire('esbuild');

if (!upstreamPackage.version || !upstreamPackage.engines?.node) {
  throw new Error('upstream package.json is missing version or engines.node');
}

await fs.rm(outputDir, { recursive: true, force: true });
await fs.mkdir(nodeSourceRoot, { recursive: true });
await fs.mkdir(browserSourceRoot, { recursive: true });
await Promise.all([
  fs.cp(sourceRoot, nodeSourceRoot, { recursive: true, dereference: true }),
  fs.cp(sourceRoot, browserSourceRoot, { recursive: true, dereference: true }),
]);

await patchFile(nodeSourceRoot, 'kiri/run/engine.js', [
  [
    "import { api } from '../app/api.js';",
    "import { api } from './node-api.js';",
  ],
  [
    "import { load } from '../../load/file.js';",
    "import { STL } from '../../load/stl.js';",
  ],
  [
    "import { newWidget } from '../app/widget.js';",
    "import { newWidget } from '../core/widget.js';",
  ],
  [
    '            client.restart();\n            client.pool.start();',
    '            client.restart();\n            // Node builds use a single isolated worker.',
  ],
  ['new load.STL()', 'new STL()', 2],
]);

await patchFile(nodeSourceRoot, 'kiri/app/workers.js', [
  [
    "import { api } from './api.js';",
    "import { api } from '../run/node-api.js';",
  ],
  [
    "import { Widget } from './widget.js';",
    "import { Widget } from '../core/widget.js';",
  ],
]);

await patchFile(nodeSourceRoot, 'kiri/core/widget.js', [
  ["import { tool as mesh_tool } from '../../mesh/tool.js';\n", ''],
  [
    "            console.time('mesh normalize')\n" +
      '            data = new mesh_tool({ precision: 0.001 }).normalizeVertices(data).toFloat32();\n' +
      "            console.timeEnd('mesh normalize');",
    "            throw new Error('mesh normalization is unavailable in the Node slicer runtime');",
  ],
]);

await patchFile(nodeSourceRoot, 'kiri/run/worker.js', fdmOnlyWorkerPatches());
await removeObjectMethods(nodeSourceRoot, 'kiri/run/worker.js', [
  'image2mesh',
  'gerber2mesh',
  'zip',
]);

await patchFile(browserSourceRoot, 'kiri/run/engine.js', [
  [
    "import { api } from '../app/api.js';",
    "import { api } from './provider-api.js';",
  ],
  [
    "import { load } from '../../load/file.js';",
    "import { STL } from '../../load/stl.js';",
  ],
  [
    "import { newWidget } from '../app/widget.js';",
    "import { newWidget } from '../core/widget.js';",
  ],
  ['new load.STL()', 'new STL()', 2],
  [
    '    slice() {\n',
    '    terminate() {\n' +
      '        if (this.terminated) return;\n' +
      '        this.terminated = true;\n' +
      '        client.pool.stop();\n' +
      '        client.terminate();\n' +
      '    }\n\n' +
      '    slice() {\n',
  ],
]);

await patchFile(browserSourceRoot, 'kiri/app/workers.js', [
  [
    "import { api } from './api.js';",
    "import { api } from '../run/provider-api.js';",
  ],
  [
    "import { Widget } from './widget.js';",
    "import { Widget } from '../core/widget.js';",
  ],
  [
    '    isBusy() {\n',
    '    terminate() {\n' +
      '        if (worker) worker.terminate();\n' +
      '        worker = null;\n' +
      '        minions = false;\n' +
      '        for (const record of Object.values(running)) {\n' +
      "            record.fn?.({ error: 'cancelled operation' });\n" +
      '        }\n' +
      '        syncd = {};\n' +
      '        running = {};\n' +
      '    },\n\n' +
      '    isBusy() {\n',
  ],
]);

await patchFile(browserSourceRoot, 'kiri/core/widget.js', [
  ["import { tool as mesh_tool } from '../../mesh/tool.js';\n", ''],
  [
    "            console.time('mesh normalize')\n" +
      '            data = new mesh_tool({ precision: 0.001 }).normalizeVertices(data).toFloat32();\n' +
      "            console.timeEnd('mesh normalize');",
    "            throw new Error('mesh normalization is unavailable in the browser slicer runtime');",
  ],
]);

await patchFile(
  browserSourceRoot,
  'kiri/run/worker.js',
  fdmOnlyWorkerPatches(),
);
await removeObjectMethods(browserSourceRoot, 'kiri/run/worker.js', [
  'image2mesh',
  'gerber2mesh',
  'zip',
]);

const providerApi = `export const api = {
  platform: { clear() {} },
  settings: { export({ engine }) { return engine; } },
  widgets: { all() { return []; } },
};
`;

await fs.writeFile(
  path.join(browserSourceRoot, 'kiri', 'run', 'provider-api.js'),
  providerApi,
);

function fdmOnlyWorkerPatches() {
  return [
    ["import { JSZip } from '../../ext/jszip-esm.js';\n", ''],
    ['import { load } from "../../load/png.js";\n', ''],
    ["import { RasterPath } from '../../gpu/raster.js';\n", ''],
    [
      "import { toMesh as gerberToMesh } from '../../load/gbr.js';\n",
      '',
      [0, 1],
    ],
    ["import { CAM } from '../mode/cam/work/init-work.js';\n", ''],
    ["import { DRAG } from '../mode/drag/init-work.js';\n", ''],
    ["import { LASER } from '../mode/laser/init-work.js';\n", ''],
    ["import { SLA } from '../mode/sla/work/init-work.js';\n", ''],
    ["import { WEDM } from '../mode/wedm/init-work.js';\n", ''],
    ["import { WJET } from '../mode/wjet/init-work.js';\n", ''],
    [
      'let drivers = {\n' +
        '        DRAG,\n' +
        '        CAM,\n' +
        '        FDM,\n' +
        '        LASER,\n' +
        '        SLA,\n' +
        '        WEDM,\n' +
        '        WJET\n' +
        '    },',
      'let drivers = { FDM },',
    ],
    [
      '/**\n' +
        ' * @returns {RasterPath} instantiated class\n' +
        ' */\n' +
        'self.get_raster_gpu = async function({ mode, resolution, rotationStep }) {\n' +
        '    let gpu = new RasterPath({\n' +
        '        mode,\n' +
        '        resolution,\n' +
        '        rotationStep,\n' +
        '        workerName: "/lib/gpu/raster-worker.js",\n' +
        '        quiet: true,\n' +
        '        debug: false,\n' +
        '    });\n' +
        '    await gpu.init();\n' +
        '    return gpu;\n' +
        '};\n\n',
      '',
    ],
  ];
}

await fs.writeFile(
  path.join(nodeSourceRoot, 'package.json'),
  '{"type":"module"}\n',
);
await fs.writeFile(
  path.join(nodeSourceRoot, 'kiri', 'run', 'node-api.js'),
  `export const api = {
  platform: { clear() {} },
  settings: { export({ engine }) { return engine; } },
  widgets: { all() { return []; } },
};
`,
);

await fs.mkdir(libDir, { recursive: true });
await fs.mkdir(wasmDir, { recursive: true });
await fs.mkdir(browserDir, { recursive: true });
await fs.mkdir(binDir, { recursive: true });
await fs.copyFile(
  path.join(sourceRoot, 'wasm', 'manifold.wasm'),
  path.join(wasmDir, 'manifold.wasm'),
);

const external = [
  '@gridspace/planegcs',
  '@salusoft89/planegcs',
  'manifold-3d',
  'three-mesh-bvh',
  './constants',
  './voronoi_structures',
  './voronoi_ctypes',
  '../thirdparty/jsbn',
  './collections',
  './voronoi_predicates',
  './voronoi_builder',
  './point_data',
  './segment_data',
  './cppgen',
  './voronoi_diagram',
  './voronoi',
];

await Promise.all([
  bundle(
    path.join(nodeSourceRoot, 'kiri', 'run', 'engine.js'),
    path.join(libDir, 'engine.mjs'),
    external,
  ),
  bundle(
    path.join(nodeSourceRoot, 'kiri', 'run', 'worker.js'),
    path.join(libDir, 'worker-core.mjs'),
    external,
  ),
]);

const [browserEngine, browserWorker, browserPool] = await Promise.all([
  bundleBrowser(
    path.join(browserSourceRoot, 'kiri', 'run', 'engine.js'),
    external,
  ),
  bundleBrowser(
    path.join(browserSourceRoot, 'kiri', 'run', 'worker.js'),
    external,
  ),
  bundleBrowser(
    path.join(browserSourceRoot, 'kiri', 'run', 'minion.js'),
    external,
  ),
]);
await fs.writeFile(
  path.join(browserDir, 'kirimoto-runtime.mjs'),
  `${browserEngine}
export const workerSource = ${JSON.stringify(browserWorker)};
export const poolSource = ${JSON.stringify(browserPool)};
export const version = ${JSON.stringify(upstreamPackage.version)};
`,
);

await fs.writeFile(path.join(libDir, 'worker.mjs'), workerRuntime());
await fs.writeFile(
  path.join(libDir, 'cli.mjs'),
  cliRuntime(upstreamPackage.version),
);
await fs.writeFile(path.join(binDir, 'kirimoto'), launcher());
await fs.chmod(path.join(binDir, 'kirimoto'), 0o755);
await fs.writeFile(
  path.join(outputDir, 'package.json'),
  `${JSON.stringify(
    {
      name: '@simplyprint/kirimoto-node',
      version: upstreamPackage.version,
      description: 'Headless Kiri:Moto FDM slicer for SimplyPrint',
      license: upstreamPackage.license ?? 'MIT',
      engines: { node: '>=22.0.0' },
      type: 'module',
    },
    null,
    2,
  )}\n`,
);

process.stdout.write(
  `Built Kiri:Moto ${upstreamPackage.version} Node package at ${outputDir}\n`,
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

async function patchFile(root, relative, replacements) {
  const file = path.join(root, relative);
  let source = await fs.readFile(file, 'utf8');
  for (const [needle, replacement, expected = 1] of replacements) {
    const actual = source.split(needle).length - 1;
    const valid = Array.isArray(expected)
      ? expected.includes(actual)
      : actual === expected;
    if (!valid) {
      throw new Error(
        `${relative}: expected ${JSON.stringify(expected)} occurrence(s) of ${JSON.stringify(needle)}, found ${actual}`,
      );
    }
    source = source.split(needle).join(replacement);
  }
  await fs.writeFile(file, source);
}

async function removeObjectMethods(root, relative, methodNames) {
  const file = path.join(root, relative);
  let source = await fs.readFile(file, 'utf8');
  for (const methodName of methodNames) {
    const pattern = new RegExp(`\\n    ${methodName}\\([^]*?\\n    },\\n`, 'g');
    const matches = source.match(pattern) ?? [];
    if (matches.length !== 1) {
      throw new Error(
        `${relative}: expected one ${methodName} worker method, found ${matches.length}`,
      );
    }
    source = source.replace(pattern, '\n');
  }
  await fs.writeFile(file, source);
}

async function bundle(entryPoint, outfile, dependencies) {
  await build({
    entryPoints: [entryPoint],
    outfile,
    bundle: true,
    platform: 'node',
    format: 'esm',
    target: 'node22',
    external: dependencies,
    sourcemap: false,
    minify: false,
    legalComments: 'eof',
    logOverride: {
      'commonjs-variable-in-esm': 'silent',
      'direct-eval': 'silent',
      'duplicate-object-key': 'silent',
      'unsupported-dynamic-import': 'silent',
    },
  });
}

async function bundleBrowser(entryPoint, dependencies) {
  const result = await build({
    entryPoints: [entryPoint],
    bundle: true,
    platform: 'browser',
    format: 'esm',
    target: 'es2022',
    external: dependencies,
    write: false,
    minify: true,
    legalComments: 'none',
    logOverride: {
      'commonjs-variable-in-esm': 'silent',
      'direct-eval': 'silent',
      'duplicate-object-key': 'silent',
      'unsupported-dynamic-import': 'silent',
    },
  });
  const output = result.outputFiles?.[0];
  if (!output)
    throw new Error(`browser bundle produced no output for ${entryPoint}`);
  return output.text;
}

function workerRuntime() {
  return `import { parentPort } from 'node:worker_threads';

globalThis.self = globalThis;
globalThis.location = { hostname: 'localhost', port: '', protocol: 'file:' };
globalThis.postMessage = (message, transfer) => parentPort.postMessage(message, transfer);

const pending = [];
parentPort.on('message', (data) => {
  if (typeof globalThis.onmessage === 'function') globalThis.onmessage({ data });
  else pending.push(data);
});

await import('./worker-core.mjs');
for (const data of pending.splice(0)) globalThis.onmessage({ data });
`;
}

function cliRuntime(version) {
  return `#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker as ThreadWorker } from 'node:worker_threads';

const options = parseArgs(process.argv.slice(2));
for (const required of ['model', 'device', 'process', 'output']) {
  if (!options[required]) fail(\`--\${required} is required\`);
  options[required] = path.resolve(options[required]);
}
process.chdir(path.dirname(fileURLToPath(import.meta.url)));

globalThis.self = globalThis;
globalThis.location = { hostname: 'localhost', port: '', protocol: 'file:' };
globalThis.Worker = class Worker {
  constructor() {
    this.worker = new ThreadWorker(new URL('./worker.mjs', import.meta.url), {
      stdout: true,
      stderr: true,
    });
    this.worker.stdout.pipe(process.stderr);
    this.worker.stderr.pipe(process.stderr);
    this.worker.on('message', (data) => this.onmessage?.({ data }));
    this.worker.on('error', (error) => this.onerror?.(workerErrorEvent(error)));
    this.worker.on('messageerror', (error) => this.onmessageerror?.(workerErrorEvent(error)));
  }
  postMessage(message, transfer) { this.worker.postMessage(message, transfer); }
  terminate() { return this.worker.terminate(); }
};

try {
  const [{ newEngine }, device, processConfig, input] = await Promise.all([
    import('./engine.mjs'),
    readJson(options.device),
    readJson(options.process),
    fs.readFile(options.model),
  ]);
  const engine = newEngine().setMode('FDM').setDevice(device).setProcess(processConfig);
  engine.setController({ threaded: false });
  let lastProgress = -1;
  engine.setListener((event) => {
    const raw = event.slice?.update ?? (event.prepare?.update == null ? null : 0.5 + event.prepare.update * 0.5);
    if (raw == null) return;
    const progress = Math.max(0, Math.min(99, Math.floor(raw * 100)));
    if (progress !== lastProgress) {
      lastProgress = progress;
      process.stderr.write(\`progress:\${progress}\\n\`);
    }
  });
  await engine.parse(input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength));
  await engine.slice();
  await engine.prepare();
  const gcode = await engine.export();
  await fs.writeFile(options.output, \`; SimplyPrint Kiri:Moto ${version}\\n\${gcode}\`);
  process.stderr.write('progress:100\\n');
  process.exit(0);
} catch (error) {
  fail(error?.stack ?? String(error));
}

function parseArgs(values) {
  const parsed = {};
  for (let index = 0; index < values.length; index += 2) {
    const key = values[index];
    if (!key?.startsWith('--') || values[index + 1] == null) fail(\`invalid arguments near \${key ?? '<end>'}\`);
    parsed[key.slice(2)] = values[index + 1];
  }
  return parsed;
}

function workerErrorEvent(error) {
  return { error, message: error?.message ?? String(error), preventDefault() {} };
}

async function readJson(file) {
  try { return JSON.parse(await fs.readFile(file, 'utf8')); }
  catch (error) { throw new Error(\`invalid JSON in \${file}: \${error.message}\`); }
}

function fail(message) {
  process.stderr.write(\`kirimoto: \${message}\\n\`);
  process.exit(1);
}
`;
}

function launcher() {
  return `#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "\${BASH_SOURCE[0]}")" && pwd -P)"
node_bin="\${KIRIMOTO_NODE:-node}"
exec "$node_bin" "$script_dir/../lib/kirimoto/cli.mjs" "$@"
`;
}
