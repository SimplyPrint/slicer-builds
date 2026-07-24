#!/usr/bin/env node

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';

const args = parseArgs(process.argv.slice(2));
const runtimePath = path.resolve(required(args, 'runtime'));
const modelPath = path.resolve(required(args, 'model'));
const chromium = args.chromium || process.env.CHROMIUM || 'chromium';

for (const file of [runtimePath, modelPath]) {
  if (!fs.statSync(file).isFile()) throw new Error(`not a file: ${file}`);
}

const server = http.createServer((request, response) => {
  if (request.url === '/runtime.mjs') {
    serveFile(response, runtimePath, 'text/javascript');
    return;
  }
  if (request.url === '/cube.stl') {
    serveFile(response, modelPath, 'model/stl');
    return;
  }

  response.writeHead(200, {
    'content-type': 'text/html; charset=utf-8',
    'cache-control': 'no-store',
  });
  response.end(testPage());
});

await new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(0, '127.0.0.1', resolve);
});

const address = server.address();
if (!address || typeof address === 'string')
  throw new Error('test server did not bind a TCP port');

const userDataDir = fs.mkdtempSync(
  path.join(os.tmpdir(), 'kirimoto-browser-test-'),
);
let browser;
try {
  browser = spawn(
    chromium,
    [
      '--headless=new',
      '--no-sandbox',
      '--disable-gpu',
      '--disable-dev-shm-usage',
      `--user-data-dir=${userDataDir}`,
      '--remote-debugging-port=0',
      'about:blank',
    ],
    {
      stdio: ['ignore', 'ignore', 'pipe'],
    },
  );
  const browserWebSocketUrl = await readDevToolsUrl(browser);
  const debuggerOrigin = new URL(browserWebSocketUrl);
  debuggerOrigin.protocol =
    debuggerOrigin.protocol === 'wss:' ? 'https:' : 'http:';
  debuggerOrigin.pathname = '';

  for (const belt of [false, true]) {
    const pageUrl = `http://127.0.0.1:${address.port}/?belt=${belt ? '1' : '0'}`;
    const target = await fetch(
      new URL(`/json/new?${encodeURIComponent(pageUrl)}`, debuggerOrigin),
      { method: 'PUT' },
    ).then((response) => response.json());
    const client = await connectCdp(target.webSocketDebuggerUrl);

    const status = await waitForStatus(client, 180000);
    const label = belt ? 'belt' : 'cartesian';
    if (status !== 'passed') {
      const body = await client.send('Runtime.evaluate', {
        expression: 'document.body.textContent',
        returnByValue: true,
      });
      throw new Error(
        `${label} browser slice failed: ${body.result?.result?.value || status}`,
      );
    }
    client.close();
    await fetch(new URL(`/json/close/${target.id}`, debuggerOrigin));
    process.stdout.write(`Kiri:Moto ${label} browser slice passed\n`);
  }
} finally {
  server.close();
  if (browser && browser.exitCode === null) {
    browser.kill('SIGTERM');
    await Promise.race([
      new Promise((resolve) => browser.once('exit', resolve)),
      new Promise((resolve) => setTimeout(resolve, 5000)),
    ]);
    if (browser.exitCode === null) browser.kill('SIGKILL');
  }
  fs.rmSync(userDataDir, { recursive: true, force: true });
}

function readDevToolsUrl(browserProcess) {
  return new Promise((resolve, reject) => {
    let stderr = '';
    const timeout = setTimeout(
      () =>
        reject(
          new Error(`Chromium did not expose DevTools: ${stderr.slice(-2000)}`),
        ),
      30000,
    );
    browserProcess.stderr.setEncoding('utf8');
    browserProcess.stderr.on('data', (chunk) => {
      stderr += chunk;
      const match = stderr.match(/DevTools listening on (ws:\/\/\S+)/);
      if (!match) return;
      clearTimeout(timeout);
      resolve(match[1]);
    });
    browserProcess.once('error', reject);
    browserProcess.once('exit', (code) => {
      clearTimeout(timeout);
      reject(new Error(`Chromium exited before DevTools was ready (${code})`));
    });
  });
}

function connectCdp(url) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url);
    let sequence = 0;
    const pending = new Map();

    socket.addEventListener('open', () => {
      resolve({
        send(method, params = {}) {
          const id = ++sequence;
          socket.send(JSON.stringify({ id, method, params }));
          return new Promise((accept, decline) =>
            pending.set(id, { accept, decline }),
          );
        },
        close() {
          socket.close();
        },
      });
    });
    socket.addEventListener('error', reject);
    socket.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      if (!message.id) return;
      const request = pending.get(message.id);
      if (!request) return;
      pending.delete(message.id);
      if (message.error) request.decline(new Error(message.error.message));
      else request.accept(message);
    });
  });
}

async function waitForStatus(client, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const response = await client.send('Runtime.evaluate', {
      expression: 'document.body?.dataset.status',
      returnByValue: true,
    });
    const status = response.result?.result?.value;
    if (status === 'passed' || status === 'failed') return status;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return 'timed out';
}

function serveFile(response, file, contentType) {
  response.writeHead(200, {
    'content-type': contentType,
    'content-length': fs.statSync(file).size,
    'cache-control': 'no-store',
  });
  fs.createReadStream(file).pipe(response);
}

function testPage() {
  return `<!doctype html>
<html>
  <body data-status="running">running</body>
  <script type="module">
    import { Engine, poolSource, version, workerSource } from '/runtime.mjs';

    const workerUrl = URL.createObjectURL(new Blob([workerSource], { type: 'text/javascript' }));
    const poolUrl = URL.createObjectURL(new Blob([poolSource], { type: 'text/javascript' }));
    const engine = new Engine({ workURL: workerUrl, poolURL: poolUrl });
    const belt = new URLSearchParams(location.search).get('belt') === '1';

    try {
      engine
        .setMode('FDM')
        .setDevice({
          bedBelt: belt,
          bedWidth: 220,
          bedDepth: 220,
          maxHeight: 250,
        })
        .setProcess({
          sliceHeight: 0.25,
          sliceAngle: belt ? 45 : 0,
          outputTemp: 210,
          firstLayerNozzleTemp: 215,
        })
        .setController({ threaded: true });

      const model = await fetch('/cube.stl').then((response) => response.arrayBuffer());
      await engine.parse(model);
      await engine.slice();
      await engine.prepare();
      const gcode = await engine.export();
      if (typeof gcode !== 'string' || gcode.length < 100 || !/\\bG[01]\\b/.test(gcode)) {
        throw new Error('runtime returned invalid G-code');
      }

      document.body.dataset.status = 'passed';
      document.body.textContent = JSON.stringify({ belt, bytes: gcode.length, version });
    } catch (error) {
      document.body.dataset.status = 'failed';
      document.body.textContent = error?.stack || String(error);
    } finally {
      engine.terminate();
      URL.revokeObjectURL(workerUrl);
      URL.revokeObjectURL(poolUrl);
    }
  </script>
</html>`;
}

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
