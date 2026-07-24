#!/usr/bin/env bash

set -euo pipefail

source_dir="${SLICER_SOURCE_DIR:-slicer-src}"
build_dir="${KIRIMOTO_BUILD_DIR:-$source_dir/build/kirimoto-node}"
output_dir="${KIRIMOTO_PACKAGE_DIR:-$source_dir/build/slicer_out}"

rm -rf "$output_dir"
mkdir -p "$output_dir/bin" "$output_dir/lib/kirimoto" "$output_dir/lib/wasm" "$output_dir/browser"
cp "$build_dir/bin/kirimoto" "$output_dir/bin/kirimoto"
cp "$build_dir/lib/kirimoto/"*.mjs "$output_dir/lib/kirimoto/"
cp "$build_dir/lib/wasm/manifold.wasm" "$output_dir/lib/wasm/manifold.wasm"
cp "$build_dir/browser/kirimoto-runtime.mjs" "$output_dir/browser/kirimoto-runtime.mjs"
cp "$build_dir/package.json" "$output_dir/package.json"
cp "$source_dir/license.md" "$output_dir/LICENSE"
chmod 0755 "$output_dir/bin/kirimoto"

mkdir -p "$source_dir/build"
node - "$source_dir" "$output_dir" "${ARCH:?ARCH is required}" <<'NODE' \
  | tee "$source_dir/build/slicer-bundle-report.json"
import fs from 'node:fs';
import path from 'node:path';

const [sourceDir, outputDir, architecture] = process.argv.slice(2);
const upstream = JSON.parse(fs.readFileSync(path.join(sourceDir, 'package.json'), 'utf8'));
const files = [];

for (const relative of [
  'bin/kirimoto',
  'lib/kirimoto/cli.mjs',
  'lib/kirimoto/engine.mjs',
  'lib/kirimoto/worker-core.mjs',
  'lib/kirimoto/worker.mjs',
  'lib/wasm/manifold.wasm',
  'browser/kirimoto-runtime.mjs',
  'package.json',
  'LICENSE',
]) {
  const full = path.join(outputDir, relative);
  files.push({ path: relative, bytes: fs.statSync(full).size });
}

process.stdout.write(`${JSON.stringify({
  schema_version: 1,
  name: 'kirimoto',
  version: upstream.version,
  architecture,
  entrypoint: 'bin/kirimoto',
  runtime: 'node >=22',
  files,
}, null, 2)}\n`);
NODE
