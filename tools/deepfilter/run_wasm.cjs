// Runs deepfilter.wasm on raw float32 mono 48 kHz: node run_wasm.cjs <wasm> <in.f32> <out.f32> <atten_db>
// Prints the milliseconds the frames took. Used by check.py.
"use strict";
const fs = require("fs");
const [wasmPath, inPath, outPath, atten] = process.argv.slice(2);
WebAssembly.instantiate(fs.readFileSync(wasmPath), {}).then(({ instance }) => {
  const E = instance.exports;
  const st = E.dfo_create(1, Number(atten));
  const hop = E.dfo_hop(st);
  const pi = E.dfo_alloc(hop * 4), po = E.dfo_alloc(hop * 4);
  const b = fs.readFileSync(inPath);
  const x = new Float32Array(b.buffer, b.byteOffset, b.length / 4);
  const y = new Float32Array(x.length);
  const t0 = Date.now();
  for (let f = 0; f + hop <= x.length; f += hop) {
    new Float32Array(E.memory.buffer, pi, hop).set(x.subarray(f, f + hop));
    E.dfo_process(st, pi, po);
    y.set(new Float32Array(E.memory.buffer, po, hop), f);
  }
  const ms = Date.now() - t0;
  fs.writeFileSync(outPath, Buffer.from(y.buffer));
  console.log(JSON.stringify({ ms: ms, hop: hop, delay: E.dfo_delay(st) }));
});
