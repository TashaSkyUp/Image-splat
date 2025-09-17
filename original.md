// @ts-nocheck
import React, { useEffect, useRef, useState } from "react";

/**
 * Image‑GS (JS‑only, CSP‑safe) — Worker Pool Edition
 * ------------------------------------------------------
 * Pure JavaScript/browser reference implementation of
 *   "Image-GS: Content-Adaptive Image Representation via 2D Gaussians"
 *
 * This build:
 *  • Avoids TFJS/WebGL/WASM and any dynamic eval — fully CSP‑safe.
 *  • Uses a pool of classic Web Workers to compute each render/EM step off the main thread.
 *  • Tile‑binned Top‑K rendering; content‑adaptive init (Sobel + uniform mix);
 *    progressive add‑by‑error; trains COLORS + μ + θ + s⁻¹ on CPU.
 *  • UI draws at a fixed ~10Hz (every 100ms) regardless of step cadence.
 *  • Downscales the preview to max 512px on the long side before training.
 *
 * NOTE: The live code now factors the worker/training orchestration into
 * `src/training.js`; this file captures the legacy single-file layout for
 * reference.
 */

// ------------------------------------------------------------
// Utilities (CSP‑safe, no eval)
// ------------------------------------------------------------
const EPS = 1e-6;
const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

function toImageDataFromF32RGB(arr, w, h) {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, j = 0; i < w * h; i++) {
    out[j++] = clamp(Math.round(arr[i * 3 + 0] * 255), 0, 255);
    out[j++] = clamp(Math.round(arr[i * 3 + 1] * 255), 0, 255);
    out[j++] = clamp(Math.round(arr[i * 3 + 2] * 255), 0, 255);
    out[j++] = 255;
  }
  return new ImageData(out, w, h);
}

function sourceToF32RGB(src, targetMax = 512) {
  // `src` can be HTMLImageElement or ImageBitmap
  const c = document.createElement("canvas");
  const w0 = Math.max(1, src.width);
  const h0 = Math.max(1, src.height);
  const scale = Math.min(1, targetMax / Math.max(w0, h0));
  const w = Math.max(1, Math.round(w0 * scale));
  const h = Math.max(1, Math.round(h0 * scale));
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Canvas 2D context unavailable");
  try {
    ctx.drawImage(src, 0, 0, w, h);
  } catch (e) {
    throw new Error(
      "Rasterization failed (drawImage). The file may be corrupt or blocked."
    );
  }
  let imgData;
  try {
    imgData = ctx.getImageData(0, 0, w, h);
  } catch (e) {
    throw new Error(
      "Reading pixels failed (getImageData). If this is an SVG with external refs, re‑export as PNG/JPG."
    );
  }
  const data = imgData.data;
  const f32 = new Float32Array(w * h * 3);
  for (let i = 0; i < w * h; i++) {
    f32[i * 3 + 0] = data[i * 4 + 0] / 255;
    f32[i * 3 + 1] = data[i * 4 + 1] / 255;
    f32[i * 3 + 2] = data[i * 4 + 2] / 255;
  }
  return { data: f32, w, h };
}

async function fileToImageSource(file) {
  if ("createImageBitmap" in window && typeof createImageBitmap === "function") {
    try {
      const bmp = await createImageBitmap(file);
      return { src: bmp, revoke: () => bmp.close?.() };
    } catch (e) {
      console.warn("createImageBitmap failed; falling back to HTMLImageElement", e);
    }
  }
  const dataUrl = await new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onerror = () => reject(new Error("FileReader failed"));
    fr.onload = () => resolve(String(fr.result));
    fr.readAsDataURL(file);
  });
  const img = await new Promise((resolve, reject) => {
    const im = new Image();
    // In case someone swaps to URL mode later, keep it CORS‑friendly
    im.crossOrigin = "anonymous";
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("Image decode failed"));
    im.src = dataUrl;
  });
  return { src: img, revoke: () => {} };
}

function sobelMagJS(rgb, w, h) {
  const mag = new Float32Array(w * h);
  const kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let gx = 0, gy = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const xx = clamp(x + dx, 0, w - 1), yy = clamp(y + dy, 0, h - 1);
          const i = yy * w + xx;
          const r = rgb[i * 3 + 0], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
          const gray = 0.299 * r + 0.587 * g + 0.114 * b;
          const kIdx = (dy + 1) * 3 + (dx + 1);
          gx += gray * kx[kIdx];
          gy += gray * ky[kIdx];
        }
      mag[y * w + x] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return mag;
}

function cpuTopKIndices(arr, K) {
  const n = arr.length;
  const k = Math.min(K, n);
  const idxs = new Array(k).fill(-1);
  const vals = new Array(k).fill(-Infinity);
  for (let i = 0; i < n; i++) {
    const v = arr[i];
    if (v > vals[k - 1]) {
      let j = k - 1;
      while (j > 0 && v > vals[j - 1]) {
        vals[j] = vals[j - 1];
        idxs[j] = idxs[j - 1];
        j--;
      }
      vals[j] = v;
      idxs[j] = i;
    }
  }
  return idxs;
}

function buildTileBinsFromArrays(muA, sInvA, H, W, tileH, tileW) {
  // Defensive: allow empty/boot state without crashing
  if (!muA || !sInvA) {
    const tilesX = Math.ceil(W / tileW);
    const tilesY = Math.ceil(H / tileH);
    return { tilesX, tilesY, bins: Array.from({ length: tilesX * tilesY }, () => []) };
  }
  const tilesX = Math.ceil(W / tileW);
  const tilesY = Math.ceil(H / tileH);
  const bins = Array.from({ length: tilesX * tilesY }, () => []);
  const N = muA.length / 2;
  for (let i = 0; i < N; i++) {
    const ux = muA[i * 2 + 0], uy = muA[i * 2 + 1];
    const sxInv = sInvA[i * 2 + 0], syInv = sInvA[i * 2 + 1];
    const cx = ux * (W - 1), cy = uy * (H - 1);
    const rx = 3 * (1 / Math.max(sxInv, EPS)) * W; // 3σ in pixels (x axis)
    const ry = 3 * (1 / Math.max(syInv, EPS)) * H; // 3σ in pixels (y axis)
    const r = Math.max(rx, ry);
    const x0 = Math.max(0, Math.floor((cx - r) / tileW));
    const y0 = Math.max(0, Math.floor((cy - r) / tileH));
    const x1 = Math.min(tilesX - 1, Math.floor((cx + r) / tileW));
    const y1 = Math.min(tilesY - 1, Math.floor((cy + r) / tileH));
    for (let ty = y0; ty <= y1; ty++) {
      const row = ty * tilesX;
      for (let tx = x0; tx <= x1; tx++) bins[row + tx].push(i);
    }
  }
  return { tilesX, tilesY, bins };
}

function samplePositionsAndColorsJS(imgF32, w, h, prob2D, N) {
  const prob = new Float32Array(prob2D);
  let s = 0; for (let i = 0; i < prob.length; i++) s += prob[i]; s = s || 1;
  for (let i = 0; i < prob.length; i++) prob[i] /= s;
  const cdf = new Float32Array(prob.length); let acc = 0;
  for (let i = 0; i < prob.length; i++) { acc += prob[i]; cdf[i] = acc; }
  const mu = new Float32Array(N * 2);
  const colors = new Float32Array(N * 3);
  for (let k = 0; k < N; k++) {
    const r = Math.random();
    let lo = 0, hi = cdf.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < r) lo = mid + 1; else hi = mid; }
    const y = Math.floor(lo / w), x = lo % w, idx = y * w + x;
    mu[k * 2 + 0] = x / (w - 1);
    mu[k * 2 + 1] = y / (h - 1);
    colors[k * 3 + 0] = imgF32[idx * 3 + 0];
    colors[k * 3 + 1] = imgF32[idx * 3 + 1];
    colors[k * 3 + 2] = imgF32[idx * 3 + 2];
  }
  return { mu, colors };
}

function psnrJS(a, b) {
  let mse = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; mse += d * d; }
  mse /= a.length || 1; if (mse <= 0) return Infinity; return 10 * Math.log10(1 / (mse + EPS));
}

function float16Quantize(x) {
  const f32 = x instanceof Float32Array ? x : new Float32Array(x);
  const buf = new ArrayBuffer(f32.length * 2);
  const dv = new DataView(buf);
  for (let i = 0; i < f32.length; i++) dv.setUint16(i * 2, float32ToFloat16(f32[i]), true);
  return new Uint8Array(buf);
}
function float32ToFloat16(val) {
  const f32 = new Float32Array(1); const i32 = new Int32Array(f32.buffer); f32[0] = val; const x = i32[0];
  const bits = (x >> 16) & 0x8000; const m = (x >> 12) & 0x07ff; const e = (x >> 23) & 0xff;
  if (e < 103) return bits; if (e > 142) return bits | 0x7c00; return bits | ((e - 112) << 10) | (m >> 1);
}

// ------------------------------------------------------------
// Worker source (classic worker via Blob, CSP‑safe; no imports)
// ------------------------------------------------------------
const WORKER_SRC = `
  const EPS = 1e-6;
  const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
  function buildTileBinsFromArrays(muA, sInvA, H, W, tileH, tileW) {
    if (!muA || !sInvA) {
      const tilesX = Math.ceil(W / tileW);
      const tilesY = Math.ceil(H / tileH);
      return { tilesX, tilesY, bins: Array.from({ length: tilesX * tilesY }, () => []) };
    }
    const tilesX = Math.ceil(W / tileW);
    const tilesY = Math.ceil(H / tileH);
    const bins = Array.from({ length: tilesX * tilesY }, () => []);
    const N = muA.length / 2;
    for (let i = 0; i < N; i++) {
      const ux = muA[i * 2 + 0], uy = muA[i * 2 + 1];
      const sxInv = sInvA[i * 2 + 0], syInv = sInvA[i * 2 + 1];
      const cx = ux * (W - 1), cy = uy * (H - 1);
      const rx = 3 * (1 / Math.max(sxInv, EPS)) * W;
      const ry = 3 * (1 / Math.max(syInv, EPS)) * H;
      const r = Math.max(rx, ry);
      const x0 = Math.max(0, Math.floor((cx - r) / tileW));
      const y0 = Math.max(0, Math.floor((cy - r) / tileH));
      const x1 = Math.min(tilesX - 1, Math.floor((cx + r) / tileW));
      const y1 = Math.min(tilesY - 1, Math.floor((cy + r) / tileH));
      for (let ty = y0; ty <= y1; ty++) {
        const row = ty * tilesX;
        for (let tx = x0; tx <= x1; tx++) bins[row + tx].push(i);
      }
    }
    return { tilesX, tilesY, bins };
  }

  let IMG = null; // Float32Array W*H*3
  let W = 0, H = 0;
  let tileW = 32, tileH = 32;
  let enableTiling = true;
  let bins = null; // {tilesX,tilesY,bins[]}

  function computeStepStripe(mu, s_inv, theta, color, K, wantImage, doRebin, y0Stripe, y1Stripe, reqId) {
    const N = mu.length / 2;
    if (!IMG) throw new Error('Worker not initialized');
    if (enableTiling && (doRebin || !bins)) bins = buildTileBinsFromArrays(mu, s_inv, H, W, tileH, tileW);

    const out = wantImage ? new Float32Array((y1Stripe - y0Stripe + 1) * W * 3) : null;
    const accW = new Float32Array(N);
    const accTW = new Float32Array(N * 3);
    const accX = new Float32Array(N);
    const accY = new Float32Array(N);
    const accXX = new Float32Array(N);
    const accYY = new Float32Array(N);
    const accXY = new Float32Array(N);

    const tilesX = bins ? bins.tilesX : Math.ceil(W / tileW);
    const tilesY = bins ? bins.tilesY : Math.ceil(H / tileH);

    const ty0 = Math.max(0, Math.floor(y0Stripe / tileH));
    const ty1 = Math.min(tilesY - 1, Math.floor(y1Stripe / tileH));

    for (let ty = ty0; ty <= ty1; ty++) {
      const xTiles = tilesX;
      const y0 = Math.max(y0Stripe, ty * tileH);
      const y1 = Math.min(y1Stripe + 1, (ty + 1) * tileH);
      for (let tx = 0; tx < xTiles; tx++) {
        const x0 = tx * tileW;
        const x1 = Math.min(W, x0 + tileW);
        const list = enableTiling && bins ? bins.bins[ty * tilesX + tx] : null;
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            const cx = x / (W - 1), cy = y / (H - 1);
            const topIdx = []; const topVal = [];
            const scan = (i) => {
              const dx = cx - mu[i * 2 + 0];
              const dy = cy - mu[i * 2 + 1];
              const th = theta[i];
              const c = Math.cos(th), s = Math.sin(th);
              const dxp = c * dx + s * dy;
              const dyp = -s * dx + c * dy;
              const sx = s_inv[i * 2 + 0], sy = s_inv[i * 2 + 1];
              const z = (dxp * sx) * (dxp * sx) + (dyp * sy) * (dyp * sy);
              const g = Math.exp(-0.5 * z);
              if (topIdx.length < K) {
                let j = topIdx.length; topIdx.push(i); topVal.push(g);
                while (j > 0 && topVal[j] > topVal[j - 1]) {
                  const ti = topIdx[j - 1]; topIdx[j - 1] = topIdx[j]; topIdx[j] = ti;
                  const tv = topVal[j - 1]; topVal[j - 1] = topVal[j]; topVal[j] = tv; j--;
                }
              } else if (g > topVal[topVal.length - 1]) {
                topIdx[topIdx.length - 1] = i; topVal[topVal.length - 1] = g; let j = topIdx.length - 1;
                while (j > 0 && topVal[j] > topVal[j - 1]) {
                  const ti = topIdx[j - 1]; topIdx[j - 1] = topIdx[j]; topIdx[j] = ti;
                  const tv = topVal[j - 1]; topVal[j - 1] = topVal[j]; topVal[j] = tv; j--;
                }
              }
            };
            if (list && list.length) { for (let ii = 0; ii < list.length; ii++) scan(list[ii]); }
            else { for (let i = 0; i < N; i++) scan(i); }
            let wsum = EPS; for (let k = 0; k < topVal.length; k++) wsum += topVal[k];
            const baseGlobal = (y * W + x) * 3;
            let r = 0, g = 0, b = 0;
            for (let k = 0; k < topIdx.length; k++) {
              const wi = topVal[k] / wsum; const gi = topIdx[k];
              r += wi * color[gi * 3 + 0]; g += wi * color[gi * 3 + 1]; b += wi * color[gi * 3 + 2];
              accW[gi] += wi;
              accTW[gi * 3 + 0] += wi * IMG[baseGlobal + 0];
              accTW[gi * 3 + 1] += wi * IMG[baseGlobal + 1];
              accTW[gi * 3 + 2] += wi * IMG[baseGlobal + 2];
              const cxn = cx, cyn = cy;
              accX[gi] += wi * cxn; accY[gi] += wi * cyn;
              accXX[gi] += wi * cxn * cxn; accYY[gi] += wi * cyn * cyn; accXY[gi] += wi * cxn * cyn;
            }
            if (out) {
              const row = y - y0Stripe; const baseLocal = (row * W + x) * 3;
              out[baseLocal + 0] = r; out[baseLocal + 1] = g; out[baseLocal + 2] = b;
            }
          }
        }
      }
    }

    const msg = { type: 'stepResult', reqId, accW, accTW, accX, accY, accXX, accYY, accXY, y0: y0Stripe, y1: y1Stripe };
    const transfers = [accW.buffer, accTW.buffer, accX.buffer, accY.buffer, accXX.buffer, accYY.buffer, accXY.buffer];
    if (out) { msg.out = out; transfers.push(out.buffer); }
    // @ts-ignore
    postMessage(msg, transfers);
  }

  // @ts-ignore
  onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'init') {
      IMG = m.img; W = m.W; H = m.H; tileW = m.tileW; tileH = m.tileH; enableTiling = !!m.enableTiling; bins = null;
      // @ts-ignore
      postMessage({ type: 'inited' });
    } else if (m.type === 'step') {
      computeStepStripe(m.mu, m.s_inv, m.theta, m.color, m.K, !!m.wantImage, !!m.doRebin, m.y0, m.y1, m.reqId);
    } else if (m.type === 'setTiling') {
      tileW = m.tileW; tileH = m.tileH; enableTiling = !!m.enableTiling; bins = null;
    }
  };
`;

function makeWorker() {
  const blob = new Blob([WORKER_SRC], { type: "text/javascript" });
  const url = URL.createObjectURL(blob);
  const w = new Worker(url, { type: "classic" });
  return { worker: w, url };
}

// ------------------------------------------------------------
// Diagnostics (optional quick tests)
// ------------------------------------------------------------
function Diagnostics() {
  const [diag, setDiag] = useState("idle");
  const run = async () => {
    try {
      setDiag("running");
      const { worker, url } = makeWorker();
      const W = 8, H = 8;
      const img = new Float32Array(W * H * 3);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 3; img[i] = x / (W - 1); img[i + 1] = y / (H - 1); img[i + 2] = 0.5;
      }
      const waitType = (type) => new Promise((res) => {
        const handler = (ev) => { if (ev.data?.type === type) { worker.removeEventListener('message', handler); res(ev.data); } };
        worker.addEventListener('message', handler);
      });
      worker.postMessage({ type: 'init', img, W, H, tileW: 4, tileH: 4, enableTiling: true }, [img.buffer]);
      await waitType('inited');
      const mu = new Float32Array([0.25, 0.25, 0.75, 0.75]);
      const s_inv = new Float32Array([W - 1, H - 1, W - 1, H - 1]);
      const theta = new Float32Array([0, 0]);
      const color = new Float32Array([1, 0, 0, 0, 1, 0]);
      const reqId = 1234;
      const got = new Promise((res) => {
        const handler = (ev) => { const d = ev.data; if (d?.type === 'stepResult' && d.reqId === reqId) { worker.removeEventListener('message', handler); res(d); } };
        worker.addEventListener('message', handler);
      });
      worker.postMessage({ type: 'step', reqId, mu, s_inv, theta, color, K: 2, wantImage: true, doRebin: true, y0: 0, y1: H - 1 });
      const d = await got;
      URL.revokeObjectURL(url); worker.terminate();
      if (!d.out || d.out.length !== W * H * 3) throw new Error('render size mismatch');
      setDiag("ok");
    } catch (e) {
      setDiag("fail: " + (e?.message || e));
    }
  };
  return (
    <div className="ml-2 inline-flex items-center gap-2">
      <button className="px-3 py-2 rounded border" onClick={run}>Diagnostics</button>
      <span className="text-sm text-gray-600">{diag}</span>
    </div>
  );
}

// ------------------------------------------------------------
// React app (JS‑only, with Worker offload)
// ------------------------------------------------------------
export default function ImageGSApp() {
  const [imgJS, setImgJS] = useState(null); // {data,w,h}
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [status, setStatus] = useState("idle");
  const [metrics, setMetrics] = useState({ step: 0, psnr: null, n: 0, mode: "js", worker: true, pool: 1 });

  // Controls
  const [K, setK] = useState(6);
  const [budget, setBudget] = useState(2000);
  const [lambdaInit, setLambdaInit] = useState(0.3);
  const [steps, setSteps] = useState(1500);
  const [stepDelayMs, setStepDelayMs] = useState(0);

  const [lrColor, setLrColor] = useState(0.4);
  const [lrMu, setLrMu] = useState(0.25);
  const [lrShape, setLrShape] = useState(0.25);

  const [enableTiling, setEnableTiling] = useState(true);
  const [tileW, setTileW] = useState(32);
  const [tileH, setTileH] = useState(32);
  const [rebinEvery, setRebinEvery] = useState(50);

  const defaultPool = Math.max(2, Math.min((navigator.hardwareConcurrency || 4), 8));
  const [poolSize, setPoolSize] = useState(defaultPool);

  const canvasRef = useRef(null);
  const stopRef = useRef(false);
  const varsRef = useRef(null); // { mu, s_inv, theta, color }
  const curNgRef = useRef(0);
  const lastDrawRef = useRef(0);

  // Worker pool
  const poolRef = useRef({ workers: [], urls: [], ready: [], resolvers: [], nextReqId: 1 });

  useEffect(() => {
    // Build initial pool
    rebuildPool(poolSize);
    return () => destroyPool();
  }, []);

  useEffect(() => {
    // Rebuild pool when size changes
    rebuildPool(poolSize, true);
  }, [poolSize]);

  function destroyPool() {
    const pool = poolRef.current;
    for (const w of pool.workers) try { w.terminate(); } catch {}
    for (const u of pool.urls) try { URL.revokeObjectURL(u); } catch {}
    poolRef.current = { workers: [], urls: [], ready: [], resolvers: [], nextReqId: 1 };
  }

  function rebuildPool(size, preserveImage = false) {
    const oldImg = preserveImage ? imgJS : null;
    destroyPool();
    const workers = []; const urls = []; const ready = []; const resolvers = [];
    for (let i = 0; i < size; i++) {
      const { worker, url } = makeWorker();
      workers.push(worker); urls.push(url); ready.push(false); resolvers.push(new Map());
      worker.onmessage = (ev) => {
        if (ev.data?.type === 'inited') {
          ready[i] = true;
        } else if (ev.data?.type === 'stepResult') {
          const resMap = resolvers[i];
          const rid = ev.data.reqId;
          if (rid && resMap.has(rid)) { const fn = resMap.get(rid); resMap.delete(rid); fn(ev.data); }
        }
      };
    }
    poolRef.current = { workers, urls, ready, resolvers, nextReqId: 1 };

    // If we already have an image, initialize workers with it
    if (preserveImage && oldImg) {
      broadcastInitImage(oldImg);
    }
  }

  function broadcastInitImage(js) {
    const pool = poolRef.current; if (!pool.workers.length) return;
    const { w, h, data } = js; setImgSize({ w, h });
    for (let i = 0; i < pool.workers.length; i++) {
      const imgCopy = new Float32Array(data); // clone for each worker; transfer the buffer
      pool.ready[i] = false;
      pool.workers[i].postMessage({ type: 'init', img: imgCopy, W: w, H: h, tileW, tileH, enableTiling }, [imgCopy.buffer]);
    }
  }

  function broadcastSetTiling() {
    const pool = poolRef.current; for (let i = 0; i < pool.workers.length; i++) {
      pool.workers[i].postMessage({ type: 'setTiling', tileW, tileH, enableTiling });
    }
  }

  async function waitAllReady() {
    const pool = poolRef.current; const deadline = performance.now() + 5000;
    while (true) {
      if (pool.ready.every(Boolean)) return;
      if (performance.now() > deadline) throw new Error('Worker pool init timeout');
      await new Promise((r) => setTimeout(r, 10));
    }
  }

  function postStepToWorker(i, payload) {
    const pool = poolRef.current; const rid = pool.nextReqId++;
    return new Promise((resolve) => {
      pool.resolvers[i].set(rid, resolve);
      pool.workers[i].postMessage({ type: 'step', reqId: rid, ...payload });
    });
  }

  function drawOnCanvasF32RGB(arr, w, h) {
    const ctx = canvasRef.current?.getContext("2d"); if (!ctx) return;
    canvasRef.current.width = w; canvasRef.current.height = h;
    const out = new Uint8ClampedArray(w * h * 4);
    for (let i = 0, j = 0; i < w * h; i++) {
      out[j++] = clamp(Math.round(arr[i * 3 + 0] * 255), 0, 255);
      out[j++] = clamp(Math.round(arr[i * 3 + 1] * 255), 0, 255);
      out[j++] = clamp(Math.round(arr[i * 3 + 2] * 255), 0, 255);
      out[j++] = 255;
    }
    ctx.putImageData(new ImageData(out, w, h), 0, 0);
  }

  async function handleFile(e) {
    try {
      const file = e.target.files?.[0]; if (!file) return;
      if (!(file.type && file.type.startsWith("image/"))) throw new Error(`Selected file is not an image (type: ${file.type || 'unknown'})`);
      const { src, revoke } = await fileToImageSource(file);
      let js; try { js = sourceToF32RGB(src, 512); } finally { try { revoke?.(); } catch {} }
      const localData = new Float32Array(js.data);
      setImgJS({ data: localData, w: js.w, h: js.h }); setImgSize({ w: js.w, h: js.h });
      drawOnCanvasF32RGB(localData, js.w, js.h);
      disposeVars();
      broadcastInitImage({ data: localData, w: js.w, h: js.h });
      setStatus("loaded");
    } catch (err) {
      console.error("handleFile failed", err);
      alert(`Failed to load image.\n\nReason: ${err?.message || err}`);
    }
  }

  function disposeVars() { varsRef.current = null; curNgRef.current = 0; }

  async function initParamsJS() {
    if (!imgJS) return;
    const { w, h, data } = imgJS;
    const grad = sobelMagJS(data, w, h);
    let sum = 0; for (let i = 0; i < grad.length; i++) sum += grad[i];
    const uniform = 1 / (w * h);
    const probs = new Float32Array(w * h);
    const lmb = lambdaInit;
    for (let i = 0; i < w * h; i++) probs[i] = (1 - lmb) * (grad[i] / (sum || 1)) + lmb * uniform;

    const Ng0 = Math.max(1, Math.floor(budget / 2));
    const { mu, colors } = samplePositionsAndColorsJS(data, w, h, probs, Ng0);
    const s_inv = new Float32Array(Ng0 * 2);
    for (let i = 0; i < Ng0; i++) { s_inv[i * 2 + 0] = (w - 1) / 5; s_inv[i * 2 + 1] = (h - 1) / 5; }
    const theta = new Float32Array(Ng0); for (let i = 0; i < Ng0; i++) theta[i] = Math.random() * Math.PI;

    varsRef.current = { mu, s_inv, theta, color: colors };
    curNgRef.current = Ng0;
    broadcastSetTiling();
  }

  function paramsForStep() {
    const v = varsRef.current; if (!v) return null;
    return {
      mu: new Float32Array(v.mu),
      s_inv: new Float32Array(v.s_inv),
      theta: new Float32Array(v.theta),
      color: new Float32Array(v.color),
    };
  }

  function mergeAccumulators(parts, N) {
    const accW = new Float32Array(N);
    const accTW = new Float32Array(N * 3);
    const accX = new Float32Array(N);
    const accY = new Float32Array(N);
    const accXX = new Float32Array(N);
    const accYY = new Float32Array(N);
    const accXY = new Float32Array(N);
    for (const p of parts) {
      const aW = p.accW, aT = p.accTW, aX = p.accX, aY = p.accY, aXX = p.accXX, aYY = p.accYY, aXY = p.accXY;
      for (let i = 0; i < N; i++) accW[i] += aW[i];
      for (let i = 0; i < N * 3; i++) accTW[i] += aT[i];
      for (let i = 0; i < N; i++) { accX[i] += aX[i]; accY[i] += aY[i]; accXX[i] += aXX[i]; accYY[i] += aYY[i]; accXY[i] += aXY[i]; }
    }
    return { accW, accTW, accX, accY, accXX, accYY, accXY };
  }

  function composeOut(parts, W, H) {
    const out = new Float32Array(W * H * 3);
    for (const p of parts) {
      if (!p.out) continue;
      const y0 = p.y0, y1 = p.y1; const rows = y1 - y0 + 1; const stripe = p.out;
      out.set(stripe, y0 * W * 3);
    }
    return out;
  }

  async function train() {
    if (!imgJS) return;
    setStatus("initializing"); stopRef.current = false; await initParamsJS();
    try { await waitAllReady(); } catch (e) { console.error(e); }
    setStatus("training"); await trainJS_pool(); setStatus("done");
  }

  function updateParamsFromAcc(N, pkg) {
    const v = varsRef.current; if (!v) return;
    for (let i = 0; i < N; i++) {
      const wsum = pkg.accW[i] + 1e-6;
      const tR = pkg.accTW[i * 3 + 0] / wsum, tG = pkg.accTW[i * 3 + 1] / wsum, tB = pkg.accTW[i * 3 + 2] / wsum;
      v.color[i * 3 + 0] = (1 - lrColor) * v.color[i * 3 + 0] + lrColor * tR;
      v.color[i * 3 + 1] = (1 - lrColor) * v.color[i * 3 + 1] + lrColor * tG;
      v.color[i * 3 + 2] = (1 - lrColor) * v.color[i * 3 + 2] + lrColor * tB;
    }
    const w = imgSize.w, h = imgSize.h; const stdMinX = 1 / Math.max(w - 1, 1); const stdMinY = 1 / Math.max(h - 1, 1);
    for (let i = 0; i < N; i++) {
      const wsum = pkg.accW[i]; if (wsum < 1e-6) continue;
      const mx = pkg.accX[i] / wsum, my = pkg.accY[i] / wsum;
      const muX = v.mu[i * 2 + 0], muY = v.mu[i * 2 + 1];
      v.mu[i * 2 + 0] = (1 - lrMu) * muX + lrMu * mx;
      v.mu[i * 2 + 1] = (1 - lrMu) * muY + lrMu * my;
      const cxx = Math.max(0, pkg.accXX[i] / wsum - mx * mx);
      const cyy = Math.max(0, pkg.accYY[i] / wsum - my * my);
      const cxy = pkg.accXY[i] / wsum - mx * my;
      const tr = cxx + cyy; const det = cxx * cyy - cxy * cxy; const disc = Math.max(0, tr * tr - 4 * det);
      const s = Math.sqrt(disc); const l1 = 0.5 * (tr + s); const l2 = 0.5 * (tr - s);
      let vx, vy; if (Math.abs(cxy) > 1e-12) { vx = l1 - cyy; vy = cxy; } else { if (cxx >= cyy) { vx = 1; vy = 0; } else { vx = 0; vy = 1; } }
      const n = Math.hypot(vx, vy) || 1; vx /= n; vy /= n;
      let th0 = v.theta[i] % Math.PI; if (th0 < 0) th0 += Math.PI; let thT = Math.atan2(vy, vx); thT %= Math.PI; if (thT < 0) thT += Math.PI;
      if (Math.abs(thT - th0) > Math.PI / 2) { if (thT > th0) thT -= Math.PI; else thT += Math.PI; }
      const thNew = th0 + (thT - th0) * lrShape; v.theta[i] = ((thNew % Math.PI) + Math.PI) % Math.PI;
      const std1 = Math.sqrt(Math.max(l1, 0)); const std2 = Math.sqrt(Math.max(l2, 0));
      const sxInvTarget = 1 / Math.max(std1, Math.min(stdMinX, stdMinY));
      const syInvTarget = 1 / Math.max(std2, Math.min(stdMinX, stdMinY));
      const sx0 = v.s_inv[i * 2 + 0], sy0 = v.s_inv[i * 2 + 1];
      v.s_inv[i * 2 + 0] = clamp((1 - lrShape) * sx0 + lrShape * sxInvTarget, 0.1, (w - 1) * 4);
      v.s_inv[i * 2 + 1] = clamp((1 - lrShape) * sy0 + lrShape * syInvTarget, 0.1, (h - 1) * 4);
    }
  }

  async function renderFrameViaPool(doRebin = false) {
    const pool = poolRef.current; const v = varsRef.current; if (!v || !pool.workers.length) return null;
    await waitAllReady();
    const { w: W, h: H } = imgSize; const parts = [];
    const params = paramsForStep();
    const stripes = Math.min(pool.workers.length, H);
    const promises = [];
    for (let i = 0; i < stripes; i++) {
      const y0 = Math.floor((H * i) / stripes);
      const y1 = Math.min(H - 1, Math.floor((H * (i + 1)) / stripes) - 1);
      const wantImage = true; const payload = { K, doRebin, wantImage, y0, y1, ...params };
      promises.push(postStepToWorker(i, payload));
    }
    const results = await Promise.all(promises);
    return composeOut(results, W, H);
  }

  async function trainJS_pool() {
    const pool = poolRef.current; if (!pool.workers.length) return;
    const Ntotal = budget; const addEvery = 400; const addChunk = Math.max(1, Math.floor(Ntotal / 8));
    const { w: W, h: H } = imgSize;

    for (let step = 1; step <= steps; step++) {
      if (stopRef.current) break;
      const v = varsRef.current; if (!v) break; const N = v.mu.length / 2;
      const params = paramsForStep();
      const stripes = Math.min(pool.workers.length, H);
      const doRebin = enableTiling && (step === 1 || (rebinEvery > 0 && step % rebinEvery === 0));
      const wantImage = performance.now() - lastDrawRef.current >= 100; // 10Hz

      const promises = [];
      for (let i = 0; i < stripes; i++) {
        const y0 = Math.floor((H * i) / stripes);
        const y1 = Math.min(H - 1, Math.floor((H * (i + 1)) / stripes) - 1);
        const payload = { K, doRebin, wantImage, y0, y1, ...params };
        promises.push(postStepToWorker(i, payload));
      }
      const results = await Promise.all(promises);
      const merged = mergeAccumulators(results, N);
      updateParamsFromAcc(N, merged);

      if (wantImage) {
        const out = composeOut(results, W, H);
        drawOnCanvasF32RGB(out, W, H);
        lastDrawRef.current = performance.now();
        const ps = psnrJS(out, imgJS.data);
        setMetrics({ step, psnr: ps, n: curNgRef.current, mode: 'js', worker: true, pool: stripes });
      } else {
        setMetrics((m) => ({ ...m, step, n: curNgRef.current, pool: stripes }));
      }

      if (step % addEvery === 0 && curNgRef.current < Ntotal) {
        await addGaussiansByErrorJS(Math.min(addChunk, Ntotal - curNgRef.current));
      }
      if (stepDelayMs > 0) await new Promise((r) => setTimeout(r, stepDelayMs));
    }

    // Final draw
    const out = await renderFrameViaPool(true); if (out) drawOnCanvasF32RGB(out, imgSize.w, imgSize.h);
  }

  async function addGaussiansByErrorJS(nNew) {
    const out = await renderFrameViaPool(false); if (!out) return;
    const { data: tgt, w: W, h: H } = imgJS; const err = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) err[i] = (Math.abs(out[i * 3 + 0] - tgt[i * 3 + 0]) + Math.abs(out[i * 3 + 1] - tgt[i * 3 + 1]) + Math.abs(out[i * 3 + 2] - tgt[i * 3 + 2])) / 3;
    const idxs = cpuTopKIndices(err, Math.min(nNew, W * H));

    const v = varsRef.current; if (!v) return;
    const NgOld = v.mu.length / 2; const NgNew = NgOld + nNew;
    const mu2 = new Float32Array(NgNew * 2); mu2.set(v.mu);
    const s2 = new Float32Array(NgNew * 2); s2.set(v.s_inv);
    const th2 = new Float32Array(NgNew); th2.set(v.theta);
    const c2 = new Float32Array(NgNew * 3); c2.set(v.color);

    for (let j = 0; j < idxs.length; j++) {
      const id = idxs[j]; const y = Math.floor(id / W), x = id % W; const base = y * W + x; const i = NgOld + j;
      mu2[i * 2 + 0] = x / (W - 1); mu2[i * 2 + 1] = y / (H - 1);
      s2[i * 2 + 0] = (W - 1) / 5; s2[i * 2 + 1] = (H - 1) / 5;
      th2[i] = Math.random() * Math.PI;
      c2[i * 3 + 0] = tgt[base * 3 + 0]; c2[i * 3 + 1] = tgt[base * 3 + 1]; c2[i * 3 + 2] = tgt[base * 3 + 2];
    }

    varsRef.current = { mu: mu2, s_inv: s2, theta: th2, color: c2 };
    curNgRef.current = NgNew;
    broadcastSetTiling(); // ensure next step rebuilds bins with new indices
  }

  function handleStop() { stopRef.current = true; setStatus("stopped"); }

  async function handleRenderOnce() { const out = await renderFrameViaPool(true); if (out) drawOnCanvasF32RGB(out, imgSize.w, imgSize.h); }

  async function handleSaveModel() {
    const H = imgSize.h, W = imgSize.w; if (!H || !W || !varsRef.current) return;
    const { mu, s_inv, theta, color } = varsRef.current; const N = mu.length / 2;
    const headerObj = { magic: "IGS1", H, W, C: 3, N }; const headerStr = JSON.stringify(headerObj) + "\n";
    const header = new Uint8Array(headerStr.split("").map((ch) => ch.charCodeAt(0) & 0xff));
    const payload = new Uint8Array([
      ...header,
      ...float16Quantize(mu),
      ...float16Quantize(s_inv),
      ...float16Quantize(theta),
      ...float16Quantize(color),
    ]);
    const blob = new Blob([payload], { type: "application/octet-stream" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "image_gs.igs"; a.click();
  }

  return (
    <div className="w-full min-h-screen p-6 flex flex-col gap-4">
      <h1 className="text-2xl font-bold">Image‑GS — 2D Gaussians (JS‑only, Worker Pool)</h1>
      <div className="text-sm text-gray-600">CSP‑safe • No TFJS/WebGL/WASM • Preview auto‑downscales ≤512px • Render/EM offloaded to a pool of Web Workers</div>

      <div className="flex flex-wrap gap-4 items-center">
        <input type="file" accept="image/*" onChange={handleFile} />
        <button className="px-3 py-2 rounded bg-black text-white" onClick={train} disabled={!imgJS || status === "training"}>
          {status === "training" ? "Training…" : "Train"}
        </button>
        <button className="px-3 py-2 rounded border" onClick={handleRenderOnce} disabled={!imgJS}>Render once</button>
        <button className="px-3 py-2 rounded border" onClick={handleStop}>Stop</button>
        <button className="px-3 py-2 rounded border" onClick={handleSaveModel} disabled={!imgJS || !varsRef.current}>Save .igs</button>
        <Diagnostics />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap gap-4 items-end">
            <label className="flex items-center gap-2">K
              <input type="number" value={K} min={1} max={64} onChange={(e) => setK(parseInt(e.target.value) || 1)} className="border p-1 w-24" />
            </label>
            <label className="flex items-center gap-2">Budget (N)
              <input type="number" value={budget} min={128} max={20000} onChange={(e) => setBudget(parseInt(e.target.value) || 128)} className="border p-1 w-28" />
            </label>
            <label className="flex items-center gap-2">λ_init
              <input type="number" step="0.05" value={lambdaInit} min={0} max={1} onChange={(e) => setLambdaInit(parseFloat(e.target.value) || 0)} className="border p-1 w-24" />
            </label>
            <label className="flex items-center gap-2">Steps
              <input type="number" value={steps} min={100} max={20000} onChange={(e) => setSteps(parseInt(e.target.value) || 100)} className="border p-1 w-28" />
            </label>
            <label className="flex items-center gap-2">Delay (ms)
              <input type="number" value={stepDelayMs} min={0} max={1000} onChange={(e) => setStepDelayMs(parseInt(e.target.value) || 0)} className="border p-1 w-28" />
            </label>
          </div>

          <div className="flex flex-wrap gap-4 items-end">
            <label className="flex items-center gap-2">Pool size
              <input type="number" value={poolSize} min={1} max={16} onChange={(e) => setPoolSize(clamp(parseInt(e.target.value) || 1, 1, 16))} className="border p-1 w-24" />
            </label>
            <label className="flex items-center gap-2">Tiling
              <input type="checkbox" checked={enableTiling} onChange={(e) => { setEnableTiling(e.target.checked); broadcastSetTiling(); }} />
            </label>
            <label className="flex items-center gap-2">Tile W
              <input type="number" value={tileW} min={8} max={128} onChange={(e) => { const v = parseInt(e.target.value) || 32; setTileW(v); broadcastSetTiling(); }} className="border p-1 w-24" />
            </label>
            <label className="flex items-center gap-2">Tile H
              <input type="number" value={tileH} min={8} max={128} onChange={(e) => { const v = parseInt(e.target.value) || 32; setTileH(v); broadcastSetTiling(); }} className="border p-1 w-24" />
            </label>
            <label className="flex items-center gap-2">Rebin every
              <input type="number" value={rebinEvery} min={0} max={500} onChange={(e) => setRebinEvery(parseInt(e.target.value) || 0)} className="border p-1 w-28" />
            </label>
          </div>

          <div className="flex flex-wrap gap-4 items-end">
            <label className="flex items-center gap-2">LR Color
              <input type="number" step="0.05" value={lrColor} min={0.05} max={1} onChange={(e) => setLrColor(parseFloat(e.target.value) || 0.4)} className="border p-1 w-24" />
            </label>
            <label className="flex items-center gap-2">LR μ
              <input type="number" step="0.05" value={lrMu} min={0.05} max={1} onChange={(e) => setLrMu(parseFloat(e.target.value) || 0.25)} className="border p-1 w-24" />
            </label>
            <label className="flex items-center gap-2">LR shape
              <input type="number" step="0.05" value={lrShape} min={0.05} max={1} onChange={(e) => setLrShape(parseFloat(e.target.value) || 0.25)} className="border p-1 w-24" />
            </label>
          </div>

          <div className="text-sm text-gray-600">Image: {imgSize.w}×{imgSize.h} • Gaussians: {curNgRef.current} • Step: {metrics.step} • Pool: {metrics.pool}</div>
          <div className="text-sm">PSNR: {Number.isFinite(metrics.psnr) ? metrics.psnr?.toFixed?.(2) : "—"} dB</div>

          <canvas ref={canvasRef} className="rounded-xl shadow border" />
        </div>
        <div className="prose max-w-none">
          <h2 className="font-semibold">Worker pool</h2>
          <p>This build runs the renderer/EM across a <b>pool</b> of workers (default ≈ your core count). The image is partitioned into row stripes; each worker returns partial accumulators and (when requested) its image stripe. The main thread reduces accumulators and composites the frame. UI still draws at a fixed ~10 Hz.</p>
          <h2 className="font-semibold mt-4">Rebin</h2>
          <p><b>Rebin</b> rebuilds each worker’s tile→Gaussian bins when μ/θ/s⁻¹ change, so per-pixel work remains local (O(K)). We keep it inside each worker for simplicity; if profiling shows duplication is hot, we can broadcast a compressed CSR index instead.</p>
          <h2 className="font-semibold mt-4">Preview size</h2>
          <p>Inputs are auto‑downscaled to ≤512px long side before training to keep compute predictable. We can expose this if you want to tune it.</p>
        </div>
      </div>
    </div>
  );
}
