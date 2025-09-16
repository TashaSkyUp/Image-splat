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
    throw new Error("Rasterization failed (drawImage). The file may be corrupt or blocked.");
  }
  let imgData;
  try {
    imgData = ctx.getImageData(0, 0, w, h);
  } catch (e) {
    throw new Error(
      "Reading pixels failed (getImageData). If this is an SVG with external refs, re-export as PNG/JPG."
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
      let gx = 0;
      let gy = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = clamp(x + dx, 0, w - 1);
          const yy = clamp(y + dy, 0, h - 1);
          const i = yy * w + xx;
          const r = rgb[i * 3 + 0];
          const g = rgb[i * 3 + 1];
          const b = rgb[i * 3 + 2];
          const gray = 0.299 * r + 0.587 * g + 0.114 * b;
          const kIdx = (dy + 1) * 3 + (dx + 1);
          gx += gray * kx[kIdx];
          gy += gray * ky[kIdx];
        }
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
    const ux = muA[i * 2 + 0];
    const uy = muA[i * 2 + 1];
    const sxInv = sInvA[i * 2 + 0];
    const syInv = sInvA[i * 2 + 1];
    const cx = ux * (W - 1);
    const cy = uy * (H - 1);
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

function samplePositionsAndColorsJS(imgF32, w, h, prob2D, N) {
  const prob = new Float32Array(prob2D);
  let s = 0;
  for (let i = 0; i < prob.length; i++) s += prob[i];
  s = s || 1;
  for (let i = 0; i < prob.length; i++) prob[i] /= s;
  const cdf = new Float32Array(prob.length);
  let acc = 0;
  for (let i = 0; i < prob.length; i++) {
    acc += prob[i];
    cdf[i] = acc;
  }
  const mu = new Float32Array(N * 2);
  const colors = new Float32Array(N * 3);
  for (let k = 0; k < N; k++) {
    const r = Math.random();
    let lo = 0;
    let hi = cdf.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] < r) lo = mid + 1;
      else hi = mid;
    }
    const y = Math.floor(lo / w);
    const x = lo % w;
    const idx = y * w + x;
    mu[k * 2 + 0] = x / (w - 1);
    mu[k * 2 + 1] = y / (h - 1);
    colors[k * 3 + 0] = imgF32[idx * 3 + 0];
    colors[k * 3 + 1] = imgF32[idx * 3 + 1];
    colors[k * 3 + 2] = imgF32[idx * 3 + 2];
  }
  return { mu, colors };
}

function psnrJS(a, b) {
  let mse = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    mse += d * d;
  }
  mse /= a.length || 1;
  if (mse <= 0) return Infinity;
  return 10 * Math.log10(1 / (mse + EPS));
}

function float16Quantize(x) {
  const f32 = x instanceof Float32Array ? x : new Float32Array(x);
  const buf = new ArrayBuffer(f32.length * 2);
  const dv = new DataView(buf);
  for (let i = 0; i < f32.length; i++) dv.setUint16(i * 2, float32ToFloat16(f32[i]), true);
  return new Uint8Array(buf);
}

function float32ToFloat16(val) {
  const f32 = new Float32Array(1);
  const i32 = new Int32Array(f32.buffer);
  f32[0] = val;
  const x = i32[0];
  const bits = (x >> 16) & 0x8000;
  const m = (x >> 12) & 0x07ff;
  const e = (x >> 23) & 0xff;
  if (e < 103) return bits;
  if (e > 142) return bits | 0x7c00;
  return bits | ((e - 112) << 10) | (m >> 1);
}
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

  let IMG = null;
  let W = 0, H = 0;
  let tileW = 32, tileH = 32;
  let enableTiling = true;
  let bins = null;

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
            const topIdx = [];
            const topVal = [];
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
                let j = topIdx.length;
                topIdx.push(i);
                topVal.push(g);
                while (j > 0 && topVal[j] > topVal[j - 1]) {
                  const ti = topIdx[j - 1];
                  topIdx[j - 1] = topIdx[j];
                  topIdx[j] = ti;
                  const tv = topVal[j - 1];
                  topVal[j - 1] = topVal[j];
                  topVal[j] = tv;
                  j--;
                }
              } else if (g > topVal[topVal.length - 1]) {
                topIdx[topIdx.length - 1] = i;
                topVal[topVal.length - 1] = g;
                let j = topIdx.length - 1;
                while (j > 0 && topVal[j] > topVal[j - 1]) {
                  const ti = topIdx[j - 1];
                  topIdx[j - 1] = topIdx[j];
                  topIdx[j] = ti;
                  const tv = topVal[j - 1];
                  topVal[j - 1] = topVal[j];
                  topVal[j] = tv;
                  j--;
                }
              }
            };
            if (list && list.length) {
              for (let ii = 0; ii < list.length; ii++) scan(list[ii]);
            } else {
              for (let i = 0; i < N; i++) scan(i);
            }
            let wsum = EPS;
            for (let k = 0; k < topVal.length; k++) wsum += topVal[k];
            const baseGlobal = (y * W + x) * 3;
            let r = 0, g = 0, b = 0;
            for (let k = 0; k < topIdx.length; k++) {
              const wi = topVal[k] / wsum;
              const gi = topIdx[k];
              r += wi * color[gi * 3 + 0];
              g += wi * color[gi * 3 + 1];
              b += wi * color[gi * 3 + 2];
              accW[gi] += wi;
              accTW[gi * 3 + 0] += wi * IMG[baseGlobal + 0];
              accTW[gi * 3 + 1] += wi * IMG[baseGlobal + 1];
              accTW[gi * 3 + 2] += wi * IMG[baseGlobal + 2];
              const cxn = cx, cyn = cy;
              accX[gi] += wi * cxn;
              accY[gi] += wi * cyn;
              accXX[gi] += wi * cxn * cxn;
              accYY[gi] += wi * cyn * cyn;
              accXY[gi] += wi * cxn * cyn;
            }
            if (out) {
              const row = y - y0Stripe;
              const baseLocal = (row * W + x) * 3;
              out[baseLocal + 0] = r;
              out[baseLocal + 1] = g;
              out[baseLocal + 2] = b;
            }
          }
        }
      }
    }

    const msg = { type: 'stepResult', reqId, accW, accTW, accX, accY, accXX, accYY, accXY, y0: y0Stripe, y1: y1Stripe };
    const transfers = [accW.buffer, accTW.buffer, accX.buffer, accY.buffer, accXX.buffer, accYY.buffer, accXY.buffer];
    if (out) {
      msg.out = out;
      transfers.push(out.buffer);
    }
    postMessage(msg, transfers);
  }

  onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'init') {
      IMG = m.img;
      W = m.W;
      H = m.H;
      tileW = m.tileW;
      tileH = m.tileH;
      enableTiling = !!m.enableTiling;
      bins = null;
      postMessage({ type: 'inited' });
    } else if (m.type === 'step') {
      computeStepStripe(m.mu, m.s_inv, m.theta, m.color, m.K, !!m.wantImage, !!m.doRebin, m.y0, m.y1, m.reqId);
    } else if (m.type === 'setTiling') {
      tileW = m.tileW;
      tileH = m.tileH;
      enableTiling = !!m.enableTiling;
      bins = null;
    }
  };
`;

function makeWorker() {
  const blob = new Blob([WORKER_SRC], { type: "text/javascript" });
  const url = URL.createObjectURL(blob);
  const worker = new Worker(url, { type: "classic" });
  return { worker, url };
}
const defaultPool = Math.max(2, Math.min(navigator.hardwareConcurrency || 4, 8));

const state = {
  imgJS: null,
  imgSize: { w: 0, h: 0 },
  status: "idle",
  metrics: { step: 0, psnr: null, n: 0, mode: "js", worker: true, pool: 1 },
  K: 6,
  budget: 2000,
  lambdaInit: 0.3,
  steps: 1500,
  stepDelayMs: 0,
  lrColor: 0.4,
  lrMu: 0.25,
  lrShape: 0.25,
  enableTiling: true,
  tileW: 32,
  tileH: 32,
  rebinEvery: 50,
  poolSize: defaultPool,
};

let vars = null;
let curNg = 0;
let stopFlag = false;
let lastDraw = 0;
let canvasEl = null;
let canvasCtx = null;
let diagStatus = "idle";

const pool = { workers: [], urls: [], ready: [], resolvers: [], nextReqId: 1 };

const ui = {
  trainButton: null,
  renderButton: null,
  stopButton: null,
  saveButton: null,
  fileInput: null,
  diagStatus: null,
  metricsInfo: null,
  psnrInfo: null,
  controls: {},
};

function updateUI() {
  if (ui.trainButton) {
    ui.trainButton.disabled = !state.imgJS || state.status === "training";
    ui.trainButton.textContent = state.status === "training" ? "Training…" : "Train";
  }
  if (ui.renderButton) ui.renderButton.disabled = !state.imgJS;
  if (ui.saveButton) ui.saveButton.disabled = !state.imgJS || !vars;
  if (ui.metricsInfo) {
    ui.metricsInfo.textContent = `Image: ${state.imgSize.w}×${state.imgSize.h} • Gaussians: ${curNg} • Step: ${state.metrics.step} • Pool: ${state.metrics.pool}`;
  }
  if (ui.psnrInfo) {
    const v = Number.isFinite(state.metrics.psnr) ? state.metrics.psnr?.toFixed?.(2) : "—";
    ui.psnrInfo.textContent = `PSNR: ${v} dB`;
  }
  if (ui.diagStatus) ui.diagStatus.textContent = diagStatus;
  const c = ui.controls;
  if (c.poolSize) c.poolSize.value = String(state.poolSize);
  if (c.K) c.K.value = String(state.K);
  if (c.budget) c.budget.value = String(state.budget);
  if (c.lambdaInit) c.lambdaInit.value = String(state.lambdaInit);
  if (c.steps) c.steps.value = String(state.steps);
  if (c.stepDelayMs) c.stepDelayMs.value = String(state.stepDelayMs);
  if (c.enableTiling) c.enableTiling.checked = !!state.enableTiling;
  if (c.tileW) c.tileW.value = String(state.tileW);
  if (c.tileH) c.tileH.value = String(state.tileH);
  if (c.rebinEvery) c.rebinEvery.value = String(state.rebinEvery);
  if (c.lrColor) c.lrColor.value = String(state.lrColor);
  if (c.lrMu) c.lrMu.value = String(state.lrMu);
  if (c.lrShape) c.lrShape.value = String(state.lrShape);
}

function destroyPool() {
  for (const w of pool.workers) {
    try {
      w.terminate();
    } catch (e) {}
  }
  for (const u of pool.urls) {
    try {
      URL.revokeObjectURL(u);
    } catch (e) {}
  }
  pool.workers = [];
  pool.urls = [];
  pool.ready = [];
  pool.resolvers = [];
  pool.nextReqId = 1;
}

function rebuildPool(size, preserveImage = false) {
  const oldImg = preserveImage ? state.imgJS : null;
  destroyPool();
  const workers = [];
  const urls = [];
  const ready = [];
  const resolvers = [];
  for (let i = 0; i < size; i++) {
    const { worker, url } = makeWorker();
    workers.push(worker);
    urls.push(url);
    ready.push(false);
    resolvers.push(new Map());
    worker.onmessage = (ev) => {
      if (ev.data?.type === "inited") {
        ready[i] = true;
      } else if (ev.data?.type === "stepResult") {
        const resMap = resolvers[i];
        const rid = ev.data.reqId;
        if (rid && resMap.has(rid)) {
          const fn = resMap.get(rid);
          resMap.delete(rid);
          fn(ev.data);
        }
      }
    };
  }
  pool.workers = workers;
  pool.urls = urls;
  pool.ready = ready;
  pool.resolvers = resolvers;
  pool.nextReqId = 1;

  if (preserveImage && oldImg) broadcastInitImage(oldImg);
}

function broadcastInitImage(js) {
  if (!pool.workers.length) return;
  const { w, h, data } = js;
  state.imgSize = { w, h };
  for (let i = 0; i < pool.workers.length; i++) {
    const imgCopy = new Float32Array(data);
    pool.ready[i] = false;
    pool.workers[i].postMessage(
      { type: "init", img: imgCopy, W: w, H: h, tileW: state.tileW, tileH: state.tileH, enableTiling: state.enableTiling },
      [imgCopy.buffer]
    );
  }
  updateUI();
}

function broadcastSetTiling() {
  for (let i = 0; i < pool.workers.length; i++) {
    pool.workers[i].postMessage({ type: "setTiling", tileW: state.tileW, tileH: state.tileH, enableTiling: state.enableTiling });
  }
}

async function waitAllReady() {
  const deadline = performance.now() + 5000;
  while (true) {
    if (pool.ready.every(Boolean)) return;
    if (performance.now() > deadline) throw new Error("Worker pool init timeout");
    await new Promise((r) => setTimeout(r, 10));
  }
}

function postStepToWorker(i, payload) {
  const rid = pool.nextReqId++;
  return new Promise((resolve) => {
    pool.resolvers[i].set(rid, resolve);
    pool.workers[i].postMessage({ type: "step", reqId: rid, ...payload });
  });
}

function drawOnCanvasF32RGB(arr, w, h) {
  if (!canvasCtx || !canvasEl) return;
  canvasEl.width = w;
  canvasEl.height = h;
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, j = 0; i < w * h; i++) {
    out[j++] = clamp(Math.round(arr[i * 3 + 0] * 255), 0, 255);
    out[j++] = clamp(Math.round(arr[i * 3 + 1] * 255), 0, 255);
    out[j++] = clamp(Math.round(arr[i * 3 + 2] * 255), 0, 255);
    out[j++] = 255;
  }
  canvasCtx.putImageData(new ImageData(out, w, h), 0, 0);
}

async function handleFile(ev) {
  try {
    const file = ev.target.files?.[0];
    if (!file) return;
    if (!(file.type && file.type.startsWith("image/"))) {
      throw new Error(`Selected file is not an image (type: ${file.type || "unknown"})`);
    }
    const { src, revoke } = await fileToImageSource(file);
    let js;
    try {
      js = sourceToF32RGB(src, 512);
    } finally {
      try {
        revoke?.();
      } catch (e) {}
    }
    const localData = new Float32Array(js.data);
    state.imgJS = { data: localData, w: js.w, h: js.h };
    state.imgSize = { w: js.w, h: js.h };
    drawOnCanvasF32RGB(localData, js.w, js.h);
    disposeVars();
    broadcastInitImage({ data: localData, w: js.w, h: js.h });
    state.status = "loaded";
    updateUI();
  } catch (err) {
    console.error("handleFile failed", err);
    alert(`Failed to load image.\n\nReason: ${err?.message || err}`);
  }
}

function disposeVars() {
  vars = null;
  curNg = 0;
  updateUI();
}

async function initParamsJS() {
  if (!state.imgJS) return;
  const { w, h, data } = state.imgJS;
  const grad = sobelMagJS(data, w, h);
  let sum = 0;
  for (let i = 0; i < grad.length; i++) sum += grad[i];
  const uniform = 1 / (w * h);
  const probs = new Float32Array(w * h);
  const lmb = state.lambdaInit;
  for (let i = 0; i < w * h; i++) probs[i] = (1 - lmb) * (grad[i] / (sum || 1)) + lmb * uniform;

  const Ng0 = Math.max(1, Math.floor(state.budget / 2));
  const { mu, colors } = samplePositionsAndColorsJS(data, w, h, probs, Ng0);
  const s_inv = new Float32Array(Ng0 * 2);
  for (let i = 0; i < Ng0; i++) {
    s_inv[i * 2 + 0] = (w - 1) / 5;
    s_inv[i * 2 + 1] = (h - 1) / 5;
  }
  const theta = new Float32Array(Ng0);
  for (let i = 0; i < Ng0; i++) theta[i] = Math.random() * Math.PI;

  vars = { mu, s_inv, theta, color: colors };
  curNg = Ng0;
  broadcastSetTiling();
  updateUI();
}

function paramsForStep() {
  if (!vars) return null;
  return {
    mu: new Float32Array(vars.mu),
    s_inv: new Float32Array(vars.s_inv),
    theta: new Float32Array(vars.theta),
    color: new Float32Array(vars.color),
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
    const aW = p.accW;
    const aT = p.accTW;
    const aX = p.accX;
    const aY = p.accY;
    const aXX = p.accXX;
    const aYY = p.accYY;
    const aXY = p.accXY;
    for (let i = 0; i < N; i++) accW[i] += aW[i];
    for (let i = 0; i < N * 3; i++) accTW[i] += aT[i];
    for (let i = 0; i < N; i++) {
      accX[i] += aX[i];
      accY[i] += aY[i];
      accXX[i] += aXX[i];
      accYY[i] += aYY[i];
      accXY[i] += aXY[i];
    }
  }
  return { accW, accTW, accX, accY, accXX, accYY, accXY };
}

function composeOut(parts, W, H) {
  const out = new Float32Array(W * H * 3);
  for (const p of parts) {
    if (!p.out) continue;
    const y0 = p.y0;
    const stripe = p.out;
    out.set(stripe, y0 * W * 3);
  }
  return out;
}

async function train() {
  if (!state.imgJS) return;
  state.status = "initializing";
  stopFlag = false;
  updateUI();
  await initParamsJS();
  try {
    await waitAllReady();
  } catch (e) {
    console.error(e);
  }
  state.status = "training";
  updateUI();
  await trainJS_pool();
  state.status = "done";
  updateUI();
}
function updateParamsFromAcc(N, pkg) {
  if (!vars) return;
  for (let i = 0; i < N; i++) {
    const wsum = pkg.accW[i] + 1e-6;
    const tR = pkg.accTW[i * 3 + 0] / wsum;
    const tG = pkg.accTW[i * 3 + 1] / wsum;
    const tB = pkg.accTW[i * 3 + 2] / wsum;
    vars.color[i * 3 + 0] = (1 - state.lrColor) * vars.color[i * 3 + 0] + state.lrColor * tR;
    vars.color[i * 3 + 1] = (1 - state.lrColor) * vars.color[i * 3 + 1] + state.lrColor * tG;
    vars.color[i * 3 + 2] = (1 - state.lrColor) * vars.color[i * 3 + 2] + state.lrColor * tB;
  }
  const w = state.imgSize.w;
  const h = state.imgSize.h;
  const stdMinX = 1 / Math.max(w - 1, 1);
  const stdMinY = 1 / Math.max(h - 1, 1);
  for (let i = 0; i < N; i++) {
    const wsum = pkg.accW[i];
    if (wsum < 1e-6) continue;
    const mx = pkg.accX[i] / wsum;
    const my = pkg.accY[i] / wsum;
    const muX = vars.mu[i * 2 + 0];
    const muY = vars.mu[i * 2 + 1];
    vars.mu[i * 2 + 0] = (1 - state.lrMu) * muX + state.lrMu * mx;
    vars.mu[i * 2 + 1] = (1 - state.lrMu) * muY + state.lrMu * my;
    const cxx = Math.max(0, pkg.accXX[i] / wsum - mx * mx);
    const cyy = Math.max(0, pkg.accYY[i] / wsum - my * my);
    const cxy = pkg.accXY[i] / wsum - mx * my;
    const tr = cxx + cyy;
    const det = cxx * cyy - cxy * cxy;
    const disc = Math.max(0, tr * tr - 4 * det);
    const s = Math.sqrt(disc);
    const l1 = 0.5 * (tr + s);
    const l2 = 0.5 * (tr - s);
    let vx;
    let vy;
    if (Math.abs(cxy) > 1e-12) {
      vx = l1 - cyy;
      vy = cxy;
    } else {
      if (cxx >= cyy) {
        vx = 1;
        vy = 0;
      } else {
        vx = 0;
        vy = 1;
      }
    }
    const n = Math.hypot(vx, vy) || 1;
    vx /= n;
    vy /= n;
    let th0 = vars.theta[i] % Math.PI;
    if (th0 < 0) th0 += Math.PI;
    let thT = Math.atan2(vy, vx);
    thT %= Math.PI;
    if (thT < 0) thT += Math.PI;
    if (Math.abs(thT - th0) > Math.PI / 2) {
      if (thT > th0) thT -= Math.PI;
      else thT += Math.PI;
    }
    const thNew = th0 + (thT - th0) * state.lrShape;
    vars.theta[i] = ((thNew % Math.PI) + Math.PI) % Math.PI;
    const std1 = Math.sqrt(Math.max(l1, 0));
    const std2 = Math.sqrt(Math.max(l2, 0));
    const sxInvTarget = 1 / Math.max(std1, Math.min(stdMinX, stdMinY));
    const syInvTarget = 1 / Math.max(std2, Math.min(stdMinX, stdMinY));
    const sx0 = vars.s_inv[i * 2 + 0];
    const sy0 = vars.s_inv[i * 2 + 1];
    vars.s_inv[i * 2 + 0] = clamp((1 - state.lrShape) * sx0 + state.lrShape * sxInvTarget, 0.1, (w - 1) * 4);
    vars.s_inv[i * 2 + 1] = clamp((1 - state.lrShape) * sy0 + state.lrShape * syInvTarget, 0.1, (h - 1) * 4);
  }
}

async function renderFrameViaPool(doRebin = false) {
  if (!vars || !pool.workers.length) return null;
  await waitAllReady();
  const { w: W, h: H } = state.imgSize;
  const params = paramsForStep();
  const stripes = Math.min(pool.workers.length, H);
  const promises = [];
  for (let i = 0; i < stripes; i++) {
    const y0 = Math.floor((H * i) / stripes);
    const y1 = Math.min(H - 1, Math.floor((H * (i + 1)) / stripes) - 1);
    const wantImage = true;
    const payload = { K: state.K, doRebin, wantImage, y0, y1, ...params };
    promises.push(postStepToWorker(i, payload));
  }
  const results = await Promise.all(promises);
  return composeOut(results, W, H);
}

async function trainJS_pool() {
  if (!pool.workers.length) return;
  const Ntotal = state.budget;
  const addEvery = 400;
  const addChunk = Math.max(1, Math.floor(Ntotal / 8));
  const { w: W, h: H } = state.imgSize;

  for (let step = 1; step <= state.steps; step++) {
    if (stopFlag) break;
    if (!vars) break;
    const N = vars.mu.length / 2;
    const params = paramsForStep();
    const stripes = Math.min(pool.workers.length, H);
    const doRebin = state.enableTiling && (step === 1 || (state.rebinEvery > 0 && step % state.rebinEvery === 0));
    const wantImage = performance.now() - lastDraw >= 100;

    const promises = [];
    for (let i = 0; i < stripes; i++) {
      const y0 = Math.floor((H * i) / stripes);
      const y1 = Math.min(H - 1, Math.floor((H * (i + 1)) / stripes) - 1);
      const payload = { K: state.K, doRebin, wantImage, y0, y1, ...params };
      promises.push(postStepToWorker(i, payload));
    }
    const results = await Promise.all(promises);
    const merged = mergeAccumulators(results, N);
    updateParamsFromAcc(N, merged);

    if (wantImage) {
      const out = composeOut(results, W, H);
      drawOnCanvasF32RGB(out, W, H);
      lastDraw = performance.now();
      const ps = psnrJS(out, state.imgJS.data);
      state.metrics = { step, psnr: ps, n: curNg, mode: "js", worker: true, pool: stripes };
      updateUI();
    } else {
      state.metrics = { ...state.metrics, step, n: curNg, pool: stripes };
      updateUI();
    }

    if (step % addEvery === 0 && curNg < Ntotal) {
      await addGaussiansByErrorJS(Math.min(addChunk, Ntotal - curNg));
    }
    if (state.stepDelayMs > 0) await new Promise((r) => setTimeout(r, state.stepDelayMs));
  }

  const out = await renderFrameViaPool(true);
  if (out) drawOnCanvasF32RGB(out, state.imgSize.w, state.imgSize.h);
}

async function addGaussiansByErrorJS(nNew) {
  const out = await renderFrameViaPool(false);
  if (!out) return;
  const { data: tgt, w: W, h: H } = state.imgJS;
  const err = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    err[i] =
      (Math.abs(out[i * 3 + 0] - tgt[i * 3 + 0]) +
        Math.abs(out[i * 3 + 1] - tgt[i * 3 + 1]) +
        Math.abs(out[i * 3 + 2] - tgt[i * 3 + 2])) /
      3;
  }
  const idxs = cpuTopKIndices(err, Math.min(nNew, W * H));

  if (!vars) return;
  const NgOld = vars.mu.length / 2;
  const NgNew = NgOld + nNew;
  const mu2 = new Float32Array(NgNew * 2);
  mu2.set(vars.mu);
  const s2 = new Float32Array(NgNew * 2);
  s2.set(vars.s_inv);
  const th2 = new Float32Array(NgNew);
  th2.set(vars.theta);
  const c2 = new Float32Array(NgNew * 3);
  c2.set(vars.color);

  for (let j = 0; j < idxs.length; j++) {
    const id = idxs[j];
    const y = Math.floor(id / W);
    const x = id % W;
    const base = y * W + x;
    const i = NgOld + j;
    mu2[i * 2 + 0] = x / (W - 1);
    mu2[i * 2 + 1] = y / (H - 1);
    s2[i * 2 + 0] = (W - 1) / 5;
    s2[i * 2 + 1] = (H - 1) / 5;
    th2[i] = Math.random() * Math.PI;
    c2[i * 3 + 0] = tgt[base * 3 + 0];
    c2[i * 3 + 1] = tgt[base * 3 + 1];
    c2[i * 3 + 2] = tgt[base * 3 + 2];
  }

  vars = { mu: mu2, s_inv: s2, theta: th2, color: c2 };
  curNg = NgNew;
  broadcastSetTiling();
  updateUI();
}

function handleStop() {
  stopFlag = true;
  state.status = "stopped";
  updateUI();
}

async function handleRenderOnce() {
  const out = await renderFrameViaPool(true);
  if (out) drawOnCanvasF32RGB(out, state.imgSize.w, state.imgSize.h);
}

async function handleSaveModel() {
  const H = state.imgSize.h;
  const W = state.imgSize.w;
  if (!H || !W || !vars) return;
  const { mu, s_inv, theta, color } = vars;
  const N = mu.length / 2;
  const headerObj = { magic: "IGS1", H, W, C: 3, N };
  const headerStr = JSON.stringify(headerObj) + "\n";
  const header = new Uint8Array(headerStr.split("").map((ch) => ch.charCodeAt(0) & 0xff));
  const payload = new Uint8Array([
    ...header,
    ...float16Quantize(mu),
    ...float16Quantize(s_inv),
    ...float16Quantize(theta),
    ...float16Quantize(color),
  ]);
  const blob = new Blob([payload], { type: "application/octet-stream" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "image_gs.igs";
  a.click();
}
function runDiagnostics() {
  const start = async () => {
    try {
      diagStatus = "running";
      updateUI();
      const { worker, url } = makeWorker();
      const W = 8;
      const H = 8;
      const img = new Float32Array(W * H * 3);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = (y * W + x) * 3;
          img[i] = x / (W - 1);
          img[i + 1] = y / (H - 1);
          img[i + 2] = 0.5;
        }
      }
      const waitType = (type) =>
        new Promise((res) => {
          const handler = (ev) => {
            if (ev.data?.type === type) {
              worker.removeEventListener("message", handler);
              res(ev.data);
            }
          };
          worker.addEventListener("message", handler);
        });
      worker.postMessage({ type: "init", img, W, H, tileW: 4, tileH: 4, enableTiling: true }, [img.buffer]);
      await waitType("inited");
      const mu = new Float32Array([0.25, 0.25, 0.75, 0.75]);
      const s_inv = new Float32Array([W - 1, H - 1, W - 1, H - 1]);
      const theta = new Float32Array([0, 0]);
      const color = new Float32Array([1, 0, 0, 0, 1, 0]);
      const reqId = 1234;
      const got = new Promise((res) => {
        const handler = (ev) => {
          const d = ev.data;
          if (d?.type === "stepResult" && d.reqId === reqId) {
            worker.removeEventListener("message", handler);
            res(d);
          }
        };
        worker.addEventListener("message", handler);
      });
      worker.postMessage({
        type: "step",
        reqId,
        mu,
        s_inv,
        theta,
        color,
        K: 2,
        wantImage: true,
        doRebin: true,
        y0: 0,
        y1: H - 1,
      });
      const d = await got;
      URL.revokeObjectURL(url);
      worker.terminate();
      if (!d.out || d.out.length !== W * H * 3) throw new Error("render size mismatch");
      diagStatus = "ok";
      updateUI();
    } catch (e) {
      diagStatus = "fail: " + (e?.message || e);
      updateUI();
    }
  };
  start();
}

function el(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function createNumberControl(labelText, key, options, onChange) {
  const label = el("label", "flex items-center gap-2");
  const span = document.createElement("span");
  span.textContent = labelText;
  const input = document.createElement("input");
  input.type = "number";
  if (options?.min !== undefined) input.min = String(options.min);
  if (options?.max !== undefined) input.max = String(options.max);
  if (options?.step !== undefined) input.step = String(options.step);
  input.value = String(state[key]);
  input.className = "border p-1 " + (options?.width || "w-24");
  input.addEventListener("change", (e) => {
    onChange(e.target.value, input);
    updateUI();
  });
  label.append(span, input);
  ui.controls[key] = input;
  return label;
}

function createCheckboxControl(labelText, key, onChange) {
  const label = el("label", "flex items-center gap-2");
  const span = document.createElement("span");
  span.textContent = labelText;
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = !!state[key];
  input.addEventListener("change", (e) => {
    onChange(e.target.checked);
    updateUI();
  });
  label.append(span, input);
  ui.controls[key] = input;
  return label;
}

function initUI() {
  const root = document.getElementById("app");
  const container = el("div", "w-full min-h-screen p-6 flex flex-col gap-4");
  root.appendChild(container);

  container.appendChild(el("h1", "text-2xl font-bold", "Image-GS — 2D Gaussians (JS-only, Worker Pool)"));
  container.appendChild(
    el(
      "div",
      "text-sm text-gray-600",
      "CSP-safe • No TFJS/WebGL/WASM • Preview auto-downscales ≤512px • Render/EM offloaded to a pool of Web Workers"
    )
  );

  const controlsRow = el("div", "flex flex-wrap gap-4 items-center");
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.addEventListener("change", handleFile);
  controlsRow.appendChild(fileInput);
  ui.fileInput = fileInput;

  const trainButton = el("button", "px-3 py-2 rounded bg-black text-white", "Train");
  trainButton.addEventListener("click", train);
  controlsRow.appendChild(trainButton);
  ui.trainButton = trainButton;

  const renderButton = el("button", "px-3 py-2 rounded border", "Render once");
  renderButton.addEventListener("click", handleRenderOnce);
  controlsRow.appendChild(renderButton);
  ui.renderButton = renderButton;

  const stopButton = el("button", "px-3 py-2 rounded border", "Stop");
  stopButton.addEventListener("click", handleStop);
  controlsRow.appendChild(stopButton);
  ui.stopButton = stopButton;

  const saveButton = el("button", "px-3 py-2 rounded border", "Save .igs");
  saveButton.addEventListener("click", handleSaveModel);
  controlsRow.appendChild(saveButton);
  ui.saveButton = saveButton;

  const diagContainer = el("div", "ml-2 inline-flex items-center gap-2");
  const diagButton = el("button", "px-3 py-2 rounded border", "Diagnostics");
  diagButton.addEventListener("click", runDiagnostics);
  const diagSpan = el("span", "text-sm text-gray-600", diagStatus);
  diagContainer.append(diagButton, diagSpan);
  controlsRow.appendChild(diagContainer);
  ui.diagStatus = diagSpan;

  container.appendChild(controlsRow);

  const grid = el("div", "grid grid-cols-1 md:grid-cols-2 gap-6");
  container.appendChild(grid);

  const leftCol = el("div", "flex flex-col gap-2");
  grid.appendChild(leftCol);

  const group1 = el("div", "flex flex-wrap gap-4 items-end");
  group1.appendChild(
    createNumberControl("K", "K", { min: 1, max: 64, width: "w-24" }, (value, input) => {
      const parsed = parseInt(value, 10);
      state.K = Number.isFinite(parsed) ? Math.max(1, Math.min(64, parsed)) : 1;
      input.value = String(state.K);
    })
  );
  group1.appendChild(
    createNumberControl("Budget (N)", "budget", { min: 128, max: 20000, width: "w-28" }, (value, input) => {
      const parsed = parseInt(value, 10);
      state.budget = Number.isFinite(parsed) ? clamp(parsed, 128, 20000) : 128;
      input.value = String(state.budget);
    })
  );
  group1.appendChild(
    createNumberControl("λ_init", "lambdaInit", { min: 0, max: 1, step: 0.05, width: "w-24" }, (value, input) => {
      const parsed = parseFloat(value);
      state.lambdaInit = Number.isFinite(parsed) ? clamp(parsed, 0, 1) : 0;
      input.value = String(state.lambdaInit);
    })
  );
  group1.appendChild(
    createNumberControl("Steps", "steps", { min: 100, max: 20000, width: "w-28" }, (value, input) => {
      const parsed = parseInt(value, 10);
      state.steps = Number.isFinite(parsed) ? clamp(parsed, 100, 20000) : 100;
      input.value = String(state.steps);
    })
  );
  group1.appendChild(
    createNumberControl("Delay (ms)", "stepDelayMs", { min: 0, max: 1000, width: "w-28" }, (value, input) => {
      const parsed = parseInt(value, 10);
      state.stepDelayMs = Number.isFinite(parsed) ? clamp(parsed, 0, 1000) : 0;
      input.value = String(state.stepDelayMs);
    })
  );
  leftCol.appendChild(group1);

  const group2 = el("div", "flex flex-wrap gap-4 items-end");
  group2.appendChild(
    createNumberControl("Pool size", "poolSize", { min: 1, max: 16, width: "w-24" }, (value, input) => {
      const parsed = parseInt(value, 10);
      state.poolSize = Number.isFinite(parsed) ? clamp(parsed, 1, 16) : 1;
      input.value = String(state.poolSize);
      rebuildPool(state.poolSize, true);
    })
  );
  group2.appendChild(
    createCheckboxControl("Tiling", "enableTiling", (checked) => {
      state.enableTiling = !!checked;
      broadcastSetTiling();
    })
  );
  group2.appendChild(
    createNumberControl("Tile W", "tileW", { min: 8, max: 128, width: "w-24" }, (value, input) => {
      const parsed = parseInt(value, 10);
      state.tileW = Number.isFinite(parsed) ? clamp(parsed, 8, 128) : 32;
      input.value = String(state.tileW);
      broadcastSetTiling();
    })
  );
  group2.appendChild(
    createNumberControl("Tile H", "tileH", { min: 8, max: 128, width: "w-24" }, (value, input) => {
      const parsed = parseInt(value, 10);
      state.tileH = Number.isFinite(parsed) ? clamp(parsed, 8, 128) : 32;
      input.value = String(state.tileH);
      broadcastSetTiling();
    })
  );
  group2.appendChild(
    createNumberControl("Rebin every", "rebinEvery", { min: 0, max: 500, width: "w-28" }, (value, input) => {
      const parsed = parseInt(value, 10);
      state.rebinEvery = Number.isFinite(parsed) ? clamp(parsed, 0, 500) : 0;
      input.value = String(state.rebinEvery);
    })
  );
  leftCol.appendChild(group2);
  const group3 = el("div", "flex flex-wrap gap-4 items-end");
  group3.appendChild(
    createNumberControl("LR Color", "lrColor", { min: 0.05, max: 1, step: 0.05, width: "w-24" }, (value, input) => {
      const parsed = parseFloat(value);
      state.lrColor = Number.isFinite(parsed) ? clamp(parsed, 0.05, 1) : 0.4;
      input.value = String(state.lrColor);
    })
  );
  group3.appendChild(
    createNumberControl("LR μ", "lrMu", { min: 0.05, max: 1, step: 0.05, width: "w-24" }, (value, input) => {
      const parsed = parseFloat(value);
      state.lrMu = Number.isFinite(parsed) ? clamp(parsed, 0.05, 1) : 0.25;
      input.value = String(state.lrMu);
    })
  );
  group3.appendChild(
    createNumberControl("LR shape", "lrShape", { min: 0.05, max: 1, step: 0.05, width: "w-24" }, (value, input) => {
      const parsed = parseFloat(value);
      state.lrShape = Number.isFinite(parsed) ? clamp(parsed, 0.05, 1) : 0.25;
      input.value = String(state.lrShape);
    })
  );
  leftCol.appendChild(group3);

  const metricsInfo = el("div", "text-sm text-gray-600", "Image: 0×0 • Gaussians: 0 • Step: 0 • Pool: 1");
  leftCol.appendChild(metricsInfo);
  ui.metricsInfo = metricsInfo;

  const psnrInfo = el("div", "text-sm", "PSNR: — dB");
  leftCol.appendChild(psnrInfo);
  ui.psnrInfo = psnrInfo;

  const canvas = document.createElement("canvas");
  canvas.className = "rounded-xl shadow border";
  leftCol.appendChild(canvas);
  canvasEl = canvas;
  canvasCtx = canvas.getContext("2d");

  const rightCol = el("div", "prose max-w-none");
  grid.appendChild(rightCol);
  rightCol.appendChild(el("h2", "font-semibold", "Worker pool"));
  rightCol.appendChild(
    el(
      "p",
      null,
      "This build runs the renderer/EM across a pool of workers (default ≈ your core count). The image is partitioned into row stripes; each worker returns partial accumulators and (when requested) its image stripe. The main thread reduces accumulators and composites the frame. UI still draws at a fixed ~10 Hz."
    )
  );
  rightCol.appendChild(el("h2", "font-semibold mt-4", "Rebin"));
  rightCol.appendChild(
    el(
      "p",
      null,
      "Rebin rebuilds each worker’s tile→Gaussian bins when μ/θ/s⁻¹ change, so per-pixel work remains local (O(K)). We keep it inside each worker for simplicity; if profiling shows duplication is hot, we can broadcast a compressed CSR index instead."
    )
  );
  rightCol.appendChild(el("h2", "font-semibold mt-4", "Preview size"));
  rightCol.appendChild(
    el(
      "p",
      null,
      "Inputs are auto-downscaled to ≤512px long side before training to keep compute predictable. We can expose this if you want to tune it."
    )
  );
}

function initApp() {
  initUI();
  rebuildPool(state.poolSize);
  window.addEventListener("beforeunload", destroyPool);
  updateUI();
}

window.addEventListener("DOMContentLoaded", initApp);
