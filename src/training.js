/**
 * Training and worker management for the Image-GS demo.
 *
 * This module owns the EM loop, worker pool orchestration, and model
 * serialization helpers so the UI layer in `main.js` can stay focused on
 * file handling and DOM updates. It is intentionally DOM-free; callers
 * register callbacks for UI updates and frame presentation via
 * `configureTrainingCallbacks`.
 */

const EPS = 1e-6;
const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

const COLOR_SIGMA = 0.25;
const INV_TWO_COLOR_SIGMA2 = 1 / (2 * COLOR_SIGMA * COLOR_SIGMA);

let updateUICallback = () => {};
let frameCallback = () => {};

export function configureTrainingCallbacks({ onUpdateUI, onFrame } = {}) {
  updateUICallback = typeof onUpdateUI === "function" ? onUpdateUI : () => {};
  frameCallback = typeof onFrame === "function" ? onFrame : () => {};
}

const WORKER_SRC = `
  const EPS = 1e-6;
  const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
  const COLOR_SIGMA = 0.25;
  const INV_TWO_COLOR_SIGMA2 = 1 / (2 * COLOR_SIGMA * COLOR_SIGMA);
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
            const baseGlobal = (y * W + x) * 3;
            const targetR = IMG[baseGlobal + 0];
            const targetG = IMG[baseGlobal + 1];
            const targetB = IMG[baseGlobal + 2];
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
              const spatial = Math.exp(-0.5 * z) * sx * sy;
              const dr = color[i * 3 + 0] - targetR;
              const dg = color[i * 3 + 1] - targetG;
              const db = color[i * 3 + 2] - targetB;
              const colorDiffSq = dr * dr + dg * dg + db * db;
              const colorWeight = Math.exp(-colorDiffSq * INV_TWO_COLOR_SIGMA2);
              const weight = spatial * colorWeight;
              if (topIdx.length < K) {
                let j = topIdx.length;
                topIdx.push(i);
                topVal.push(weight);
                while (j > 0 && topVal[j] > topVal[j - 1]) {
                  const ti = topIdx[j - 1];
                  topIdx[j - 1] = topIdx[j];
                  topIdx[j] = ti;
                  const tv = topVal[j - 1];
                  topVal[j - 1] = topVal[j];
                  topVal[j] = tv;
                  j--;
                }
              } else if (weight > topVal[topVal.length - 1]) {
                topIdx[topIdx.length - 1] = i;
                topVal[topVal.length - 1] = weight;
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
            let r = 0, g = 0, b = 0;
            for (let k = 0; k < topIdx.length; k++) {
              const wi = topVal[k] / wsum;
              const gi = topIdx[k];
              r += wi * color[gi * 3 + 0];
              g += wi * color[gi * 3 + 1];
              b += wi * color[gi * 3 + 2];
              accW[gi] += wi;
              accTW[gi * 3 + 0] += wi * targetR;
              accTW[gi * 3 + 1] += wi * targetG;
              accTW[gi * 3 + 2] += wi * targetB;
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

export const state = {
  imgJS: null,
  imgSize: { w: 0, h: 0 },
  status: "idle",
  metrics: { step: 0, psnr: null, n: 0, mode: "js", worker: true, pool: defaultPool },
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

const pool = { workers: [], urls: [], ready: [], resolvers: [], nextReqId: 1 };

const notifyUI = () => updateUICallback();
const emitFrame = (out, W, H) => frameCallback(out, W, H);

function destroyPoolInternal() {
  for (const w of pool.workers) {
    try {
      w.terminate();
    } catch (e) {
      console.warn("Failed to terminate worker:", e);
    }
  }
  for (const u of pool.urls) {
    try {
      URL.revokeObjectURL(u);
    } catch (e) {
      console.warn("Failed to revoke worker URL:", e);
    }
  }
  pool.workers = [];
  pool.urls = [];
  pool.ready = [];
  pool.resolvers = [];
  pool.nextReqId = 1;
  state.metrics = { ...state.metrics, pool: 0 };
}

export function destroyPool() {
  destroyPoolInternal();
}

export function rebuildPool(size, preserveImage = false) {
  state.poolSize = size;
  const oldImg = preserveImage ? state.imgJS : null;
  destroyPoolInternal();
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
  state.metrics = { ...state.metrics, pool: pool.workers.length };

  if (preserveImage && oldImg) broadcastInitImage(oldImg);
  notifyUI();
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
  notifyUI();
}

export function broadcastSetTiling() {
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

export function setImageData(js) {
  if (!js) return;
  state.imgJS = { data: js.data, w: js.w, h: js.h };
  state.imgSize = { w: js.w, h: js.h };
  vars = null;
  curNg = 0;
  state.metrics = { step: 0, psnr: null, n: curNg, mode: "js", worker: true, pool: pool.workers.length || state.poolSize };
  state.status = "loaded";
  broadcastInitImage(js);
  notifyUI();
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
  state.metrics = { ...state.metrics, n: curNg };
  broadcastSetTiling();
  notifyUI();
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

export async function renderFrameViaPool(doRebin = false) {
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
      emitFrame(out, W, H);
      lastDraw = performance.now();
      const ps = psnrJS(out, state.imgJS.data);
      state.metrics = { step, psnr: ps, n: curNg, mode: "js", worker: true, pool: stripes };
      notifyUI();
    } else {
      state.metrics = { ...state.metrics, step, n: curNg, pool: stripes };
      notifyUI();
    }

    if (step % addEvery === 0 && curNg < Ntotal) {
      await addGaussiansByErrorJS(Math.min(addChunk, Ntotal - curNg));
    }
    if (state.stepDelayMs > 0) await new Promise((r) => setTimeout(r, state.stepDelayMs));
  }

  const out = await renderFrameViaPool(true);
  if (out) emitFrame(out, state.imgSize.w, state.imgSize.h);
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
  state.metrics = { ...state.metrics, n: curNg };
  broadcastSetTiling();
  notifyUI();
}

export async function startTraining() {
  if (!state.imgJS) return;
  state.status = "initializing";
  stopFlag = false;
  notifyUI();
  await initParamsJS();
  try {
    await waitAllReady();
  } catch (e) {
    console.error("Worker pool initialization failed:", e);
    state.status = "error";
    notifyUI();
    return;
  }
  state.status = "training";
  notifyUI();
  await trainJS_pool();
  state.status = stopFlag ? "stopped" : "done";
  notifyUI();
}

export function stopTraining() {
  stopFlag = true;
  state.status = "stopped";
  notifyUI();
}

export function getGaussianCount() {
  return curNg;
}

export function serializeModel() {
  const H = state.imgSize.h;
  const W = state.imgSize.w;
  if (!H || !W || !vars) return null;
  const { mu, s_inv, theta, color } = vars;
  const N = mu.length / 2;
  const headerObj = { magic: "IGS1", H, W, C: 3, N };
  const headerStr = JSON.stringify(headerObj) + "\n";
  const header = new TextEncoder().encode(headerStr);
  const muQ = float16Quantize(mu);
  const sQ = float16Quantize(s_inv);
  const thQ = float16Quantize(theta);
  const cQ = float16Quantize(color);
  const out = new Uint8Array(header.length + muQ.length + sQ.length + thQ.length + cQ.length);
  let offset = 0;
  out.set(header, offset);
  offset += header.length;
  out.set(muQ, offset);
  offset += muQ.length;
  out.set(sQ, offset);
  offset += sQ.length;
  out.set(thQ, offset);
  offset += thQ.length;
  out.set(cQ, offset);
  return out;
}

export async function runDiagnosticsOnce() {
  const { worker, url } = makeWorker();
  try {
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
    if (!d.out || d.out.length !== W * H * 3) throw new Error("render size mismatch");
  } finally {
    URL.revokeObjectURL(url);
    worker.terminate();
  }
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
