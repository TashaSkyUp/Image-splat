export const WORKER_SRC = `
  const EPS = 1e-6;
  const COLOR_PRECISION = 20;
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
    const accErrW = new Float32Array(N);
    const accErrX = new Float32Array(N);
    const accErrY = new Float32Array(N);
    const accErrXX = new Float32Array(N);
    const accErrYY = new Float32Array(N);
    const accErrXY = new Float32Array(N);

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
            const tR = IMG[baseGlobal + 0];
            const tG = IMG[baseGlobal + 1];
            const tB = IMG[baseGlobal + 2];
            const topIdx = [];
            const topVal = [];
            const scan = (i) => {
              const dx = cx - mu[i * 2 + 0];
              const dy = cy - mu[i * 2 + 1];
              const th = theta[i];
              const c = Math.cos(th), s = Math.sin(th);
              const dxp = c * dx + s * dy;
              const dyp = -s * dx + c * dy;
              const sx = Math.max(s_inv[i * 2 + 0], EPS);
              const sy = Math.max(s_inv[i * 2 + 1], EPS);
              const z = (dxp * sx) * (dxp * sx) + (dyp * sy) * (dyp * sy);
              const spatial = Math.exp(-0.5 * z) * sx * sy;
              if (spatial <= 0) return;
              const cr = color[i * 3 + 0];
              const cg = color[i * 3 + 1];
              const cb = color[i * 3 + 2];
              const dR = tR - cr;
              const dG = tG - cg;
              const dB = tB - cb;
              const colorDist = dR * dR + dG * dG + dB * dB;
              const colorWeight = Math.exp(-0.5 * colorDist * COLOR_PRECISION);
              const weight = spatial * colorWeight;
              if (weight <= 0) return;
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
            const cxn = cx;
            const cyn = cy;
            let r = 0, gOut = 0, b = 0;
            for (let k = 0; k < topIdx.length; k++) {
              const wi = topVal[k] / wsum;
              topVal[k] = wi;
              const gi = topIdx[k];
              r += wi * color[gi * 3 + 0];
              gOut += wi * color[gi * 3 + 1];
              b += wi * color[gi * 3 + 2];
              accW[gi] += wi;
              accTW[gi * 3 + 0] += wi * IMG[baseGlobal + 0];
              accTW[gi * 3 + 1] += wi * IMG[baseGlobal + 1];
              accTW[gi * 3 + 2] += wi * IMG[baseGlobal + 2];
              accX[gi] += wi * cxn;
              accY[gi] += wi * cyn;
              accXX[gi] += wi * cxn * cxn;
              accYY[gi] += wi * cyn * cyn;
              accXY[gi] += wi * cxn * cyn;
            }
            const residual =
              (Math.abs(tR - r) + Math.abs(tG - gOut) + Math.abs(tB - b)) / 3;
            const cx2 = cxn * cxn;
            const cy2 = cyn * cyn;
            const cxy = cxn * cyn;
            for (let k = 0; k < topIdx.length; k++) {
              const gi = topIdx[k];
              const wi = topVal[k];
              const wErr = wi * residual;
              accErrW[gi] += wErr;
              accErrX[gi] += wErr * cxn;
              accErrY[gi] += wErr * cyn;
              accErrXX[gi] += wErr * cx2;
              accErrYY[gi] += wErr * cy2;
              accErrXY[gi] += wErr * cxy;
            }
            if (out) {
              const row = y - y0Stripe;
              const baseLocal = (row * W + x) * 3;
              out[baseLocal + 0] = r;
              out[baseLocal + 1] = gOut;
              out[baseLocal + 2] = b;
            }
          }
        }
      }
    }

    const msg = {
      type: 'stepResult',
      reqId,
      accW,
      accTW,
      accX,
      accY,
      accXX,
      accYY,
      accXY,
      accErrW,
      accErrX,
      accErrY,
      accErrXX,
      accErrYY,
      accErrXY,
      y0: y0Stripe,
      y1: y1Stripe,
    };
    const transfers = [
      accW.buffer,
      accTW.buffer,
      accX.buffer,
      accY.buffer,
      accXX.buffer,
      accYY.buffer,
      accXY.buffer,
      accErrW.buffer,
      accErrX.buffer,
      accErrY.buffer,
      accErrXX.buffer,
      accErrYY.buffer,
      accErrXY.buffer,
    ];
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
`

export function makeWorker() {
  const blob = new Blob([WORKER_SRC], { type: 'text/javascript' })
  const url = URL.createObjectURL(blob)
  const worker = new Worker(url, { type: 'classic' })
  return { worker, url }
}
