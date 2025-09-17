import { clamp, cpuTopKIndices, sobelMagJS, samplePositionsAndColorsJS } from './gaussianMath'

export function initializeParameters(image, startCount, lambdaInit) {
  const { w, h, data } = image
  const grad = sobelMagJS(data, w, h)
  let sum = 0
  for (let i = 0; i < grad.length; i++) sum += grad[i]
  const uniform = 1 / (w * h)
  const probs = new Float32Array(w * h)
  for (let i = 0; i < w * h; i++) probs[i] = (1 - lambdaInit) * (grad[i] / (sum || 1)) + lambdaInit * uniform

  const Ng0 = Math.max(1, Math.min(Math.floor(startCount) || 1, w * h))
  const { mu, colors } = samplePositionsAndColorsJS(data, w, h, probs, Ng0)
  const s_inv = new Float32Array(Ng0 * 2)
  for (let i = 0; i < Ng0; i++) {
    s_inv[i * 2 + 0] = (w - 1) / 5
    s_inv[i * 2 + 1] = (h - 1) / 5
  }
  const theta = new Float32Array(Ng0)
  for (let i = 0; i < Ng0; i++) theta[i] = Math.random() * Math.PI

  return { vars: { mu, s_inv, theta, color: colors }, count: Ng0 }
}

export function mergeAccumulators(parts, N) {
  const accW = new Float32Array(N)
  const accTW = new Float32Array(N * 3)
  const accX = new Float32Array(N)
  const accY = new Float32Array(N)
  const accXX = new Float32Array(N)
  const accYY = new Float32Array(N)
  const accXY = new Float32Array(N)
  const accErrW = new Float32Array(N)
  const accErrX = new Float32Array(N)
  const accErrY = new Float32Array(N)
  const accErrXX = new Float32Array(N)
  const accErrYY = new Float32Array(N)
  const accErrXY = new Float32Array(N)
  for (const p of parts) {
    const aW = p.accW
    const aT = p.accTW
    const aX = p.accX
    const aY = p.accY
    const aXX = p.accXX
    const aYY = p.accYY
    const aXY = p.accXY
    const aErrW = p.accErrW
    const aErrX = p.accErrX
    const aErrY = p.accErrY
    const aErrXX = p.accErrXX
    const aErrYY = p.accErrYY
    const aErrXY = p.accErrXY
    for (let i = 0; i < N; i++) accW[i] += aW[i]
    for (let i = 0; i < N * 3; i++) accTW[i] += aT[i]
    for (let i = 0; i < N; i++) {
      accX[i] += aX[i]
      accY[i] += aY[i]
      accXX[i] += aXX[i]
      accYY[i] += aYY[i]
      accXY[i] += aXY[i]
      if (aErrW) accErrW[i] += aErrW[i]
      if (aErrX) accErrX[i] += aErrX[i]
      if (aErrY) accErrY[i] += aErrY[i]
      if (aErrXX) accErrXX[i] += aErrXX[i]
      if (aErrYY) accErrYY[i] += aErrYY[i]
      if (aErrXY) accErrXY[i] += aErrXY[i]
    }
  }
  return {
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
  }
}

export function composeOutput(parts, W, H) {
  const out = new Float32Array(W * H * 3)
  for (const p of parts) {
    if (!p.out) continue
    const y0 = p.y0
    const stripe = p.out
    out.set(stripe, y0 * W * 3)
  }
  return out
}

export function updateParametersFromAccumulators(vars, accumulators, config) {
  if (!vars) return
  const { lrColor, lrMu, lrShape, size } = config
  const { w, h } = size
  const N = vars.mu.length / 2
  for (let i = 0; i < N; i++) {
    const wsum = accumulators.accW[i] + 1e-6
    const tR = accumulators.accTW[i * 3 + 0] / wsum
    const tG = accumulators.accTW[i * 3 + 1] / wsum
    const tB = accumulators.accTW[i * 3 + 2] / wsum
    vars.color[i * 3 + 0] = (1 - lrColor) * vars.color[i * 3 + 0] + lrColor * tR
    vars.color[i * 3 + 1] = (1 - lrColor) * vars.color[i * 3 + 1] + lrColor * tG
    vars.color[i * 3 + 2] = (1 - lrColor) * vars.color[i * 3 + 2] + lrColor * tB
  }
  const stdMinX = 1 / Math.max(w - 1, 1)
  const stdMinY = 1 / Math.max(h - 1, 1)
  const accErrW = accumulators.accErrW ?? accumulators.accW
  const accErrX = accumulators.accErrX ?? accumulators.accX
  const accErrY = accumulators.accErrY ?? accumulators.accY
  const accErrXX = accumulators.accErrXX ?? accumulators.accXX
  const accErrYY = accumulators.accErrYY ?? accumulators.accYY
  const accErrXY = accumulators.accErrXY ?? accumulators.accXY
  for (let i = 0; i < N; i++) {
    const wsum = accErrW[i]
    if (wsum < 1e-6) continue
    const mx = accErrX[i] / wsum
    const my = accErrY[i] / wsum
    const muX = vars.mu[i * 2 + 0]
    const muY = vars.mu[i * 2 + 1]
    vars.mu[i * 2 + 0] = (1 - lrMu) * muX + lrMu * mx
    vars.mu[i * 2 + 1] = (1 - lrMu) * muY + lrMu * my
    const cxx = Math.max(0, accErrXX[i] / wsum - mx * mx)
    const cyy = Math.max(0, accErrYY[i] / wsum - my * my)
    const cxy = accErrXY[i] / wsum - mx * my
    const tr = cxx + cyy
    const det = cxx * cyy - cxy * cxy
    const disc = Math.max(0, tr * tr - 4 * det)
    const s = Math.sqrt(disc)
    const l1 = 0.5 * (tr + s)
    const l2 = 0.5 * (tr - s)
    let vx
    let vy
    if (Math.abs(cxy) > 1e-12) {
      vx = l1 - cyy
      vy = cxy
    } else if (cxx >= cyy) {
      vx = 1
      vy = 0
    } else {
      vx = 0
      vy = 1
    }
    const n = Math.hypot(vx, vy) || 1
    vx /= n
    vy /= n
    let th0 = vars.theta[i] % Math.PI
    if (th0 < 0) th0 += Math.PI
    let thT = Math.atan2(vy, vx)
    thT %= Math.PI
    if (thT < 0) thT += Math.PI
    if (Math.abs(thT - th0) > Math.PI / 2) {
      if (thT > th0) thT -= Math.PI
      else thT += Math.PI
    }
    const thNew = th0 + (thT - th0) * lrShape
    vars.theta[i] = ((thNew % Math.PI) + Math.PI) % Math.PI
    const std1 = Math.sqrt(Math.max(l1, 0))
    const std2 = Math.sqrt(Math.max(l2, 0))
    const sxInvTarget = 1 / Math.max(std1, Math.min(stdMinX, stdMinY))
    const syInvTarget = 1 / Math.max(std2, Math.min(stdMinX, stdMinY))
    const sx0 = vars.s_inv[i * 2 + 0]
    const sy0 = vars.s_inv[i * 2 + 1]
    vars.s_inv[i * 2 + 0] = clamp((1 - lrShape) * sx0 + lrShape * sxInvTarget, 0.1, (w - 1) * 4)
    vars.s_inv[i * 2 + 1] = clamp((1 - lrShape) * sy0 + lrShape * syInvTarget, 0.1, (h - 1) * 4)
  }
}

export function computeErrorField(out, target, width, height) {
  const err = new Float32Array(width * height)
  for (let i = 0; i < width * height; i++) {
    err[i] =
      (Math.abs(out[i * 3 + 0] - target[i * 3 + 0]) +
        Math.abs(out[i * 3 + 1] - target[i * 3 + 1]) +
        Math.abs(out[i * 3 + 2] - target[i * 3 + 2])) /
      3
  }
  return err
}

export function selectErrorIndices(errorField, count) {
  return cpuTopKIndices(errorField, count)
}

export function extendModelWithErrors(vars, indices, image) {
  const { data: tgt, w: W, h: H } = image
  const NgOld = vars.mu.length / 2
  const nNew = indices.length
  const NgNew = NgOld + nNew
  const mu2 = new Float32Array(NgNew * 2)
  mu2.set(vars.mu)
  const s2 = new Float32Array(NgNew * 2)
  s2.set(vars.s_inv)
  const th2 = new Float32Array(NgNew)
  th2.set(vars.theta)
  const c2 = new Float32Array(NgNew * 3)
  c2.set(vars.color)

  for (let j = 0; j < indices.length; j++) {
    const id = indices[j]
    const y = Math.floor(id / W)
    const x = id % W
    const base = y * W + x
    const i = NgOld + j
    mu2[i * 2 + 0] = x / (W - 1)
    mu2[i * 2 + 1] = y / (H - 1)
    s2[i * 2 + 0] = (W - 1) / 5
    s2[i * 2 + 1] = (H - 1) / 5
    th2[i] = Math.random() * Math.PI
    c2[i * 3 + 0] = tgt[base * 3 + 0]
    c2[i * 3 + 1] = tgt[base * 3 + 1]
    c2[i * 3 + 2] = tgt[base * 3 + 2]
  }

  return { vars: { mu: mu2, s_inv: s2, theta: th2, color: c2 }, count: NgNew }
}
