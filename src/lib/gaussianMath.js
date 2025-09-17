export const EPS = 1e-6

export const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi)

export function sobelMagJS(rgb, w, h) {
  const mag = new Float32Array(w * h)
  const kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
  const ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let gx = 0
      let gy = 0
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const xx = clamp(x + dx, 0, w - 1)
          const yy = clamp(y + dy, 0, h - 1)
          const i = yy * w + xx
          const r = rgb[i * 3 + 0]
          const g = rgb[i * 3 + 1]
          const b = rgb[i * 3 + 2]
          const gray = 0.299 * r + 0.587 * g + 0.114 * b
          const kIdx = (dy + 1) * 3 + (dx + 1)
          gx += gray * kx[kIdx]
          gy += gray * ky[kIdx]
        }
      mag[y * w + x] = Math.sqrt(gx * gx + gy * gy)
    }
  }
  return mag
}

export function cpuTopKIndices(arr, K) {
  const n = arr.length
  const k = Math.min(K, n)
  const idxs = new Array(k).fill(-1)
  const vals = new Array(k).fill(-Infinity)
  for (let i = 0; i < n; i++) {
    const v = arr[i]
    if (v > vals[k - 1]) {
      let j = k - 1
      while (j > 0 && v > vals[j - 1]) {
        vals[j] = vals[j - 1]
        idxs[j] = idxs[j - 1]
        j--
      }
      vals[j] = v
      idxs[j] = i
    }
  }
  return idxs
}

export function samplePositionsAndColorsJS(imgF32, w, h, prob2D, N) {
  const prob = new Float32Array(prob2D)
  let s = 0
  for (let i = 0; i < prob.length; i++) s += prob[i]
  s = s || 1
  for (let i = 0; i < prob.length; i++) prob[i] /= s
  const cdf = new Float32Array(prob.length)
  let acc = 0
  for (let i = 0; i < prob.length; i++) {
    acc += prob[i]
    cdf[i] = acc
  }
  const mu = new Float32Array(N * 2)
  const colors = new Float32Array(N * 3)
  for (let k = 0; k < N; k++) {
    const r = Math.random()
    let lo = 0
    let hi = cdf.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (cdf[mid] < r) lo = mid + 1
      else hi = mid
    }
    const y = Math.floor(lo / w)
    const x = lo % w
    const idx = y * w + x
    mu[k * 2 + 0] = x / (w - 1)
    mu[k * 2 + 1] = y / (h - 1)
    colors[k * 3 + 0] = imgF32[idx * 3 + 0]
    colors[k * 3 + 1] = imgF32[idx * 3 + 1]
    colors[k * 3 + 2] = imgF32[idx * 3 + 2]
  }
  return { mu, colors }
}

export function psnrJS(a, b) {
  let mse = 0
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i]
    mse += d * d
  }
  mse /= a.length || 1
  if (mse <= 0) return Infinity
  return 10 * Math.log10(1 / (mse + EPS))
}
