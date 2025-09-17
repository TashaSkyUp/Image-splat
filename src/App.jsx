/* eslint-disable react-hooks/exhaustive-deps */
import { useEffect, useRef, useState } from 'react'

import InfoPopover from './components/InfoPopover'
import { fileToImageSource, sourceToF32RGB } from './lib/imageLoader'
import { clamp, psnrJS } from './lib/gaussianMath'
import { float16Quantize } from './lib/quantization'
import {
  composeOutput,
  computeErrorField,
  extendModelWithErrors,
  initializeParameters,
  mergeAccumulators,
  selectErrorIndices,
  updateParametersFromAccumulators,
} from './lib/training'
import { makeWorker } from './lib/workerFactory'

const buttonBase =
  'inline-flex items-center justify-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400 focus-visible:ring-offset-slate-950 disabled:cursor-not-allowed disabled:opacity-60'
const primaryButton =
  buttonBase +
  ' bg-gradient-to-r from-sky-500 to-cyan-400 text-slate-950 shadow-lg shadow-sky-500/30 hover:from-sky-400 hover:to-cyan-300'
const secondaryButton =
  buttonBase + ' border border-white/15 bg-white/10 text-slate-50 hover:bg-white/20'
const ghostButton = buttonBase + ' border border-white/10 bg-transparent text-slate-200 hover:bg-white/10'

const inputClass =
  'w-full rounded-xl border border-white/15 bg-slate-950/40 px-3 py-2 text-sm text-slate-50 shadow-inner shadow-black/40 transition focus:border-sky-300 focus:outline-none focus:ring-2 focus:ring-sky-400/40 placeholder:text-slate-400'

const glassCardClass =
  'rounded-3xl border border-white/10 bg-white/10 p-6 shadow-2xl shadow-slate-950/40 ring-1 ring-white/10 backdrop-blur'
const metricCardClass = 'rounded-2xl border border-white/10 bg-slate-900/60 px-4 py-3 shadow-inner shadow-black/40'

const statusBadgeBase =
  'inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium shadow-sm backdrop-blur'
const statusStyles = {
  idle: {
    label: 'Idle',
    classes: 'border-white/10 bg-white/10 text-slate-200',
    dot: 'bg-slate-200',
  },
  loaded: {
    label: 'Ready',
    classes: 'border-emerald-400/30 bg-emerald-500/15 text-emerald-100',
    dot: 'bg-emerald-300',
  },
  initializing: {
    label: 'Initializing',
    classes: 'border-indigo-400/40 bg-indigo-500/20 text-indigo-100',
    dot: 'bg-indigo-300',
  },
  training: {
    label: 'Training',
    classes: 'border-sky-400/40 bg-sky-500/20 text-sky-100',
    dot: 'bg-sky-300',
  },
  stopped: {
    label: 'Stopped',
    classes: 'border-amber-400/40 bg-amber-500/20 text-amber-100',
    dot: 'bg-amber-300',
  },
  done: {
    label: 'Finished',
    classes: 'border-emerald-400/40 bg-emerald-500/15 text-emerald-100',
    dot: 'bg-emerald-300',
  },
}

const MAX_HISTORY_POINTS = 720

export default function ImageGSApp() {
  const [imgJS, setImgJS] = useState(null)
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 })
  const [status, setStatus] = useState('idle')
  const [metrics, setMetrics] = useState({
    step: 0,
    psnr: null,
    n: 0,
    mode: 'js',
    worker: true,
    pool: 1,
    igsBytes: null,
  })

  const [K, setK] = useState(6)
  const [budget, setBudget] = useState(2000)
  const [lambdaInit, setLambdaInit] = useState(0.3)
  const [steps, setSteps] = useState(1500)
  const [stepDelayMs, setStepDelayMs] = useState(0)
  const [addEvery, setAddEvery] = useState(400)

  const [lrColor, setLrColor] = useState(0.4)
  const [lrMu, setLrMu] = useState(0.25)
  const [lrShape, setLrShape] = useState(0.25)

  const [enableTiling, setEnableTiling] = useState(true)
  const [tileW, setTileW] = useState(32)
  const [tileH, setTileH] = useState(32)
  const [rebinEvery, setRebinEvery] = useState(50)

  const defaultPool = Math.max(2, Math.min(navigator.hardwareConcurrency || 4, 8))
  const [poolSize, setPoolSize] = useState(defaultPool)

  const [psnrHistory, setPsnrHistory] = useState([])
  const [compressionHistory, setCompressionHistory] = useState([])
  const [baselineSizes, setBaselineSizes] = useState({ source: null, png: null, jpg: null })
  const baselineRef = useRef(baselineSizes)
  const baselineJobRef = useRef(0)
  const runStartedAtRef = useRef(null)

  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)
  const stopRef = useRef(false)
  const varsRef = useRef(null)
  const curNgRef = useRef(0)
  const lastDrawRef = useRef(0)

  const poolRef = useRef({ workers: [], urls: [], ready: [], resolvers: [], nextReqId: 1 })

  useEffect(() => {
    rebuildPool(poolSize)
    return () => destroyPool()
  }, [])

  useEffect(() => {
    rebuildPool(poolSize, true)
  }, [poolSize])

  useEffect(() => {
    baselineRef.current = baselineSizes
  }, [baselineSizes])

  function destroyPool() {
    const pool = poolRef.current
    for (const w of pool.workers) {
      try {
        w.terminate()
      } catch (err) {
        console.warn('terminate worker failed', err)
      }
    }
    for (const u of pool.urls) {
      try {
        URL.revokeObjectURL(u)
      } catch (err) {
        console.warn('revokeObjectURL failed', err)
      }
    }
    poolRef.current = { workers: [], urls: [], ready: [], resolvers: [], nextReqId: 1 }
  }

  function rebuildPool(size, preserveImage = false) {
    const oldImg = preserveImage ? imgJS : null
    destroyPool()
    const workers = []
    const urls = []
    const ready = []
    const resolvers = []
    for (let i = 0; i < size; i++) {
      const { worker, url } = makeWorker()
      workers.push(worker)
      urls.push(url)
      ready.push(false)
      resolvers.push(new Map())
      worker.onmessage = (ev) => {
        if (ev.data?.type === 'inited') {
          ready[i] = true
        } else if (ev.data?.type === 'stepResult') {
          const resMap = resolvers[i]
          const rid = ev.data.reqId
          if (rid && resMap.has(rid)) {
            const fn = resMap.get(rid)
            resMap.delete(rid)
            fn(ev.data)
          }
        }
      }
    }
    poolRef.current = { workers, urls, ready, resolvers, nextReqId: 1 }
    if (preserveImage && oldImg) {
      broadcastInitImage(oldImg)
    }
  }

  function broadcastInitImage(js) {
    const pool = poolRef.current
    if (!pool.workers.length) return
    const { w, h, data } = js
    setImgSize({ w, h })
    for (let i = 0; i < pool.workers.length; i++) {
      const imgCopy = new Float32Array(data)
      pool.ready[i] = false
      pool.workers[i].postMessage(
        { type: 'init', img: imgCopy, W: w, H: h, tileW, tileH, enableTiling },
        [imgCopy.buffer],
      )
    }
  }

  function broadcastSetTiling() {
    const pool = poolRef.current
    for (let i = 0; i < pool.workers.length; i++) {
      pool.workers[i].postMessage({ type: 'setTiling', tileW, tileH, enableTiling })
    }
  }

  async function waitAllReady() {
    const pool = poolRef.current
    const deadline = performance.now() + 5000
    while (true) {
      if (pool.ready.every(Boolean)) return
      if (performance.now() > deadline) throw new Error('Worker pool init timeout')
      await new Promise((r) => setTimeout(r, 10))
    }
  }

  function postStepToWorker(i, payload) {
    const pool = poolRef.current
    const rid = pool.nextReqId++
    return new Promise((resolve) => {
      pool.resolvers[i].set(rid, resolve)
      pool.workers[i].postMessage({ type: 'step', reqId: rid, ...payload })
    })
  }

  function drawOnCanvasF32RGB(arr, w, h) {
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return
    canvasRef.current.width = w
    canvasRef.current.height = h
    const out = new Uint8ClampedArray(w * h * 4)
    for (let i = 0, j = 0; i < w * h; i++) {
      out[j++] = clamp(Math.round(arr[i * 3 + 0] * 255), 0, 255)
      out[j++] = clamp(Math.round(arr[i * 3 + 1] * 255), 0, 255)
      out[j++] = clamp(Math.round(arr[i * 3 + 2] * 255), 0, 255)
      out[j++] = 255
    }
    ctx.putImageData(new ImageData(out, w, h), 0, 0)
  }

  async function handleFile(e) {
    try {
      const file = e.target.files?.[0]
      if (!file) return
      if (!(file.type && file.type.startsWith('image/')))
        throw new Error(`Selected file is not an image (type: ${file.type || 'unknown'})`)
      const { src, revoke } = await fileToImageSource(file)
      let js
      try {
        js = sourceToF32RGB(src, 512)
      } finally {
        try {
          revoke?.()
        } catch (err) {
          console.warn('revoke failed', err)
        }
      }
      const localData = new Float32Array(js.data)
      setImgJS({ data: localData, w: js.w, h: js.h })
      setImgSize({ w: js.w, h: js.h })
      drawOnCanvasF32RGB(localData, js.w, js.h)
      setMetrics({ step: 0, psnr: null, n: 0, mode: 'js', worker: true, pool: poolSize, igsBytes: null })
      setPsnrHistory([])
      setCompressionHistory([])
      setBaselineSizes({ source: file.size ?? null, png: null, jpg: null })
      runStartedAtRef.current = null
      computeBaselines({ data: localData, w: js.w, h: js.h }, file.size ?? null)
      disposeVars()
      broadcastInitImage({ data: localData, w: js.w, h: js.h })
      setStatus('loaded')
    } catch (err) {
      console.error('handleFile failed', err)
      window.alert(`Failed to load image.\n\nReason: ${err?.message || err}`)
    } finally {
      if (e.target) e.target.value = ''
    }
  }

  function disposeVars() {
    varsRef.current = null
    curNgRef.current = 0
  }

  async function computeBaselines(image, fileSize) {
    if (!image || typeof document === 'undefined' || typeof ImageData === 'undefined') return
    const jobId = baselineJobRef.current + 1
    baselineJobRef.current = jobId
    setBaselineSizes({ source: fileSize ?? null, png: null, jpg: null })
    try {
      const imgData = floatImageToImageData(image)
      if (!imgData) return
      const [pngSize, jpgSize] = await Promise.all([
        encodeImageFromImageData(imgData, 'image/png'),
        encodeImageFromImageData(imgData, 'image/jpeg', 0.92),
      ])
      if (baselineJobRef.current === jobId) {
        setBaselineSizes({
          source: fileSize ?? null,
          png: pngSize ?? null,
          jpg: jpgSize ?? null,
        })
      }
    } catch (err) {
      console.error('computeBaselines failed', err)
      if (baselineJobRef.current === jobId) {
        setBaselineSizes({ source: fileSize ?? null, png: null, jpg: null })
      }
    }
  }

  async function initParamsJS() {
    if (!imgJS) return
    const { vars, count } = initializeParameters(imgJS, budget, lambdaInit)
    varsRef.current = vars
    curNgRef.current = count
    broadcastSetTiling()
  }

  function paramsForStep() {
    const v = varsRef.current
    if (!v) return null
    return {
      mu: new Float32Array(v.mu),
      s_inv: new Float32Array(v.s_inv),
      theta: new Float32Array(v.theta),
      color: new Float32Array(v.color),
    }
  }

  async function train() {
    if (!imgJS) return
    setStatus('initializing')
    stopRef.current = false
    setPsnrHistory([])
    setCompressionHistory([])
    runStartedAtRef.current = typeof performance !== 'undefined' ? performance.now() : Date.now()
    await initParamsJS()
    const initialBytes = estimateIgsSizeBytes(varsRef.current, imgSize)
    setMetrics((prev) => ({
      ...prev,
      step: 0,
      psnr: null,
      n: curNgRef.current,
      igsBytes: initialBytes,
    }))
    recordModelStats(0, { psnr: null, igsBytes: initialBytes })
    try {
      await waitAllReady()
    } catch (e) {
      console.error(e)
    }
    setStatus('training')
    await trainJS_pool()
    setStatus('done')
  }

  function recordModelStats(step, { psnr, igsBytes }) {
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now()
    const startedAt = runStartedAtRef.current
    const timeSec = typeof startedAt === 'number' ? (now - startedAt) / 1000 : step

    if (Number.isFinite(psnr)) {
      setPsnrHistory((prev) => {
        const next = [...prev, { time: timeSec, psnr, step }]
        if (next.length > MAX_HISTORY_POINTS) next.splice(0, next.length - MAX_HISTORY_POINTS)
        return next
      })
    }

    if (Number.isFinite(igsBytes) && igsBytes > 0) {
      const base = baselineRef.current
      const pngBytes = base?.png
      const jpgBytes = base?.jpg
      if (Number.isFinite(pngBytes) && pngBytes > 0 && Number.isFinite(jpgBytes) && jpgBytes > 0) {
        setCompressionHistory((prev) => {
          const next = [
            ...prev,
            {
              time: timeSec,
              step,
              igsBytes,
              pngRatio: pngBytes / igsBytes,
              jpgRatio: jpgBytes / igsBytes,
            },
          ]
          if (next.length > MAX_HISTORY_POINTS) next.splice(0, next.length - MAX_HISTORY_POINTS)
          return next
        })
      }
    }
  }

  function applyAccumulatorUpdate(pkg) {
    if (!varsRef.current) return
    updateParametersFromAccumulators(varsRef.current, pkg, {
      lrColor,
      lrMu,
      lrShape,
      size: imgSize,
    })
  }

  async function renderFrameViaPool(doRebin = false) {
    const pool = poolRef.current
    const v = varsRef.current
    if (!v || !pool.workers.length) return null
    await waitAllReady()
    const { w: W, h: H } = imgSize
    const params = paramsForStep()
    if (!params) return null
    const stripes = Math.min(pool.workers.length, H)
    const promises = []
    for (let i = 0; i < stripes; i++) {
      const y0 = Math.floor((H * i) / stripes)
      const y1 = Math.min(H - 1, Math.floor((H * (i + 1)) / stripes) - 1)
      const wantImage = true
      const payload = { K, doRebin, wantImage, y0, y1, ...params }
      promises.push(postStepToWorker(i, payload))
    }
    const results = await Promise.all(promises)
    return composeOutput(results, W, H)
  }

  async function trainJS_pool() {
    const pool = poolRef.current
    if (!pool.workers.length) return
    const Ntotal = budget
    const addChunk = Math.max(1, Math.floor(Ntotal / 8))
    const { w: W, h: H } = imgSize
    const addFrequency = addEvery > 0 ? addEvery : null

    let lastStep = 0

    for (let step = 1; step <= steps; step++) {
      if (stopRef.current) break
      const v = varsRef.current
      if (!v) break
      const N = v.mu.length / 2
      const params = paramsForStep()
      if (!params) break
      const stripes = Math.min(pool.workers.length, H)
      const doRebin = enableTiling && (step === 1 || (rebinEvery > 0 && step % rebinEvery === 0))
      const nowTime = typeof performance !== 'undefined' ? performance.now() : Date.now()
      const wantImage = nowTime - lastDrawRef.current >= 100

      const promises = []
      for (let i = 0; i < stripes; i++) {
        const y0 = Math.floor((H * i) / stripes)
        const y1 = Math.min(H - 1, Math.floor((H * (i + 1)) / stripes) - 1)
        const payload = { K, doRebin, wantImage, y0, y1, ...params }
        promises.push(postStepToWorker(i, payload))
      }
      const results = await Promise.all(promises)
      const merged = mergeAccumulators(results, N)
      applyAccumulatorUpdate(merged)

      const igsBytes = estimateIgsSizeBytes(varsRef.current, imgSize)
      let ps = null
      if (wantImage) {
        const out = composeOutput(results, W, H)
        drawOnCanvasF32RGB(out, W, H)
        lastDrawRef.current = typeof performance !== 'undefined' ? performance.now() : Date.now()
        ps = psnrJS(out, imgJS.data)
        setMetrics({
          step,
          psnr: ps,
          n: curNgRef.current,
          mode: 'js',
          worker: true,
          pool: stripes,
          igsBytes,
        })
      } else {
        setMetrics((m) => ({ ...m, step, n: curNgRef.current, pool: stripes, igsBytes }))
      }

      recordModelStats(step, { psnr: ps, igsBytes })

      if (addFrequency && step % addFrequency === 0 && curNgRef.current < Ntotal) {
        await addGaussiansByErrorJS(Math.min(addChunk, Ntotal - curNgRef.current))
      }
      if (stepDelayMs > 0) await new Promise((r) => setTimeout(r, stepDelayMs))
      lastStep = step
    }

    const out = await renderFrameViaPool(true)
    if (out) {
      drawOnCanvasF32RGB(out, imgSize.w, imgSize.h)
      const ps = psnrJS(out, imgJS.data)
      const igsBytes = estimateIgsSizeBytes(varsRef.current, imgSize)
      const stepForRecord = lastStep || 0
      setMetrics((m) => ({
        ...m,
        psnr: ps,
        igsBytes,
        step: stepForRecord || m.step || 0,
        n: curNgRef.current,
      }))
      recordModelStats(stepForRecord || 0, { psnr: ps, igsBytes })
    }
  }

  async function addGaussiansByErrorJS(nNew) {
    const out = await renderFrameViaPool(false)
    if (!out || !imgJS) return
    const { data: tgt, w: W, h: H } = imgJS
    const err = computeErrorField(out, tgt, W, H)
    const idxs = selectErrorIndices(err, Math.min(nNew, W * H))

    const v = varsRef.current
    if (!v) return
    const { vars, count } = extendModelWithErrors(v, idxs, imgJS)
    varsRef.current = vars
    curNgRef.current = count
    broadcastSetTiling()
  }

  function handleStop() {
    stopRef.current = true
    setStatus('stopped')
  }

  async function handleRenderOnce() {
    const out = await renderFrameViaPool(true)
    if (out) drawOnCanvasF32RGB(out, imgSize.w, imgSize.h)
  }

  async function handleSaveModel() {
    const H = imgSize.h
    const W = imgSize.w
    if (!H || !W || !varsRef.current) return
    const { mu, s_inv, theta, color } = varsRef.current
    const N = mu.length / 2
    const headerObj = { magic: 'IGS1', H, W, C: 3, N }
    const headerStr = JSON.stringify(headerObj) + '\n'
    const header = new Uint8Array(headerStr.split('').map((ch) => ch.charCodeAt(0) & 0xff))
    const payload = new Uint8Array([
      ...header,
      ...float16Quantize(mu),
      ...float16Quantize(s_inv),
      ...float16Quantize(theta),
      ...float16Quantize(color),
    ])
    const blob = new Blob([payload], { type: 'application/octet-stream' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'image_gs.igs'
    a.click()
  }


  const statusMeta =
    statusStyles[status] ?? {
      ...statusStyles.idle,
      label: status ? status.charAt(0).toUpperCase() + status.slice(1) : statusStyles.idle.label,
    }
  const psnrDisplay = Number.isFinite(metrics.psnr) ? metrics.psnr?.toFixed?.(2) : '—'
  const imageDimensions =
    imgSize.w && imgSize.h ? `${imgSize.w}×${imgSize.h}` : '—'
  const gaussianCount = curNgRef.current
  const gaussianDisplay = Number.isFinite(gaussianCount) ? gaussianCount.toLocaleString() : '—'
  const stepDisplay = Number.isFinite(metrics.step) ? metrics.step.toLocaleString() : '0'
  const poolDisplay = Number.isFinite(metrics.pool) ? metrics.pool.toLocaleString() : '—'
  const modelSizeDisplay = formatBytes(metrics.igsBytes)
  const latestPsnr = psnrHistory.length ? psnrHistory[psnrHistory.length - 1] : null
  const latestCompression =
    compressionHistory.length ? compressionHistory[compressionHistory.length - 1] : null
  const pngRatioDisplay =
    Number.isFinite(latestCompression?.pngRatio) && latestCompression.pngRatio > 0
      ? latestCompression.pngRatio >= 10
        ? latestCompression.pngRatio.toFixed(0)
        : latestCompression.pngRatio.toFixed(2)
      : null
  const jpgRatioDisplay =
    Number.isFinite(latestCompression?.jpgRatio) && latestCompression.jpgRatio > 0
      ? latestCompression.jpgRatio >= 10
        ? latestCompression.jpgRatio.toFixed(0)
        : latestCompression.jpgRatio.toFixed(2)
      : null

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-950 text-slate-100">
      <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute left-1/2 top-[-30%] h-[560px] w-[560px] -translate-x-1/2 rounded-full bg-sky-500/20 blur-3xl" />
        <div className="absolute bottom-[-20%] right-[-10%] h-[460px] w-[460px] rounded-full bg-cyan-500/15 blur-3xl" />
        <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(15,23,42,0.85),rgba(2,6,23,0.95))]" />
      </div>

      <div className="relative mx-auto flex w-full max-w-6xl flex-col gap-10 px-6 py-12 lg:px-10">
        <header className="space-y-6">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
            <div className="space-y-4">
              <span className="inline-flex w-fit items-center gap-2 rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.3em] text-slate-300/80">
                Client showcase
              </span>
              <h1 className="text-4xl font-semibold tracking-tight text-white sm:text-5xl">
                Image-GS — 2D Gaussian Splatting in the Browser
              </h1>
              <p className="max-w-2xl text-base text-slate-300">
                Upload an image to see a refined Gaussian splat reinterpretation bloom in real time. The entire pipeline runs in pure JavaScript—no WebGL, WASM, or servers required—making it ideal for polished, client-facing showcases.
              </p>
            </div>
            <span className={`${statusBadgeBase} ${statusMeta.classes}`}>
              <span className={`h-2.5 w-2.5 rounded-full ${statusMeta.dot}`} />
              Status: {statusMeta.label}
            </span>
          </div>
          <p className="max-w-3xl text-sm text-slate-400">
            Monitor fidelity and compression in real time with the live analytics dashboard while multi-threaded workers keep playback responsive.
          </p>
        </header>

        <div className="grid gap-10 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
          <section className="space-y-8">
            <div className={glassCardClass}>
              <div className="flex flex-col gap-6">
                <div className="flex flex-wrap items-center gap-3">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFile}
                    className="hidden"
                  />
                  <button
                    type="button"
                    className={secondaryButton}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Upload image
                  </button>
                  <button
                    className={primaryButton}
                    onClick={train}
                    disabled={!imgJS || status === 'training'}
                  >
                    {status === 'training' ? 'Training…' : 'Train'}
                  </button>
                  <button className={secondaryButton} onClick={handleRenderOnce} disabled={!imgJS}>
                    Render once
                  </button>
                  <button className={ghostButton} onClick={handleStop}>
                    Stop
                  </button>
                  <button
                    className={ghostButton}
                    onClick={handleSaveModel}
                    disabled={!imgJS || !varsRef.current}
                  >
                    Save .igs
                  </button>
                </div>
                <p className="text-sm text-slate-400">
                  Training orchestrates a worker pool tuned to each visitor’s hardware, delivering smooth, reliable updates. Capture snapshots at any moment or export the learned splats as a compact <code className="rounded bg-white/10 px-1 py-0.5 text-xs">.igs</code> file.
                </p>
              </div>
            </div>

            <div className={glassCardClass}>
              <div className="flex flex-col gap-6">
                <div>
                  <h2 className="text-lg font-semibold text-white">Training parameters</h2>
                  <p className="mt-2 text-sm text-slate-400">
                    Refine the optimization budget, tiling strategy, and learning rates to balance speed and fidelity.
                  </p>
                </div>

                <div className="space-y-6">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Core setup
                    </h3>
                    <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">K</span>
                          <InfoPopover
                            title="Nearest splats (K)"
                            description="Controls how many Gaussians blend together for each pixel. Higher values can sharpen detail at the cost of more computation."
                          />
                        </div>
                        <input
                          type="number"
                          value={K}
                          min={1}
                          max={64}
                          onChange={(e) => setK(parseInt(e.target.value, 10) || 1)}
                          className={inputClass}
                        />
                      </label>
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Budget (N)</span>
                          <InfoPopover
                            title="Gaussian budget"
                            description="Sets the maximum number of splats the optimizer can allocate. Larger budgets capture more nuance but require more memory and processing."
                          />
                        </div>
                        <input
                          type="number"
                          value={budget}
                          min={128}
                          max={20000}
                          onChange={(e) => setBudget(parseInt(e.target.value, 10) || 128)}
                          className={inputClass}
                        />
                      </label>
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">λ_init</span>
                          <InfoPopover
                            title="Initialization blend"
                            description="Balances gradient-based seeding with a uniform distribution. Higher values produce broader coverage before training begins."
                          />
                        </div>
                        <input
                          type="number"
                          step="0.05"
                          value={lambdaInit}
                          min={0}
                          max={1}
                          onChange={(e) => setLambdaInit(parseFloat(e.target.value) || 0)}
                          className={inputClass}
                        />
                      </label>
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Steps</span>
                          <InfoPopover
                            title="Optimization steps"
                            description="Defines how many EM iterations run during training. More steps yield higher fidelity until convergence."
                          />
                        </div>
                        <input
                          type="number"
                          value={steps}
                          min={100}
                          max={20000}
                          onChange={(e) => setSteps(parseInt(e.target.value, 10) || 100)}
                          className={inputClass}
                        />
                      </label>
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Delay (ms)</span>
                          <InfoPopover
                            title="Step delay"
                            description="Adds a pause between iterations in milliseconds. Useful for showcasing the progression at a relaxed pace."
                          />
                        </div>
                        <input
                          type="number"
                          value={stepDelayMs}
                          min={0}
                          max={1000}
                          onChange={(e) => setStepDelayMs(parseInt(e.target.value, 10) || 0)}
                          className={inputClass}
                        />
                      </label>
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Add splats every</span>
                          <InfoPopover
                            title="Auto-growth cadence"
                            description="Determines how often new Gaussians are introduced based on reconstruction error. Set to 0 to keep the count fixed."
                          />
                        </div>
                        <input
                          type="number"
                          value={addEvery}
                          min={0}
                          max={5000}
                          onChange={(e) => {
                            const v = parseInt(e.target.value, 10)
                            setAddEvery(Number.isFinite(v) ? Math.max(0, v) : 0)
                          }}
                          className={inputClass}
                        />
                      </label>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Worker pool
                    </h3>
                    <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Pool size</span>
                          <InfoPopover
                            title="Worker pool"
                            description="Specifies how many web workers optimize the image in parallel. Match this to the number of CPU cores available for smoother playback."
                          />
                        </div>
                        <input
                          type="number"
                          value={poolSize}
                          min={1}
                          max={16}
                          onChange={(e) => setPoolSize(clamp(parseInt(e.target.value, 10) || 1, 1, 16))}
                          className={inputClass}
                        />
                      </label>
                      <label className="flex flex-col gap-2 text-sm text-slate-300 sm:col-span-2 xl:col-span-1">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Tiling</span>
                          <InfoPopover
                            title="Adaptive tiling"
                            description="Caches Gaussians into spatial bins so each worker touches fewer splats per pixel. Disable to profile the raw solver."
                          />
                        </div>
                        <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-slate-950/50 px-4 py-3">
                          <input
                            type="checkbox"
                            checked={enableTiling}
                            onChange={(e) => {
                              setEnableTiling(e.target.checked)
                              broadcastSetTiling()
                            }}
                            className="h-5 w-5 rounded border-white/30 bg-slate-900/70 accent-sky-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-1 focus-visible:ring-sky-400/60 focus-visible:ring-offset-slate-950"
                          />
                          <span className="text-sm text-slate-300">
                            {enableTiling ? 'Enabled' : 'Disabled'}
                          </span>
                        </div>
                      </label>
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Tile W</span>
                          <InfoPopover
                            title="Tile width"
                            description="Width of each tiling bucket in pixels. Smaller tiles improve accuracy but require more bin rebuilds."
                          />
                        </div>
                        <input
                          type="number"
                          value={tileW}
                          min={8}
                          max={128}
                          onChange={(e) => {
                            const v = parseInt(e.target.value, 10) || 32
                            setTileW(v)
                            broadcastSetTiling()
                          }}
                          className={inputClass}
                        />
                      </label>
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Tile H</span>
                          <InfoPopover
                            title="Tile height"
                            description="Height of each tiling bucket in pixels. Tune alongside tile width for the desired performance profile."
                          />
                        </div>
                        <input
                          type="number"
                          value={tileH}
                          min={8}
                          max={128}
                          onChange={(e) => {
                            const v = parseInt(e.target.value, 10) || 32
                            setTileH(v)
                            broadcastSetTiling()
                          }}
                          className={inputClass}
                        />
                      </label>
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Rebin every</span>
                          <InfoPopover
                            title="Rebin cadence"
                            description="How frequently (in steps) the tiling bins are rebuilt. Set to 0 to reuse bins indefinitely."
                          />
                        </div>
                        <input
                          type="number"
                          value={rebinEvery}
                          min={0}
                          max={500}
                          onChange={(e) => setRebinEvery(parseInt(e.target.value, 10) || 0)}
                          className={inputClass}
                        />
                      </label>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Learning rates
                    </h3>
                    <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">LR Color</span>
                          <InfoPopover
                            title="Color learning rate"
                            description="Controls how quickly Gaussian colors adapt toward the target image. Lower values smooth the transition; higher values react faster."
                          />
                        </div>
                        <input
                          type="number"
                          step="0.05"
                          value={lrColor}
                          min={0.05}
                          max={1}
                          onChange={(e) => setLrColor(parseFloat(e.target.value) || 0.4)}
                          className={inputClass}
                        />
                      </label>
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">LR μ</span>
                          <InfoPopover
                            title="Position learning rate"
                            description="Adjusts how rapidly Gaussian centers move toward the weighted average of the pixels they explain."
                          />
                        </div>
                        <input
                          type="number"
                          step="0.05"
                          value={lrMu}
                          min={0.05}
                          max={1}
                          onChange={(e) => setLrMu(parseFloat(e.target.value) || 0.25)}
                          className={inputClass}
                        />
                      </label>
                      <label className="flex flex-col gap-2 text-sm text-slate-300">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">LR shape</span>
                          <InfoPopover
                            title="Shape learning rate"
                            description="Controls the update speed for Gaussian covariance and orientation. Lower settings stabilize anisotropic splats."
                          />
                        </div>
                        <input
                          type="number"
                          step="0.05"
                          value={lrShape}
                          min={0.05}
                          max={1}
                          onChange={(e) => setLrShape(parseFloat(e.target.value) || 0.25)}
                          className={inputClass}
                        />
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className={glassCardClass}>
              <div className="flex flex-col gap-6">
                <div className="flex flex-col gap-2">
                  <h2 className="text-lg font-semibold text-white">Live preview</h2>
                  <p className="text-sm text-slate-400">
                    Workers composite a new frame roughly every 100&nbsp;ms. Capture progress or inspect the final result below.
                  </p>
                </div>
                <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                  <div className={metricCardClass}>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Image</p>
                    <p className="mt-1 text-lg font-semibold text-white">{imageDimensions}</p>
                  </div>
                  <div className={metricCardClass}>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Gaussians</p>
                    <p className="mt-1 text-lg font-semibold text-white">{gaussianDisplay}</p>
                  </div>
                  <div className={metricCardClass}>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Step</p>
                    <p className="mt-1 text-lg font-semibold text-white">{stepDisplay}</p>
                  </div>
                  <div className={metricCardClass}>
                    <p className="text-xs uppercase tracking-wide text-slate-400">PSNR</p>
                    <p className="mt-1 text-lg font-semibold text-white">
                      {psnrDisplay}
                      {psnrDisplay !== '—' && (
                        <span className="ml-1 text-xs font-medium text-slate-400">dB</span>
                      )}
                    </p>
                  </div>
                  <div className={metricCardClass}>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Worker pool</p>
                    <p className="mt-1 text-lg font-semibold text-white">{poolDisplay}</p>
                  </div>
                  <div className={metricCardClass}>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Model size</p>
                    <p className="mt-1 text-lg font-semibold text-white">{modelSizeDisplay}</p>
                  </div>
                </div>
                <div className="overflow-hidden rounded-3xl border border-white/10 bg-slate-950/70 shadow-[0_25px_50px_-12px_rgba(15,23,42,0.8)] ring-1 ring-white/10">
                  <canvas ref={canvasRef} className="block h-auto w-full" />
                </div>
              </div>
            </div>
          </section>

          <aside className="space-y-8">
            <div className={glassCardClass}>
              <div className="flex flex-col gap-6">
                <div>
                  <h2 className="text-lg font-semibold text-white">PSNR over time</h2>
                  <p className="mt-2 text-sm text-slate-400">
                    Track reconstruction fidelity as the optimizer progresses. Measurements update whenever a new frame is rendered.
                  </p>
                </div>
                <TimeSeriesChart
                  series={[
                    {
                      label: 'PSNR (dB)',
                      color: '#38bdf8',
                      data: psnrHistory.map((entry) => ({ x: entry.time, y: entry.psnr })),
                    },
                  ]}
                  xLabel="Time (s)"
                  yLabel="PSNR (dB)"
                  xFormatter={(value) => `${value.toFixed(1)}s`}
                  yFormatter={(value) => value.toFixed(1)}
                  emptyLabel="PSNR measurements will appear once training produces frames."
                />
                {latestPsnr ? (
                  <p className="text-xs text-slate-400">
                    Latest: step {latestPsnr.step.toLocaleString()} — {latestPsnr.psnr.toFixed(2)} dB
                  </p>
                ) : (
                  <p className="text-xs text-slate-500">Load an image and start training to monitor PSNR.</p>
                )}
              </div>
            </div>

            <div className={glassCardClass}>
              <div className="flex flex-col gap-6">
                <div>
                  <h2 className="text-lg font-semibold text-white">Compression comparison</h2>
                  <p className="mt-2 text-sm text-slate-400">
                    Compare the learned splat payload against PNG and JPEG encodings of the same resolution. Ratios refresh in real time as new splats are added.
                  </p>
                </div>
                <TimeSeriesChart
                  series={[
                    {
                      label: 'PNG / .igs',
                      color: '#a855f7',
                      data: compressionHistory.map((entry) => ({ x: entry.time, y: entry.pngRatio })),
                    },
                    {
                      label: 'JPEG / .igs',
                      color: '#f97316',
                      data: compressionHistory.map((entry) => ({ x: entry.time, y: entry.jpgRatio })),
                    },
                  ]}
                  xLabel="Time (s)"
                  yLabel="Compression ratio"
                  xFormatter={(value) => `${value.toFixed(1)}s`}
                  yFormatter={(value) => (Math.abs(value) >= 10 ? value.toFixed(0) : value.toFixed(2))}
                  emptyLabel={
                    baselineSizes.png == null || baselineSizes.jpg == null
                      ? 'Preparing PNG/JPEG baselines…'
                      : 'Start training to compare compression ratios.'
                  }
                />
                <div className="flex flex-wrap gap-4 text-xs text-slate-400">
                  <span>Model size: {modelSizeDisplay}</span>
                  <span>
                    PNG: {formatBytes(baselineSizes.png)}
                    {pngRatioDisplay ? ` (${pngRatioDisplay}×)` : ''}
                  </span>
                  <span>
                    JPEG: {formatBytes(baselineSizes.jpg)}
                    {jpgRatioDisplay ? ` (${jpgRatioDisplay}×)` : ''}
                  </span>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  )
}

function estimateIgsSizeBytes(vars, size) {
  if (!vars || !size?.w || !size?.h) return null
  const { mu, s_inv: sInv, theta, color } = vars
  if (!mu || !sInv || !theta || !color) return null
  const N = mu.length / 2
  if (!Number.isFinite(N) || N <= 0) return null
  let headerBytes = 0
  try {
    const headerStr = JSON.stringify({ magic: 'IGS1', H: size.h, W: size.w, C: 3, N }) + '\n'
    if (typeof TextEncoder !== 'undefined') {
      headerBytes = new TextEncoder().encode(headerStr).length
    } else {
      headerBytes = headerStr.length
    }
  } catch (err) {
    console.warn('estimateIgsSizeBytes header encode failed', err)
    headerBytes = 0
  }
  const totalValues = (mu.length || 0) + (sInv.length || 0) + (theta.length || 0) + (color.length || 0)
  return headerBytes + totalValues * 2
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '—'
  if (bytes <= 0) return bytes === 0 ? '0 B' : '—'
  if (bytes < 1024) return `${bytes.toFixed(0)} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function floatImageToImageData(image) {
  if (!image) return null
  const { data, w, h } = image
  if (!data || !Number.isFinite(w) || !Number.isFinite(h)) return null
  const width = Math.max(1, Math.floor(w))
  const height = Math.max(1, Math.floor(h))
  const out = new Uint8ClampedArray(width * height * 4)
  for (let i = 0, j = 0; i < width * height; i++) {
    const r = data[i * 3 + 0] ?? 0
    const g = data[i * 3 + 1] ?? 0
    const b = data[i * 3 + 2] ?? 0
    out[j++] = clamp(Math.round(r * 255), 0, 255)
    out[j++] = clamp(Math.round(g * 255), 0, 255)
    out[j++] = clamp(Math.round(b * 255), 0, 255)
    out[j++] = 255
  }
  try {
    return new ImageData(out, width, height)
  } catch (err) {
    console.error('floatImageToImageData failed', err)
    return null
  }
}

async function encodeImageFromImageData(imageData, type, quality) {
  if (!imageData || typeof document === 'undefined') return null
  return new Promise((resolve) => {
    try {
      const canvas = document.createElement('canvas')
      canvas.width = imageData.width
      canvas.height = imageData.height
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        resolve(null)
        return
      }
      ctx.putImageData(imageData, 0, 0)
      if (typeof canvas.toBlob === 'function') {
        canvas.toBlob(
          (blob) => {
            resolve(blob?.size ?? null)
          },
          type,
          quality,
        )
      } else {
        try {
          const dataUrl = canvas.toDataURL(type, quality)
          const comma = dataUrl.indexOf(',')
          if (comma >= 0) {
            const base64 = dataUrl.slice(comma + 1)
            const padding = base64.endsWith('==') ? 2 : base64.endsWith('=') ? 1 : 0
            resolve(Math.floor((base64.length * 3) / 4) - padding)
          } else {
            resolve(null)
          }
        } catch (err) {
          console.error('encodeImageFromImageData fallback failed', err)
          resolve(null)
        }
      }
    } catch (err) {
      console.error('encodeImageFromImageData failed', err)
      resolve(null)
    }
  })
}

function TimeSeriesChart({
  series = [],
  xLabel,
  yLabel,
  xFormatter,
  yFormatter,
  emptyLabel,
}) {
  const processedSeries = series.map((s, idx) => {
    const points = (s.data ?? [])
      .map((d) => ({ x: Number(d.x), y: Number(d.y) }))
      .filter((d) => Number.isFinite(d.x) && Number.isFinite(d.y))
      .sort((a, b) => a.x - b.x)
    return { ...s, points, _idx: idx }
  })
  const allPoints = processedSeries.flatMap((s) => s.points)
  if (!allPoints.length) {
    return (
      <div className="flex h-48 items-center justify-center rounded-2xl border border-white/10 bg-slate-950/40 text-sm text-slate-500">
        {emptyLabel || 'No data yet.'}
      </div>
    )
  }

  const viewWidth = 640
  const viewHeight = 260
  const margin = { top: 20, right: 16, bottom: 42, left: 60 }
  const innerWidth = viewWidth - margin.left - margin.right
  const innerHeight = viewHeight - margin.top - margin.bottom

  const xValues = allPoints.map((p) => p.x)
  const yValues = allPoints.map((p) => p.y)

  const actualXMin = Math.min(...xValues)
  const actualXMax = Math.max(...xValues)
  let xMin = Math.min(0, actualXMin)
  let xMax = actualXMax
  if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) {
    return (
      <div className="flex h-48 items-center justify-center rounded-2xl border border-white/10 bg-slate-950/40 text-sm text-slate-500">
        {emptyLabel || 'No data yet.'}
      </div>
    )
  }
  if (xMax === xMin) xMax = xMin + 1

  const actualYMin = Math.min(...yValues)
  const actualYMax = Math.max(...yValues)
  let yMin = actualYMin
  let yMax = actualYMax
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    return (
      <div className="flex h-48 items-center justify-center rounded-2xl border border-white/10 bg-slate-950/40 text-sm text-slate-500">
        {emptyLabel || 'No data yet.'}
      </div>
    )
  }
  if (yMax === yMin) {
    const delta = Math.abs(yMin) || 1
    yMin -= delta * 0.5
    yMax += delta * 0.5
  } else {
    const pad = (yMax - yMin) * 0.08
    yMin -= pad
    yMax += pad
    if (actualYMin >= 0 && yMin < 0) yMin = 0
  }

  const xScale = (value) => margin.left + ((value - xMin) / (xMax - xMin)) * innerWidth
  const yScale = (value) => margin.top + innerHeight - ((value - yMin) / (yMax - yMin)) * innerHeight

  const xTicks = linearTicks(xMin, xMax, 4)
  const yTicks = linearTicks(yMin, yMax, 4)

  const axisColor = 'rgba(148, 163, 184, 0.4)'
  const gridColor = 'rgba(148, 163, 184, 0.15)'
  const labelColor = '#cbd5f5'
  const tickLabelColor = '#94a3b8'

  return (
    <div className="flex flex-col gap-4">
      <svg viewBox={`0 0 ${viewWidth} ${viewHeight}`} className="h-56 w-full" preserveAspectRatio="none">
        {yTicks.map((tick) => {
          const y = yScale(tick)
          return (
            <line
              key={`grid-y-${tick}`}
              x1={margin.left}
              x2={viewWidth - margin.right}
              y1={y}
              y2={y}
              stroke={gridColor}
              strokeWidth={1}
            />
          )
        })}
        {xTicks.map((tick) => {
          const x = xScale(tick)
          return (
            <line
              key={`grid-x-${tick}`}
              x1={x}
              x2={x}
              y1={margin.top}
              y2={viewHeight - margin.bottom}
              stroke={gridColor}
              strokeWidth={1}
            />
          )
        })}
        <line
          x1={margin.left}
          x2={margin.left}
          y1={margin.top}
          y2={viewHeight - margin.bottom}
          stroke={axisColor}
          strokeWidth={1.5}
        />
        <line
          x1={margin.left}
          x2={viewWidth - margin.right}
          y1={viewHeight - margin.bottom}
          y2={viewHeight - margin.bottom}
          stroke={axisColor}
          strokeWidth={1.5}
        />
        {xTicks.map((tick) => {
          const x = xScale(tick)
          return (
            <g key={`tick-x-${tick}`}>
              <line
                x1={x}
                x2={x}
                y1={viewHeight - margin.bottom}
                y2={viewHeight - margin.bottom + 6}
                stroke={axisColor}
                strokeWidth={1}
              />
              <text
                x={x}
                y={viewHeight - margin.bottom + 20}
                fill={tickLabelColor}
                fontSize={12}
                textAnchor="middle"
              >
                {typeof xFormatter === 'function' ? xFormatter(tick) : tick.toFixed(1)}
              </text>
            </g>
          )
        })}
        {yTicks.map((tick) => {
          const y = yScale(tick)
          return (
            <g key={`tick-y-${tick}`}>
              <line
                x1={margin.left - 6}
                x2={margin.left}
                y1={y}
                y2={y}
                stroke={axisColor}
                strokeWidth={1}
              />
              <text
                x={margin.left - 10}
                y={y + 4}
                fill={tickLabelColor}
                fontSize={12}
                textAnchor="end"
              >
                {typeof yFormatter === 'function' ? yFormatter(tick) : tick.toFixed(1)}
              </text>
            </g>
          )
        })}
        {xLabel && (
          <text
            x={(margin.left + viewWidth - margin.right) / 2}
            y={viewHeight - 6}
            fill={labelColor}
            fontSize={12}
            textAnchor="middle"
          >
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text
            x={16}
            y={(margin.top + viewHeight - margin.bottom) / 2}
            fill={labelColor}
            fontSize={12}
            textAnchor="middle"
            transform={`rotate(-90 16 ${(margin.top + viewHeight - margin.bottom) / 2})`}
          >
            {yLabel}
          </text>
        )}
        {processedSeries.map((seriesItem) => {
          if (!seriesItem.points.length) return null
          const d = seriesItem.points
            .map((pt, idx) => `${idx === 0 ? 'M' : 'L'}${xScale(pt.x)} ${yScale(pt.y)}`)
            .join(' ')
          const lastPoint = seriesItem.points[seriesItem.points.length - 1]
          return (
            <g key={seriesItem.label ?? seriesItem._idx}>
              <path
                d={d}
                fill="none"
                stroke={seriesItem.color || '#38bdf8'}
                strokeWidth={2.4}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              {lastPoint ? (
                <circle
                  cx={xScale(lastPoint.x)}
                  cy={yScale(lastPoint.y)}
                  r={3.6}
                  fill={seriesItem.color || '#38bdf8'}
                />
              ) : null}
            </g>
          )
        })}
      </svg>
      <div className="flex flex-wrap gap-4 text-xs text-slate-400">
        {series.map((s, idx) => {
          const hasData = processedSeries[idx]?.points?.length > 0
          return (
            <span key={s.label ?? idx} className={`flex items-center gap-2 ${hasData ? 'text-slate-300' : 'text-slate-600'}`}>
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{
                  backgroundColor: s.color || '#38bdf8',
                  opacity: hasData ? 1 : 0.35,
                }}
              />
              {s.label}
            </span>
          )
        })}
      </div>
    </div>
  )
}

function linearTicks(min, max, count) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return []
  if (count <= 0 || min === max) return [min]
  const step = (max - min) / Math.max(count, 1)
  const ticks = []
  for (let i = 0; i <= count; i++) ticks.push(min + step * i)
  return ticks
}
