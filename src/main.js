// UI entrypoint – the worker-based trainer lives in ./training.js
import {
  state,
  configureTrainingCallbacks,
  rebuildPool,
  broadcastSetTiling,
  renderFrameViaPool,
  startTraining,
  stopTraining,
  serializeModel,
  setImageData,
  destroyPool,
  runDiagnosticsOnce,
  getGaussianCount,
} from "./training.js";

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

let canvasEl = null;
let canvasCtx = null;
let diagStatus = "idle";

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

function updateUI() {
  if (ui.trainButton) {
    ui.trainButton.disabled = !state.imgJS || state.status === "training";
    ui.trainButton.textContent = state.status === "training" ? "Training…" : "Train";
  }
  if (ui.renderButton) ui.renderButton.disabled = !state.imgJS;
  if (ui.saveButton) {
    const hasModel = (state.metrics?.n ?? 0) > 0;
    ui.saveButton.disabled = !state.imgJS || !hasModel;
  }
  if (ui.metricsInfo) {
    const gaussians = state.metrics?.n ?? getGaussianCount();
    ui.metricsInfo.textContent = `Image: ${state.imgSize.w}×${state.imgSize.h} • Gaussians: ${gaussians} • Step: ${state.metrics.step} • Pool: ${state.metrics.pool}`;
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
    setImageData({ data: localData, w: js.w, h: js.h });
    drawOnCanvasF32RGB(localData, js.w, js.h);
    updateUI();
  } catch (err) {
    console.error("handleFile failed", err);
    alert(`Failed to load image.\n\nReason: ${err?.message || err}`);
  }
}

async function handleStartTraining() {
  try {
    await startTraining();
  } catch (err) {
    console.error("Training failed", err);
  }
}

async function handleRenderOnce() {
  const out = await renderFrameViaPool(true);
  if (out) drawOnCanvasF32RGB(out, state.imgSize.w, state.imgSize.h);
}

async function handleSaveModel() {
  const payload = serializeModel();
  if (!payload) return;
  const blob = new Blob([payload], { type: "application/octet-stream" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "image_gs.igs";
  a.click();
}

function handleStop() {
  stopTraining();
}

function runDiagnostics() {
  const start = async () => {
    try {
      diagStatus = "running";
      updateUI();
      await runDiagnosticsOnce();
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
  trainButton.addEventListener("click", () => {
    handleStartTraining();
  });
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
  configureTrainingCallbacks({
    onUpdateUI: updateUI,
    onFrame: (out, w, h) => drawOnCanvasF32RGB(out, w, h),
  });
  rebuildPool(state.poolSize);
  window.addEventListener("beforeunload", destroyPool);
  updateUI();
}

window.addEventListener("DOMContentLoaded", initApp);
