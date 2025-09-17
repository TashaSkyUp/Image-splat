export function sourceToF32RGB(src, targetMax = 512) {
  const c = document.createElement('canvas')
  const w0 = Math.max(1, src.width)
  const h0 = Math.max(1, src.height)
  const scale = Math.min(1, targetMax / Math.max(w0, h0))
  const w = Math.max(1, Math.round(w0 * scale))
  const h = Math.max(1, Math.round(h0 * scale))
  c.width = w
  c.height = h
  const ctx = c.getContext('2d', { willReadFrequently: true })
  if (!ctx) throw new Error('Canvas 2D context unavailable')
  try {
    ctx.imageSmoothingEnabled = true
    if ('imageSmoothingQuality' in ctx) {
      ctx.imageSmoothingQuality = 'high'
    }
    ctx.drawImage(src, 0, 0, w, h)
  } catch (err) {
    console.error('Rasterization failed (drawImage)', err)
    throw new Error('Rasterization failed (drawImage). The file may be corrupt or blocked.')
  }
  let imgData
  try {
    imgData = ctx.getImageData(0, 0, w, h)
  } catch (err) {
    console.error('Reading pixels failed (getImageData)', err)
    throw new Error('Reading pixels failed (getImageData). If this is an SVG with external refs, re-export as PNG/JPG.')
  }
  const data = imgData.data
  const f32 = new Float32Array(w * h * 3)
  for (let i = 0; i < w * h; i++) {
    f32[i * 3 + 0] = data[i * 4 + 0] / 255
    f32[i * 3 + 1] = data[i * 4 + 1] / 255
    f32[i * 3 + 2] = data[i * 4 + 2] / 255
  }
  return { data: f32, w, h }
}

export async function fileToImageSource(file) {
  if ('createImageBitmap' in window && typeof createImageBitmap === 'function') {
    try {
      const bmp = await createImageBitmap(file)
      return { src: bmp, revoke: () => bmp.close?.() }
    } catch (err) {
      console.warn('createImageBitmap failed; falling back to HTMLImageElement', err)
    }
  }
  const dataUrl = await new Promise((resolve, reject) => {
    const fr = new FileReader()
    fr.onerror = () => reject(new Error('FileReader failed'))
    fr.onload = () => resolve(String(fr.result))
    fr.readAsDataURL(file)
  })
  const img = await new Promise((resolve, reject) => {
    const im = new Image()
    im.crossOrigin = 'anonymous'
    im.onload = () => resolve(im)
    im.onerror = () => reject(new Error('Image decode failed'))
    im.src = dataUrl
  })
  return { src: img, revoke: () => {} }
}
