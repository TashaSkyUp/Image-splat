export function float16Quantize(x) {
  const f32 = x instanceof Float32Array ? x : new Float32Array(x)
  const buf = new ArrayBuffer(f32.length * 2)
  const dv = new DataView(buf)
  for (let i = 0; i < f32.length; i++) dv.setUint16(i * 2, float32ToFloat16(f32[i]), true)
  return new Uint8Array(buf)
}

export function float32ToFloat16(val) {
  const f32 = new Float32Array(1)
  const i32 = new Int32Array(f32.buffer)
  f32[0] = val
  const x = i32[0]
  const bits = (x >> 16) & 0x8000
  const m = (x >> 12) & 0x07ff
  const e = (x >> 23) & 0xff
  if (e < 103) return bits
  if (e > 142) return bits | 0x7c00
  return bits | ((e - 112) << 10) | (m >> 1)
}
