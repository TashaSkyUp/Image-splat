import { useEffect, useId, useRef, useState } from 'react'

export default function InfoPopover({ title, description }) {
  const [open, setOpen] = useState(false)
  const containerRef = useRef(null)
  const tooltipId = useId()

  useEffect(() => {
    if (!open) return
    const handleClick = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setOpen(false)
      }
    }
    const handleKey = (event) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('mousedown', handleClick)
      document.removeEventListener('keydown', handleKey)
    }
  }, [open])

  return (
    <div ref={containerRef} className="relative inline-flex">
      <button
        type="button"
        aria-label={`More info about ${title}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? tooltipId : undefined}
        onClick={() => setOpen((prev) => !prev)}
        className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-white/20 bg-white/10 text-[10px] font-semibold text-slate-200 transition hover:border-sky-400 hover:text-sky-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-1 focus-visible:ring-offset-slate-950"
      >
        i
      </button>
      {open && (
        <div
          role="dialog"
          id={tooltipId}
          aria-modal="false"
          className="absolute right-0 top-full z-40 mt-2 w-64 rounded-2xl border border-white/10 bg-slate-950/95 p-4 text-left shadow-xl shadow-slate-950/60"
        >
          <p className="text-sm font-semibold text-white">{title}</p>
          <p className="mt-2 text-xs leading-relaxed text-slate-300">{description}</p>
        </div>
      )}
    </div>
  )
}
