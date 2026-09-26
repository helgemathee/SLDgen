/**
 * Overlaying an artwork tab on the job's input (the artwork pane's "overlay"
 * toggle): which tabs can be overlaid, and how an inlined SVG is made to
 * stretch to the input's box.
 *
 * Every canvas-space artefact (mask, condition, edge target, frames, the final
 * SVG) is drawn at the render size, like input.png; the final PNG is twice
 * that. Stretching the top layer to the input's box therefore lines them up.
 */

/** Tabs that are never overlaid: the input itself, and landmarks (already drawn on it). */
const NO_OVERLAY = new Set(['input', 'landmarks'])

export function canOverlay(tab: string, hasInput: boolean): boolean {
  return hasInput && !NO_OVERLAY.has(tab)
}

/**
 * The SVG markup with a viewBox, so that sizing it to 100% of a box scales the
 * drawing instead of cropping it. The run writes `width`/`height` only.
 */
export function withViewBox(markup: string): string {
  const open = /<svg\b[^>]*>/i.exec(markup)
  if (!open || /\bviewBox\s*=/i.test(open[0])) return markup
  const size = (name: string) => {
    const match = new RegExp(`\\b${name}\\s*=\\s*["']\\s*([\\d.]+)`, 'i').exec(open[0])
    return match ? Number(match[1]) : null
  }
  const width = size('width')
  const height = size('height')
  if (!width || !height) return markup
  const tag = open[0].replace(/^<svg\b/i, `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"`)
  return markup.slice(0, open.index) + tag + markup.slice(open.index + open[0].length)
}

export type OverlayBlend = 'normal' | 'multiply'

/** CSS for the top layer. Multiply lets white paper vanish, so only the lines sit on the photo. */
export function overlayStyle(opacity: number, blend: OverlayBlend): { opacity: number; mixBlendMode: OverlayBlend } {
  return { opacity: Math.min(1, Math.max(0, opacity)), mixBlendMode: blend }
}
