/**
 * Carriage-return cooking, client side.
 *
 * The API cooks too (Spec 2 SS13.2) and is the authority on what the bytes mean,
 * but it cooks *the chunk it was asked for*. The log viewer fetches
 * incrementally by byte offset, and a tqdm progress line routinely spans a chunk
 * boundary — so cooking each chunk in isolation and concatenating the results
 * produces a line the terminal would never show.
 *
 * The viewer therefore fetches `raw=true`, accumulates, and cooks the whole
 * buffer here. That also makes the raw/cooked toggle instant instead of a
 * refetch. This function must agree with `sldgen_service/logs.py`; the vitest
 * suite asserts the same cases the Python tests use.
 */

/** `"abcdef\rXY"` becomes `"XYcdef"`, exactly as a terminal would render it. */
export function cookLine(line: string): string {
  if (!line.includes('\r')) return line
  let out = ''
  for (const fragment of line.split('\r')) {
    // Overlay rather than replace: a shorter repaint must not silently truncate
    // what it did not cover.
    out = fragment + out.slice(fragment.length)
  }
  return out
}

export function cook(text: string): string {
  return text.split('\n').map(cookLine).join('\n')
}

/** Lines matching a filter, keeping the original text when the filter is empty. */
export function filterLog(text: string, needle: string): string {
  const trimmed = needle.trim().toLowerCase()
  if (!trimmed) return text
  return text
    .split('\n')
    .filter((line) => line.toLowerCase().includes(trimmed))
    .join('\n')
}

/**
 * Split a line around every match, so the viewer can mark them.
 *
 * Returned as alternating [plain, match, plain, match, …] starting with plain,
 * which is the shape a renderer can map over without tracking parity.
 */
export function highlightParts(line: string, needle: string): string[] {
  const trimmed = needle.trim()
  if (!trimmed) return [line]
  const parts: string[] = []
  const haystack = line.toLowerCase()
  const target = trimmed.toLowerCase()
  let cursor = 0
  for (;;) {
    const found = haystack.indexOf(target, cursor)
    if (found === -1) break
    parts.push(line.slice(cursor, found), line.slice(found, found + target.length))
    cursor = found + target.length
  }
  parts.push(line.slice(cursor))
  return parts
}
