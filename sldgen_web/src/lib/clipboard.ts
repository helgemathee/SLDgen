/**
 * Copy text to the clipboard, including from an insecure origin.
 *
 * `navigator.clipboard` exists only in a secure context: HTTPS, or the
 * localhost exemption. The service is plain HTTP on whatever tailnet and LAN
 * addresses start.sh resolved, so every browser that is not on the box itself
 * has no async clipboard at all -- undefined, not merely unpermitted. That is
 * the normal way this UI is used, not an edge case, so the deprecated
 * execCommand path below is the one that usually runs.
 *
 * Returns whether the text actually made it. Callers are expected to say so
 * either way; a copy button that silently does nothing is the bug this
 * replaces.
 */
export async function copyText(text: string): Promise<boolean> {
  if (navigator.clipboard) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      // Denied, or the document was not focused. The fallback often still
      // works, so try it rather than reporting failure here.
    }
  }
  return legacyCopy(text)
}

function legacyCopy(text: string): boolean {
  const area = document.createElement('textarea')
  area.value = text
  // execCommand ignores a textarea that is not rendered, so it goes off-screen
  // rather than display:none. readOnly keeps a mobile keyboard from appearing
  // for the instant it exists.
  area.readOnly = true
  area.style.position = 'fixed'
  area.style.top = '0'
  area.style.left = '-9999px'
  document.body.appendChild(area)

  // Copying steals the selection, so put back whatever the reader had
  // highlighted -- plausibly the command itself, out of the panel below.
  const selection = document.getSelection()
  const previous = selection && selection.rangeCount > 0 ? selection.getRangeAt(0) : null

  try {
    area.select()
    area.setSelectionRange(0, text.length)
    return document.execCommand('copy')
  } catch {
    return false
  } finally {
    area.remove()
    if (selection && previous) {
      selection.removeAllRanges()
      selection.addRange(previous)
    }
  }
}
