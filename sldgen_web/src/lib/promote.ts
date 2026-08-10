/**
 * Quick-step sizes for the promote control (Spec 3 SS6.3).
 *
 * Promotion is the only operation that extends a job in place, so its steps
 * have to make sense for whatever horizon the job has: fixed +500/+1000 buttons
 * are dead weight on a 100-iteration smoke test, where every press clamps to
 * the same number. These scale to the horizon and drop out entirely once they
 * would overshoot it -- past the horizon there is nothing to promote to.
 */

/** Round up to 1, 2 or 5 times a power of ten, so the buttons read as steps. */
function niceRound(value: number): number {
  if (!(value > 1)) return 1
  const magnitude = 10 ** Math.floor(Math.log10(value))
  for (const factor of [1, 2, 5]) {
    const candidate = factor * magnitude
    if (value <= candidate) return candidate
  }
  return 10 * magnitude
}

export function promoteSteps(numIter: number, currentEpoch: number): number[] {
  const room = numIter - currentEpoch
  if (!(room > 0)) return []
  const steps = [numIter / 20, numIter / 10, numIter / 4].map(niceRound)
  // Strictly less than the room left: a step that lands exactly on the horizon
  // duplicates the "to {horizon}" button that always sits beside these.
  return [...new Set(steps)].filter((step) => step < room).sort((a, b) => a - b)
}
