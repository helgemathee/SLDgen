/** Reconnection timing (Spec 3 SS13).
 *
 * The API is restarted often during development and the UI must survive it
 * without a toast storm. Exponential with full jitter: doubling alone makes a
 * fleet of tabs retry in lockstep, and this app legitimately runs several.
 */
export function backoffDelay(
  attempt: number,
  { base = 500, cap = 15_000, random = Math.random }: {
    base?: number
    cap?: number
    random?: () => number
  } = {},
): number {
  const ceiling = Math.min(cap, base * 2 ** Math.max(0, attempt))
  // Full jitter, floored at base/2 so the first retry is still prompt -- an API
  // restart takes under a second and should not cost a visible stall.
  return Math.max(base / 2, Math.round(random() * ceiling))
}

/** After this many consecutive SSE failures, give up on SSE and poll instead.
 *
 * SSE through a misconfigured proxy fails in ways that are hard to diagnose
 * (SS13), so the UI stops trying rather than reconnecting forever behind a proxy
 * that will never deliver a stream. */
export const SSE_FAILURES_BEFORE_POLLING = 4
