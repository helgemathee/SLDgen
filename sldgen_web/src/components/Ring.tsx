import { ringGeometry, ringPoint, type RingInput } from '../lib/ring'

/**
 * The status iconography of the whole application (Spec 3 SS5): a progress ring
 * whose filled arc is progress against the horizon and whose tick is the budget
 * this run is headed for.
 */
export function Ring({ size = 20, ...input }: RingInput & { size?: number }) {
  const geometry = ringGeometry({ ...input, radius: input.radius ?? size / 2 - 2 })
  const center = size / 2
  const tick = ringPoint(geometry.radius, geometry.tickAngle, center)
  const tickOuter = ringPoint(geometry.radius + 3, geometry.tickAngle, center)
  const lead = ringPoint(geometry.radius, geometry.leadingAngle, center)

  const label =
    input.state === 'waiting'
      ? `waiting at ${input.currentEpoch} of ${input.numIter}`
      : `${input.state}, ${input.currentEpoch} of ${input.numIter}`

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label={label}
      style={{ opacity: geometry.opacity, flex: '0 0 auto', display: 'block' }}
    >
      <circle
        cx={center}
        cy={center}
        r={geometry.radius}
        fill="none"
        stroke="var(--rule-strong)"
        strokeWidth={input.state === 'waiting' ? 2 : 1}
      />
      <circle
        cx={center}
        cy={center}
        r={geometry.radius}
        fill="none"
        stroke={geometry.colorVar}
        strokeWidth={input.state === 'waiting' ? 3 : 2.5}
        strokeDasharray={geometry.strokeDash ?? geometry.dashArray}
        strokeLinecap="butt"
        transform={`rotate(-90 ${center} ${center})`}
        className={geometry.animated ? 'spin' : undefined}
        // A running job's arc advances with real progress; the slow rotation is
        // the only continuous motion in the interface and it means "on the GPU".
        style={geometry.animated ? { transformOrigin: '50% 50%' } : undefined}
      />
      {/* Where this run is headed. Sitting at the arc's leading edge is what
          "complete against its budget, incomplete against the horizon" looks
          like, and it needs no words. */}
      <line
        x1={tick.x}
        y1={tick.y}
        x2={tickOuter.x}
        y2={tickOuter.y}
        stroke={geometry.colorVar}
        strokeWidth={1.5}
      />
      {input.state === 'running' && (
        <circle cx={lead.x} cy={lead.y} r={1.8} fill={geometry.colorVar} />
      )}
      {input.state === 'failed' && (
        <line
          x1={center - geometry.radius * 0.7}
          y1={center - geometry.radius * 0.7}
          x2={center + geometry.radius * 0.7}
          y2={center + geometry.radius * 0.7}
          stroke="var(--st-failed)"
          strokeWidth={1.5}
        />
      )}
    </svg>
  )
}
