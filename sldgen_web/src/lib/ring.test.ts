import { describe, expect, it } from 'vitest'
import { ringGeometry, ringPoint } from './ring'

describe('ringGeometry', () => {
  it('encodes the partially-completed case the rail exists to show', () => {
    // 400 of a 4000 horizon, budgeted to 400: complete against its budget,
    // incomplete against the horizon. The tick sits exactly at the arc's edge.
    const geometry = ringGeometry({
      state: 'waiting',
      currentEpoch: 400,
      targetEpoch: 400,
      numIter: 4000,
    })
    expect(geometry.progress).toBeCloseTo(0.1, 6)
    expect(geometry.target).toBeCloseTo(0.1, 6)
    expect(geometry.tickAngle).toBeCloseTo(geometry.leadingAngle, 6)
    expect(geometry.colorVar).toBe('var(--st-waiting)')
  })

  it('puts the tick ahead of the arc while a run is under way', () => {
    const geometry = ringGeometry({
      state: 'running',
      currentEpoch: 150,
      targetEpoch: 400,
      numIter: 4000,
    })
    expect(geometry.leadingAngle).toBeLessThan(geometry.tickAngle)
  })

  it('fills the ring for a complete job regardless of rounding', () => {
    const geometry = ringGeometry({
      state: 'complete',
      currentEpoch: 3999,
      targetEpoch: 4000,
      numIter: 4000,
    })
    const [filled, total] = geometry.dashArray.split(' ').map(Number)
    expect(filled).toBeCloseTo(total, 3)
  })

  it('clamps rather than overdrawing when the epoch overshoots', () => {
    const geometry = ringGeometry({
      state: 'running',
      currentEpoch: 5000,
      targetEpoch: 6000,
      numIter: 4000,
    })
    expect(geometry.progress).toBe(1)
    expect(geometry.target).toBe(1)
  })

  it('survives a zero horizon instead of dividing by it', () => {
    const geometry = ringGeometry({
      state: 'queued',
      currentEpoch: 0,
      targetEpoch: 0,
      numIter: 0,
    })
    expect(Number.isFinite(geometry.progress)).toBe(true)
    expect(geometry.progress).toBe(0)
  })

  it('distinguishes paused from queued by dash, since they share a colour', () => {
    const paused = ringGeometry({ state: 'paused', currentEpoch: 10, targetEpoch: 20, numIter: 100 })
    const queued = ringGeometry({ state: 'queued', currentEpoch: 10, targetEpoch: 20, numIter: 100 })
    expect(paused.colorVar).toBe(queued.colorVar)
    expect(paused.strokeDash).toBeDefined()
    expect(queued.strokeDash).toBeUndefined()
  })

  it('dims a job that is being deleted', () => {
    expect(
      ringGeometry({ state: 'deleting', currentEpoch: 1, targetEpoch: 2, numIter: 10 }).opacity,
    ).toBeLessThan(1)
  })
})

describe('ringPoint', () => {
  it('measures clockwise from twelve o clock', () => {
    expect(ringPoint(10, 0, 10)).toMatchObject({ x: expect.closeTo(10, 6), y: expect.closeTo(0, 6) })
    expect(ringPoint(10, 90, 10)).toMatchObject({
      x: expect.closeTo(20, 6),
      y: expect.closeTo(10, 6),
    })
    expect(ringPoint(10, 180, 10)).toMatchObject({
      x: expect.closeTo(10, 6),
      y: expect.closeTo(20, 6),
    })
  })
})
