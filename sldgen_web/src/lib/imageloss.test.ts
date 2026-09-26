import { describe, expect, it } from 'vitest'
import {
  alphaAt,
  parseImageLossLog,
  scheduleSeries,
  sparkline,
  summarizeImageLoss,
} from './imageloss'

const CSV = [
  'epoch,alpha,sds_norm,img_norm,cosine,skipped,loss_sds,loss_img,chamfer,pyramid,landmark',
  '0,0.5,4,0.02,-0.05,0,249,2.08,2.08,,',
  '1,0.46,48,0.02,0.05,0,13679,1.84,1.84,,',
  '2,0.42,0.3,0.02,0.2,1,988,1.6,1.6,,',
  'garbage',
  '',
].join('\n')

describe('alphaAt', () => {
  it('mirrors the core schedule', () => {
    expect(alphaAt(50, 101, 'constant', 0.3)).toBe(0.3)
    expect(alphaAt(0, 101, 'decay', 0.1)).toBeCloseTo(0.5)
    expect(alphaAt(50, 101, 'decay', 0.1)).toBeCloseTo(0.3)
    expect(alphaAt(100, 101, 'decay', 0.1)).toBeCloseTo(0.1)
    expect(alphaAt(0, 11, 'ramp', 0.3)).toBeCloseTo(0.05)
    expect(alphaAt(5, 11, 'decay', 0.2, 0.8)).toBeCloseTo(0.5)
  })

  it('samples a schedule from start to end', () => {
    const series = scheduleSeries(8000, 'decay', 0.1, 0.5)
    expect(series[0]).toBeCloseTo(0.5)
    expect(series[series.length - 1]).toBeCloseTo(0.1)
  })
})

describe('parseImageLossLog', () => {
  it('parses rows, empty cells as null, drops malformed lines', () => {
    const rows = parseImageLossLog(CSV)
    expect(rows).toHaveLength(3)
    expect(rows[0]).toMatchObject({ epoch: 0, alpha: 0.5, chamfer: 2.08, pyramid: null, landmark: null })
    expect(rows[2].skipped).toBe(1)
  })

  it('returns nothing without a usable header', () => {
    expect(parseImageLossLog('')).toEqual([])
    expect(parseImageLossLog('a,b\n1,2')).toEqual([])
  })
})

describe('summarizeImageLoss', () => {
  it('reports what the diagnostics panel shows', () => {
    const summary = summarizeImageLoss(parseImageLossLog(CSV))
    expect(summary.rows).toBe(3)
    expect(summary.lastEpoch).toBe(2)
    expect(summary.alpha).toBe(0.42)
    expect(summary.cosineMean).toBeCloseTo(0.2 / 3)
    expect(summary.cosineNegativeShare).toBeCloseTo(1 / 3)
    expect(summary.sdsNorm).toEqual({ min: 0.3, median: 4, max: 48 })
    expect(summary.skippedShare).toBeCloseTo(1 / 3)
    expect(summary.latest.chamfer).toBe(1.6)
  })

  it('is empty-safe', () => {
    const summary = summarizeImageLoss([])
    expect(summary.alpha).toBeNull()
    expect(summary.sdsNorm).toBeNull()
  })
})

describe('sparkline', () => {
  it('maps values into the box, low values at the bottom', () => {
    expect(sparkline([0, 1], 10, 10, [0, 1])).toBe('0.0,10.0 10.0,0.0')
  })

  it('skips nulls and caps the vertex count, keeping the last point', () => {
    const values = Array.from({ length: 1000 }, (_unused, index) => (index % 7 ? index : null))
    const points = sparkline(values, 100, 10, undefined, 50).split(' ')
    expect(points.length).toBeLessThanOrEqual(51)
    expect(points[points.length - 1].startsWith('100.0,')).toBe(true)
    expect(sparkline([null, null], 10, 10)).toBe('')
  })
})
