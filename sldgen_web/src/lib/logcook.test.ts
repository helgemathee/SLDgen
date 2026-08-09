import { describe, expect, it } from 'vitest'
import { cook, cookLine, filterLog, highlightParts } from './logcook'

/**
 * These cases mirror `sldgen_service/logs.py`. The two implementations must
 * agree: the API cooks whole-file downloads, the browser cooks the incrementally
 * fetched buffer, and a difference between them would show up as a log that
 * changes when you press "raw" twice.
 */
describe('cookLine', () => {
  it('overlays rather than discarding, exactly as a terminal would', () => {
    expect(cookLine('abcdef\rXY')).toBe('XYcdef')
  })

  it('leaves a line without carriage returns untouched', () => {
    expect(cookLine('plain output')).toBe('plain output')
  })

  it('collapses a full tqdm repaint to its final state', () => {
    const repaints = ' 10%|# | 400/4000\r 20%|## | 800/4000\r 30%|### | 1200/4000'
    expect(cookLine(repaints)).toBe(' 30%|### | 1200/4000')
  })

  it('keeps the tail of a longer previous paint that a shorter one did not cover', () => {
    expect(cookLine('123456789\rabc')).toBe('abc456789')
  })

  it('handles a leading carriage return', () => {
    expect(cookLine('\rXY')).toBe('XY')
  })

  it('handles a trailing carriage return', () => {
    expect(cookLine('abc\r')).toBe('abc')
  })
})

describe('cook', () => {
  it('cooks each line independently and preserves the line structure', () => {
    expect(cook('one\ntwo\rTWO\nthree')).toBe('one\nTWO\nthree')
  })

  it('preserves empty lines', () => {
    expect(cook('a\n\nb')).toBe('a\n\nb')
  })

  it('is idempotent -- cooking cooked output changes nothing', () => {
    const once = cook('abcdef\rXY\nplain')
    expect(cook(once)).toBe(once)
  })

  it('cooks correctly across what would have been a chunk boundary', () => {
    // This is the case the client-side implementation exists for: the server
    // cooks the chunk it was asked for, and a progress line spanning two chunks
    // would cook wrong if each half were cooked alone.
    const wholeBuffer = ' 10%| 400/4000\r 20%| 800/4000'
    const chunkA = ' 10%| 400/4000'
    const chunkB = '\r 20%| 800/4000'
    expect(cook(wholeBuffer)).toBe(' 20%| 800/4000')
    expect(cook(chunkA) + cook(chunkB)).not.toBe(cook(wholeBuffer))
  })
})

describe('filterLog', () => {
  const text = 'starting up\nCUDA out of memory\nDone!'

  it('returns everything for an empty filter', () => {
    expect(filterLog(text, '')).toBe(text)
    expect(filterLog(text, '   ')).toBe(text)
  })

  it('keeps only matching lines, case-insensitively', () => {
    expect(filterLog(text, 'cuda')).toBe('CUDA out of memory')
  })

  it('returns nothing when there is no match', () => {
    expect(filterLog(text, 'zzz')).toBe('')
  })
})

describe('highlightParts', () => {
  it('returns the whole line as one part with no needle', () => {
    expect(highlightParts('hello', '')).toEqual(['hello'])
  })

  it('alternates plain and matched, starting with plain', () => {
    expect(highlightParts('a-bug-and-a-bug', 'bug')).toEqual(['a-', 'bug', '-and-a-', 'bug', ''])
  })

  it('preserves the original casing of a case-insensitive match', () => {
    expect(highlightParts('CUDA error', 'cuda')).toEqual(['', 'CUDA', ' error'])
  })

  it('reassembles to the original line', () => {
    const line = 'epoch 400 of 4000, loss 0.4'
    expect(highlightParts(line, '0').join('')).toBe(line)
  })
})
