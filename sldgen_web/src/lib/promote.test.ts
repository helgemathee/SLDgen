import { describe, expect, it } from 'vitest'
import { promoteSteps } from './promote'

describe('promoteSteps', () => {
  it('scales to a short horizon instead of offering +500 on a 100-iteration job', () => {
    // The smoke-test case: 80 of 100 done, so only 20 iterations are left.
    expect(promoteSteps(100, 80)).toEqual([5, 10])
  })

  it('offers coarse, round steps on a full-length run', () => {
    // 4000/20, /10, /4 rounded up to 1/2/5 x 10^k.
    expect(promoteSteps(4000, 0)).toEqual([200, 500, 1000])
  })

  it('drops steps that would reach or overshoot the horizon', () => {
    expect(promoteSteps(4000, 3900)).toEqual([])
    // 500 left: the 500 step would land exactly on the horizon, which is what
    // the "to 4000" button beside these already does.
    expect(promoteSteps(4000, 3500)).toEqual([200])
  })

  it('is empty at or past the horizon -- there is nothing to promote to', () => {
    expect(promoteSteps(100, 100)).toEqual([])
    expect(promoteSteps(100, 140)).toEqual([])
  })

  it('never offers a zero step', () => {
    for (const [numIter, epoch] of [
      [10, 0],
      [4, 1],
      [1, 0],
    ]) {
      for (const step of promoteSteps(numIter, epoch)) {
        expect(step).toBeGreaterThan(0)
      }
    }
  })
})
