import { describe, expect, it } from 'vitest'
import { PARAM_HELP, SECTION_HELP, paramTooltip } from './help'
import { PARAM_SPECS, SECTION_LABELS, SPEC_BY_NAME } from './params'

describe('the hover help', () => {
  it('explains every parameter', () => {
    const missing = PARAM_SPECS.filter((spec) => !PARAM_HELP[spec.name]?.trim()).map((spec) => spec.name)
    expect(missing).toEqual([])
  })

  it('explains nothing that is not a parameter', () => {
    expect(Object.keys(PARAM_HELP).filter((name) => !SPEC_BY_NAME[name])).toEqual([])
  })

  it('explains every section', () => {
    for (const section of Object.keys(SECTION_LABELS)) {
      expect(SECTION_HELP[section as keyof typeof SECTION_HELP]).toBeTruthy()
    }
  })

  it('ends a tooltip with the parameter name', () => {
    expect(paramTooltip(SPEC_BY_NAME.image_loss_chamfer)).toMatch(/\n\n\(image_loss_chamfer\)$/)
  })
})
