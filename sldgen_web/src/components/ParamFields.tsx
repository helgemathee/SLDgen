import type { ParamValue, Params } from '../api/types'
import {
  PARAM_SPECS,
  SECTION_LABELS,
  coerceParam,
  type ParamSection,
  type ParamSpec,
} from '../lib/params'

/**
 * The parameter form, shared by the new-job flow (Spec 3 SS8.3) and run-again
 * (SS6.5), so both show the same fields with the same names in the same order.
 *
 * `changedAgainst` marks every field edited relative to a parent job, which is
 * what makes "Queue job (3 changes)" trustworthy: the count and the marks come
 * from the same comparison.
 */
export function ParamFields({
  params,
  onChange,
  sections,
  changedAgainst,
  collapsed = ['losses'],
  hide = [],
}: {
  params: Params
  onChange: (name: string, value: ParamValue) => void
  sections: ParamSection[]
  changedAgainst?: Params
  collapsed?: ParamSection[]
  hide?: string[]
}) {
  return (
    <>
      {sections.map((section) => {
        const specs = PARAM_SPECS.filter(
          (spec) => spec.section === section && !spec.optional && !hide.includes(spec.name),
        )
        if (specs.length === 0) return null
        const changedHere = changedAgainst
          ? specs.filter((spec) => !same(params[spec.name], changedAgainst[spec.name])).length
          : 0
        return (
          <details key={section} className="group" open={!collapsed.includes(section)}>
            <summary>
              <span className="eyebrow">{SECTION_LABELS[section]}</span>
              {changedHere > 0 && <span className="mono">{changedHere} changed</span>}
            </summary>
            <div className="group__body">
              {specs.map((spec) => (
                <Field
                  key={spec.name}
                  spec={spec}
                  value={params[spec.name] ?? null}
                  changed={
                    changedAgainst ? !same(params[spec.name], changedAgainst[spec.name]) : false
                  }
                  onChange={(value) => onChange(spec.name, value)}
                />
              ))}
            </div>
          </details>
        )
      })}
    </>
  )
}

function same(a: ParamValue | undefined, b: ParamValue | undefined): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((value, index) => value === b[index])
  }
  return (a ?? null) === (b ?? null)
}

export function Field({
  spec,
  value,
  changed,
  onChange,
}: {
  spec: ParamSpec
  value: ParamValue
  changed?: boolean
  onChange: (value: ParamValue) => void
}) {
  const boolean = spec.kind === 'true_flag' || spec.kind === 'false_flag'

  return (
    <div className={`field${changed ? ' changed' : ''}`}>
      <label htmlFor={`param-${spec.name}`} title={spec.name}>
        {spec.label}
      </label>
      <div>
        {boolean ? (
          <input
            id={`param-${spec.name}`}
            type="checkbox"
            checked={Boolean(value)}
            onChange={(event) => onChange(event.target.checked)}
          />
        ) : spec.choices ? (
          <select
            id={`param-${spec.name}`}
            value={String(value ?? '')}
            onChange={(event) => onChange(event.target.value)}
          >
            {spec.choices.map((choice) => (
              <option key={choice} value={choice}>
                {choice}
              </option>
            ))}
          </select>
        ) : spec.name === 'caption' ? (
          <textarea
            id={`param-${spec.name}`}
            value={String(value ?? '')}
            placeholder="Leave empty to let BLIP-2 caption the image"
            onChange={(event) => onChange(event.target.value)}
          />
        ) : spec.kind === 'int' || spec.kind === 'float' ? (
          <input
            id={`param-${spec.name}`}
            type="number"
            step={spec.step ?? (spec.kind === 'int' ? 1 : 'any')}
            min={spec.min}
            max={spec.max}
            value={value === null ? '' : String(value)}
            onChange={(event) => onChange(coerceParam(spec, event.target.value))}
          />
        ) : (
          <input
            id={`param-${spec.name}`}
            type="text"
            value={value === null ? '' : String(value)}
            onChange={(event) => onChange(coerceParam(spec, event.target.value))}
          />
        )}
        {spec.hint && <div className="note">{spec.hint}</div>}
      </div>
    </div>
  )
}
