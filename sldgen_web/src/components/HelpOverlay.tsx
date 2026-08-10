import { Fragment } from 'react'

const SHORTCUTS: [string, string][] = [
  ['j / k', 'Move through the rail'],
  ['Enter', 'Open the focused job'],
  ['Space', 'Play or pause the filmstrip'],
  ['← / →', 'Step one frame (shift: ten)'],
  ['Home / End', 'First or last frame'],
  ['f', 'Star this frame (starred SVGs download together)'],
  ['p', 'Promote a waiting job'],
  ['l', 'Open the log'],
  ['n', 'Start a new job'],
  ['/', 'Focus the rail filter'],
  ['?', 'This list'],
]

export function HelpOverlay({ onClose }: { onClose: () => void }) {
  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div
        className="overlay__card"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-label="Keyboard shortcuts"
      >
        <div className="panel__head">
          <span className="eyebrow">Keyboard</span>
          <button type="button" className="btn btn--small" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="panel__body">
          <div className="shortcuts">
            {SHORTCUTS.map(([keys, description]) => (
              <Fragment key={keys}>
                <span className="kbd">{keys}</span>
                <span>{description}</span>
              </Fragment>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
