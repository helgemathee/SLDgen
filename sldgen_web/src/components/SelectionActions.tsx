import { useState } from 'react'
import { navigate } from '../router'
import { useApp } from '../state/store'
import { DeleteJobsDialog } from './DeleteJobsDialog'
import { RenameJobsDialog } from './RenameJobsDialog'

/**
 * What you can do with a ticked selection, wherever you ticked it.
 *
 * The rail and the overview grid share the one selection in the store, so they
 * share these buttons too — the alternative is compare living in one place and
 * delete in the other, and then having to remember which.
 */
export function SelectionActions({ compact = false }: { compact?: boolean }) {
  const { selection, setSelection } = useApp()
  const [confirming, setConfirming] = useState(false)
  const [renaming, setRenaming] = useState(false)

  if (selection.length === 0) return null

  return (
    <>
      <div className="btn-row">
        <button
          type="button"
          className={`btn btn--primary${compact ? ' btn--small' : ''}`}
          disabled={selection.length < 2}
          onClick={() => navigate({ name: 'compare', ids: selection })}
        >
          Compare {selection.length}
        </button>
        <button
          type="button"
          className={`btn${compact ? ' btn--small' : ''}`}
          onClick={() => setRenaming(true)}
        >
          Rename {selection.length}
        </button>
        <button
          type="button"
          className={`btn btn--danger${compact ? ' btn--small' : ''}`}
          onClick={() => setConfirming(true)}
        >
          Delete {selection.length}
        </button>
        <button
          type="button"
          className={`btn${compact ? ' btn--small' : ''}`}
          onClick={() => setSelection([])}
        >
          Clear
        </button>
      </div>

      {renaming && <RenameJobsDialog ids={selection} onClose={() => setRenaming(false)} />}
      {confirming && (
        <DeleteJobsDialog ids={selection} onClose={() => setConfirming(false)} />
      )}
    </>
  )
}
