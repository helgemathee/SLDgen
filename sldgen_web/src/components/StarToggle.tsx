import type { JobSummary } from '../api/types'
import { jobLabel } from '../lib/format'
import { useApp } from '../state/store'

/**
 * The job's own star, drawn next to its ring.
 *
 * Solid yellow when set. When not set it is an outline the row reveals on
 * hover (`.star-toggle--off` in app.css), so a rail of fifty seeds is not fifty
 * grey stars -- only the ones you chose.
 */
export function StarToggle({ job }: { job: JobSummary }) {
  const { toggleStar } = useApp()
  const label = job.starred
    ? `Unmark ${jobLabel(job)} as favourite`
    : `Mark ${jobLabel(job)} as favourite`
  return (
    <button
      type="button"
      className={`star-toggle${job.starred ? ' star-toggle--on' : ' star-toggle--off'}`}
      aria-pressed={job.starred}
      aria-label={label}
      title={label}
      onClick={(event) => {
        // Rows and cells navigate on click; the star must not.
        event.stopPropagation()
        toggleStar(job.id)
      }}
    >
      {job.starred ? '★' : '☆'}
    </button>
  )
}
