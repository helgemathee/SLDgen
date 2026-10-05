import { useState } from "react";
import { JobThumb } from "../components/JobThumb";
import { useVisibleJobs } from "../components/JobRail";
import { JobStatus } from "../components/JobStatus";
import { SelectionActions } from "../components/SelectionActions";
import { StarToggle } from "../components/StarToggle";
import { formatAgo, jobLabel } from "../lib/format";
import { queueLabel, queuePositions } from "../lib/queue";
import { navigate } from "../router";
import { useApp } from "../state/store";

/**
 * The overview grid.
 *
 * The rail is the authoritative list; this is the same set (same filters, same
 * order) at a size where the
 * artwork is actually judgeable, which is what you want when four candidates
 * have just finished and the rail's 44px thumbnails are too small to choose by.
 */
/** Cell width at 100%: the size the grid has always had. */
const BASE_CELL_PX = 300;
const SCALE_MIN = 50;
const SCALE_MAX = 200;
const SCALE_KEY = "sldgen.jobsThumbScale";

function loadScale(): number {
  try {
    const stored = Number(localStorage.getItem(SCALE_KEY));
    if (stored >= SCALE_MIN && stored <= SCALE_MAX) return stored;
  } catch {
    // Storage unavailable (private window etc.): fall back to the default.
  }
  return 100;
}

function saveScale(scale: number) {
  try {
    localStorage.setItem(SCALE_KEY, String(scale));
  } catch {
    // Not worth surfacing: the size just won't be remembered.
  }
}

export function JobsPage() {
  const { jobs, selection, toggleSelected } = useApp();
  const visible = useVisibleJobs();
  const [scale, setScaleState] = useState(loadScale);
  const setScale = (next: number) => {
    setScaleState(next);
    saveScale(next);
  };
  // Positions over the whole queue, not just the filtered cells.
  const positions = queuePositions(jobs);

  if (jobs.length === 0) {
    return (
      <div className="empty">
        <strong>No jobs yet.</strong>
        <span className="note">Prepare an image to get started.</span>
        <button
          type="button"
          className="btn btn--primary"
          onClick={() => navigate({ name: "new" })}
        >
          New job
        </button>
      </div>
    );
  }

  return (
    <div
      className="compare"
      style={{
        gridTemplateColumns: `repeat(auto-fill, minmax(${Math.round((BASE_CELL_PX * scale) / 100)}px, 1fr))`,
      }}
    >
      <div className="compare__toolbar">
        <span className="eyebrow">
          {visible.length === jobs.length
            ? `${jobs.length} jobs`
            : `${visible.length} of ${jobs.length} jobs`}
        </span>
        <span className="note">
          Tick two or more to compare them, or any number to delete them.
          Shift-click a rail row does the same.
        </span>
        <span style={{ flex: 1 }} />
        <SelectionActions compact />
        <label
          className="note"
          title="Thumbnail size. Double-click to reset to 100%."
          style={{ display: "flex", alignItems: "center", gap: 6 }}
        >
          size
          <input
            type="range"
            style={{ width: 120, accentColor: "var(--ink)" }}
            min={SCALE_MIN}
            max={SCALE_MAX}
            step={10}
            value={scale}
            aria-label="Thumbnail size"
            onChange={(event) => setScale(Number(event.target.value))}
            onDoubleClick={() => setScale(100)}
          />
          <span
            className="mono"
            style={{ minWidth: "4ch", textAlign: "right" }}
          >
            {scale}%
          </span>
        </label>
      </div>
      {visible.length === 0 && (
        <div className="note">Nothing matches the rail's filter.</div>
      )}
      {visible.map((job) => (
        <div className="cell" key={job.id}>
          <div
            className="cell__art"
            role="button"
            tabIndex={0}
            onClick={() => navigate({ name: "job", id: job.id })}
            onKeyDown={(event) => {
              if (event.key === "Enter") navigate({ name: "job", id: job.id });
            }}
          >
            <JobThumb job={job} alt={jobLabel(job)} />
          </div>
          <div className="cell__body">
            <div className="cell__head">
              <input
                type="checkbox"
                checked={selection.includes(job.id)}
                aria-label={`Select ${jobLabel(job)}`}
                onChange={() => toggleSelected(job.id, true)}
              />
              <StarToggle job={job} />
              <JobStatus job={job} size={18} position={positions.get(job.id)} />
              <strong>{jobLabel(job)}</strong>
            </div>
            <div className="mono muted">
              {job.current_epoch}/{job.num_iter} · {job.state}
              {positions.has(job.id)
                ? ` (${queueLabel(positions.get(job.id))})`
                : ""}{" "}
              · {formatAgo(job.created_at)}
            </div>
            {scale >= 100 && job.resolved_caption && (
              <div className="note">“{job.resolved_caption}”</div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
