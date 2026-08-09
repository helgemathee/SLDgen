import { Fragment, useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { JobDetail, Segment } from '../api/types'
import { ERROR_COPY, formatDuration } from '../lib/format'
import { cook, filterLog, highlightParts } from '../lib/logcook'

/**
 * The console (Spec 3 SS6.4, Spec 2 SS13).
 *
 * A full panel rather than a modal, because when a run fails at 3800 iterations
 * the log is the only thing that explains why, and you read it alongside the
 * parameters rather than on top of them.
 *
 * Raw bytes are what gets fetched and accumulated; cooking happens on the whole
 * buffer at render time (see `lib/logcook`). Offsets refer to the file, so a
 * reconnect resumes exactly where it left off.
 */
export function LogViewer({ job, segment }: { job: JobDetail; segment: Segment | null }) {
  const [raw, setRaw] = useState(false)
  const [search, setSearch] = useState('')
  const [text, setText] = useState('')
  const [offset, setOffset] = useState(0)
  const [follow, setFollow] = useState(true)
  const [truncated, setTruncated] = useState(false)
  const pane = useRef<HTMLPreElement>(null)
  const seq = segment?.seq ?? null

  // Reset when the segment changes: each segment has its own argv header and
  // exit status, and concatenating them would misrepresent both (SS13.3).
  useEffect(() => {
    setText('')
    setOffset(0)
    setFollow(true)
    setTruncated(false)
  }, [job.id, seq])

  const pull = useCallback(
    async (from: number) => {
      const chunk = await api.log(job.id, {
        segment: seq ?? undefined,
        from,
        raw: true, // always raw on the wire; cooked at render time
      })
      if (chunk.to > from || (from === 0 && chunk.text)) {
        setText((current) => (from === 0 ? chunk.text : current + chunk.text))
        setOffset(chunk.to)
        // A 1 MiB cap per request means a very long log arrives in pieces; say so
        // rather than looking like it stopped.
        setTruncated(!chunk.eof)
      }
      return chunk
    },
    [job.id, seq],
  )

  useEffect(() => {
    let cancelled = false
    let cursor = 0
    let timer: ReturnType<typeof setTimeout> | null = null

    const tick = async () => {
      if (cancelled) return
      try {
        const chunk = await pull(cursor)
        cursor = chunk.to
        // Poll only while the segment is live. A finished segment's log is
        // immutable, so re-reading it forever would be pure waste.
        if (!cancelled && chunk.running) timer = setTimeout(tick, 1000)
      } catch {
        if (!cancelled) timer = setTimeout(tick, 3000)
      }
    }
    tick()

    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [pull])

  // Auto-scroll disengages the moment you scroll up and re-engages at the bottom.
  useEffect(() => {
    if (!follow) return
    const element = pane.current
    if (element) element.scrollTop = element.scrollHeight
  }, [text, follow])

  const display = filterLog(raw ? text : cook(text), search)
  const failed = job.state === 'failed' && job.error_class
  const copy = failed ? ERROR_COPY[job.error_class!] ?? ERROR_COPY.unknown : null

  return (
    <div className="panel">
      <div className="log__toolbar">
        <span className="eyebrow" style={{ flex: 1 }}>
          Console{segment ? ` · segment ${segment.seq}` : ''}
        </span>
        <input
          className="log__search"
          placeholder="Filter lines"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <button
          type="button"
          className="btn btn--small"
          aria-pressed={raw}
          onClick={() => setRaw((value) => !value)}
          title="Raw shows every carriage return; cooked shows what a terminal would"
        >
          {raw ? 'raw' : 'cooked'}
        </button>
        <button
          type="button"
          className="btn btn--small"
          aria-pressed={follow}
          onClick={() => {
            setFollow(true)
            if (pane.current) pane.current.scrollTop = pane.current.scrollHeight
          }}
        >
          follow
        </button>
        <a
          className="btn btn--small"
          href={`/api/jobs/${job.id}/log/download${seq !== null ? `?segment=${seq}` : ''}`}
          download
        >
          download
        </a>
      </div>

      {copy && (
        <div className="panel__body" style={{ borderBottom: '1px solid var(--rule)' }}>
          <strong className="state-failed">{copy.headline}</strong>
          <div className="note">{copy.advice}</div>
          {job.error_message && (
            <div className="mono" style={{ marginTop: 4 }}>
              {job.error_message}
            </div>
          )}
        </div>
      )}

      <pre
        className="log"
        ref={pane}
        onScroll={(event) => {
          const element = event.currentTarget
          const atBottom =
            element.scrollHeight - element.scrollTop - element.clientHeight < 24
          setFollow(atBottom)
        }}
      >
        {display
          ? display.split('\n').map((line, index) => (
              <Fragment key={index}>
                {highlightParts(line, search).map((part, position) =>
                  position % 2 === 1 ? <mark key={position}>{part}</mark> : <Fragment key={position}>{part}</Fragment>,
                )}
                {'\n'}
              </Fragment>
            ))
          : search
            ? '(no matching lines)'
            : '(no output yet)'}
      </pre>

      <div className="panel__body mono" style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <span>{offset.toLocaleString()} bytes read</span>
        {truncated && (
          <button type="button" className="btn btn--small" onClick={() => pull(offset)}>
            load more
          </button>
        )}
        {segment?.finished_at && (
          <>
            <span>exit {segment.exit_code ?? 'adopted'}</span>
            <span>
              {formatDuration(
                (Date.parse(segment.finished_at) - Date.parse(segment.started_at)) / 1000,
              )}
            </span>
          </>
        )}
      </div>
    </div>
  )
}
