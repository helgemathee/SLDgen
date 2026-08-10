import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import { ActionsPanel } from '../components/ActionsPanel'
import { ArtworkPane, availableArtwork, defaultTab, type ArtworkTab } from '../components/ArtworkPane'
import { Filmstrip } from '../components/Filmstrip'
import { LineagePanel, ParamTable, SegmentList } from '../components/JobData'
import { LogViewer } from '../components/LogViewer'
import { PartitionPanel } from '../components/PartitionPanel'
import { Ring } from '../components/Ring'
import { RunAgainDialog } from '../components/RunAgainDialog'
import { jobLabel } from '../lib/format'
import { navigate } from '../router'
import { useApp } from '../state/store'
import { useJob } from '../state/useJob'

export function JobPage({ jobId }: { jobId: string }) {
  const { job, frames, lineage, error, refresh } = useJob(jobId)
  const { setSelection, toast } = useApp()
  const [tab, setTab] = useState<ArtworkTab | null>(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [segmentSeq, setSegmentSeq] = useState<number | null>(null)
  const [showLog, setShowLog] = useState(false)
  const [runAgainOpen, setRunAgainOpen] = useState(false)

  // A failed job opens on the log, scrolled to the end, with the classified
  // error stated above it (Spec 3 SS6.4) -- that is the only thing you came for.
  useEffect(() => {
    if (job?.state === 'failed') setShowLog(true)
  }, [job?.state, job?.id])

  useEffect(() => {
    setTab(null)
    setFrameIndex(0)
    setPlaying(false)
    setPinned(true)
    setSegmentSeq(null)
    setShowLog(false)
  }, [jobId])

  // Follow the head of the strip while a job runs, but stop the moment the user
  // scrubs -- yanking the view away mid-comparison would be intolerable.
  const [pinned, setPinned] = useState(true)
  useEffect(() => {
    if (pinned && frames && frames.frames.length > 0) setFrameIndex(frames.frames.length - 1)
  }, [frames, pinned])

  /**
   * The starred frames, held here rather than read straight off the job so a
   * click lands immediately. The server's list wins whenever it changes, which
   * is how a star set on the laptop turns up on the phone.
   */
  const [favorites, setFavorites] = useState<number[]>([])
  const serverFavorites = job?.favorite_epochs
  const favoritesKey = (serverFavorites ?? []).join(',')
  // Keyed on the list's *contents*, not on the job object: the detail is
  // refetched whenever the job changes state, and re-running this on an
  // unchanged list would stamp on a star that was just clicked.
  useEffect(() => {
    setFavorites(serverFavorites ?? [])
  }, [jobId, favoritesKey])

  const currentFrame = frames?.frames[frameIndex] ?? null

  const toggleFavorite = useCallback(
    (epoch: number) => {
      const marked = favorites.includes(epoch)
      const call = marked ? api.removeFavorite(jobId, epoch) : api.addFavorite(jobId, epoch)
      // Optimistic: the star is a judgement made at speed, going through 4000
      // frames, and a round-trip per click would be felt.
      setFavorites((current) =>
        marked ? current.filter((value) => value !== epoch) : [...current, epoch].sort((a, b) => a - b),
      )
      call
        .then((response) => setFavorites(response.favorites.map((favorite) => favorite.epoch)))
        .catch((cause) => {
          setFavorites(serverFavorites ?? [])
          toast(cause instanceof Error ? cause.message : 'Could not save that star.')
        })
    },
    [favorites, jobId, serverFavorites, toast],
  )

  /**
   * Where this job is parked, restored once per job and then written back as
   * the user scrubs.
   *
   * `saved` tracks what the server already has, so scrubbing does not write the
   * same epoch twice; the ref also gates the writer until the restore has run,
   * because saving before restoring would overwrite the mark with the default
   * position of a page that has not been placed yet.
   */
  const mark = useRef<{ jobId: string | null; saved: number | null }>({ jobId: null, saved: null })

  useEffect(() => {
    if (!job || !frames || frames.frames.length === 0) return
    if (mark.current.jobId === job.id) return
    mark.current = { jobId: job.id, saved: job.viewed_epoch }
    if (job.viewed_epoch === null) return
    const position = frames.frames.findIndex((frame) => frame.epoch === job.viewed_epoch)
    if (position < 0) return
    setPinned(false)
    setFrameIndex(position)
    // The frame is why the job was left parked here, so show it rather than the
    // final SVG the page would otherwise open on.
    setTab('preview')
  }, [job, frames])

  useEffect(() => {
    const state = mark.current
    if (!job || !frames || frames.frames.length === 0 || state.jobId !== job.id) return
    // Parking on the newest frame is not parking: it is the default, and a job
    // that is still running must go on showing whatever it produces next.
    const atHead = frameIndex >= frames.frames.length - 1
    const wanted = atHead ? null : frames.frames[frameIndex]?.epoch ?? null
    if (wanted === state.saved) return
    // Debounced because arrow-key scrubbing changes this several times a second.
    const timer = setTimeout(() => {
      state.saved = wanted
      api.setViewedEpoch(job.id, wanted).catch(() => undefined)
    }, 600)
    return () => clearTimeout(timer)
  }, [job, frames, frameIndex])

  const available = useMemo(
    () => (job ? availableArtwork(job, currentFrame?.png_url ?? null) : null),
    [job, currentFrame],
  )
  const activeTab = tab ?? (available ? defaultTab(available) : 'result')
  const segment =
    job?.segments.find((entry) => entry.seq === segmentSeq) ??
    job?.segments[job.segments.length - 1] ??
    null

  // Filmstrip keys, live only while this page is mounted (SS14).
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (
        target &&
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
      )
        return
      if (event.metaKey || event.ctrlKey || event.altKey) return
      const count = frames?.frames.length ?? 0

      const step = (delta: number) => {
        if (count === 0) return
        event.preventDefault()
        setPinned(false)
        setPlaying(false)
        setFrameIndex((index) => Math.min(count - 1, Math.max(0, index + delta)))
      }

      switch (event.key) {
        case ' ':
          if (count > 0) {
            event.preventDefault()
            setPinned(false)
            setPlaying((value) => !value)
          }
          break
        case 'ArrowLeft':
          step(event.shiftKey ? -10 : -1)
          break
        case 'ArrowRight':
          step(event.shiftKey ? 10 : 1)
          break
        case 'Home':
          if (count > 0) {
            event.preventDefault()
            setPinned(false)
            setFrameIndex(0)
          }
          break
        case 'End':
          if (count > 0) {
            event.preventDefault()
            setPinned(true)
            setFrameIndex(count - 1)
          }
          break
        case 'f':
          if (currentFrame) {
            event.preventDefault()
            toggleFavorite(currentFrame.epoch)
          }
          break
        case 'l':
          event.preventDefault()
          setShowLog((value) => !value)
          break
        case 'p':
          if (job?.state === 'waiting') {
            event.preventDefault()
            document.querySelector<HTMLButtonElement>('.panel button.btn--primary')?.click()
          }
          break
        default:
          break
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [frames, job?.state, currentFrame, toggleFavorite])

  if (error) {
    return (
      <div className="empty">
        <strong>That job is not here.</strong>
        <span className="note">{error}</span>
        <button type="button" className="btn" onClick={() => navigate({ name: 'jobs' })}>
          Back to jobs
        </button>
      </div>
    )
  }

  if (!job) return <div className="empty muted">Loading…</div>

  return (
    <div className="detail">
      <div className="detail__title">
        <Ring
          size={22}
          state={job.state}
          currentEpoch={job.current_epoch}
          targetEpoch={job.target_epoch}
          numIter={job.num_iter}
        />
        <h1>{jobLabel(job)}</h1>
        <span className="mono muted">
          {job.current_epoch}/{job.num_iter} · budget {job.target_epoch}
        </span>
        {job.resolved_caption && (
          <span className="note" style={{ flex: 1 }}>
            “{job.resolved_caption}”
          </span>
        )}
        {lineage && (
          <LineagePanel
            lineage={lineage}
            onOpen={(id) => navigate({ name: 'job', id })}
            onCompare={(ids) => {
              const all = [job.id, ...ids]
              setSelection(all)
              navigate({ name: 'compare', ids: all })
            }}
          />
        )}
      </div>

      <div className="detail__left">
        <ArtworkPane
          job={job}
          frameUrl={currentFrame?.png_url ?? null}
          frameEpoch={currentFrame?.epoch ?? null}
          tab={activeTab}
          onTab={setTab}
        />
        {frames && (
          <Filmstrip
            frames={frames}
            index={frameIndex}
            onIndex={(index) => {
              setPinned(index === (frames.frames.length ?? 1) - 1)
              setFrameIndex(index)
              setTab('preview')
            }}
            playing={playing}
            onPlaying={setPlaying}
            favorites={favorites}
            onToggleFavorite={toggleFavorite}
          />
        )}
        {showLog && <LogViewer job={job} segment={segment} />}
      </div>

      <div className="detail__right">
        <ActionsPanel
          job={job}
          favorites={favorites}
          onRunAgain={() => setRunAgainOpen(true)}
          onChanged={refresh}
        />
        <SegmentList
          job={job}
          selected={segment?.seq ?? null}
          onSelect={(seq) => {
            setSegmentSeq(seq)
            setShowLog(true)
          }}
        />
        {!showLog && (
          <button type="button" className="btn" onClick={() => setShowLog(true)}>
            Open the log
          </button>
        )}
        <ParamTable job={job} onRunAgain={() => setRunAgainOpen(true)} />
        {job.state === 'complete' && <PartitionPanel job={job} />}
      </div>

      {runAgainOpen && (
        <RunAgainDialog
          job={job}
          onClose={() => setRunAgainOpen(false)}
          onQueued={(ids) => {
            setRunAgainOpen(false)
            setSelection(ids)
            if (ids.length > 1) navigate({ name: 'compare', ids })
            else if (ids[0]) navigate({ name: 'job', id: ids[0] })
          }}
        />
      )}
    </div>
  )
}
