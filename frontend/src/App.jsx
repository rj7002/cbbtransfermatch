import { useState, useEffect, useRef } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, Tooltip
} from 'recharts'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:5002'

const SCORE_LABELS = {
  ShotFit: 'Shot Fit',
  OpportunityFit: 'Opportunity',
  GapFit: 'Gap Fill',
  Efficiency: 'Efficiency',
}

const DEFAULT_WEIGHTS = {
  ShotFit: 35,
  OpportunityFit: 25,
  GapFit: 20,
  Efficiency: 20,
}

function applyWeights(items, weights) {
  const total = Object.values(weights).reduce((a, b) => a + b, 0) || 1
  return [...items]
    .map(item => ({
      ...item,
      FinalScore: Object.keys(weights).reduce(
        (sum, k) => sum + (item[k] || 0) * (weights[k] / total), 0
      ),
    }))
    .sort((a, b) => b.FinalScore - a.FinalScore)
}

function BudgetFilter({ budget, onChange, max, filtered, total }) {
  const step = Math.max(1000, Math.round(max / 100))
  const [lo, hi] = budget ?? [0, max]
  const isDefault = lo === 0 && hi === max

  const leftPct  = (lo / max) * 100
  const rightPct = (hi / max) * 100

  function fmt(v) {
    if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`
    if (v >= 1_000)     return `$${Math.round(v / 1_000)}K`
    return `$${v}`
  }

  function setLo(v) {
    const next = Math.min(Number(v), hi - step)
    onChange([next, hi])
  }
  function setHi(v) {
    const next = Math.max(Number(v), lo + step)
    onChange(next >= max && lo === 0 ? null : [lo, next >= max ? max : next])
  }

  return (
    <div className="budget-panel">
      <div className="weights-header">
        <span className="weights-title">NIL Budget</span>
        <span className="budget-value">
          {isDefault ? 'All ranges' : `${fmt(lo)} – ${hi >= max ? 'No limit' : fmt(hi)}`}
        </span>
        {!isDefault && (
          <button className="weights-reset" onClick={() => onChange(null)}>Clear</button>
        )}
      </div>

      <div className="dual-range">
        <div
          className="dual-range-track-fill"
          style={{ left: `${leftPct}%`, width: `${rightPct - leftPct}%` }}
        />
        <input
          type="range"
          className="dual-range-input"
          min={0} max={max} step={step}
          value={lo}
          onChange={e => setLo(e.target.value)}
        />
        <input
          type="range"
          className="dual-range-input"
          min={0} max={max} step={step}
          value={hi}
          onChange={e => setHi(e.target.value)}
        />
      </div>

      <div className="budget-labels">
        <span>{fmt(0)}</span>
        <span className="budget-count">{filtered} of {total} players</span>
        <span>No limit</span>
      </div>
    </div>
  )
}

function ChipFilter({ label, options, selected, onChange }) {
  if (!options.length) return null
  function toggle(val) {
    onChange(selected.includes(val) ? selected.filter(v => v !== val) : [...selected, val])
  }
  return (
    <div className="budget-panel">
      <div className="weights-header">
        <span className="weights-title">{label}</span>
        {selected.length > 0 && (
          <button className="weights-reset" onClick={() => onChange([])}>Clear</button>
        )}
      </div>
      <div className="chip-group">
        {options.map(opt => (
          <button
            key={opt}
            className={`chip ${selected.includes(opt) ? 'chip--on' : ''}`}
            onClick={() => toggle(opt)}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  )
}

function WeightsPanel({ weights, onChange }) {
  const total = Object.values(weights).reduce((a, b) => a + b, 0)
  const isDefault = JSON.stringify(weights) === JSON.stringify(DEFAULT_WEIGHTS)

  return (
    <div className="weights-panel">
      <div className="weights-header">
        <span className="weights-title">Score Weights</span>
        <span className="weights-total" style={{ color: total === 100 ? '#60c060' : '#e8334a' }}>
          {total}%
        </span>
        {!isDefault && (
          <button className="weights-reset" onClick={() => onChange(DEFAULT_WEIGHTS)}>
            Reset
          </button>
        )}
      </div>
      {Object.entries(SCORE_LABELS).map(([key, label]) => (
        <div key={key} className="weight-row">
          <span className="weight-label">{label}</span>
          <input
            type="range"
            min={0}
            max={60}
            step={5}
            value={weights[key]}
            className="weight-slider"
            onChange={e => onChange({ ...weights, [key]: Number(e.target.value) })}
          />
          <span className="weight-val">{weights[key]}%</span>
        </div>
      ))}
    </div>
  )
}

function scoreColor(val) {
  const v = Math.max(0, Math.min(1, val))
  if (v < 0.5) {
    const t = v * 2
    return `rgb(220, ${Math.round(60 + 100 * t)}, 60)`
  }
  const t = (v - 0.5) * 2
  return `rgb(${Math.round(220 - 140 * t)}, ${Math.round(160 + 40 * t)}, 60)`
}

function ScoreBar({ label, value }) {
  return (
    <div className="score-bar">
      <span className="score-bar-label">{label}</span>
      <div className="bar-track">
        <div
          className="bar-fill"
          style={{ width: `${Math.round(value * 100)}%`, background: scoreColor(value) }}
        />
      </div>
      <span className="score-bar-val">{Math.round(value * 100)}</span>
    </div>
  )
}

function DetailPanel({ item, mode, gender, onClose }) {
  const [overview, setOverview] = useState(null)
  const [overviewLoading, setOverviewLoading] = useState(false)

  const name = mode === 'team' ? item.Player : item.Team

  useEffect(() => {
    let cancelled = false
    async function load() {
      setOverview(null)
      setOverviewLoading(true)
      try {
        const endpoint = mode === 'team'
          ? `${API}/get_player_overview/${encodeURIComponent(name)}?gender=${gender}`
          : `${API}/get_team_overview/${encodeURIComponent(name)}?gender=${gender}`
        const d = await fetch(endpoint).then(r => r.json())
        if (!cancelled) setOverview(d.overview || null)
      } catch {
        if (!cancelled) setOverview(null)
      } finally {
        if (!cancelled) setOverviewLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [name, mode, gender])

  const radarData = Object.entries(SCORE_LABELS).map(([key, label]) => ({
    subject: label,
    value: Math.round((item[key] || 0) * 100),
  }))

  const sub = mode === 'team'
    ? [item.Position, item.Year, item.PrevTeam].filter(Boolean).join(' · ')
    : item.Conference

  return (
    <div className="detail-panel">
      <button className="detail-close" onClick={onClose}>✕</button>

      <div className="detail-hero">
        {mode === 'team'
          ? <PlayerAvatar teamId={item.PrevTeamId} playerId={item.PlayerId} name={item.Player} size="lg" />
          : <TeamLogo teamId={item.TeamId} name={item.Team} size="lg" />
        }
        <div className="detail-hero-info">
          <div className="detail-name">{name}</div>
          {sub && <div className="detail-sub">{sub}</div>}
        </div>
      </div>

      <div className="overview-box">
        {overviewLoading
          ? <div className="overview-loading"><div className="spinner spinner--sm" /><span>{mode === 'team' ? 'Generating scouting report...' : 'Generating program overview...'}</span></div>
          : overview
            ? <p className="overview-text">{overview}</p>
            : null
        }
      </div>

      {mode === 'player' && (
        <div className="detail-stats">
          <StatPill label="PPG"  value={item.ptsScoredPg} />
          <StatPill label="ORtg" value={item.ortg} />
          <StatPill label="DRtg" value={item.drtg} />
          <StatPill label="Net"  value={item.netRtg} />
          <StatPill label="Pace" value={item.pace} />
          <StatPill label="eFG%" value={item.efgPct} isPercent />
          <StatPill label="3P%"  value={item.fg3Pct}  isPercent />
          <StatPill label="REB"  value={item.rebPg} />
          <StatPill label="AST"  value={item.astPg} />
          <StatPill label="TOV"  value={item.tovPg} />
        </div>
      )}

      {mode === 'team' && (
        <div className="detail-stats">
          <StatPill label="PTS"  value={item.ptsScoredPg} />
          <StatPill label="REB"  value={item.rebPg} />
          <StatPill label="AST"  value={item.astPg} />
          <StatPill label="STL"  value={item.stlPg} />
          <StatPill label="BLK"  value={item.blkPg} />
          <StatPill label="TOV"  value={item.tovPg} />
          <StatPill label="FG%"  value={item.fgPct}  isPercent />
          <StatPill label="2P%"  value={item.fg2Pct} isPercent />
          <StatPill label="3P%"  value={item.fg3Pct} isPercent />
          <StatPill label="FT%"  value={item.ftPct}  isPercent />
        </div>
      )}

      <div className="detail-score-hero" style={{ color: scoreColor(item.FinalScore) }}>
        {(item.FinalScore * 100).toFixed(1)}
        <span className="detail-score-denom"> / 100</span>
      </div>

      <div className="radar-wrap">
        <ResponsiveContainer width="100%" height={200}>
          <RadarChart data={radarData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
            <PolarGrid stroke="#1e1e38" />
            <PolarAngleAxis
              dataKey="subject"
              tick={{ fill: '#6666a0', fontSize: 10, fontFamily: 'inherit' }}
            />
            <Tooltip
              contentStyle={{
                background: '#12121e',
                border: '1px solid #1e1e38',
                borderRadius: 8,
                fontSize: 12,
                color: '#f0f0f8',
              }}
            />
            <Radar
              dataKey="value"
              stroke="#e8334a"
              fill="#e8334a"
              fillOpacity={0.18}
              strokeWidth={2}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <div className="score-bars">
        {Object.entries(SCORE_LABELS).map(([key, label]) => (
          <ScoreBar key={key} label={label} value={item[key] || 0} />
        ))}
      </div>

      {item.Explanation?.length > 0 && (
        <div className="detail-tags">
          {item.Explanation.map((e, i) => <span key={i} className="tag">{e}</span>)}
        </div>
      )}
    </div>
  )
}

const IMG_BASE = 'https://storage.googleapis.com/cbb-image-files'

function PlayerAvatar({ teamId, playerId, name, size }) {
  const [err, setErr] = useState(false)
  const src = `${IMG_BASE}/player-headshots/${teamId}-${playerId}.png`
  const cls = `avatar${size === 'lg' ? ' avatar--lg' : ''} avatar--fallback`
  return err
    ? <div className={cls}>{(name || '?')[0]}</div>
    : <img className={`avatar${size === 'lg' ? ' avatar--lg' : ''}`} src={src} alt={name} onError={() => setErr(true)} />
}

function TeamLogo({ teamId, name, size }) {
  const [err, setErr] = useState(false)
  const src = `${IMG_BASE}/team-logos/${teamId}.png`
  const cls = `avatar avatar--team${size === 'lg' ? ' avatar--lg' : ''} avatar--fallback`
  return err
    ? <div className={cls}>{(name || '?')[0]}</div>
    : <img className={`avatar avatar--team${size === 'lg' ? ' avatar--lg' : ''}`} src={src} alt={name} onError={() => setErr(true)} />
}

function StatPill({ label, value, isPercent }) {
  if (value == null) return null
  const display = isPercent ? `${Math.round(value * 100)}%` : value.toFixed(1)
  return (
    <div className="stat-pill">
      <span className="stat-val">{display}</span>
      <span className="stat-label">{label}</span>
    </div>
  )
}

function PlayerCard({ player, rank, selected, onClick }) {
  return (
    <div className={`card card--player ${selected ? 'card--active' : ''}`} onClick={() => onClick(player)}>
      <div className="card-rank">#{rank}</div>
      <PlayerAvatar teamId={player.PrevTeamId} playerId={player.PlayerId} name={player.Player} />
      <div className="card-body">
        <div className="card-top">
          <span className="card-name">{player.Player}</span>
          <div className="card-badges">
            {player.Position && <span className="badge">{player.Position}</span>}
            {player.Year && <span className="badge badge--dim">{player.Year}</span>}
            {player.NilTier && <span className={`badge badge--nil badge--nil-${player.NilTier.split(' ')[0].toLowerCase()}`}>{player.NilTier}</span>}
          </div>
        </div>
        <div className="card-sub">
          {player.PrevTeam}
          {player.NilValue != null && <span className="nil-value">~${player.NilValue.toLocaleString()}</span>}
        </div>
        <div className="stat-row">
          <StatPill label="PTS" value={player.ptsScoredPg} />
          <StatPill label="REB" value={player.rebPg} />
          <StatPill label="AST" value={player.astPg} />
          <StatPill label="STL" value={player.stlPg} />
          <StatPill label="BLK" value={player.blkPg} />
          <StatPill label="TOV" value={player.tovPg} />
        </div>
        <div className="stat-row stat-row--pct">
          <StatPill label="FG%" value={player.fgPct} isPercent />
          <StatPill label="2P%" value={player.fg2Pct} isPercent />
          <StatPill label="3P%" value={player.fg3Pct} isPercent />
          <StatPill label="FT%" value={player.ftPct} isPercent />
        </div>
      </div>
      <div className="card-score" style={{ color: scoreColor(player.FinalScore) }}>
        <span className="score-big">{Math.round(player.FinalScore * 100)}</span>
      </div>
    </div>
  )
}

function TeamCard({ team, rank, selected, onClick }) {
  const record = (team.overallWins != null && team.overallLosses != null)
    ? `${team.overallWins}-${team.overallLosses}` : null

  return (
    <div className={`card ${selected ? 'card--active' : ''}`} onClick={() => onClick(team)}>
      <div className="card-rank">#{rank}</div>
      <TeamLogo teamId={team.TeamId} name={team.Team} />
      <div className="card-body">
        <div className="card-top">
          <span className="card-name">{team.Team}</span>
          <div className="card-badges">
            {team.Conference && <span className="badge badge--dim">{team.Conference}</span>}
            {record && <span className="badge">{record}</span>}
            {team.netRanking != null && <span className="badge badge--dim">NET #{team.netRanking}</span>}
          </div>
        </div>
        <div className="card-stats">
          <StatPill label="PPG"  value={team.ptsScoredPg} />
          <StatPill label="ORtg" value={team.ortg} />
          <StatPill label="DRtg" value={team.drtg} />
          <StatPill label="Pace" value={team.pace} />
          <StatPill label="eFG%" value={team.efgPct} isPercent />
          <StatPill label="3P%"  value={team.fg3Pct}  isPercent />
        </div>
      </div>
      <div className="card-score" style={{ color: scoreColor(team.FinalScore) }}>
        <span className="score-big">{Math.round(team.FinalScore * 100)}</span>
      </div>
    </div>
  )
}

function Combobox({ options, value, onChange, onSelect, placeholder }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  const filtered = value.length > 0
    ? options.filter(o => o.toLowerCase().includes(value.toLowerCase())).slice(0, 8)
    : []

  useEffect(() => {
    function handler(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div className="combobox" ref={ref}>
      <div className="search-box">
        <svg className="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.35-4.35" />
        </svg>
        <input
          className="search-input"
          placeholder={placeholder}
          value={value}
          onChange={e => { onChange(e.target.value); setOpen(true) }}
          onFocus={() => value && setOpen(true)}
        />
        {value && (
          <button className="search-clear" onClick={() => { onChange(''); setOpen(false) }}>✕</button>
        )}
      </div>
      {open && filtered.length > 0 && (
        <div className="suggestions">
          {filtered.map(s => (
            <div key={s} className="suggestion" onMouseDown={() => { onSelect(s); setOpen(false) }}>
              {s}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [gender, setGender] = useState('MALE')
  const [mode, setMode] = useState('team')
  const [teams, setTeams] = useState([])
  const [players, setPlayers] = useState([])
  const [query, setQuery] = useState('')
  const [rawResults, setRawResults] = useState([])
  const [weights, setWeights] = useState(DEFAULT_WEIGHTS)
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(null)
  const [error, setError] = useState(null)

  const [nilBudget, setNilBudget]       = useState(null) // null = no filter, or [lo, hi]
  const [yearFilter, setYearFilter]     = useState([])
  const [posFilter, setPosFilter]       = useState([])
  const [confFilter, setConfFilter]     = useState([])

  const [activeQuery, setActiveQuery] = useState(null)
  const [allOptions, setAllOptions] = useState({ years: [], positions: [], conferences: [], maxNil: 0, total: 0 })
  const filterRefetchTimer = useRef(null)

  // Filters are applied server-side — results is just weights applied to whatever the backend returned
  const results = rawResults.length > 0 ? applyWeights(rawResults, weights) : []

  function buildFetchUrl(val, currentMode, currentGender, filters) {
    const base = currentMode === 'team'
      ? `${API}/get_team_fit/${encodeURIComponent(val)}?gender=${currentGender}`
      : `${API}/get_player_fit/${encodeURIComponent(val)}?gender=${currentGender}`
    const p = new URLSearchParams()
    if (currentMode === 'team') {
      if (filters.nilBudget) {
        p.set('nil_min', filters.nilBudget[0])
        p.set('nil_max', filters.nilBudget[1])
      }
      if (filters.yearFilter?.length) p.set('years', filters.yearFilter.join(','))
      if (filters.posFilter?.length)  p.set('positions', filters.posFilter.join(','))
    } else {
      if (filters.confFilter?.length) p.set('conferences', filters.confFilter.join(','))
    }
    const qs = p.toString()
    return qs ? `${base}&${qs}` : base
  }

  async function doFilterFetch(val, currentMode, currentGender, filters) {
    setLoading(true)
    try {
      const url = buildFetchUrl(val, currentMode, currentGender, filters)
      console.log('[filter] fetching:', url)
      const res = await fetch(url)
      const data = await res.json()
      console.log('[filter] response:', Array.isArray(data) ? `${data.length} results` : data)
      if (Array.isArray(data)) setRawResults(data)
    } catch (e) {
      console.error('[filter] error:', e)
    } finally { setLoading(false) }
  }

  // Immediate refetch for chip clicks
  function refetchNow(val, currentMode, currentGender, filters) {
    if (!val) return
    clearTimeout(filterRefetchTimer.current)
    doFilterFetch(val, currentMode, currentGender, filters)
  }

  // Debounced refetch for the NIL slider (fires constantly while dragging)
  function refetchDebounced(val, currentMode, currentGender, filters) {
    if (!val) return
    clearTimeout(filterRefetchTimer.current)
    filterRefetchTimer.current = setTimeout(() => {
      doFilterFetch(val, currentMode, currentGender, filters)
    }, 350)
  }

  useEffect(() => {
    setQuery('')
    setRawResults([])
    setSelected(null)
    setError(null)
    fetch(`${API}/get_teams?gender=${gender}`).then(r => r.json()).then(setTeams).catch(() => {})
    fetch(`${API}/get_players?gender=${gender}`).then(r => r.json()).then(setPlayers).catch(() => {})
  }, [gender])

  function clearFilters() {
    setNilBudget(null)
    setYearFilter([])
    setPosFilter([])
    setConfFilter([])
  }

  function switchMode(m) {
    setMode(m)
    setQuery('')
    setRawResults([])
    setSelected(null)
    setError(null)
    setActiveQuery(null)
    setAllOptions({ years: [], positions: [], conferences: [], maxNil: 0, total: 0 })
    clearFilters()
  }

  function switchGender(g) {
    setGender(g)
    setMode('team')
    setActiveQuery(null)
    setAllOptions({ years: [], positions: [], conferences: [], maxNil: 0, total: 0 })
    clearFilters()
  }

  async function handleSelect(val) {
    setQuery(val)
    setActiveQuery(val)
    setSelected(null)
    setError(null)
    setLoading(true)
    clearFilters()
    try {
      const url = buildFetchUrl(val, mode, gender, {})
      const data = await fetch(url).then(r => r.json())
      if (data.error) { setError(data.error); setRawResults([]) }
      else {
        setRawResults(data)
        setAllOptions({
          years: [...new Set(data.map(r => r.Year).filter(Boolean))].sort(),
          positions: [...new Set(data.map(r => r.Position).filter(Boolean))].sort(),
          conferences: [...new Set(data.map(r => r.Conference).filter(Boolean))].sort(),
          maxNil: data.reduce((mx, r) => Math.max(mx, r.NilValue ?? 0), 0),
          total: data.length,
        })
      }
    } catch {
      setError('Could not reach the server.')
    } finally {
      setLoading(false)
    }
  }

  const hasResults = results.length > 0

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-text">Portal<span className="logo-accent">Match</span></span>
          </div>
          <div className="gender-toggle">
            <button
              className={`gender-btn ${gender === 'MALE' ? 'gender-btn--on' : ''}`}
              onClick={() => switchGender('MALE')}
            >
              Men's
            </button>
            <button
              className={`gender-btn ${gender === 'FEMALE' ? 'gender-btn--on' : ''}`}
              onClick={() => switchGender('FEMALE')}
            >
              Women's
            </button>
          </div>
          <div className="mode-toggle">
            <button
              className={`mode-btn ${mode === 'team' ? 'mode-btn--on' : ''}`}
              onClick={() => switchMode('team')}
            >
              Find Players
            </button>
            <button
              className={`mode-btn ${mode === 'player' ? 'mode-btn--on' : ''}`}
              onClick={() => switchMode('player')}
            >
              Find Teams
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        <div className={`search-section ${hasResults ? 'search-section--compact' : ''}`}>
          {!hasResults && (
            <div className="hero">
              <h1 className="hero-title">
                {mode === 'team'
                  ? <>Find your next<br /><span className="hero-accent">transfer target</span></>
                  : <>Find the right<br /><span className="hero-accent">program fit</span></>
                }
              </h1>
              <p className="hero-sub">
                {mode === 'team'
                  ? "Match portal players to your system using shot profile & gap analysis"
                  : "See which programs align with a player's game and style"
                }
              </p>
            </div>
          )}
          <Combobox
            options={mode === 'team' ? teams : players}
            value={query}
            onChange={setQuery}
            onSelect={handleSelect}
            placeholder={mode === 'team' ? 'Search a team...' : 'Search a player...'}
          />
        </div>

        {loading && (
          <div className="status-row">
            <div className="spinner" />
            <span>Analyzing fit...</span>
          </div>
        )}

        {error && <div className="status-row status-row--error">{error}</div>}

        {hasResults && (
          <div className={`results-layout ${selected ? 'results-layout--split' : ''}`}>
            <div className="results-col">
              <div className="results-header">
                <strong>{results.length}</strong>&nbsp;matches for&nbsp;
                <span className="results-query">{query}</span>
              </div>
              <WeightsPanel weights={weights} onChange={setWeights} />
              {mode === 'team' && allOptions.maxNil > 0 && (
                <BudgetFilter
                  budget={nilBudget}
                  onChange={v => {
                    setNilBudget(v)
                    refetchDebounced(activeQuery, mode, gender, { nilBudget: v, yearFilter, posFilter, confFilter })
                  }}
                  max={allOptions.maxNil}
                  filtered={results.length}
                  total={allOptions.total}
                />
              )}
              {mode === 'team' && (
                <ChipFilter
                  label="Class Year"
                  options={allOptions.years}
                  selected={yearFilter}
                  onChange={v => {
                    setYearFilter(v)
                    refetchNow(activeQuery, mode, gender, { nilBudget, yearFilter: v, posFilter, confFilter })
                  }}
                />
              )}
              {mode === 'team' && (
                <ChipFilter
                  label="Position"
                  options={allOptions.positions}
                  selected={posFilter}
                  onChange={v => {
                    setPosFilter(v)
                    refetchNow(activeQuery, mode, gender, { nilBudget, yearFilter, posFilter: v, confFilter })
                  }}
                />
              )}
              {mode === 'player' && (
                <ChipFilter
                  label="Conference"
                  options={allOptions.conferences}
                  selected={confFilter}
                  onChange={v => {
                    setConfFilter(v)
                    refetchNow(activeQuery, mode, gender, { nilBudget, yearFilter, posFilter, confFilter: v })
                  }}
                />
              )}
              <div className={`results-list${loading ? ' results-list--loading' : ''}`}>
                {results.map((item, i) =>
                  mode === 'team'
                    ? <PlayerCard
                        key={item.PlayerId ?? i}
                        player={item}
                        rank={i + 1}
                        selected={selected?.PlayerId === item.PlayerId}
                        onClick={setSelected}
                      />
                    : <TeamCard
                        key={item.TeamId ?? i}
                        team={item}
                        rank={i + 1}
                        selected={selected?.TeamId === item.TeamId}
                        onClick={setSelected}
                      />
                )}
              </div>
            </div>

            {selected && (
              <div className="detail-col">
                <DetailPanel item={selected} mode={mode} gender={gender} onClose={() => setSelected(null)} />
              </div>
            )}
          </div>
        )}
      </main>
      <ChatPopup gender={gender} />
    </div>
  )
}

function ChatPopup({ gender }) {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    if (open) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
      inputRef.current?.focus()
    }
  }, [messages, open])

  async function send() {
    const text = input.trim()
    if (!text || loading) return
    const next = [...messages, { role: 'user', content: text }]
    setMessages(next)
    setInput('')
    setLoading(true)
    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history: messages,
          gender,
        }),
      }).then(r => r.json())
      setMessages([...next, { role: 'assistant', content: res.response || res.error || 'No response.' }])
    } catch {
      setMessages([...next, { role: 'assistant', content: 'Could not reach the server.' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <button className="chat-fab" onClick={() => setOpen(o => !o)} title="Ask the analyst">
        {open ? '✕' : '💬'}
      </button>

      {open && (
        <div className="chat-popup">
          <div className="chat-header">
            <span className="chat-title">CBB Analyst</span>
            <span className="chat-subtitle">Powered by Gemini</span>
          </div>

          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="chat-empty">
                Ask anything about players, teams, or matchups.
                <div className="chat-suggestions">
                  {["Who are the best portal scorers?", "How does Maryland shoot the ball?", "Compare rim frequency for big ten teams"].map(s => (
                    <button key={s} className="chat-suggestion" onClick={() => { setInput(s); inputRef.current?.focus() }}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`chat-msg chat-msg--${m.role}`}>
                <div className="chat-bubble">{m.content}</div>
              </div>
            ))}
            {loading && (
              <div className="chat-msg chat-msg--assistant">
                <div className="chat-bubble chat-bubble--loading">
                  <span className="dot" /><span className="dot" /><span className="dot" />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="chat-input-row">
            <input
              ref={inputRef}
              className="chat-input"
              placeholder="Ask about any player or team..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && send()}
            />
            <button className="chat-send" onClick={send} disabled={!input.trim() || loading}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                <path d="M22 2L11 13M22 2L15 22l-4-9-9-4 20-7z" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </>
  )
}
