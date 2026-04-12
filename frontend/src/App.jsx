import { useState, useEffect, useRef } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, Tooltip
} from 'recharts'
import './App.css'

const API = 'http://localhost:5002'

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

function DetailPanel({ item, mode, onClose }) {
  const [overview, setOverview] = useState(null)
  const [overviewLoading, setOverviewLoading] = useState(false)

  const name = mode === 'team' ? item.Player : item.Team

  useEffect(() => {
    if (mode !== 'team') return
    setOverview(null)
    setOverviewLoading(true)
    fetch(`${API}/get_player_overview/${encodeURIComponent(name)}`)
      .then(r => r.json())
      .then(d => setOverview(d.overview || null))
      .catch(() => setOverview(null))
      .finally(() => setOverviewLoading(false))
  }, [name, mode])

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
      <div className="detail-name">{name}</div>
      {sub && <div className="detail-sub">{sub}</div>}

      {mode === 'team' && (
        <div className="overview-box">
          {overviewLoading
            ? <div className="overview-loading"><div className="spinner spinner--sm" /><span>Generating scouting report...</span></div>
            : overview
              ? <p className="overview-text">{overview}</p>
              : null
          }
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

function PlayerAvatar({ teamId, playerId, name }) {
  const [err, setErr] = useState(false)
  const src = `${IMG_BASE}/player-headshots/${teamId}-${playerId}.png`
  return err
    ? <div className="avatar avatar--fallback">{(name || '?')[0]}</div>
    : <img className="avatar" src={src} alt={name} onError={() => setErr(true)} />
}

function TeamLogo({ teamId, name }) {
  const [err, setErr] = useState(false)
  const src = `${IMG_BASE}/team-logos/${teamId}.png`
  return err
    ? <div className="avatar avatar--fallback avatar--team">{(name || '?')[0]}</div>
    : <img className="avatar avatar--team" src={src} alt={name} onError={() => setErr(true)} />
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
          </div>
        </div>
        <div className="card-sub">{player.PrevTeam}</div>
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
  return (
    <div className={`card ${selected ? 'card--active' : ''}`} onClick={() => onClick(team)}>
      <div className="card-rank">#{rank}</div>
      <TeamLogo teamId={team.TeamId} name={team.Team} />
      <div className="card-body">
        <div className="card-top">
          <span className="card-name">{team.Team}</span>
          {team.Conference && (
            <div className="card-badges">
              <span className="badge">{team.Conference}</span>
            </div>
          )}
        </div>
        {team.Explanation?.length > 0 && (
          <div className="card-tags">
            {team.Explanation.slice(0, 2).map((e, i) => (
              <span key={i} className="tag tag--sm">{e}</span>
            ))}
          </div>
        )}
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
  const [mode, setMode] = useState('team')
  const [teams, setTeams] = useState([])
  const [players, setPlayers] = useState([])
  const [query, setQuery] = useState('')
  const [rawResults, setRawResults] = useState([])
  const [weights, setWeights] = useState(DEFAULT_WEIGHTS)
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(null)
  const [error, setError] = useState(null)

  const results = rawResults.length > 0 ? applyWeights(rawResults, weights) : []

  useEffect(() => {
    fetch(`${API}/get_teams`).then(r => r.json()).then(setTeams).catch(() => {})
    fetch(`${API}/get_players`).then(r => r.json()).then(setPlayers).catch(() => {})
  }, [])

  function switchMode(m) {
    setMode(m)
    setQuery('')
    setRawResults([])
    setSelected(null)
    setError(null)
  }

  async function handleSelect(val) {
    setQuery(val)
    setSelected(null)
    setError(null)
    setLoading(true)
    try {
      const url = mode === 'team'
        ? `${API}/get_team_fit/${encodeURIComponent(val)}`
        : `${API}/get_player_fit/${encodeURIComponent(val)}`
      const data = await fetch(url).then(r => r.json())
      if (data.error) { setError(data.error); setRawResults([]) }
      else setRawResults(data.slice(0, 40))
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
              <div className="results-list">
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
                <DetailPanel item={selected} mode={mode} onClose={() => setSelected(null)} />
              </div>
            )}
          </div>
        )}
      </main>
      <ChatPopup />
    </div>
  )
}

function ChatPopup() {
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
