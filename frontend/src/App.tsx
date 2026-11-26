import { useEffect, useMemo, useState } from 'react'
import type { ChangeEvent, ReactNode } from 'react'
import {
  Activity, ArrowDownRight, ArrowRight, BarChart3, Boxes, BrainCircuit, Check,
  ChevronDown, ChevronRight, CircleHelp, Clock3, Database, Download, FileSpreadsheet,
  FlaskConical, Gauge, Layers3, Menu, Search, ShieldCheck, Sparkles, Target,
  TrendingDown, UploadCloud, X, Zap, RefreshCw, Package, Truck, Ruler, AlertTriangle
} from 'lucide-react'

const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const labels = ['Sizing', 'Damaged Item', 'Wrong Product', 'Late Delivery']
const descriptions = ['Fit, size or dimensions do not match', 'Broken, defective or damaged on arrival', 'The delivered item differs from the order', 'Arrived after the promised delivery date']
const examples = [
  'The shirt is too tight around my waist and does not fit.',
  'The glass arrived cracked and the box was damaged.',
  'I ordered black shoes but received a pair of brown sandals.',
  'My package arrived five days after the promised delivery date.'
]
type Page = 'Overview' | 'Prediction Studio' | 'Batch Intelligence' | 'Dataset Explorer' | 'Model Laboratory' | 'How It Works'
type Model = 'tfidf' | 'distilbert'
type Prediction = { predicted_category: string; confidence: number; probabilities: Record<string, number>; model_used: string; detected_language?: string; demo_fallback: boolean; explanation?: string }
type BatchRow = { text: string; actual?: string; predicted_category: string; confidence: number; model_used: string; probabilities?: Record<string, number> }
type ImportedRow = { text: string; actual?: string }
type Health = { database_connected: boolean; tfidf_model_available: boolean; distilbert_model_available: boolean; classes: string[] }
const palette = ['#82977a', '#c58c7c', '#c3a36c', '#9185a6']
const icons = [Ruler, AlertTriangle, Package, Truck]
const pct = (n: number) => `${(n * 100).toFixed(1)}%`
const safe = (n: number) => Number.isFinite(n) ? n : 0

// RFC-4180-style CSV parser: handles quoted commas, escaped quotes and newlines.
function parseCSV(input: string): string[][] {
  const rows: string[][] = []
  let row: string[] = [], cell = '', quoted = false
  for (let i = 0; i < input.length; i++) {
    const c = input[i]
    if (quoted) {
      if (c === '"' && input[i + 1] === '"') { cell += '"'; i++ }
      else if (c === '"') quoted = false
      else cell += c
    } else if (c === '"') quoted = true
    else if (c === ',') { row.push(cell); cell = '' }
    else if (c === '\n') { row.push(cell.replace(/\r$/, '')); rows.push(row); row = []; cell = '' }
    else cell += c
  }
  if (cell.length || row.length) { row.push(cell.replace(/\r$/, '')); rows.push(row) }
  return rows.filter(r => r.some(v => v.trim()))
}
function csvEscape(v: unknown) { return `"${String(v ?? '').replace(/"/g, '""')}"` }
function downloadCSV(rows: string[][], filename: string) {
  const blob = new Blob([rows.map(r => r.map(csvEscape).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob), a = document.createElement('a')
  a.href = url; a.download = filename; a.click(); URL.revokeObjectURL(url)
}
function Panel({ children, className = '' }: { children: ReactNode; className?: string }) { return <section className={`panel ${className}`}>{children}</section> }
function Heading({ eyebrow, title, detail, action }: { eyebrow: string; title: string; detail?: string; action?: React.ReactNode }) {
  return <div className="section-heading"><div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2>{detail && <p>{detail}</p>}</div>{action}</div>
}
function Kpi({ icon: Icon, label, value, hint, tone = 0 }: any) {
  return <div className="kpi-card"><div className={`kpi-icon tone-${tone}`}><Icon size={18}/></div><span>{label}</span><strong>{value}</strong><small>{hint}</small></div>
}
function Bar({ value, color }: { value: number; color: string }) { return <div className="bar-bg"><i style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: color }}/></div> }

export default function App() {
  const [entered, setEntered] = useState(false)
  const [page, setPage] = useState<Page>('Overview')
  const [health, setHealth] = useState<Health | null>(null)
  const [history, setHistory] = useState<any[]>([])
  const [text, setText] = useState(examples[0])
  const [model, setModel] = useState<Model>('tfidf')
  const [result, setResult] = useState<Prediction | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [fileName, setFileName] = useState('')
  const [imported, setImported] = useState<ImportedRow[]>([])
  const [batch, setBatch] = useState<BatchRow[]>([])
  const [batchBusy, setBatchBusy] = useState(false)
  const [batchProgress, setBatchProgress] = useState(0)
  const [batchError, setBatchError] = useState('')
  const [textColumn, setTextColumn] = useState('text')
  const [labelColumn, setLabelColumn] = useState('')
  const [headers, setHeaders] = useState<string[]>([])
  const [selectedCategory, setSelectedCategory] = useState('Sizing')
  const [query, setQuery] = useState('')
  const [filterCategory, setFilterCategory] = useState('All categories')
  const [menuOpen, setMenuOpen] = useState(false)
  const [metrics, setMetrics] = useState<any>(null)

  const nav: { name: Page; icon: any; group: string }[] = [
    { name: 'Overview', icon: Gauge, group: 'WORKSPACE' },
    { name: 'Prediction Studio', icon: BrainCircuit, group: 'WORKSPACE' },
    { name: 'Batch Intelligence', icon: UploadCloud, group: 'WORKSPACE' },
    { name: 'Dataset Explorer', icon: Database, group: 'INTELLIGENCE' },
    { name: 'Model Laboratory', icon: FlaskConical, group: 'INTELLIGENCE' },
    { name: 'How It Works', icon: CircleHelp, group: 'LEARN' }
  ]
  async function get(path: string) {
    const r = await fetch(API + path)
    if (!r.ok) throw new Error(`API ${r.status}`)
    return r.json()
  }
  async function refresh() {
    try {
      const [h, p, m] = await Promise.all([get('/api/health'), get('/api/predictions/recent?limit=500'), get('/api/evaluation/latest')])
      setHealth(h); setHistory(p.items || []); setMetrics(m)
    } catch { setHealth(null) }
  }
  useEffect(() => {
  refresh()
  const t = window.setInterval(refresh, 15000)
  return () => window.clearInterval(t)
}, [])

useEffect(() => {
  window.history.replaceState(
    { ...window.history.state, stanvinView: 'landing' },
    '',
    window.location.href
  )

  const handlePopState = (event: PopStateEvent) => {
    const isOverview = event.state?.stanvinView === 'overview'
    setEntered(isOverview)

    if (isOverview) {
      setPage('Overview')
    }

    setMenuOpen(false)
  }

  window.addEventListener('popstate', handlePopState)
  return () => window.removeEventListener('popstate', handlePopState)
}, [])

const go = (p: Page) => {
  setPage(p)
  setMenuOpen(false)
  setError('')
  setBatchError('')
}

const enterWorkspace = (target: Page = 'Overview') => {
  window.history.pushState(
    { stanvinView: 'overview' },
    '',
    window.location.href
  )
  setPage(target)
  setEntered(true)
}

const returnToLanding = () => {
  window.history.back()
}

  async function predict() {
    if (text.trim().length < 3) { setError('Enter at least 3 characters.'); return }
    setBusy(true); setError(''); setResult(null)
    try {
      const r = await fetch(API + '/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, model }) })
      const d = await r.json(); if (!r.ok) throw new Error(d.detail || 'Prediction failed')
      setResult(d); refresh()
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not reach the API. Start FastAPI first.') }
    finally { setBusy(false) }
  }

  async function onFile(file?: File) {
    if (!file) return
    setBatchError(''); setBatch([]); setImported([]); setFileName(file.name)
    try {
      const raw = await file.text(), parsed = parseCSV(raw)
      if (parsed.length < 2) throw new Error('The CSV needs a header row and at least one data row.')
      const heads = parsed[0].map(h => h.trim().replace(/^\uFEFF/, ''))
      const normalized = heads.map(h => h.toLowerCase().trim())
      const ti = normalized.findIndex(h => ['text', 'comment', 'return_comment', 'customer_comment', 'review', 'review_text', 'feedback'].includes(h))
      const li = normalized.findIndex(h => ['label', 'actual_label', 'category', 'return_reason', 'ground_truth'].includes(h))
      setHeaders(heads); setTextColumn(heads[ti >= 0 ? ti : 0]); setLabelColumn(li >= 0 ? heads[li] : '')
      const records = parsed.slice(1).map(r => ({ text: (r[ti >= 0 ? ti : 0] || '').trim(), actual: li >= 0 ? (r[li] || '').trim() : undefined })).filter(r => r.text.length >= 3)
      if (!records.length) throw new Error('No usable review text found. Choose the correct text column.')
      setImported(records)
    } catch (e) { setBatchError(e instanceof Error ? e.message : 'Could not read this CSV.') }
  }
  function updateColumn(kind: 'text' | 'label', value: string) {
    if (kind === 'text') setTextColumn(value); else setLabelColumn(value)
    const fileInput = document.getElementById('csv-file') as HTMLInputElement | null
    if (!fileInput?.files?.[0]) return
    fileInput.files[0].text().then(raw => {
      const parsed = parseCSV(raw), heads = parsed[0].map(h => h.trim().replace(/^\uFEFF/, ''))
      const ti = heads.indexOf(kind === 'text' ? value : textColumn)
      const li = heads.indexOf(kind === 'label' ? value : labelColumn)
      setImported(parsed.slice(1).map(r => ({ text: (r[ti] || '').trim(), actual: li >= 0 ? (r[li] || '').trim() : undefined })).filter(r => r.text.length >= 3))
    })
  }
  async function runBatch() {
    if (!imported.length) return
    setBatchBusy(true); setBatchError(''); setBatch([]); setBatchProgress(0)
    const all: BatchRow[] = []
    try {
      // Backend currently accepts max 500 per request. Send every valid row in sequential chunks.
      const chunkSize = 500, totalChunks = Math.ceil(imported.length / chunkSize)
      for (let offset = 0; offset < imported.length; offset += chunkSize) {
        const chunk = imported.slice(offset, offset + chunkSize)
        const r = await fetch(API + '/api/predict/batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ texts: chunk.map(x => x.text), model }) })
        const d = await r.json()
        if (!r.ok) throw new Error(d.detail || `Batch request failed at row ${offset + 1}`)
        const results = (d.results || []) as Prediction[]
        if (results.length !== chunk.length) throw new Error(`Backend returned ${results.length} of ${chunk.length} rows in this batch.`)
        results.forEach((p, i) => all.push({ text: chunk[i].text, actual: chunk[i].actual, predicted_category: p.predicted_category, confidence: p.confidence, model_used: p.model_used, probabilities: p.probabilities }))
        setBatch([...all]); setBatchProgress(Math.round(Math.min(all.length, imported.length) / imported.length * 100))
      }
      if (all.length !== imported.length) throw new Error(`Only ${all.length} of ${imported.length} rows were classified.`)
    } catch (e) { setBatchError(e instanceof Error ? e.message : 'Batch classification failed.') }
    finally { setBatchBusy(false) }
  }
  function exportBatch() {
    downloadCSV([['text', 'actual_label', 'predicted_category', 'confidence', 'model_used', 'probabilities'], ...batch.map(r => [r.text, r.actual || '', r.predicted_category, r.confidence, r.model_used, JSON.stringify(r.probabilities || {})])], 'StanVin_batch_predictions.csv')
  }

  const total = batch.length
  const categoryStats = useMemo(() => labels.map((label, i) => {
    const rows = batch.filter(r => r.predicted_category === label)
    const avg = rows.length ? rows.reduce((s, r) => s + safe(r.confidence), 0) / rows.length : 0
    return { label, count: rows.length, share: total ? rows.length / total : 0, avg, color: palette[i] }
  }), [batch, total])
  const avgConfidence = total ? batch.reduce((s, r) => s + safe(r.confidence), 0) / total : 0
  const lowConfidence = batch.filter(r => r.confidence < 0.55).length
  const labeled = batch.filter(r => r.actual && labels.includes(r.actual))
  const correct = labeled.filter(r => r.actual === r.predicted_category).length
  const accuracy = labeled.length ? correct / labeled.length : null
  const filteredBatch = batch.filter(r => (filterCategory === 'All categories' || r.predicted_category === filterCategory) && (!query || r.text.toLowerCase().includes(query.toLowerCase())))
  const selectedRows = batch.filter(r => r.predicted_category === selectedCategory)
  const classRows = labels.map((l, i) => ({ label: l, count: batch.filter(r => r.predicted_category === l).length, color: palette[i] }))
  const dbTotal = history.length
  const recentCounts = labels.map((label, i) => ({ label, count: history.filter(r => r.category === label).length, color: palette[i] }))

  return !entered ? (
  <Landing
    onEnter={() => enterWorkspace('Overview')}
    onHow={() => enterWorkspace('How It Works')}
  />
) : (
    <div className="app-shell">
      <aside className={`sidebar ${menuOpen ? 'open' : ''}`}>
        <div className="brand"><div className="brand-mark"><Boxes size={20}/></div><div><b>StanVin</b><small>RETURNS INTELLIGENCE</small></div><button className="mobile-close" onClick={() => setMenuOpen(false)}><X size={18}/></button></div>
        {['WORKSPACE', 'INTELLIGENCE', 'LEARN'].map(group => <div className="nav-group" key={group}><span className="nav-label">{group}</span>{nav.filter(n => n.group === group).map(n => { const I = n.icon; return <button key={n.name} className={`nav-item ${page === n.name ? 'active' : ''}`} onClick={() => go(n.name)}><I size={17}/><span>{n.name}</span>{page === n.name && <ChevronRight size={14}/>}</button> })}</div>)}
        <div className="sidebar-foot"><span className={`live-dot ${health?.database_connected ? '' : 'offline'}`}/><div><b>Workspace status</b><small>{health?.database_connected ? 'PostgreSQL connected' : 'Database unavailable'}</small></div></div>
        <button className="back-home" onClick={returnToLanding}>
  ← Back to website
</button>
      </aside>
      <main className="main-area">
        <header className="topbar"><button className="mobile-menu" onClick={() => setMenuOpen(true)}><Menu/></button><div className="breadcrumb">StanVin <span>/</span> {page}</div><div className="top-actions"><span className={`connection ${health?.database_connected ? 'connected' : ''}`}><i/>{health?.database_connected ? 'DATABASE LIVE' : 'DATABASE CHECK'}</span><button className="icon-button" onClick={refresh} title="Refresh"><RefreshCw size={16}/></button><div className="avatar">R</div></div></header>
        <div className="page-content">
          {page === 'Overview' && <Overview history={history} total={dbTotal} health={health} counts={recentCounts} onNavigate={go} />}
          {page === 'Prediction Studio' && <><Heading eyebrow="NLP WORKSPACE" title="Prediction Studio" detail="Classify an individual customer comment and inspect the model's score distribution."/><div className="two-col"><Panel><label className="field-label">CUSTOMER RETURN COMMENT</label><textarea className="comment-input" value={text} onChange={e => setText(e.target.value)} maxLength={3000}/><div className="field-meta"><span>{text.length}/3000 characters</span><button className="subtle-link" onClick={() => setText(examples[Math.floor(Math.random() * examples.length)])}>Load example</button></div><label className="field-label">CLASSIFICATION MODEL</label><ModelPicker model={model} setModel={setModel} health={health}/><button className="primary-button full" onClick={predict} disabled={busy}>{busy ? 'Classifying…' : 'Classify return reason'} <ArrowRight size={16}/></button>{error && <div className="error-box">{error}</div>}</Panel><Panel className="result-panel">{result ? <PredictionResult result={result}/> : <div className="empty-state"><div className="empty-orb"><BrainCircuit size={30}/></div><b>Prediction results appear here</b><p>Run a classification to see the predicted reason, confidence and score breakdown.</p></div>}</Panel></div></>}
          {page === 'Batch Intelligence' && <><Heading eyebrow="DATA IMPORT & ANALYTICS" title="Returns intelligence workspace" detail="Upload a CSV to classify every valid review and build an interactive dashboard from the results."/><Panel className="upload-panel"><div className="upload-layout"><label className="drop-zone"><input id="csv-file" type="file" accept=".csv,text/csv" onChange={(e: ChangeEvent<HTMLInputElement>) => onFile(e.target.files?.[0])}/><div className="upload-symbol"><FileSpreadsheet size={25}/></div><b>Drop a CSV here or browse files</b><span>Any CSV with a customer review / text column. Up to your browser's available memory.</span></label><div className="upload-instructions"><b>What StanVin will do</b><ul><li>Detect the review text column automatically</li><li>Classify every valid row in chunks</li><li>Build category, confidence and quality insights</li><li>Let you drill into individual customer comments</li></ul></div></div>
            {fileName && <div className="file-info"><FileSpreadsheet size={18}/><div><b>{fileName}</b><small>{imported.length.toLocaleString()} valid reviews detected · {headers.length} columns</small></div><button className="text-button" onClick={() => { setImported([]); setBatch([]); setFileName('') }}>Remove</button></div>}
            {headers.length > 0 && <div className="column-selectors"><label>Review text column<select value={textColumn} onChange={e => updateColumn('text', e.target.value)}>{headers.map(h => <option key={h}>{h}</option>)}</select></label><label>Actual label column (optional)<select value={labelColumn} onChange={e => updateColumn('label', e.target.value)}><option value="">Not available</option>{headers.map(h => <option key={h}>{h}</option>)}</select></label></div>}
            {imported.length > 0 && <div className="batch-actions"><div><b>{imported.length.toLocaleString()} reviews ready</b><small>All valid rows will be classified. No 500-row truncation.</small></div><select value={model} onChange={e => setModel(e.target.value as Model)}><option value="tfidf">TF-IDF + Logistic Regression</option><option value="distilbert" disabled={!health?.distilbert_model_available}>Fine-tuned DistilBERT</option></select><button className="primary-button" onClick={runBatch} disabled={batchBusy}>{batchBusy ? 'Processing…' : 'Build intelligence dashboard'} <ArrowRight size={16}/></button></div>}
            {batchBusy && <div className="progress-wrap"><div><span>Classifying all reviews</span><b>{batchProgress}%</b></div><Bar value={batchProgress} color="#748d70"/></div>}
            {batchError && <div className="error-box">{batchError}</div>}
          </Panel>
          {batch.length > 0 && <BatchDashboard rows={batch} stats={categoryStats} average={avgConfidence} low={lowConfidence} accuracy={accuracy} labeled={labeled.length} selected={selectedCategory} setSelected={setSelectedCategory} onExport={exportBatch} />}
          </>}
          {page === 'Dataset Explorer' && <><Heading eyebrow="DATA EXPLORER" title="Inspect classified comments" detail="Search and filter the uploaded dataset, inspect confidence and open category-level details."/><DatasetTable rows={batch} filtered={filteredBatch} query={query} setQuery={setQuery} filter={filterCategory} setFilter={setFilterCategory} onCategory={c => { setSelectedCategory(c); go('Batch Intelligence') }}/></>}
          {page === 'Model Laboratory' && <><Heading eyebrow="MODEL PERFORMANCE" title="Model laboratory" detail="Compare TF-IDF + Logistic Regression with your fine-tuned DistilBERT model on the same held-out test split."/><div className="model-grid"><Panel><div className="model-head"><div className="model-icon"><Layers3/></div><span className="status-tag">BASELINE</span></div><h3>TF-IDF + Logistic Regression</h3><p>Converts text into word and bigram importance features, then learns a linear decision boundary for the four return classes.</p><div className="model-facts"><span>Fast inference</span><span>Interpretable feature weights</span><span>Low resource footprint</span></div><b className="availability">{health?.tfidf_model_available ? 'Model artifact available' : 'Model artifact missing'}</b></Panel><Panel><div className="model-head"><div className="model-icon"><BrainCircuit/></div><span className="status-tag advanced">TRANSFORMER</span></div><h3>Fine-tuned DistilBERT</h3><p>Uses contextual token representations to interpret the meaning of a comment beyond isolated word counts.</p><div className="model-facts"><span>Context-aware</span><span>Fine-tuned locally</span><span>Higher compute cost</span></div><b className={`availability ${health?.distilbert_model_available ? 'ready' : ''}`}>{health?.distilbert_model_available ? 'Model artifact available' : 'Model artifact missing'}</b></Panel></div><Panel><h3 className="panel-title">Head-to-head evaluation</h3><p className="muted">Both models were evaluated on the same 3,000-row synthetic test split (750 examples per class).</p><div className="table-scroll"><table><thead><tr><th>Model</th><th>Accuracy</th><th>Macro precision</th><th>Macro recall</th><th>Macro F1</th><th>Weighted F1</th></tr></thead><tbody><tr><td>TF-IDF + Logistic Regression</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>Fine-tuned DistilBERT</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr></tbody></table></div><p className="footnote">These are development results on synthetic data, not evidence of real-world performance. Validate with representative, independently labelled customer returns before deployment.</p></Panel><div className="analytics-grid"><ConfusionMatrix title="TF-IDF confusion matrix" /><ConfusionMatrix title="DistilBERT confusion matrix" /></div><Panel><h3 className="panel-title">Latest saved evaluation snapshot</h3>{metrics?.available ? <div className="metric-strip"><Kpi icon={Target} label="Test accuracy" value={pct(metrics.accuracy || 0)} hint="Saved evaluation"/><Kpi icon={Activity} label="Macro F1" value={pct(metrics.macro_f1 || 0)} hint="Equal class weighting"/><Kpi icon={BarChart3} label="Weighted F1" value={pct(metrics.weighted_f1 || 0)} hint="Support-weighted"/></div> : <p className="muted">No separate backend evaluation snapshot is available. The comparison above uses the generated model-comparison report.</p>}</Panel></>}
          {page === 'How It Works' && <><Heading eyebrow="PRODUCT & METHODOLOGY" title="How StanVin turns comments into return intelligence" detail="An interactive walkthrough of the multilingual classification pipeline, from customer text to model evaluation and operational action."/><InteractivePipeline/><div className="value-grid"><Panel><h3>Why returns intelligence matters</h3><p>Return reasons are often buried in free-text notes, support tickets and reviews. Consistent categories help teams identify recurring product, quality, catalog and delivery issues.</p></Panel><Panel><h3>What the four classes mean</h3><p><b>Sizing:</b> fit or dimensions are wrong. <b>Damaged Item:</b> broken or defective on arrival. <b>Wrong Product:</b> delivered item differs from the order. <b>Late Delivery:</b> arrives after the promised date.</p></Panel><Panel><h3>What the models do not do</h3><p>They classify the stated return reason. They do not verify order status, prove a product is defective, approve refunds, translate every language perfectly or replace human review.</p></Panel></div></>}
        </div>
        <footer className="app-footer"><span>StanVin · E-COMMERCE RETURNS INTELLIGENCE</span><span>LOCAL NLP · AUDITABLE PREDICTIONS · HUMAN REVIEW</span></footer>
      </main>
    </div>
  )
}


function InteractivePipeline() {
  const steps = [
    {title:'1. Capture customer feedback',tag:'INPUT',detail:'StanVin accepts a customer comment in Prediction Studio or a CSV file in Batch Intelligence. The original text is retained for review.',example:'“నేను నల్ల షూస్ ఆర్డర్ చేశాను కానీ తెల్ల సాండల్స్ వచ్చాయి.”',output:'Raw customer comment and optional CSV ground-truth label',note:'Supported input styles include English, Telugu, Hindi, Hinglish and Romanized Telugu/Hindi.'},
    {title:'2. Detect language',tag:'LANGUAGE',detail:'The API checks Telugu and Devanagari Unicode scripts first. Latin-script text is checked for common Romanized Telugu and Hindi cues.',example:'Telugu script → Telugu  |  “naaku ... kavali” → Romanized Telugu  |  “mujhe ... chahiye” → Hinglish / Romanized Hindi',output:'Detected-language label; original text remains unchanged',note:'This is a lightweight heuristic, not full translation. Mixed-language or uncommon transliterations can be ambiguous.'},
    {title:'3. Prepare model input',tag:'PREPROCESS',detail:'TF-IDF lowercases text, removes URLs and punctuation while preserving Unicode letters, then builds word and bigram features. DistilBERT tokenizes the original sentence into contextual subword tokens.',example:'“Shoes size chart se bhi chhote nikle” → TF-IDF features or DistilBERT token IDs and attention mask.',output:'Sparse TF-IDF feature vector or transformer token tensors',note:'The two models use different text representations, so their predictions and scores need not match.'},
    {title:'4. Classify with the selected model',tag:'MODEL',detail:'TF-IDF + Logistic Regression is the fast baseline. Fine-tuned multilingual DistilBERT uses contextual transformer representations and is available after its trained checkpoint is saved.',example:'“The parcel arrived with a cracked screen.” → expected category: Damaged Item.',output:'Sizing, Damaged Item, Wrong Product or Late Delivery',note:'Both models are trained on the same labelled dataset and fixed split, but different algorithms can disagree on individual comments.'},
    {title:'5. Score and review confidence',tag:'SCORES',detail:'Prediction Studio shows the predicted category, top-class score and score distribution across all four categories. Low-confidence results should be reviewed by a person.',example:'Sizing 0.72 · Damaged Item 0.12 · Wrong Product 0.09 · Late Delivery 0.07',output:'Predicted category and per-class scores',note:'Model scores are not guarantees of correctness and may not be calibrated probabilities.'},
    {title:'6. Process a batch',tag:'BATCH',detail:'Batch Intelligence reads each valid text row, applies the selected model and summarizes category distribution, average confidence and agreement with supplied labels.',example:'CSV columns: text, label, language. The label is expected ground truth for evaluation, not input to the prediction.',output:'Row-level predictions, dashboard and downloadable results',note:'Different model selections can produce different outputs. Label agreement compares predictions to CSV labels; it is not the same as held-out test accuracy.'},
    {title:'7. Evaluate and improve',tag:'EVALUATE',detail:'Train both models on the same versioned dataset and fixed train/test split. Compare accuracy, macro-F1, weighted-F1 and confusion matrices, then review errors and correct verified labels.',example:'A verified sizing complaint predicted as damaged item is a false positive to inspect before retraining.',output:'Evaluation reports and a human-reviewed improvement queue',note:'The included 1,500 examples are synthetic development data. Replace them with representative, manually verified customer examples before real-world claims.'}
  ];
  const [active,setActive]=useState(0);
  const [playing,setPlaying]=useState(false);
  useEffect(()=>{
    if(!playing) return;
    const timer=window.setInterval(()=>{
      setActive(current=>{
        if(current>=steps.length-1){setPlaying(false);return 0;}
        return current+1;
      });
    },4200);
    return ()=>window.clearInterval(timer);
  },[playing]);
  const step=steps[active];
  return <Panel>
    <div className="interactive-pipeline-head">
      <div><span className="eyebrow">INTERACTIVE WALKTHROUGH</span><h3 className="panel-title">Explore the StanVin pipeline</h3><p className="muted">Select a stage or play the guided walkthrough.</p></div>
      <button className="pipeline-play" onClick={()=>setPlaying(v=>!v)}>{playing?'Pause walkthrough':'▶ Play walkthrough'}</button>
    </div>
    <div className="pipeline-progress" aria-label={`Step ${active+1} of ${steps.length}`}>
      {steps.map((item,index)=><button key={item.tag} className={`pipeline-step-dot ${index===active?'active':''} ${index<active?'complete':''}`} onClick={()=>{setActive(index);setPlaying(false)}} aria-label={`Show ${item.title}`}><span>{String(index+1).padStart(2,'0')}</span><small>{item.tag}</small></button>)}
    </div>
    <div className="pipeline-detail" key={step.tag}>
      <div className="pipeline-detail-top"><span className="pipeline-stage-tag">{step.tag}</span><span className="muted">STEP {active+1} / {steps.length}</span></div>
      <h3>{step.title}</h3><p>{step.detail}</p>
      <div className="pipeline-example"><b>Example</b><p>{step.example}</p></div>
      <div className="pipeline-output"><b>Output</b><p>{step.output}</p></div>
      <p className="pipeline-note"><b>Important:</b> {step.note}</p>
    </div>
    <div className="pipeline-controls">
      <button className="pipeline-nav" disabled={active===0} onClick={()=>{setActive(v=>Math.max(0,v-1));setPlaying(false)}}>← Previous</button>
      <div className="pipeline-count">{active+1} of {steps.length}</div>
      <button className="pipeline-nav primary" onClick={()=>{setActive(v=>v===steps.length-1?0:v+1);setPlaying(false)}}>{active===steps.length-1?'Start again ↻':'Next step →'}</button>
    </div>
  </Panel>;
}

function Landing({ onEnter, onHow }: { onEnter: () => void; onHow: () => void }) {
  return <div className="landing-page">
    <nav className="landing-nav"><div className="brand light"><div className="brand-mark"><Boxes size={20}/></div><div><b>StanVin</b><small>RETURNS INTELLIGENCE</small></div></div><div className="landing-links"><a href="#problem">The problem</a><a href="#platform">Platform</a><a href="#workflow">How it works</a></div><button className="nav-cta" onClick={onEnter}>Explore demo <ArrowRight size={15}/></button></nav>
    <section className="landing-hero"><div className="landing-copy"><span className="eyebrow light-eyebrow"><Sparkles size={14}/> CUSTOMER SIGNAL. BUSINESS CLARITY.</span><h1>Every return has<br/><em>a reason. Find it.</em></h1><p>StanVin turns unstructured e-commerce return comments into clear, measurable signals — helping teams understand product fit, quality, catalog accuracy and delivery experience.</p><div className="landing-actions"><button className="primary-button" onClick={onEnter}>Explore interactive demo <ArrowRight size={17}/></button><button className="outline-light" onClick={onHow}>Explore the method</button></div><div className="trust-line"><span><Check size={14}/> Local NLP models</span><span><Check size={14}/> Four actionable categories</span><span><Check size={14}/> Review-level traceability</span></div></div>
      <div className="hero-3d" aria-label="Abstract 3D return intelligence visualization"><div className="scene-grid"/><div className="orbit3d orbit-one"/><div className="orbit3d orbit-two"/><div className="glass-cube cube-a"><span>RETURN</span><b>IQ</b></div><div className="glass-cube cube-b"><Activity size={25}/><small>INSIGHT</small></div><div className="glass-cube cube-c"><span>4</span><small>REASONS</small></div><div className="sphere"><div className="sphere-shine"/></div><div className="hero-caption"><span className="live-dot"/> TEXT → SIGNAL → ACTION</div></div>
    </section>
    <section className="landing-proof"><div><strong>01</strong><span>Understand the reason behind returns</span></div><div><strong>02</strong><span>Spot repeat issues across categories</span></div><div><strong>03</strong><span>Prioritize operational improvements</span></div></section>
    <section className="landing-section" id="problem"><span className="eyebrow">THE BUSINESS PROBLEM</span><h2>Returns are expensive.<br/><em>Unstructured feedback hides why.</em></h2><p>Teams may receive hundreds or thousands of return comments, but those comments are difficult to compare consistently. StanVin helps turn that text into a structured view that teams can investigate.</p><div className="problem-cards"><article><Ruler/><h3>Fit & sizing friction</h3><p>Find recurring fit complaints that may point to size charts, product dimensions or unclear product guidance.</p></article><article><ShieldCheck/><h3>Product quality signals</h3><p>Surface repeated reports of damage or defects for quality and packaging investigation.</p></article><article><Package/><h3>Catalog mismatches</h3><p>Identify reports that the received item does not match what the customer ordered.</p></article><article><Truck/><h3>Delivery experience</h3><p>Understand how often late delivery appears in customer return explanations.</p></article></div></section>
    <section className="landing-section platform-section" id="platform"><span className="eyebrow">THE StanVin PLATFORM</span><h2>One workspace. From comment<br/>to operational insight.</h2><div className="platform-grid"><div className="platform-feature"><BarChart3/><b>Returns intelligence dashboard</b><p>Category mix, confidence distribution, low-confidence review queues and category drill-downs.</p></div><div className="platform-feature"><UploadCloud/><b>CSV-to-dashboard workflow</b><p>Upload review data and classify all valid rows in manageable chunks, with progress and export.</p></div><div className="platform-feature"><BrainCircuit/><b>Two local NLP approaches</b><p>Compare a TF-IDF baseline with a fine-tuned DistilBERT model without paid inference APIs.</p></div><div className="platform-feature"><Search/><b>Review-level traceability</b><p>Filter, search and inspect the original customer comment behind each predicted category.</p></div></div></section>
    <section className="landing-section workflow-section" id="workflow"><span className="eyebrow">HOW IT WORKS</span><h2>Turn customer language<br/>into a clearer action plan.</h2><div className="landing-steps">{[['01','Upload','Bring a CSV of return comments or test a single review.'],['02','Classify','Run local NLP models to assign one of four return reasons.'],['03','Explore','Inspect the category distribution, confidence and examples.'],['04','Improve','Use repeated signals to guide product, quality and delivery reviews.']].map(s=><div key={s[0]}><b>{s[0]}</b><h3>{s[1]}</h3><p>{s[2]}</p></div>)}</div></section>
    <section className="landing-bottom"><div><span className="eyebrow light-eyebrow">SEE THE SIGNALS IN YOUR DATA</span><h2>Make every return comment<br/><em>count toward a better decision.</em></h2><p>Explore the working StanVin workspace.</p></div><button className="primary-button" onClick={onEnter}>Enter StanVin <ArrowRight size={17}/></button></section>
    <footer className="landing-footer"><b>StanVin</b><span>Local NLP · Four return reason classes · Decision support for e-commerce teams</span></footer>
  </div>
}

function Overview({ history, total, health, counts, onNavigate }: any) {
  const avg = total ? history.reduce((s: number, r: any) => s + safe(r.confidence), 0) / total : 0
  const low = history.filter((r: any) => r.confidence < .55).length
  return <>
    <div className="overview-welcome"><div><span className="eyebrow">EXECUTIVE WORKSPACE</span><h1>Returns intelligence <em>at a glance.</em></h1><p>Monitor classified return reasons, confidence and operational signals from your saved prediction history or uploaded datasets.</p></div><button className="primary-button" onClick={() => onNavigate('Batch Intelligence')}><UploadCloud size={16}/> Analyze a CSV</button></div>
    <div className="kpi-grid"><Kpi icon={Database} label="Recent saved predictions" value={total.toLocaleString()} hint="PostgreSQL prediction history" tone={0}/><Kpi icon={Layers3} label="Return categories" value="04" hint="Standardized reason taxonomy" tone={1}/><Kpi icon={Target} label="Average confidence" value={total ? pct(avg) : '—'} hint="Across recent saved predictions" tone={2}/><Kpi icon={AlertTriangle} label="Review recommended" value={low.toLocaleString()} hint="Predictions below 55% confidence" tone={3}/></div>
    <div className="analytics-grid"><Panel><div className="panel-top"><div><span className="eyebrow">CATEGORY MIX</span><h3>What is driving returns?</h3><p>Click a category to inspect its customer comments.</p></div><span className="data-tag">SAVED HISTORY</span></div><div className="category-bars">{counts.map((c: any, i: number) => <button className="category-bar-row" key={c.label} onClick={() => onNavigate('Dataset Explorer')}><span className="cat-name"><i style={{ background: c.color }}/>{c.label}</span><div className="category-bar-track"><i style={{ width: `${total ? c.count / total * 100 : 0}%`, background: c.color }}/></div><b>{c.count}</b></button>)}</div>{!total && <div className="inline-empty">Your live analytics will populate as predictions are saved. Upload a CSV to create a dataset-specific dashboard.</div>}</Panel><Panel><div className="panel-top"><div><span className="eyebrow">OPERATING SIGNALS</span><h3>Where to investigate</h3><p>Suggested areas linked to return categories.</p></div><Sparkles size={19}/></div><div className="signal-list"><div><Ruler/><span><b>Fit consistency</b><small>Review size guides and product measurements.</small></span></div><div><ShieldCheck/><span><b>Product condition</b><small>Investigate damage reports and packaging.</small></span></div><div><Package/><span><b>Catalog accuracy</b><small>Check product images, variants and pick accuracy.</small></span></div><div><Truck/><span><b>Delivery promise</b><small>Compare promised dates with carrier performance.</small></span></div></div></Panel></div>
    <Panel><div className="panel-top"><div><span className="eyebrow">RECENT ACTIVITY</span><h3>Latest classified comments</h3></div><button className="subtle-link" onClick={() => onNavigate('Dataset Explorer')}>Open explorer <ArrowRight size={14}/></button></div>{history.length ? <div className="table-scroll"><table><thead><tr><th>Customer comment</th><th>Predicted reason</th><th>Confidence</th><th>Model</th></tr></thead><tbody>{history.slice(0, 8).map((r: any) => <tr key={r.id}><td className="comment-cell">{r.text}</td><td><span className="category-pill">{r.category}</span></td><td>{pct(r.confidence)}</td><td>{r.model}</td></tr>)}</tbody></table></div> : <div className="inline-empty">No saved predictions yet. Run a single prediction or upload a CSV to begin.</div>}</Panel>
    <div className="system-note"><span className={`live-dot ${health?.database_connected ? '' : 'offline'}`}/>{health?.database_connected ? 'PostgreSQL connected' : 'PostgreSQL connection not detected'} · Dashboard cards use saved prediction history; CSV analytics are generated separately in Batch Intelligence.</div>
  </>
}

function BatchDashboard({ rows, stats, average, low, accuracy, labeled, selected, setSelected, onExport }: any) {
  const chosen = rows.filter((r: BatchRow) => r.predicted_category === selected)
  const stopWords = new Set(['the','and','that','this','with','was','were','for','from','but','not','too','my','i','it','is','a','an','to','of','in','on','me','has','have','had','because','arrived','order','ordered','item','product','return','received'])
  const wordCounts = new Map<string, number>()
  chosen.forEach((r: BatchRow) => (r.text.toLowerCase().match(/[a-z]{3,}/g) || []).forEach(w => { if (!stopWords.has(w)) wordCounts.set(w, (wordCounts.get(w) || 0) + 1) }))
  const topTerms = [...wordCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([word, count]) => `${word} · ${count}`)
  const [showAll, setShowAll] = useState(false)
  const max = Math.max(1, ...stats.map((s: any) => s.count))
  return <div className="batch-dashboard">
    <div className="dashboard-title-row"><div><span className="eyebrow">DATASET INTELLIGENCE</span><h2>Analysis dashboard</h2><p>{rows.length.toLocaleString()} reviews classified · all uploaded valid rows included</p></div><button className="secondary-button" onClick={onExport}><Download size={15}/> Export results</button></div>
    <div className="kpi-grid five"><Kpi icon={FileSpreadsheet} label="Reviews classified" value={rows.length.toLocaleString()} hint="Rows processed successfully" tone={0}/><Kpi icon={Target} label="Average confidence" value={pct(average)} hint="Mean top-class score" tone={1}/><Kpi icon={AlertTriangle} label="Low confidence" value={low.toLocaleString()} hint="Below 55% · review queue" tone={3}/><Kpi icon={Layers3} label="Categories found" value={stats.filter((s: any) => s.count > 0).length} hint="Of 4 supported categories" tone={2}/><Kpi icon={Check} label="Label agreement" value={accuracy === null ? 'N/A' : pct(accuracy)} hint={labeled ? `${labeled} rows with known labels` : 'No ground-truth label column'} tone={0}/></div>
    <div className="analytics-grid batch-charts"><Panel><div className="panel-top"><div><span className="eyebrow">PREDICTED DISTRIBUTION</span><h3>Where are returns concentrated?</h3><p>Select a category to drill into its reviews.</p></div><span className="data-tag">CLICK TO FILTER</span></div><div className="distribution-chart">{stats.map((s: any, i: number) => <button className={`distribution-column ${selected === s.label ? 'chosen' : ''}`} key={s.label} onClick={() => { setSelected(s.label); setShowAll(false) }}><div className="column-value">{s.count}</div><div className="column-track"><i style={{ height: `${Math.max(2, s.count / max * 100)}%`, background: s.color }}/></div><b>{s.label}</b><small>{pct(rows.length ? s.count / rows.length : 0)}</small></button>)}</div></Panel>
      <Panel><div className="panel-top"><div><span className="eyebrow">CONFIDENCE PROFILE</span><h3>Model confidence by category</h3><p>Average top-class probability for each predicted group.</p></div></div><div className="confidence-bars">{stats.map((s: any) => <div key={s.label}><div><span>{s.label}</span><b>{pct(s.avg)}</b></div><Bar value={s.avg * 100} color={s.color}/></div>)}</div><div className="confidence-foot">Confidence is a model score, not a guarantee of correctness.</div></Panel></div>
    <Panel className="category-drill"><div className="drill-head"><div><span className="eyebrow">CATEGORY DRILL-DOWN</span><h3>{selected}</h3><p>{descriptions[labels.indexOf(selected)]} · {chosen.length.toLocaleString()} reviews ({pct(rows.length ? chosen.length / rows.length : 0)} of dataset)</p></div><span className="drill-count">{chosen.length}<small>reviews</small></span></div>
      <div className="drill-insights"><div><span>Average confidence</span><b>{pct(chosen.length ? chosen.reduce((s: number, r: BatchRow) => s + r.confidence, 0) / chosen.length : 0)}</b></div><div><span>Below 55% confidence</span><b>{chosen.filter((r: BatchRow) => r.confidence < .55).length}</b></div><div><span>Known labels</span><b>{chosen.filter((r: BatchRow) => r.actual).length}</b></div></div>
      <div className="sample-terms"><span>FREQUENT WORDS IN THIS CATEGORY</span>{topTerms.length ? topTerms.map(t => <i key={t}>{t}</i>) : <i>Not enough text to summarize</i>}</div>
      <div className="table-scroll"><table><thead><tr><th>Customer comment</th><th>Confidence</th><th>Ground truth</th><th>Assessment</th></tr></thead><tbody>{chosen.slice(0, showAll ? chosen.length : 8).map((r: BatchRow, i: number) => <tr key={`${i}-${r.text}`}><td className="comment-cell">{r.text}</td><td>{pct(r.confidence)}</td><td>{r.actual || '—'}</td><td><span className={`confidence-tag ${r.confidence < .55 ? 'needs-review' : 'strong'}`}>{r.confidence < .55 ? 'Review suggested' : 'Higher score'}</span></td></tr>)}</tbody></table></div>
      {chosen.length > 8 && <button className="show-more" onClick={() => setShowAll(!showAll)}>{showAll ? 'Show fewer' : `Show all ${chosen.length} ${selected} reviews`} <ChevronDown size={14}/></button>}
    </Panel>
  </div>
}

function DatasetTable({ rows, filtered, query, setQuery, filter, setFilter, onCategory }: any) {
  const [visible, setVisible] = useState(100)
  useEffect(() => { setVisible(100) }, [query, filter, rows.length])
  return <Panel><div className="explorer-controls"><div className="search-box"><Search size={16}/><input placeholder="Search customer comments…" value={query} onChange={e => setQuery(e.target.value)}/></div><select value={filter} onChange={e => setFilter(e.target.value)}><option>All categories</option>{labels.map(l => <option key={l}>{l}</option>)}</select><span>{filtered.length.toLocaleString()} of {rows.length.toLocaleString()} records</span></div>{rows.length ? <div className="table-scroll"><table><thead><tr><th>Customer comment</th><th>Predicted reason</th><th>Confidence</th><th>Model</th></tr></thead><tbody>{filtered.slice(0, visible).map((r: BatchRow, i: number) => <tr key={i}><td className="comment-cell">{r.text}</td><td><button className="category-pill clickable" onClick={() => onCategory(r.predicted_category)}>{r.predicted_category}</button></td><td>{pct(r.confidence)}</td><td>{r.model_used}</td></tr>)}</tbody></table></div> : <div className="inline-empty">Upload and classify a CSV to explore its records.</div>}{filtered.length > visible && <button className="show-more" onClick={() => setVisible(v => Math.min(v + 100, filtered.length))}>Load next 100 records ({filtered.length - visible} remaining) <ChevronDown size={14}/></button>}</Panel>
}

function ConfusionMatrix({ title }: { title: string }) {
  const matrixLabels = ['Sizing', 'Damaged', 'Wrong product', 'Late delivery']
  const values = [[750, 0, 0, 0], [0, 750, 0, 0], [0, 0, 750, 0], [0, 0, 0, 750]]
  return <Panel><h3 className="panel-title">{title}</h3><p className="muted">Actual classes are rows; predicted classes are columns.</p><div className="table-scroll"><table><thead><tr><th>Actual / Predicted</th>{matrixLabels.map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{values.map((row, i) => <tr key={matrixLabels[i]}><th>{matrixLabels[i]}</th>{row.map((value, j) => <td key={j} style={{ background: value ? 'rgba(116,141,112,.22)' : 'transparent', fontWeight: value ? 700 : 400, textAlign: 'center' }}>{value}</td>)}</tr>)}</tbody></table></div></Panel>
}

function PredictionResult({ result }: { result: Prediction }) {
  return <div className="prediction-result"><div className="result-kicker"><Check size={15}/> CLASSIFICATION COMPLETE</div>{result.detected_language && <div className="language-detected">Detected language: <b>{result.detected_language}</b></div>}<span className="eyebrow">PREDICTED RETURN REASON</span><h2>{result.predicted_category}</h2><div className="result-confidence">{pct(result.confidence)} <small>top-class score</small></div><span className="field-label">ALL CLASS SCORES</span><div className="confidence-bars">{labels.map((l, i) => <div key={l}><div><span>{l}</span><b>{pct(result.probabilities?.[l] || 0)}</b></div><Bar value={(result.probabilities?.[l] || 0) * 100} color={palette[i]}/></div>)}</div><div className="model-explain"><b>{result.model_used}</b><p>{result.explanation || 'The model assigned scores to each supported return category.'}</p><small>Review the original comment when confidence is low. Probabilities are not calibrated certainty.</small></div></div>
}
function ModelPicker({ model, setModel, health }: any) {
  return <div className="model-picker"><button className={model === 'tfidf' ? 'selected' : ''} onClick={() => setModel('tfidf')}><Layers3/><span><b>TF-IDF + Logistic Regression</b><small>Fast baseline classifier</small></span></button><button className={model === 'distilbert' ? 'selected' : ''} disabled={!health?.distilbert_model_available} onClick={() => setModel('distilbert')}><BrainCircuit/><span><b>Fine-tuned DistilBERT</b><small>{health?.distilbert_model_available ? 'Context-aware transformer' : 'Train model to enable'}</small></span></button></div>
}
