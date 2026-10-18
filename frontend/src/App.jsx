import { useEffect, useState } from 'react'
import { MessageSquare, Database, Activity, KeyRound, Settings } from 'lucide-react'
import ChatPage from './pages/ChatPage'
import KnowledgePage from './pages/KnowledgePage'
import PerformancePage from './pages/PerformancePage'
import ModelSettingsPage from './pages/ModelSettingsPage'
import ApiKeySettings from './components/ApiKeySettings'
import {
  getKeyPolicy,
  getStoredApiKey,
  saveStoredApiKey,
  clearStoredApiKey,
  maskApiKey,
} from './services/api'

const NAV_ITEMS = [
  { key: 'chat', label: '智能问答', icon: MessageSquare },
  { key: 'knowledge', label: '知识库管理', icon: Database },
  { key: 'performance', label: '性能监控', icon: Activity },
  { key: 'models', label: '模型设置', icon: Settings },
]

export default function App() {
  const [page, setPage] = useState('chat')
  const [apiKey, setApiKey] = useState(() => getStoredApiKey())
  const [keyPolicy, setKeyPolicy] = useState(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  // 后端返回 401 时置为 true：即使策略接口没拿到，也必须配置后才能继续
  const [forceRequired, setForceRequired] = useState(false)

  // 拉取后端的密钥策略：是否需要用户自带 Key
  useEffect(() => {
    let alive = true
    getKeyPolicy()
      .then(policy => { if (alive) setKeyPolicy(policy) })
      .catch(() => { /* 后端未就绪时忽略，后续 401 会兜底 */ })
    return () => { alive = false }
  }, [])

  // 策略要求自带 Key，或服务端没有兜底 Key 时，也同样必须要用户配置
  const policyNeedsKey = keyPolicy
    ? (keyPolicy.require_user_api_key || !keyPolicy.server_key_configured)
    : false
  const keyRequired = !apiKey && (policyNeedsKey || forceRequired)

  // 首次进入：要求自带 Key 且本地没有 → 自动弹出设置
  useEffect(() => {
    if (keyRequired) setSettingsOpen(true)
  }, [keyRequired])

  // 任意模型接口返回 401（缺 Key / Key 失效）→ 自动打开设置
  useEffect(() => {
    const onRequired = () => {
      setForceRequired(true)
      setSettingsOpen(true)
    }
    window.addEventListener('legal-rag:api-key-required', onRequired)
    return () => window.removeEventListener('legal-rag:api-key-required', onRequired)
  }, [])

  const handleSaved = (key) => {
    saveStoredApiKey(key)
    setApiKey(key)
    setForceRequired(false)
    setSettingsOpen(false)
  }

  const handleCleared = () => {
    clearStoredApiKey()
    setApiKey('')
  }

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="icon">&#9878;</div>
            法律RAG系统
          </div>
        </div>
        <nav className="sidebar-nav">
          {NAV_ITEMS.map(item => (
            <button
              key={item.key}
              className={`nav-item ${page === item.key ? 'active' : ''}`}
              onClick={() => setPage(item.key)}
            >
              <item.icon size={18} />
              {item.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <button
            className="api-key-btn"
            onClick={() => setPage('models')}
            title="配置 API Key 与模型"
          >
            <KeyRound size={14} />
            <span className="api-key-btn-label">
              {apiKey ? maskApiKey(apiKey) : '配置 API Key'}
            </span>
            <span className={`api-key-dot ${apiKey ? 'ok' : 'missing'}`} />
          </button>
          <div className="sidebar-note">
            DeepSeek-V3.2 + BGE-M3<br />
            LangChain + ChromaDB
          </div>
        </div>
      </aside>
      <main className="main-content">
        {/* 用 display 控制显隐，避免切换 tab 时丢失 ChatPage 状态 */}
        <div style={{ display: page === 'chat' ? 'block' : 'none', height: '100%' }}>
          <ChatPage />
        </div>
        <div style={{ display: page === 'knowledge' ? 'block' : 'none' }}>
          <KnowledgePage />
        </div>
        <div style={{ display: page === 'performance' ? 'block' : 'none' }}>
          <PerformancePage />
        </div>
        <div style={{ display: page === 'models' ? 'block' : 'none' }}>
          <ModelSettingsPage
            apiKey={apiKey}
            keyPolicy={keyPolicy}
            active={page === 'models'}
            onApiKeySaved={handleSaved}
            onApiKeyCleared={handleCleared}
          />
        </div>
      </main>

      <ApiKeySettings
        open={settingsOpen}
        required={keyRequired}
        initialKey={apiKey}
        onClose={() => setSettingsOpen(false)}
        onSaved={handleSaved}
        onCleared={handleCleared}
      />
    </div>
  )
}
