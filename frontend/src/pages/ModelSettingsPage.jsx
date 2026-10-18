import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle,
  Cpu,
  ExternalLink,
  Eye,
  EyeOff,
  KeyRound,
  Plug,
  RefreshCw,
  RotateCcw,
  Save,
  Trash2,
  XCircle,
} from 'lucide-react'
import {
  clearStoredModelSettings,
  getStoredModelSettings,
  listProviderModels,
  maskApiKey,
  saveStoredModelSettings,
  testConnection,
} from '../services/api'

const EMPTY_GROUPS = { llm: [], embedding: [], rerank: [], other: [] }

const MODEL_FIELDS = [
  {
    name: 'llm_model',
    label: 'LLM 模型',
    hint: '用于回答生成、查询改写、质量评估',
    group: 'llm',
    listId: 'model-options-llm',
  },
  {
    name: 'embedding_model',
    label: '嵌入模型',
    hint: '用于向量检索与知识库索引',
    group: 'embedding',
    listId: 'model-options-embedding',
  },
  {
    name: 'rerank_model',
    label: '重排序模型',
    hint: '用于 BGE 精排阶段',
    group: 'rerank',
    listId: 'model-options-rerank',
  },
]

/**
 * 模型设置页：API Key + 连接测试 + 模型列表 + LLM / 嵌入 / 重排模型配置。
 * 配置只保存在当前浏览器，请求时通过请求头传给后端。
 */
export default function ModelSettingsPage({
  apiKey,
  keyPolicy,
  active = true,
  onApiKeySaved,
  onApiKeyCleared,
}) {
  const [key, setKey] = useState(apiKey || '')
  const [showKey, setShowKey] = useState(false)
  const [form, setForm] = useState(() => {
    const stored = getStoredModelSettings()
    return {
      llm_model: stored.llm_model,
      embedding_model: stored.embedding_model,
      rerank_model: stored.rerank_model,
    }
  })
  const [groups, setGroups] = useState(EMPTY_GROUPS)
  const [allModels, setAllModels] = useState([])
  const [modelCount, setModelCount] = useState(0)
  const [loadingModels, setLoadingModels] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)
  // 请求本身失败（网络不可达 / 401 / 422 / 502）时，也要在结果区展示具体报错
  const [testError, setTestError] = useState(null)
  const [message, setMessage] = useState(null)
  const autoLoadedRef = useRef(false)

  // 后端 .env 中的默认模型（旧版后端可能没有 rerank_model 字段）
  const defaults = useMemo(() => ({
    llm_model: keyPolicy?.llm_model || '',
    embedding_model: keyPolicy?.embedding_model || '',
    rerank_model: keyPolicy?.rerank_model || '',
  }), [keyPolicy])

  // 默认值到达后只填空字段，不覆盖用户已保存的自定义值
  useEffect(() => {
    setForm(prev => ({
      llm_model: prev.llm_model || defaults.llm_model,
      embedding_model: prev.embedding_model || defaults.embedding_model,
      rerank_model: prev.rerank_model || defaults.rerank_model,
    }))
  }, [defaults])

  useEffect(() => {
    setKey(apiKey || '')
  }, [apiKey])

  const fetchModels = useCallback(async (silent = false) => {
    const value = key.trim()
    if (!value) {
      if (!silent) setMessage({ type: 'error', text: '请先填写 SiliconFlow API Key' })
      return
    }
    setLoadingModels(true)
    if (!silent) setMessage(null)
    try {
      const data = await listProviderModels(value)
      setGroups(data?.groups || EMPTY_GROUPS)
      setAllModels((data?.models || []).map(item => item.id))
      setModelCount(data?.count || 0)
      if (!silent) setMessage({ type: 'success', text: `已获取 ${data?.count || 0} 个可用模型` })
    } catch (err) {
      if (!silent) {
        setMessage({
          type: 'error',
          text: err.response?.data?.detail || err.message || '获取模型列表失败',
        })
      }
    } finally {
      setLoadingModels(false)
    }
  }, [key])

  // 首次切换到本页且已有 Key 时自动拉取一次，填充各模型的候选列表
  useEffect(() => {
    if (active && !autoLoadedRef.current && apiKey) {
      autoLoadedRef.current = true
      fetchModels(true)
    }
  }, [active, apiKey, fetchModels])

  const keyRequired = keyPolicy
    ? (keyPolicy.require_user_api_key || !keyPolicy.server_key_configured)
    : false

  const updateField = (name, value) => {
    setForm(prev => ({ ...prev, [name]: value }))
    setMessage(null)
  }

  const handleTest = async () => {
    const value = key.trim()
    if (!value) {
      setMessage({ type: 'error', text: '请先填写 SiliconFlow API Key' })
      return
    }
    setTesting(true)
    setTestResult(null)
    setTestError(null)
    setMessage(null)
    try {
      const result = await testConnection({
        key: value,
        llmModel: form.llm_model.trim(),
        embeddingModel: form.embedding_model.trim(),
        rerankModel: form.rerank_model.trim(),
      })
      setTestResult(result)
    } catch (err) {
      const detail = err.response?.data?.detail
      setTestError({
        status: err.response?.status ?? null,
        message: typeof detail === 'string'
          ? detail
          : detail
            ? JSON.stringify(detail, null, 2)
            : (err.message || '连接测试请求失败'),
      })
    } finally {
      setTesting(false)
    }
  }

  const handleSave = () => {
    const value = key.trim()
    if (!value) {
      setMessage({ type: 'error', text: '请先填写 SiliconFlow API Key' })
      return
    }
    const models = {
      llm_model: form.llm_model.trim(),
      embedding_model: form.embedding_model.trim(),
      rerank_model: form.rerank_model.trim(),
    }
    onApiKeySaved?.(value)
    saveStoredModelSettings(models)
    setForm(models)
    setMessage({
      type: 'success',
      text: '设置已保存，后续问答、知识库与性能测试请求将使用新配置',
    })
    window.dispatchEvent(new CustomEvent('legal-rag:model-settings-changed'))
  }

  const handleClearKey = () => {
    setKey('')
    setTestResult(null)
    setTestError(null)
    setMessage(null)
    onApiKeyCleared?.()
  }

  const handleRestoreDefaults = () => {
    setForm({ ...defaults })
    clearStoredModelSettings()
    setTestResult(null)
    setTestError(null)
    setMessage({
      type: 'success',
      text: '已恢复后端默认模型（立即生效）；可再次点击「保存设置」固定配置',
    })
    window.dispatchEvent(new CustomEvent('legal-rag:model-settings-changed'))
  }

  return (
    <div className="settings-page">
      <div className="page-header">
        <h1>模型设置</h1>
        <p>配置 SiliconFlow API Key，并自定义 LLM、嵌入与重排序模型。设置只保存在当前浏览器。</p>
      </div>

      {/* ---------- API Key ---------- */}
      <section className="card">
        <div className="card-header">
          <KeyRound size={18} />
          SiliconFlow API Key
        </div>

        {keyRequired && !key && (
          <div className="modal-alert">
            本站已开启「用户自带 Key」模式，请先填写你自己的 SiliconFlow API Key。
          </div>
        )}

        <div className="settings-key-row">
          <div className="key-input-wrap">
            <input
              className="input key-input"
              type={showKey ? 'text' : 'password'}
              value={key}
              autoComplete="off"
              spellCheck={false}
              placeholder="sk-..."
              onChange={(e) => {
                setKey(e.target.value)
                setTestResult(null)
                setTestError(null)
                setMessage(null)
              }}
              disabled={testing}
            />
            <button
              type="button"
              className="key-visibility-btn"
              onClick={() => setShowKey(v => !v)}
              title={showKey ? '隐藏' : '显示'}
            >
              {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
          {key && (
            <button className="btn btn-outline" onClick={handleClearKey} disabled={testing}>
              <Trash2 size={14} /> 清除
            </button>
          )}
        </div>

        {apiKey && (
          <div className="modal-hint">当前已保存：{maskApiKey(apiKey)}</div>
        )}

        <div className="modal-hint" style={{ marginTop: 10 }}>
          获取 Key：
          <a
            href="https://cloud.siliconflow.cn/account/ak"
            target="_blank"
            rel="noreferrer noopener"
            className="modal-link"
          >
            硅基流动控制台 <ExternalLink size={11} />
          </a>
          。Key 只保存在当前浏览器，请求时发送到本站后端转发调用，不会写入服务器磁盘。
        </div>
      </section>

      {/* ---------- 模型配置 ---------- */}
      <section className="card">
        <div className="card-header">
          <Cpu size={18} />
          模型配置
        </div>

        <div className="settings-grid">
          {MODEL_FIELDS.map(field => (
            <label className="settings-field" key={field.name}>
              <span className="settings-field-label">{field.label}</span>
              <input
                className="input"
                list={field.listId}
                value={form[field.name]}
                spellCheck={false}
                placeholder="输入或选择模型 ID"
                onChange={e => updateField(field.name, e.target.value)}
              />
              <span className="settings-field-hint">{field.hint}</span>
            </label>
          ))}
        </div>

        {MODEL_FIELDS.map(field => (
          <datalist id={field.listId} key={field.listId}>
            {(groups[field.group]?.length ? groups[field.group] : allModels).map(id => (
              <option key={id} value={id} />
            ))}
          </datalist>
        ))}

        <div className="settings-actions">
          <button
            className="btn btn-outline"
            onClick={() => fetchModels(false)}
            disabled={loadingModels || !key.trim()}
          >
            {loadingModels ? <span className="spinner spinner-sm" /> : <RefreshCw size={15} />}
            {loadingModels ? '获取中...' : '获取模型列表'}
          </button>
          <button
            className="btn btn-outline"
            onClick={handleTest}
            disabled={testing || !key.trim()}
          >
            {testing ? <span className="spinner spinner-sm" /> : <Plug size={15} />}
            {testing ? '测试中...' : '连接测试'}
          </button>
          <button
            className="btn btn-outline"
            onClick={handleRestoreDefaults}
            disabled={!defaults.llm_model && !defaults.embedding_model && !defaults.rerank_model}
          >
            <RotateCcw size={15} /> 恢复默认
          </button>
          <button className="btn btn-primary" onClick={handleSave}>
            <Save size={15} /> 保存设置
          </button>
        </div>

        {message && (
          <div className={message.type === 'error' ? 'modal-error' : 'modal-success'}>
            {message.text}
          </div>
        )}

        <div className="settings-note">
          {modelCount > 0
            ? `已加载 ${modelCount} 个模型（LLM ${groups.llm.length} / 嵌入 ${groups.embedding.length} / 重排 ${groups.rerank.length}）。`
            : '点击「获取模型列表」从 SiliconFlow 拉取可用模型。'}
          也可以直接手动输入未列出的模型 ID。
        </div>

        <div className="settings-note settings-note-warning">
          <AlertTriangle size={14} />
          <span>
            更换嵌入模型后，其向量维度必须与现有知识库索引一致（BGE-M3 为 1024 维），
            否则检索会提示维度不匹配。此时请恢复默认嵌入模型，或使用当前模型重建知识库索引。
          </span>
        </div>
      </section>

      {/* ---------- 连接测试结果 ---------- */}
      {(testResult || testError) && (
        <section className="card">
          <div className="card-header">
            {testResult?.ok
              ? <CheckCircle size={18} color="var(--success)" />
              : <XCircle size={18} color="var(--danger)" />}
            连接测试结果
            <span className={`test-summary ${testResult?.ok ? 'ok' : 'fail'}`}>
              {testError ? '请求失败' : (testResult?.ok ? '全部通过' : '存在异常')}
            </span>
          </div>

          {testError ? (
            <div className="test-checks">
              <div className="test-check fail">
                <XCircle size={16} className="test-check-icon" />
                <div className="test-check-main">
                  <div className="test-check-title">
                    连接测试请求失败
                    {testError.status != null && <code>HTTP {testError.status}</code>}
                  </div>
                  <pre className="test-check-error">{testError.message}</pre>
                  <div className="test-check-detail">
                    请检查后端服务是否已启动、网络是否可达，以及模型名称是否符合要求。
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="test-checks">
              {(testResult.checks || []).map(check => (
                <div key={check.key} className={`test-check ${check.ok ? 'ok' : 'fail'}`}>
                  {check.ok
                    ? <CheckCircle size={16} className="test-check-icon" />
                    : <XCircle size={16} className="test-check-icon" />}
                  <div className="test-check-main">
                    <div className="test-check-title">
                      {check.label}
                      {check.model && <code>{check.model}</code>}
                    </div>
                    <div className={`test-check-detail ${!check.ok && !check.skipped ? 'error' : ''}`}>
                      {check.detail}
                    </div>
                    {!check.ok && !check.skipped && check.error_type && (
                      <div className="test-check-detail">异常类型：{check.error_type}</div>
                    )}
                  </div>
                  <div className="test-check-meta">
                    {check.skipped
                      ? '跳过'
                      : [
                          check.status_code != null ? `HTTP ${check.status_code}` : null,
                          check.latency_ms != null ? `${check.latency_ms} ms` : null,
                        ].filter(Boolean).join(' · ')}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  )
}
