import axios from 'axios'

// 用户自己的 SiliconFlow API Key 保存在本浏览器 localStorage，
// 每次请求通过请求头传给后端；不会写入服务器磁盘。
const API_KEY_STORAGE = 'legal_rag.siliconflow_api_key'
export const API_KEY_HEADER = 'X-SiliconFlow-API-Key'

// 用户自定义的模型配置保存在本浏览器 localStorage，随请求头传给后端。
// 空字符串表示使用后端 .env 中的默认模型。
const MODEL_SETTINGS_STORAGE = 'legal_rag.model_settings'
export const MODEL_HEADERS = {
  llm: 'X-LLM-Model',
  embedding: 'X-Embedding-Model',
  rerank: 'X-Rerank-Model',
}
const EMPTY_MODEL_SETTINGS = { llm_model: '', embedding_model: '', rerank_model: '' }

// 后端可通过 API_KEY_HEADER 环境变量自定义头名，拉到策略后动态更新
let currentApiKeyHeader = API_KEY_HEADER

export function getStoredApiKey() {
  try {
    return localStorage.getItem(API_KEY_STORAGE) || ''
  } catch {
    return ''
  }
}

export function saveStoredApiKey(key) {
  try {
    localStorage.setItem(API_KEY_STORAGE, key)
  } catch {
    /* 隐私模式下可能不可写，忽略即可 */
  }
}

export function clearStoredApiKey() {
  try {
    localStorage.removeItem(API_KEY_STORAGE)
  } catch {
    /* ignore */
  }
}

// ===================== 模型配置（localStorage） =====================

export function getStoredModelSettings() {
  try {
    const raw = localStorage.getItem(MODEL_SETTINGS_STORAGE)
    if (!raw) return { ...EMPTY_MODEL_SETTINGS }
    const parsed = JSON.parse(raw)
    return {
      llm_model: parsed?.llm_model || '',
      embedding_model: parsed?.embedding_model || '',
      rerank_model: parsed?.rerank_model || '',
    }
  } catch {
    return { ...EMPTY_MODEL_SETTINGS }
  }
}

export function saveStoredModelSettings(settings) {
  try {
    localStorage.setItem(MODEL_SETTINGS_STORAGE, JSON.stringify({
      llm_model: settings?.llm_model || '',
      embedding_model: settings?.embedding_model || '',
      rerank_model: settings?.rerank_model || '',
    }))
  } catch {
    /* 隐私模式下可能不可写，忽略即可 */
  }
}

export function clearStoredModelSettings() {
  try {
    localStorage.removeItem(MODEL_SETTINGS_STORAGE)
  } catch {
    /* ignore */
  }
}

// Key 脱敏展示：sk-abcd...wxyz
export function maskApiKey(key) {
  if (!key) return ''
  return key.length > 12 ? `${key.slice(0, 6)}...${key.slice(-4)}` : 'sk-****'
}

// 后端可能通过 /settings/key-policy 下发自定义请求头名
export function getApiKeyHeader() {
  return currentApiKeyHeader
}

const api = axios.create({
  baseURL: '/api',
  timeout: 300000,
})

// 所有请求自动携带当前浏览器保存的用户 Key 与模型配置
api.interceptors.request.use((config) => {
  const key = getStoredApiKey()
  config.headers = config.headers || {}
  // 不覆盖调用方显式传入的 Key（如「验证并保存」时待校验的新 Key）
  if (key && !config.headers[currentApiKeyHeader]) {
    config.headers[currentApiKeyHeader] = key
  }

  const models = getStoredModelSettings()
  if (models.llm_model && !config.headers[MODEL_HEADERS.llm]) {
    config.headers[MODEL_HEADERS.llm] = models.llm_model
  }
  if (models.embedding_model && !config.headers[MODEL_HEADERS.embedding]) {
    config.headers[MODEL_HEADERS.embedding] = models.embedding_model
  }
  if (models.rerank_model && !config.headers[MODEL_HEADERS.rerank]) {
    config.headers[MODEL_HEADERS.rerank] = models.rerank_model
  }
  return config
})

// 后端返回 401 时通知界面打开「模型设置」（校验 Key 的请求本身除外）
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const url = err.config?.url || ''
    if (err.response?.status === 401 && !url.includes('/settings/verify-key')) {
      window.dispatchEvent(new CustomEvent('legal-rag:api-key-required', {
        detail: err.response?.data?.detail || '',
      }))
    }
    return Promise.reject(err)
  },
)

// ===================== 密钥设置 =====================

export async function getKeyPolicy() {
  const res = await api.get('/settings/key-policy')
  const policy = res.data.data
  if (policy?.header) currentApiKeyHeader = policy.header
  return policy
}

// 显式传入待校验的 Key（用户可能还没保存）
export async function verifyApiKey(key) {
  const res = await api.post('/settings/verify-key', null, {
    headers: key ? { [currentApiKeyHeader]: key } : undefined,
  })
  return res.data.data
}

// 获取当前 Key 可用的模型列表（按 LLM / 嵌入 / 重排分组）
export async function listProviderModels(key) {
  const res = await api.get('/settings/models', {
    headers: key ? { [currentApiKeyHeader]: key } : undefined,
  })
  return res.data.data
}

// 连接测试：显式传入 Key 与三个模型，测试尚未保存的表单配置
export async function testConnection({ key, llmModel, embeddingModel, rerankModel } = {}) {
  const headers = {}
  if (key) headers[currentApiKeyHeader] = key
  if (llmModel) headers[MODEL_HEADERS.llm] = llmModel
  if (embeddingModel) headers[MODEL_HEADERS.embedding] = embeddingModel
  if (rerankModel) headers[MODEL_HEADERS.rerank] = rerankModel
  const res = await api.post('/settings/test-connection', null, { headers })
  return res.data.data
}

// ===================== 问答 =====================

function buildChatPayload({
  question,
  useRerank = true,
  useQueryRewrite = false,
  topK = 5,
  collection = 'all',
  evaluateQuality = false,
  monitorSystem = false,
  queryTransform = 'none',
  rerankStrategy = 'bge',
  generationStrategy = 'standard',
  useKg = false,
  enableThinking = true,
}) {
  return {
    question,
    use_rerank: useRerank,
    use_query_rewrite: useQueryRewrite,
    top_k: topK,
    collection,
    evaluate_quality: evaluateQuality,
    monitor_system: monitorSystem,
    query_transform: queryTransform,
    rerank_strategy: rerankStrategy,
    generation_strategy: generationStrategy,
    use_kg: useKg,
    enable_thinking: enableThinking,
  }
}

export async function sendChat(params) {
  const res = await api.post('/chat', buildChatPayload(params))
  return res.data.data
}

// 解析单个 SSE 事件块（event: xxx\ndata: {...}）
function parseSSEBlock(raw) {
  let event = 'message'
  const dataLines = []
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).replace(/^ /, ''))
    }
  }
  if (!dataLines.length) return null
  const text = dataLines.join('\n')
  let data = text
  try {
    data = JSON.parse(text)
  } catch {
    /* 非 JSON 数据保留原始文本 */
  }
  return { event, data }
}

/**
 * 流式问答：通过 SSE 逐事件回调
 *
 * onEvent 收到 { event, data }，事件类型见后端 stream_service：
 * stage / sources / reasoning / answer / answer_replace / metrics / quality /
 * done / error。
 */
export async function streamChat(params, { onEvent, signal } = {}) {
  const key = getStoredApiKey()
  const headers = { 'Content-Type': 'application/json' }
  if (key) headers[currentApiKeyHeader] = key

  const models = getStoredModelSettings()
  if (models.llm_model) headers[MODEL_HEADERS.llm] = models.llm_model
  if (models.embedding_model) headers[MODEL_HEADERS.embedding] = models.embedding_model
  if (models.rerank_model) headers[MODEL_HEADERS.rerank] = models.rerank_model

  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers,
    body: JSON.stringify(buildChatPayload(params)),
    signal,
  })

  // fetch 不走 axios 拦截器，这里复刻 401 → 打开模型设置的行为
  if (!response.ok) {
    let detail = `请求失败 (${response.status})`
    try {
      const body = await response.json()
      detail = body?.detail || detail
    } catch {
      /* 响应体不是 JSON 时沿用默认提示 */
    }
    if (response.status === 401) {
      window.dispatchEvent(new CustomEvent('legal-rag:api-key-required', { detail }))
    }
    const err = new Error(detail)
    err.status = response.status
    throw err
  }
  if (!response.body) throw new Error('当前浏览器不支持流式响应')

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const emitBlock = (raw) => {
    const parsed = parseSSEBlock(raw)
    if (parsed) onEvent?.(parsed)
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    // 统一换行符；跨 chunk 的 \r\n 会在后续拼接后修正
    buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, '\n')
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      if (raw.trim()) emitBlock(raw)
    }
  }

  // 处理末尾没有空行结尾的残留事件
  buffer = (buffer + decoder.decode()).replace(/\r\n/g, '\n')
  if (buffer.trim()) emitBlock(buffer)
}

// ===================== 知识库 =====================

export async function uploadDocument(file, docType) {
  const form = new FormData()
  form.append('file', file)
  if (docType) form.append('doc_type', docType)
  const res = await api.post('/knowledge/upload', form)
  return res.data.data
}

export async function listDocuments() {
  const res = await api.get('/knowledge/list')
  return res.data.data
}

export async function deleteDocument(filename) {
  const res = await api.delete(`/knowledge/${encodeURIComponent(filename)}`)
  return res.data
}

export async function rebuildIndex() {
  const res = await api.post('/knowledge/rebuild')
  return res.data.data
}

export async function getKBStats() {
  const res = await api.get('/knowledge/stats')
  return res.data.data
}

// ===================== 性能 =====================

export async function getSystemInfo() {
  const res = await api.get('/performance/system')
  return res.data.data
}

export async function runBenchmark(queries, useRerank = true, evaluateQuality = false) {
  const res = await api.post('/performance/bench', null, {
    params: { use_rerank: useRerank, evaluate_quality: evaluateQuality },
  })
  return res.data.data
}

// ===================== 报告 =====================

export async function createReport(queries, useRerank = true, evaluateQuality = true) {
  const res = await api.post('/performance/report', null, {
    params: { use_rerank: useRerank, evaluate_quality: evaluateQuality },
  })
  return res.data.data
}

export async function listReports() {
  const res = await api.get('/performance/reports')
  return res.data.data
}

export function getReportDownloadUrl(reportId) {
  return `/api/performance/reports/${encodeURIComponent(reportId)}`
}

// ===================== 问答记录 =====================

export async function saveChatRecord(record) {
  const res = await api.post('/chat/save-record', record)
  return res.data.data
}

export async function listChatRecords() {
  const res = await api.get('/chat/records')
  return res.data.data
}

export function getChatRecordDownloadUrl(recordId) {
  return `/api/chat/records/${encodeURIComponent(recordId)}`
}
