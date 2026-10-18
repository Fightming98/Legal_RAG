import axios from 'axios'

// 用户自己的 SiliconFlow API Key 保存在本浏览器 localStorage，
// 每次请求通过请求头传给后端；不会写入服务器磁盘。
const API_KEY_STORAGE = 'legal_rag.siliconflow_api_key'
export const API_KEY_HEADER = 'X-SiliconFlow-API-Key'

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

// Key 脱敏展示：sk-abcd...wxyz
export function maskApiKey(key) {
  if (!key) return ''
  return key.length > 12 ? `${key.slice(0, 6)}...${key.slice(-4)}` : 'sk-****'
}

const api = axios.create({
  baseURL: '/api',
  timeout: 300000,
})

// 所有请求自动携带当前浏览器保存的用户 Key
api.interceptors.request.use((config) => {
  const key = getStoredApiKey()
  config.headers = config.headers || {}
  // 不覆盖调用方显式传入的 Key（如「验证并保存」时待校验的新 Key）
  if (key && !config.headers[currentApiKeyHeader]) {
    config.headers[currentApiKeyHeader] = key
  }
  return config
})

// 后端返回 401 时通知界面打开「API 设置」（校验 Key 的请求本身除外）
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

// ===================== 问答 =====================

export async function sendChat({
  question,
  useRerank = true,
  useQueryRewrite = false,
  topK = 5,
  collection = 'all',
  evaluateQuality = false,
  monitorSystem = false,
  queryTransform = 'none',
  rerankStrategy = 'simple',
  generationStrategy = 'standard',
  useKg = false,
}) {
  const res = await api.post('/chat', {
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
  })
  return res.data.data
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
