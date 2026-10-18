import { useEffect, useState } from 'react'
import { Eye, EyeOff, ExternalLink, KeyRound, ShieldCheck, Trash2, X } from 'lucide-react'
import { maskApiKey, verifyApiKey } from '../services/api'

/**
 * API Key 设置弹窗
 *
 * required=true 时不可关闭：本站要求每个访客使用自己的 SiliconFlow Key。
 */
export default function ApiKeySettings({
  open,
  required = false,
  initialKey = '',
  onClose,
  onSaved,
  onCleared,
}) {
  const [key, setKey] = useState(initialKey)
  const [show, setShow] = useState(false)
  const [testing, setTesting] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  // 每次打开时重置表单状态
  useEffect(() => {
    if (open) {
      setKey(initialKey || '')
      setShow(false)
      setError('')
      setSuccess('')
    }
  }, [open, initialKey])

  if (!open) return null

  const handleSave = async () => {
    const value = key.trim()
    if (!value) {
      setError('请输入你的 SiliconFlow API Key')
      return
    }
    setTesting(true)
    setError('')
    setSuccess('')
    try {
      const result = await verifyApiKey(value)
      setSuccess(`验证通过，可用模型 ${result.model_count} 个`)
      onSaved?.(value, result)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '验证失败，请检查网络或 Key')
    } finally {
      setTesting(false)
    }
  }

  const handleClear = () => {
    setKey('')
    setError('')
    setSuccess('')
    onCleared?.()
  }

  const canClose = !required

  return (
    <div
      className="modal-overlay"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && canClose) onClose?.()
      }}
    >
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="api-key-title">
        <div className="modal-header">
          <div className="modal-title" id="api-key-title">
            <KeyRound size={18} />
            配置你的 SiliconFlow API Key
          </div>
          {canClose && (
            <button className="modal-close" onClick={onClose} aria-label="关闭">
              <X size={18} />
            </button>
          )}
        </div>

        <div className="modal-body">
          <p className="modal-desc">
            本站不提供公共额度，提问、上传文档、重建索引和性能测试都会消耗
            <strong>你自己的</strong>硅基流动额度。Key 只保存在当前浏览器，随请求发送到本站后端
            转发调用，不会写入服务器磁盘。
          </p>

          {required && (
            <div className="modal-alert">
              本站已开启「用户自带 Key」模式，配置后才能使用问答功能。
            </div>
          )}

          <label className="modal-label" htmlFor="api-key-input">API Key</label>
          <div className="key-input-wrap">
            <input
              id="api-key-input"
              className="input key-input"
              type={show ? 'text' : 'password'}
              value={key}
              autoComplete="off"
              spellCheck={false}
              placeholder="sk-..."
              onChange={(e) => { setKey(e.target.value); setError(''); setSuccess('') }}
              onKeyDown={(e) => { if (e.key === 'Enter') handleSave() }}
              disabled={testing}
            />
            <button
              type="button"
              className="key-visibility-btn"
              onClick={() => setShow(v => !v)}
              title={show ? '隐藏' : '显示'}
            >
              {show ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>

          {initialKey && !error && !success && (
            <div className="modal-hint">当前已保存：{maskApiKey(initialKey)}</div>
          )}
          {error && <div className="modal-error">{error}</div>}
          {success && (
            <div className="modal-success">
              <ShieldCheck size={14} /> {success}
            </div>
          )}

          <div className="modal-hint" style={{ marginTop: 12 }}>
            获取 Key：
            <a
              href="https://cloud.siliconflow.cn/account/ak"
              target="_blank"
              rel="noreferrer noopener"
              className="modal-link"
            >
              硅基流动控制台 <ExternalLink size={11} />
            </a>
            。建议单独创建一个低限额 Key，避免泄露后损失过大。
          </div>
        </div>

        <div className="modal-footer">
          {initialKey && (
            <button className="modal-clear" onClick={handleClear} disabled={testing}>
              <Trash2 size={13} /> 清除
            </button>
          )}
          <div className="modal-footer-right">
            {canClose && (
              <button className="btn btn-outline" onClick={onClose} disabled={testing}>
                取消
              </button>
            )}
            <button className="btn btn-primary" onClick={handleSave} disabled={testing}>
              {testing ? <><span className="spinner" /> 验证中...</> : '验证并保存'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
