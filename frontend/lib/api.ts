const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export interface Source {
  file_name: string
  source: string
  page: number
  chunk_index: number
  total_chunks: number
  rerank_score: number
  content_preview: string
}

export interface QueryResponse {
  query: string
  answer: string
  answered: boolean
  sources: Source[]
  prompt_version: string
  latency_ms: number
}

export async function queryLegalRAG(question: string): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: question }),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API error ${res.status}: ${text}`)
  }
  return res.json()
}

export async function fetchStats(): Promise<{ total_chunks: number; collection_name: string }> {
  const res = await fetch(`${API_BASE}/api/stats`)
  if (!res.ok) throw new Error('Could not fetch stats')
  return res.json()
}