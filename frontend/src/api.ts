import type { ProductSummary } from "./types"

const BASE = "/api"

export async function fetchProducts(): Promise<ProductSummary[]> {
  const res = await fetch(`${BASE}/products`)
  if (!res.ok) throw new Error("Failed to fetch products")
  return res.json()
}

export async function fetchProduct(id: string): Promise<ProductSummary> {
  const res = await fetch(`${BASE}/products/${id}`)
  if (!res.ok) throw new Error(`Product '${id}' not found`)
  return res.json()
}
