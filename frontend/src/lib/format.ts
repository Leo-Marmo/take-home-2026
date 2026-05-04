import type { Price } from "../types"

export function formatPrice(price: Price): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: price.currency,
  }).format(price.price)
}

export function formatCompareAt(price: Price): string | null {
  if (!price.compare_at_price || price.compare_at_price <= price.price) return null
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: price.currency,
  }).format(price.compare_at_price)
}
