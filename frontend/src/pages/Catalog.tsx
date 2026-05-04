import { useEffect, useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import { fetchProducts } from "../api"
import type { ProductSummary } from "../types"
import { formatPrice, formatCompareAt } from "../lib/format"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select"
import { ErrorBanner } from "@/components/common"

function lastSegment(category: string): string {
  const parts = category.split(">")
  return parts[parts.length - 1].trim()
}

function ProductCard({ summary }: { summary: ProductSummary }) {
  const navigate = useNavigate()
  const { id, product } = summary
  const image = product.image_urls[0]
  const compareAt = formatCompareAt(product.price)

  return (
    <Card
      className="group cursor-pointer overflow-hidden transition-shadow hover:shadow-md"
      onClick={() => navigate(`/products/${id}`)}
    >
      <div className="relative aspect-square overflow-hidden bg-muted">
        {image ? (
          <img
            src={image}
            alt={product.name}
            className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-muted-foreground text-sm">
            No image
          </div>
        )}
        {compareAt && (
          <Badge className="absolute top-2 left-2" variant="destructive">
            Sale
          </Badge>
        )}
      </div>
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">{product.brand}</p>
        <h3 className="font-medium text-sm leading-snug line-clamp-2 mb-2">{product.name}</h3>
        <div className="flex items-baseline gap-2">
          <span className="font-semibold text-sm">{formatPrice(product.price)}</span>
          {compareAt && (
            <span className="text-xs text-muted-foreground line-through">{compareAt}</span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function CardSkeleton() {
  return (
    <div className="overflow-hidden rounded-xl border">
      <Skeleton className="aspect-square w-full" />
      <div className="p-4 space-y-2">
        <Skeleton className="h-3 w-16" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-2/3" />
        <Skeleton className="h-4 w-1/4" />
      </div>
    </div>
  )
}

export default function Catalog() {
  const [products, setProducts] = useState<ProductSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [category, setCategory] = useState<string>("all")

  useEffect(() => {
    fetchProducts()
      .then(setProducts)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const categories = useMemo(() => {
    const unique = [...new Set(products.map((s) => s.product.category.name))]
    return unique.sort()
  }, [products])

  const filtered = useMemo(() =>
    category === "all"
      ? products
      : products.filter((s) => s.product.category.name === category),
    [products, category]
  )

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="flex items-end justify-between mb-10">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">Products</h1>
            {!loading && !error && (
              <p className="text-muted-foreground mt-1">{filtered.length} items</p>
            )}
          </div>

          {!loading && categories.length > 0 && (
            <Select value={category} onValueChange={(v) => setCategory(v ?? "all")}>
              <SelectTrigger className="w-auto min-w-40">
                <span className="flex-1 text-left text-sm">
                  {category === "all" ? "All Categories" : lastSegment(category)}
                </span>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                {categories.map((c) => (
                  <SelectItem key={c} value={c}>
                    {c}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>

        {error && <ErrorBanner message={error} />}

        <div className="grid grid-cols-2 gap-6 sm:grid-cols-3 lg:grid-cols-4">
          {loading
            ? Array.from({ length: 8 }).map((_, i) => <CardSkeleton key={i} />)
            : filtered.map((s) => <ProductCard key={s.id} summary={s} />)}
        </div>
      </div>
    </div>
  )
}
