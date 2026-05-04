import { useEffect, useState } from "react"
import { useParams, useNavigate } from "react-router-dom"
import { fetchProduct } from "../api"
import type { Product, Variant, Price } from "../types"
import { formatPrice, formatCompareAt } from "../lib/format"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { ChevronLeft } from "lucide-react"
import { ErrorBanner, SectionHeader, OptionPill } from "@/components/common"

function ImageGallery({ urls }: { urls: string[] }) {
  const [active, setActive] = useState(0)
  const [failedUrls, setFailedUrls] = useState<Set<number>>(new Set())

  const markFailed = (i: number) => setFailedUrls((prev) => new Set(prev).add(i))
  const validUrls = urls.filter((_, i) => !failedUrls.has(i))

  if (urls.length === 0 || validUrls.length === 0) {
    return (
      <div className="aspect-square w-full rounded-xl bg-muted flex items-center justify-center text-muted-foreground text-sm">
        No image
      </div>
    )
  }

  const safeActive = Math.min(active, validUrls.length - 1)

  return (
    <div className="space-y-3">
      <div className="aspect-square w-full overflow-hidden rounded-xl bg-muted">
        <img
          src={validUrls[safeActive]}
          alt="Product"
          className="h-full w-full object-cover"
          onError={() => markFailed(urls.indexOf(validUrls[safeActive]))}
        />
      </div>
      {validUrls.length > 1 && (
        <div className="flex gap-2 overflow-x-auto pb-1">
          {validUrls.map((url, i) => (
            <button
              key={url}
              onClick={() => setActive(i)}
              className={`shrink-0 w-16 h-16 rounded-lg overflow-hidden border-2 transition-colors ${
                i === safeActive ? "border-primary" : "border-transparent"
              }`}
            >
              <img src={url} alt="" className="h-full w-full object-cover" />
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function VariantSelector({
  variants,
  onPriceChange,
}: {
  variants: Variant[]
  onPriceChange: (price: Price | null) => void
}) {
  const optionNames = [...new Set(variants.flatMap((v) => v.options.map((o) => o.name)))].sort()
  const [selections, setSelections] = useState<Record<string, string>>({})

  if (variants.length === 0 || optionNames.length === 0) return null

  const allSelected = optionNames.every((name) => !!selections[name])

  const selectedVariant = allSelected
    ? variants.find((v) =>
        optionNames.every((name) => v.options.some((o) => o.name === name && o.value === selections[name]))
      ) ?? null
    : null

  const handleSelect = (optName: string, val: string) => {
    const next = { ...selections, [optName]: val }
    setSelections(next)
    const allNowSelected = optionNames.every((name) => !!next[name])
    const matched = allNowSelected
      ? variants.find((v) =>
          optionNames.every((name) => v.options.some((o) => o.name === name && o.value === next[name]))
        )
      : null
    onPriceChange(matched?.price ?? null)
  }

  const isOutOfStock = selectedVariant ? !selectedVariant.in_stock : false

  return (
    <div className="space-y-4">
      {optionNames.map((optName) => {
        const values = [...new Set(
          variants.flatMap((v) => v.options.filter((o) => o.name === optName).map((o) => o.value))
        )]
        return (
          <div key={optName}>
            <SectionHeader>{optName}</SectionHeader>
            <div className="flex flex-wrap gap-2">
              {values.map((val) => {
                const variantForVal = variants.find((v) =>
                  v.options.some((o) => o.name === optName && o.value === val)
                )
                const outOfStock = variantForVal ? !variantForVal.in_stock : false
                const isSelected = selections[optName] === val

                return (
                  <OptionPill
                    key={val}
                    selected={isSelected}
                    disabled={outOfStock}
                    onClick={() => handleSelect(optName, val)}
                  >
                    {val}
                  </OptionPill>
                )
              })}
            </div>
          </div>
        )
      })}
      {isOutOfStock && (
        <p className="text-sm text-destructive">Out of stock</p>
      )}
    </div>
  )
}

function PDPSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-10 lg:grid-cols-2">
      <Skeleton className="aspect-square w-full rounded-xl" />
      <div className="space-y-4">
        <Skeleton className="h-4 w-20" />
        <Skeleton className="h-8 w-3/4" />
        <Skeleton className="h-6 w-24" />
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-10 w-32" />
      </div>
    </div>
  )
}

function PDPContent({ product }: { product: Product }) {
  const [variantPrice, setVariantPrice] = useState<Price | null>(null)
  const activePrice = variantPrice ?? product.price
  const compareAt = formatCompareAt(activePrice)

  return (
    <div className="grid grid-cols-1 gap-10 lg:grid-cols-2">
      <ImageGallery urls={product.image_urls} />

      <div className="space-y-6">
        <div>
          <p className="text-sm text-muted-foreground uppercase tracking-wide mb-1">{product.brand}</p>
          <h1 className="text-2xl font-semibold leading-snug">{product.name}</h1>
          <p className="text-xs text-muted-foreground mt-1">{product.category.name}</p>
        </div>

        <div className="flex items-baseline gap-3">
          <span className="text-2xl font-bold">{formatPrice(activePrice)}</span>
          {compareAt && (
            <>
              <span className="text-base text-muted-foreground line-through">{compareAt}</span>
              <Badge variant="destructive">Sale</Badge>
            </>
          )}
        </div>

        <Separator />

        <p className="text-sm text-muted-foreground leading-relaxed">{product.description}</p>

        {product.key_features.length > 0 && (
          <div>
            <SectionHeader>Key Features</SectionHeader>
            <ul className="space-y-1">
              {product.key_features.map((f, i) => (
                <li key={i} className="text-sm text-muted-foreground flex gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-muted-foreground" />
                  {f}
                </li>
              ))}
            </ul>
          </div>
        )}

        {product.colors.length > 0 && !product.variants.some((v) => v.options.some((o) => o.name === "Color")) && (
          <div>
            <SectionHeader>Colors</SectionHeader>
            <div className="flex flex-wrap gap-2">
              {product.colors.map((c) => (
                <Badge key={c} variant="secondary">{c}</Badge>
              ))}
            </div>
          </div>
        )}

        <VariantSelector
          variants={product.variants}
          onPriceChange={setVariantPrice}
        />
      </div>
    </div>
  )
}

export default function PDP() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [product, setProduct] = useState<Product | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!id) return
    fetchProduct(id)
      .then((s) => setProduct(s.product))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [id])

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-5xl mx-auto px-6 py-12">
        <Button
          variant="ghost"
          size="sm"
          className="mb-8 -ml-2 text-muted-foreground"
          onClick={() => navigate("/products")}
        >
          <ChevronLeft className="mr-1 h-4 w-4" />
          All Products
        </Button>

        {error && <ErrorBanner message={error} />}

        {loading && <PDPSkeleton />}
        {product && <PDPContent product={product} />}
      </div>
    </div>
  )
}
