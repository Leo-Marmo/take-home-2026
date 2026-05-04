import type { ReactNode } from "react"

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-destructive text-sm">
      {message}
    </div>
  )
}

export function SectionHeader({ children }: { children: ReactNode }) {
  return <p className="text-sm font-medium mb-2">{children}</p>
}

export function OptionPill({
  selected,
  disabled,
  onClick,
  children,
}: {
  selected: boolean
  disabled: boolean
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      className={[
        "px-3 py-1.5 text-sm rounded-md border transition-colors",
        selected
          ? "border-primary bg-primary text-primary-foreground"
          : "hover:border-primary",
        disabled ? "opacity-40 line-through cursor-not-allowed" : "cursor-pointer",
      ].join(" ")}
    >
      {children}
    </button>
  )
}
