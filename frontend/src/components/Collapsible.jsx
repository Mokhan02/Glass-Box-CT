/** Native <details>: keyboard accessible by default. Used for the secondary sections of the report. */
export default function Collapsible({ title, children, open = false, hint }) {
  return (
    <details open={open} className="group border-t border-white/[0.07] py-4">
      <summary className="focus-ring flex cursor-pointer list-none items-center gap-2 rounded text-[13px] text-ink">
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" className="text-ink-dim transition-transform duration-200 group-open:rotate-90"><path d="M4 2.5l4 3.5-4 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
        <span className="font-medium">{title}</span>{hint ? <span className="ml-auto text-[12px] text-ink-dim">{hint}</span> : null}
      </summary>
      <div className="mt-3 pl-5">{children}</div>
    </details>
  );
}
