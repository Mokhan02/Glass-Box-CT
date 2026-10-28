import { NavLink, Outlet, useLocation, Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { pageVariants } from "../lib/motion";

const NAV = [["/", "Scans"], ["/evidence", "Evidence"], ["/follow-up", "Follow-up (simulated)"], ["/about", "About"]];

function Header() {
  return (
    <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-bg/70 backdrop-blur-glass">
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2.5 sm:flex-nowrap sm:px-6 sm:py-3">
        <Link to="/" className="focus-ring flex items-center gap-2.5 rounded-md">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="3.5" y="3.5" width="17" height="17" rx="5" stroke="#5EE6D0" strokeWidth="1.4" /><circle cx="12" cy="12" r="3.2" stroke="#5EE6D0" strokeWidth="1.4" /></svg>
          <span className="text-[15px] font-medium tracking-tight">Glass-Box CT</span>
        </Link>
        <nav aria-label="Main" className="order-last -mx-1 flex w-full flex-wrap items-center gap-1 sm:order-none sm:mx-0 sm:ml-auto sm:w-auto sm:flex-nowrap">
          {NAV.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `focus-ring rounded-full px-2.5 py-1.5 text-[13px] transition-colors sm:px-3 ${isActive ? "bg-white/[0.08] text-ink" : "text-ink-dim hover:text-ink"}`}>{label}</NavLink>
          ))}
        </nav>
        <span className="hidden rounded-full border border-caution/40 px-2.5 py-0.5 text-[11px] text-caution md:inline">Research use only</span>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="mx-auto mt-16 max-w-[1500px] border-t border-white/[0.06] px-4 py-6 text-[12px] leading-relaxed text-ink-dim sm:px-6">
      <p><strong className="font-medium text-ink">Research use only. Not for clinical decisions.</strong> Do not upload patient data. This tool describes a model, not a patient.</p>
      <p className="mt-1.5">Sample scans: HCC-TACE-Seg, The Cancer Imaging Archive, CC BY 4.0 (Moawad et al., 2021, <a className="focus-ring rounded text-accent underline-offset-2 hover:underline" href="https://doi.org/10.7937/TCIA.5FNA-0924">doi:10.7937/TCIA.5FNA-0924</a>). Segmentation: NVIDIA NV-Segment-CT (VISTA-3D). <Link className="focus-ring rounded text-accent underline-offset-2 hover:underline" to="/about">Full attribution</Link>.</p>
    </footer>
  );
}

export default function Layout() {
  const loc = useLocation();
  return (
    <div className="backdrop-glow min-h-screen">
      <a href="#main" className="focus-ring sr-only left-4 top-4 z-50 rounded-md bg-accent px-3 py-2 text-bg focus:not-sr-only focus:absolute">Skip to content</a>
      <Header />
      <AnimatePresence mode="wait" initial={false}>
        <motion.main id="main" key={loc.pathname.startsWith("/scan/") ? "scan" : loc.pathname} variants={pageVariants} initial="initial" animate="animate" exit="exit">
          <Outlet />
          <Footer />
        </motion.main>
      </AnimatePresence>
    </div>
  );
}
