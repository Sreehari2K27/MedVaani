import type { ReactNode } from "react";
import GlobalHeader, { type GlobalNavId } from "./GlobalHeader";

interface GlobalLayoutProps {
  activeId?: GlobalNavId;
  onNavigate: (id: GlobalNavId) => void;
  chatMode?: boolean;
  children: ReactNode;
}

function GlobalLayout({
  activeId,
  onNavigate,
  chatMode = false,
  children,
}: GlobalLayoutProps) {
  return (
    <div
      className={
        chatMode ? "global-root global-root--chat" : "global-root"
      }
    >
      <GlobalHeader activeId={activeId} onNavigate={onNavigate} />
      <main className="global-main">{children}</main>
    </div>
  );
}

export default GlobalLayout;