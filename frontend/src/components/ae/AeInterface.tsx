import { useState } from "react";
import AeEntry from "./AeEntry";
import AeChat from "./AeChat";

type AeView = "entry" | "chat";
type AeChatMode = "demo" | "live";

interface AeInterfaceProps {
  onChatModeChange?: (chatMode: boolean) => void;
}

function AeInterface({ onChatModeChange }: AeInterfaceProps) {
  const [view, setView] = useState<AeView>("entry");
  const [chatInitialMode, setChatInitialMode] = useState<AeChatMode>("demo");

  const startChat = (mode: AeChatMode) => {
    setChatInitialMode(mode);
    setView("chat");
    onChatModeChange?.(true);
  };

  return (
    <div
      className={view === "chat" ? "ae-shell ae-shell--chat" : "ae-shell"}
    >
      {view === "entry" ? (
        <AeEntry
          onStartChat={() => startChat("demo")}
          onStartVoice={() => startChat("live")}
        />
      ) : (
        <AeChat initialMode={chatInitialMode} />
      )}
    </div>
  );
}

export default AeInterface;