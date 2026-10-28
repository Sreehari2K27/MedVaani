// Composer control visibility for the AE chat, by mode.
//
// Pure decision (the same convention as micClickAction and
// languageSelectDisabled in aeVoiceStartup.ts) so the mode rules are
// testable without a DOM: the typed ("demo") composer owns the text field
// and the send button; the voice-only ("live") composer offers ONLY the
// microphone — the text input is absent entirely, which is a rendering
// decision at this shared boundary, never a CSS hiding workaround.
// AeChat's mode is fixed for the lifetime of a mount (useState initializer;
// the entry screen unmounts it), so a pure decision is also the stale-state
// guarantee: every call reflects exactly the mode it was given.

export type AeChatMode = "demo" | "live";

export interface ComposerVisibility {
  textInput: boolean;
  micButton: boolean;
  sendButton: boolean;
}

export function composerControls(input: {
  mode: AeChatMode;
}): ComposerVisibility {
  const typed = input.mode === "demo";
  return {
    textInput: typed,
    micButton: !typed,
    sendButton: typed,
  };
}
