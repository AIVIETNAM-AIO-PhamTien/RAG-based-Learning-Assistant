import { ChatInput } from "./ChatInput";

type ChatComposerProps = {
  disabled?: boolean;
  sendDisabled?: boolean;
  isFlashcardsOpen: boolean;
  sidebarOffsetClass: string;
  onSend: (message: string) => void;
};

export function ChatComposer({
  disabled,
  sendDisabled,
  isFlashcardsOpen,
  sidebarOffsetClass,
  onSend,
}: ChatComposerProps) {
  return (
    <div
      className={`fixed bottom-0 left-0 right-0 z-30 transition-[left,right] duration-300 ${sidebarOffsetClass} ${
        isFlashcardsOpen ? "xl:right-[380px]" : "xl:right-0"
      }`}
    >
      <ChatInput disabled={disabled} sendDisabled={sendDisabled} onSend={onSend} />
    </div>
  );
}
