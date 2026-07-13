import { KeyboardEvent, useState } from "react";

type MessageComposerProps = {
  value: string;
  disabled?: boolean;
  placeholder: string;
  submitLabel?: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
};

export default function MessageComposer({
  value,
  disabled = false,
  placeholder,
  submitLabel = "Send",
  onChange,
  onSubmit
}: MessageComposerProps) {
  const [isComposing, setIsComposing] = useState(false);

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !isComposing &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      onSubmit();
    }
  }

  return (
    <div className="composer">
      <textarea
        aria-label="Message"
        value={value}
        disabled={disabled}
        placeholder={placeholder}
        rows={2}
        onChange={(event) => onChange(event.target.value)}
        onCompositionEnd={() => setIsComposing(false)}
        onCompositionStart={() => setIsComposing(true)}
        onKeyDown={handleKeyDown}
      />
      <button type="button" disabled={disabled || !value.trim()} onClick={onSubmit}>
        {submitLabel}
      </button>
    </div>
  );
}
