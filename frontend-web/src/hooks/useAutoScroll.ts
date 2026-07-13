import { RefObject, useCallback, useRef } from "react";

const NEAR_BOTTOM_THRESHOLD = 80;

export function useAutoScroll(ref: RefObject<HTMLElement | null>) {
  const followRef = useRef(true);

  const handleScroll = useCallback(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
    followRef.current = distance < NEAR_BOTTOM_THRESHOLD;
  }, [ref]);

  const scrollToBottom = useCallback(
    (options: { force?: boolean } = {}) => {
      const element = ref.current;
      if (!element || (!options.force && !followRef.current)) {
        return;
      }
      element.scrollTop = element.scrollHeight;
    },
    [ref]
  );

  const enableFollow = useCallback(() => {
    followRef.current = true;
    scrollToBottom({ force: true });
  }, [scrollToBottom]);

  return {
    handleScroll,
    scrollToBottom,
    enableFollow
  };
}
