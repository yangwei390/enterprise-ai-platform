type ErrorStateProps = {
  title?: string;
  message: string;
};

export default function ErrorState({ title = "Something went wrong", message }: ErrorStateProps) {
  return (
    <div className="state-box error">
      <strong>{title}</strong>
      <span>{message}</span>
    </div>
  );
}
