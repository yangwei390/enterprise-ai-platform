type EmptyStateProps = {
  title?: string;
  description: string;
};

export default function EmptyState({ title = "No data", description }: EmptyStateProps) {
  return (
    <div className="state-box empty">
      <strong>{title}</strong>
      <span>{description}</span>
    </div>
  );
}
