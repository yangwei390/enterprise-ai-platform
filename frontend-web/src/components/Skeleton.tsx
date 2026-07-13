type SkeletonProps = {
  rows?: number;
  variant?: "list" | "card";
};

export default function Skeleton({ rows = 3, variant = "list" }: SkeletonProps) {
  return (
    <div className={`skeleton-stack ${variant}`} aria-label="Loading">
      {Array.from({ length: rows }).map((_, index) => (
        <div className="skeleton-row" key={index}>
          <span />
          <small />
        </div>
      ))}
    </div>
  );
}
