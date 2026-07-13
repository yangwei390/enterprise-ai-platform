type LoadingProps = {
  label?: string;
};

export default function Loading({ label = "Loading..." }: LoadingProps) {
  return (
    <div className="state-box">
      <div className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}
