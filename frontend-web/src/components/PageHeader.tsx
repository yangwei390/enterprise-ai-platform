type PageHeaderProps = {
  title: string;
  description: string;
  eyebrow?: string;
};

export default function PageHeader({ title, description, eyebrow }: PageHeaderProps) {
  return (
    <header className="page-header">
      {eyebrow && <span className="eyebrow">{eyebrow}</span>}
      <h1>{title}</h1>
      <p>{description}</p>
    </header>
  );
}
