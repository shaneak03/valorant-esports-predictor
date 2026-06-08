interface Props {
  title: string;
  subtitle?: string;
}

export default function SectionHeading({ title, subtitle }: Props) {
  return (
    <div className="mb-8 flex items-start gap-3">
      <span className="mt-1 h-5 w-[3px] shrink-0 bg-accent" />
      <div>
        <h2 className="font-display text-2xl font-extrabold uppercase tracking-[0.15em] text-vcream">
          {title}
        </h2>
        {subtitle && (
          <p className="mt-0.5 font-sans text-xs text-vmuted">{subtitle}</p>
        )}
      </div>
    </div>
  );
}
