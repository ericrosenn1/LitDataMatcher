import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { X } from "lucide-react";
import type { Workspace } from "../lib/types";

function shortSource(source: string) {
  return source.replace("Metabolomics Workbench", "Metabolomics WB");
}

export function Score({
  value,
  small = false,
}: {
  value: number;
  small?: boolean;
}) {
  return (
    <div
      className={`score ${small ? "score-small" : ""}`}
      style={{ "--score": `${value}%` } as React.CSSProperties}
      title={`Ranking score ${Math.round(value)} out of 100`}
    >
      <span>
        {Math.round(value)}
        <small>/100</small>
      </span>
    </div>
  );
}

export function Modal({
  children,
  title,
  onClose,
  wide = false,
}: {
  children: ReactNode;
  title: string;
  onClose: () => void;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    dialog?.showModal();
    dialog
      ?.querySelector<HTMLElement>("#command-query, .analyze-form textarea")
      ?.focus();
    return () => {
      dialog?.close();
    };
  }, []);
  return (
    <dialog
      ref={ref}
      className={`modal ${wide ? "modal-wide" : ""}`}
      aria-label={title}
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="modal-title">
        <span>{title}</span>
        <button
          className="icon-button"
          aria-label="Close dialog"
          onClick={onClose}
        >
          <X size={19} />
        </button>
      </div>
      {children}
    </dialog>
  );
}

export function EvidenceMap({
  workspace,
  onSelect,
  large = false,
}: {
  workspace: Workspace;
  onSelect: (id: string) => void;
  large?: boolean;
}) {
  const points = [
    { x: 257, y: 53 },
    { x: 302, y: 117 },
    { x: 272, y: 203 },
    { x: 157, y: 229 },
    { x: 76, y: 188 },
    { x: 54, y: 95 },
  ];
  return (
    <svg
      viewBox="0 0 350 280"
      className={`evidence-map ${large ? "map-large" : ""}`}
      role="group"
      aria-label="Connections between research questions and candidate dataset sources"
    >
      <defs>
        <radialGradient id={large ? "glowHero" : "glowMap"}>
          <stop offset="0%" stopColor="#b3d7c9" stopOpacity=".5" />
          <stop offset="100%" stopColor="#e7eee6" stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle
        cx="177"
        cy="135"
        r="120"
        fill={`url(#${large ? "glowHero" : "glowMap"})`}
      />
      <circle cx="177" cy="135" r="95" className="map-orbit" />
      <circle cx="177" cy="135" r="62" className="map-orbit inner" />
      {[
        ...new Map(
          workspace.datasets.map((dataset) => [dataset.source, dataset]),
        ).values(),
      ]
        .slice(0, 6)
        .map((dataset, i) => {
          const point = points[i];
          return (
            <g key={dataset.id}>
              <path
                d={`M177 135 Q${point.x} 135 ${point.x} ${point.y}`}
                className="map-link"
              />
              <g
                role="button"
                tabIndex={0}
                aria-label={`Show matches for ${dataset.source}`}
                onClick={() => onSelect(dataset.source)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelect(dataset.source);
                  }
                }}
                className={`map-node node-${i % 3}`}
              >
                <circle cx={point.x} cy={point.y} r="19" />
                <circle
                  cx={point.x}
                  cy={point.y}
                  r="5"
                  className="node-center"
                />
                <text x={point.x} y={point.y + 34} textAnchor="middle">
                  {shortSource(dataset.source).slice(0, 21)}
                </text>
              </g>
            </g>
          );
        })}
      <circle cx="177" cy="135" r="31" fill="#244b43" />
      <path
        d="M165 127l9-4 14 5v16l-14-5-9 4zM174 123v16"
        fill="none"
        stroke="#d7dfc5"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <circle cx="128" cy="77" r="4" fill="#d1ad61" />
      <circle cx="219" cy="184" r="3" fill="#d1ad61" />
      <circle cx="110" cy="147" r="2.5" fill="#82aaa0" />
    </svg>
  );
}
