/** Animated Sahel / Africa risk globe for hero bento card */
export default function SahelVisual({ className = "" }) {
  return (
    <div className={`relative ${className}`}>
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-[260px] h-[260px] rounded-full border border-white/10 animate-pulse-soft" />
        <div className="absolute w-[200px] h-[200px] rounded-full border border-gold/20" />
      </div>
      <svg viewBox="0 0 320 320" className="w-full h-full drop-shadow-2xl" aria-hidden>
        <defs>
          <radialGradient id="globeGlow" cx="40%" cy="35%">
            <stop offset="0%" stopColor="#d6a24a" stopOpacity="0.85" />
            <stop offset="45%" stopColor="#312e81" stopOpacity="0.75" />
            <stop offset="100%" stopColor="#0f172a" stopOpacity="0.4" />
          </radialGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <circle cx="160" cy="160" r="110" fill="url(#globeGlow)" opacity="0.9" />
        <path
          d="M 70 180 Q 120 120 160 110 Q 200 105 250 130 Q 270 150 260 190 Q 200 220 120 210 Q 80 200 70 180 Z"
          fill="rgba(251,191,36,0.4)"
          stroke="rgba(214,162,74,0.7)"
          strokeWidth="1.5"
        />
        {[
          [130, 155, "#b83227", 8],
          [175, 140, "#d9822b", 6],
          [210, 165, "#b89b4a", 5],
          [155, 175, "#b83227", 7],
          [190, 185, "#d9822b", 5],
        ].map(([cx, cy, fill, r], i) => (
          <g key={i} filter="url(#glow)">
            <circle cx={cx} cy={cy} r={r + 4} fill={fill} opacity="0.25" className="animate-pulse-soft" />
            <circle cx={cx} cy={cy} r={r} fill={fill} />
          </g>
        ))}
        <ellipse cx="160" cy="160" rx="130" ry="45" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="1" transform="rotate(-20 160 160)" />
      </svg>
    </div>
  );
}
