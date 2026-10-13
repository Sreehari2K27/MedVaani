import medvaaniLogo from "../assets/medvaani-logo.png.png";
import sanofiLogo from "../assets/sanofi-logo.png";

export type GlobalNavId = "home" | "about" | "how" | "mi" | "contact";

interface GlobalHeaderProps {
  activeId?: GlobalNavId;
  onNavigate: (id: GlobalNavId) => void;
}

const NAV_ITEMS: { id: GlobalNavId; label: string }[] = [
  { id: "home", label: "Home" },
  { id: "about", label: "About" },
  { id: "how", label: "How it works" },
  { id: "mi", label: "Lantus SoloStar" },
  { id: "contact", label: "Contact us" },
];

function GlobalHeader({ activeId, onNavigate }: GlobalHeaderProps) {
  return (
    <header className="gh-header">
      <div className="container gh-row">
        <button
          type="button"
          className="gh-logo-btn"
          onClick={() => onNavigate("home")}
          aria-label="MedVaani — Home"
        >
          <img src={medvaaniLogo} alt="MedVaani" className="site-logo" />
        </button>

        <nav className="gh-nav" aria-label="Primary">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`gh-nav-item${
                activeId === item.id ? " gh-nav-item--active" : ""
              }`}
              onClick={() => onNavigate(item.id)}
              aria-current={activeId === item.id ? "page" : undefined}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="gh-right" aria-label="Language selector">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#475569"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="2" y1="12" x2="22" y2="12"></line>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
          </svg>
          <span className="gh-lang">English</span>
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#475569"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </div>
      </div>

      <div className="container gh-sanofi">
        <span className="gh-association">In association with</span>
        <img src={sanofiLogo} alt="Sanofi" className="gh-sanofi-img" />
      </div>
    </header>
  );
}

export default GlobalHeader;