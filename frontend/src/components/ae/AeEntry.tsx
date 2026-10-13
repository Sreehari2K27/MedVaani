import lantusPen from "../../assets/lantus-solostar.png.png";

interface AeEntryProps {
  onStartChat: () => void;
  onStartVoice: () => void;
}

function AeEntry({ onStartChat, onStartVoice }: AeEntryProps) {
  return (
    <section className="ae-entry">
      <div className="container ae-entry-inner">
        <div className="ae-hero">
          <div className="ae-intro">
            <h1 className="ae-title">
              Report a
              <br />
              Side Effect
            </h1>
            <p className="ae-lead">
              Tell us what happened in your own words. You can speak or type —
              MedVaani will guide you through the report.
            </p>
          </div>

          <div className="ae-product">
            <span className="ae-product-name">Lantus® SoloStar®</span>
            <p className="ae-product-ingredient">
              Insulin glargine (rDNA origin) Injection
            </p>
            <img
              src={lantusPen}
              alt="Lantus SoloStar pen"
              className="ae-pen"
            />
          </div>
        </div>

        <div className="ae-cards">
          <button type="button" className="ae-card ae-card--voice" onClick={onStartVoice}>
            <span className="ae-mic" aria-hidden="true">
              <svg
                width="44"
                height="44"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="22"></line>
              </svg>
            </span>
            <span className="ae-card-body">
              <span className="ae-card-title">Speak about a side effect</span>
              <span className="ae-card-desc">
                Describe what happened using your voice.
              </span>
            </span>
            <span className="ae-cta ae-cta--primary">
              Start speaking
              <span className="ae-cta-arrow" aria-hidden="true">
                →
              </span>
            </span>
          </button>

          <button
            type="button"
            className="ae-card ae-card--chat"
            onClick={onStartChat}
          >
            <span className="ae-chat-icon" aria-hidden="true">
              <svg
                width="40"
                height="40"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                <line x1="8" y1="9" x2="16" y2="9"></line>
                <line x1="8" y1="13" x2="13" y2="13"></line>
              </svg>
            </span>
            <span className="ae-card-body">
              <span className="ae-card-title">Type your experience</span>
              <span className="ae-card-desc">
                Tell us what happened in your own words.
              </span>
            </span>
            <span className="ae-cta ae-cta--soft">
              Start typing
              <span className="ae-cta-arrow" aria-hidden="true">
                →
              </span>
            </span>
          </button>
        </div>

        <div className="ae-safety-card" role="note">
          <span className="ae-safety-icon" aria-hidden="true">
            <svg
              width="22"
              height="22"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
              <path d="M8.5 12.5h7"></path>
              <path d="M12 9v7"></path>
            </svg>
          </span>
          <div className="ae-safety-content">
            <h2 className="ae-safety-title">If you need urgent medical attention</h2>
            <p className="ae-safety-body">
              If you are experiencing a serious or life-threatening reaction,
              seek immediate medical attention.
            </p>
          </div>
        </div>

        <p className="ae-disclaimer">
          MedVaani helps document safety reports and provides product
          information from referenced product documents. It does not diagnose
          conditions or provide treatment advice. Consult a qualified healthcare
          professional for personal medical guidance.
        </p>
      </div>
    </section>
  );
}

export default AeEntry;
