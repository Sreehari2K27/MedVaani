interface ActionCardsProps {
  onReportSideEffect?: () => void;
  onGetProductInfo?: () => void;
}

function ActionCards({ onReportSideEffect, onGetProductInfo }: ActionCardsProps) {
  return (
    <section style={styles.section}>
      <div className="container cards-grid">
        <div
          style={styles.card}
          className="action-card lavender"
          role="button"
          tabIndex={0}
          onClick={onGetProductInfo}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              onGetProductInfo?.();
            }
          }}
        >
          <div style={styles.iconWrapper} className="card-icon icon-lavender">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
              <line x1="9" y1="10" x2="15" y2="10"></line>
              <line x1="9" y1="14" x2="13" y2="14"></line>
            </svg>
          </div>
          <div style={styles.content}>
            <h3 style={styles.title}>Get Product Information</h3>
            <p style={styles.description}>
              Ask anything about Lantus® SoloStar® —
              <br />
              uses, dosing, storage, and more.
            </p>
          </div>
          <div style={styles.arrowBtn} aria-label="Get Product Information">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="5" y1="12" x2="19" y2="12"></line>
              <polyline points="12 5 19 12 12 19"></polyline>
            </svg>
          </div>
        </div>
        <div
          style={styles.card}
          className="action-card peach"
          role="button"
          tabIndex={0}
          onClick={onReportSideEffect}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              onReportSideEffect?.();
            }
          }}
        >
          <div style={styles.iconWrapper} className="card-icon icon-peach">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#f97316" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
              <polyline points="9 11 12 14 16 8"></polyline>
            </svg>
          </div>
          <div style={styles.content}>
            <h3 style={styles.title}>Report a Side Effect</h3>
            <p style={styles.description}>
              Share your experience and help
              <br />
              improve medicine safety.
            </p>
          </div>
          <div style={styles.arrowBtn} aria-label="Report a Side Effect">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#f97316" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="5" y1="12" x2="19" y2="12"></line>
              <polyline points="12 5 19 12 12 19"></polyline>
            </svg>
          </div>
        </div>
      </div>
    </section>
  );
}

const styles: Record<string, React.CSSProperties> = {
  section: {
    padding: "3rem 0 2.25rem",
    background:
      "radial-gradient(55% 95% at 100% 0%, rgba(224, 214, 255, 0.55) 0%, rgba(224, 214, 255, 0) 65%), linear-gradient(180deg, #ffffff 0%, #fbfaff 100%)",
  },
  card: { display: "flex", alignItems: "center", gap: "1.75rem", padding: "2.5rem", borderRadius: "1.5rem", cursor: "pointer", transition: "transform 0.2s ease, box-shadow 0.2s ease" },
  iconWrapper: { display: "flex", alignItems: "center", justifyContent: "center", width: "6rem", height: "6rem", borderRadius: "1.5rem", flexShrink: 0 },
  content: { flex: 1, minWidth: 0 },
  title: { fontSize: "1.5rem", fontWeight: 700, marginBottom: "0.6rem", color: "#0f1b2b", letterSpacing: "-0.01em" },
  description: { fontSize: "1.125rem", lineHeight: 1.75, color: "#475569", margin: 0 },
  arrowBtn: { display: "flex", alignItems: "center", justifyContent: "center", width: "3.5rem", height: "3.5rem", borderRadius: "50%", background: "rgba(255, 255, 255, 0.95)", flexShrink: 0, boxShadow: "0 6px 16px rgba(15, 27, 43, 0.08)" },
};

export default ActionCards;
