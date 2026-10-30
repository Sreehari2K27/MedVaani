function DisclaimerCard() {
  return (
    <section style={styles.section}>
      <div className="container">
        <div style={styles.card}>
          <div style={styles.iconWrapper}>
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="16" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
          </div>
          <div style={styles.content}>
            <h3 style={styles.title}>Disclaimer</h3>
            <p style={styles.description}>
              MedVaani provides information based on approved product documents
              for Lantus® SoloStar®. It does not provide medical advice, diagnosis,
              or treatment. Always consult your doctor or healthcare professional
              for personal medical guidance.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

const styles: Record<string, React.CSSProperties> = {
  section: { padding: "2rem 0 4.5rem" },
  card: { display: "flex", alignItems: "flex-start", gap: "1.5rem", padding: "2.25rem", background: "linear-gradient(135deg, #f3f2ff 0%, #faf9ff 100%)", borderRadius: "1.5rem", border: "none" },
  iconWrapper: { display: "flex", alignItems: "center", justifyContent: "center", width: "3.5rem", height: "3.5rem", borderRadius: "1.125rem", background: "rgba(99, 102, 241, 0.12)", flexShrink: 0, marginTop: "0.125rem" },
  content: { flex: 1 },
  title: { fontSize: "1.25rem", fontWeight: 700, marginBottom: "0.6rem", color: "#0f1b2b" },
  description: { fontSize: "1.0625rem", lineHeight: 1.85, color: "#475569", margin: 0 },
};

export default DisclaimerCard;
