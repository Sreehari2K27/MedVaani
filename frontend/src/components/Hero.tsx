import lantusPen from "../assets/lantus-solostar.png.png";

function Hero() {
  return (
    <section className="hero-section">
      <div className="container hero-grid">
        <div style={styles.content}>
          <h1 style={styles.heading}>
            Ask. Report.
            <br />
            Stay <span style={styles.gradient}>Informed.</span>
          </h1>
          <p style={styles.supporting}>
            Your voice-based assistant for Lantus® SoloStar®.
            <br />
            Get accurate product information and report side effects —
            <br />
            in your language, instantly.
          </p>
        </div>
        <div style={styles.visual}>
          <div style={styles.penWrapper}>
            <div style={styles.lantusLabel}>
              <span style={styles.lantus}>Lantus® SoloStar®</span>
            </div>
            <p style={styles.ingredient}>Insulin glargine (rDNA origin) Injection</p>
            <img src={lantusPen} alt="Lantus SoloStar pen" style={styles.penImg} />
          </div>
        </div>
      </div>
    </section>
  );
}

const styles: Record<string, React.CSSProperties> = {
  content: {},
  heading: {
    fontSize: "clamp(2.25rem, 5.4vw, 6.5rem)",
    fontWeight: 700,
    lineHeight: 1.06,
    marginBottom: "1.75rem",
    color: "#0f1b2b",
    letterSpacing: "-0.02em",
  },
  gradient: {
    background: "linear-gradient(90deg, #6366f1 0%, #ec4899 50%, #f59e0b 100%)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    backgroundClip: "text",
  },
  supporting: {
    fontSize: "1.3rem",
    lineHeight: 1.7,
    color: "#475569",
  },
  visual: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  penWrapper: {
    textAlign: "center" as const,
    width: "100%",
  },
  lantusLabel: {
    marginBottom: "0.5rem",
  },
  lantus: {
    fontWeight: 700,
    fontSize: "2.5rem",
    color: "#4f46e5",
    letterSpacing: "-0.01em",
  },
  ingredient: {
    fontSize: "1.125rem",
    color: "#64748b",
    marginBottom: "1.75rem",
  },
  penImg: {
    display: "block",
    margin: "0 auto",
    maxWidth: "720px",
    width: "100%",
    height: "auto",
    mixBlendMode: "multiply",
    maskImage:
      "linear-gradient(to right, transparent 0%, #000 7%, #000 96%, transparent 100%), linear-gradient(to bottom, transparent 0%, #000 15%, #000 85%, transparent 100%)",
    WebkitMaskImage:
      "linear-gradient(to right, transparent 0%, #000 7%, #000 96%, transparent 100%), linear-gradient(to bottom, transparent 0%, #000 15%, #000 85%, transparent 100%)",
    maskComposite: "intersect",
    WebkitMaskComposite: "source-in",
  },
};

export default Hero;
