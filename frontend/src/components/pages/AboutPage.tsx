function AboutPage() {
  return (
    <section className="info-page">
      <div className="hero-aura" aria-hidden="true" />
      <div className="container info-wrap">
        <h1>About MedVaani</h1>
        <p className="info-lead">
          MedVaani is a voice-first health companion that helps you understand
          Lantus® SoloStar® and document side effects — in your own language,
          whenever you need it.
        </p>

        <div className="info-card">
          <h2>Our purpose</h2>
          <p>
            We believe important product information should be easy to
            understand and hard to miss. MedVaani makes it conversational:
            ask questions naturally, get clear answers, and report an adverse
            event in minutes.
          </p>
        </div>

        <div className="info-card">
          <h2>Built on referenced product documents</h2>
          <p>
            Product information in MedVaani comes from the referenced official
            product documents. MedVaani does not diagnose conditions or
            provide treatment advice — for personal medical guidance, always
            consult a qualified healthcare professional.
          </p>
        </div>

        <div className="info-card">
          <h2>In association with Sanofi</h2>
          <p>
            MedVaani is provided in association with Sanofi to support people
            using Lantus® SoloStar®.
          </p>
        </div>
      </div>
    </section>
  );
}

export default AboutPage;