function HowItWorksPage() {
  return (
    <section className="info-page">
      <div className="hero-aura" aria-hidden="true" />
      <div className="container info-wrap">
        <h1>How it works</h1>
        <p className="info-lead">
          Three simple ways to use MedVaani — ask, report, and stay informed.
        </p>

        <ol className="info-steps">
          <li className="info-card">
            <h2>1. Ask</h2>
            <p>
              Ask questions about Lantus® SoloStar® in your own words and get
              clear answers drawn from the referenced product documents.
            </p>
          </li>
          <li className="info-card">
            <h2>2. Report</h2>
            <p>
              Report a side effect by typing or speaking what happened.
              MedVaani guides you through the details to build a complete
              safety report.
            </p>
          </li>
          <li className="info-card">
            <h2>3. Stay informed</h2>
            <p>
              Track your report as it comes together and review every detail
              before it is completed.
            </p>
          </li>
        </ol>

        <p className="info-lead" style={{ marginBottom: 0 }}>
          MedVaani helps document safety reports and provides product
          information from referenced product documents. It does not diagnose
          conditions or provide treatment advice.
        </p>
      </div>
    </section>
  );
}

export default HowItWorksPage;