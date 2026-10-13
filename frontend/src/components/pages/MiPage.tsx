function MiPage() {
  return (
    <section className="info-page">
      <div className="hero-aura" aria-hidden="true" />
      <div className="container info-wrap">
        <h1>Lantus® SoloStar®</h1>
        <p className="info-lead">
          Insulin glargine (rDNA origin) injection — 100 units/mL, for
          subcutaneous use.
        </p>

        <div className="info-card">
          <h2>What it is used for</h2>
          <p>
            Lantus® SoloStar® is a long-acting insulin used to improve
            glycemic control in adults and pediatric patients with diabetes
            mellitus. It is intended for once-daily subcutaneous administration.
          </p>
        </div>

        <div className="info-card">
          <h2>Important safety information</h2>
          <ul>
            <li>
              Lantus® SoloStar® is a prescription medicine that must be used
              as directed by a healthcare professional.
            </li>
            <li>
              Hypoglycemia is the most common adverse reaction of all
              insulins, including Lantus®.
            </li>
            <li>
              Do not share insulin pens, cartridges, or reusable pens between
              patients, even if the needle is changed.
            </li>
            <li>
              Serious, life-threatening hypersensitivity reactions, including
              generalized allergy and anaphylaxis, may occur.
            </li>
          </ul>
        </div>

        <div className="info-card">
          <h2>How to learn more</h2>
          <p>
            Detailed prescribing information is available from the referenced
            product documents. For personal medical guidance, always consult
            a qualified healthcare professional.
          </p>
        </div>
      </div>
    </section>
  );
}

export default MiPage;