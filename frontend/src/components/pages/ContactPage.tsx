function ContactPage() {
  return (
    <section className="info-page">
      <div className="hero-aura" aria-hidden="true" />
      <div className="container info-wrap">
        <h1>Contact us</h1>
        <p className="info-lead">
          We are here to help with questions about MedVaani and Lantus®
          SoloStar®.
        </p>

        <div className="info-card">
          <h2>Get in touch</h2>
          <p>
            For questions about the MedVaani experience or the information
            presented here, reach out through your usual Sanofi contact
            channels.
          </p>
        </div>

        <div className="info-card">
          <h2>Reporting a side effect</h2>
          <p>
            Suspect a side effect? Use the Report a side effect flow to
            document what happened — you can type or speak your experience.
          </p>
        </div>

        <div className="info-card">
          <h2>Emergency</h2>
          <p>
            If you are experiencing a serious or life-threatening reaction,
            seek immediate medical attention.
          </p>
        </div>
      </div>
    </section>
  );
}

export default ContactPage;