import { useState, type ReactNode } from "react";
import GlobalLayout from "./components/GlobalLayout";
import type { GlobalNavId } from "./components/GlobalHeader";
import Hero from "./components/Hero";
import ActionCards from "./components/ActionCards";
import DisclaimerCard from "./components/DisclaimerCard";
import AeInterface from "./components/ae/AeInterface";
import AboutPage from "./components/pages/AboutPage";
import HowItWorksPage from "./components/pages/HowItWorksPage";
import MiPage from "./components/pages/MiPage";
import ContactPage from "./components/pages/ContactPage";
import "./styles.css";
import "./ae-styles.css";

type RouteId = GlobalNavId | "ae";

function App() {
  const [route, setRoute] = useState<RouteId>("home");
  const [chatMode, setChatMode] = useState(false);

  const navigate = (id: GlobalNavId) => {
    setChatMode(false);
    setRoute(id);
  };

  const openAe = () => {
    setChatMode(false);
    setRoute("ae");
  };

  const activeId: GlobalNavId | undefined =
    route === "ae" ? undefined : route;

  let content: ReactNode;
  switch (route) {
    case "about":
      content = <AboutPage />;
      break;
    case "how":
      content = <HowItWorksPage />;
      break;
    case "mi":
      content = <MiPage />;
      break;
    case "contact":
      content = <ContactPage />;
      break;
    case "ae":
      content = <AeInterface onChatModeChange={setChatMode} />;
      break;
    default:
      content = (
        <>
          <Hero />
          <ActionCards
            onReportSideEffect={openAe}
            onGetProductInfo={() => navigate("mi")}
          />
          <DisclaimerCard />
        </>
      );
  }

  return (
    <GlobalLayout
      activeId={activeId}
      onNavigate={navigate}
      chatMode={route === "ae" && chatMode}
    >
      {content}
    </GlobalLayout>
  );
}

export default App;