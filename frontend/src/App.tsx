import Index from "./index";
import { Button } from "./components/ui/button";
import { CardTitle, CardDescription } from "./components/ui/card";

const App = () => {
  return (
    <>
      <header className="flex justify-end items-center h-14 px-2 w-full border-b border-gray-200">
        <div className="mx-auto max-w-5xl w-full flex justify-between items-center">
          <a
            href="https://github.com/julienlucas/pneumonia-diagnostic-assistant"
            target="_blank"
          >
            <Button size="lg">Repo Github</Button>
          </a>
        </div>
      </header>
      <Index />
      <footer className="mx-auto max-w-2xl w-full border-t border-zinc-200 mt-auto">
        <CardTitle variant="h4">Mes autres projets IA</CardTitle>
        <CardDescription className="text-sm flex items-center gap-1">
          Fashion Style Analyzer MM-RAG
        </CardDescription>
        <CardDescription className="text-sm flex items-center gap-1">
          DocChat RAG Agentique pour docs techniques
        </CardDescription>
        <div className="max-w-[1100px] mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="text-sm flex justify-center">
            © {new Date().getFullYear()} Julien Lucas. Tous droits réservés.
          </div>
        </div>
      </footer>
    </>
  );
};

export default App;
