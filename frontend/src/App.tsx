import Index from "./index";
import { Button } from "./components/ui/button";

const App = () => {
  return (
    <>
      <header className="flex justify-end items-center h-14 px-2 w-full border-b border-gray-200">
        <div className="mx-auto max-w-5xl w-full flex justify-between items-center">
          <a
            href="https://github.com/julienlucas/fake-detector-nanobananapro"
            target="_blank"
          >
            <Button size="lg">
              Repo Github
            </Button>
          </a>
        </div>
      </header>
      <Index />
      <footer className="mx-auto border-t border-zinc-200 mt-auto">
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
