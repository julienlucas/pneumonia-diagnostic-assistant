import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Upload, X, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import ContactForm from "@/components/ui/contact-form";
import { useState } from "react";

const exampleImages = [
  {
    value: "/static/BACTERIAL_PNEUMONIA/person78_bacteria_378.jpeg",
    label: "Pneumonie bactérienne",
  },
  {
    value: "/static/BACTERIAL_PNEUMONIA/person78_bacteria_385.jpeg",
    label: "Pneumonie bactérienne",
  },
  {
    value: "/static/NORMAL/IM-0001-0001.jpeg",
    label: "Normal",
  },
  {
    value: "/static/NORMAL/IM-0005-0001.jpeg",
    label: "Normal",
  },
  {
    value: "/static/VIRAL_PNEUMONIA/person1672_virus_2888.jpeg",
    label: "Pneumonie virale",
  },
  {
    value: "/static/VIRAL_PNEUMONIA/person1673_virus_2889.jpeg",
    label: "Pneumonie virale",
  },
];

export default function Index() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(
    "/static/BACTERIAL_PNEUMONIA/person78_bacteria_385.jpeg"
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    label: string;
    confidence: number;
    normal_confidence: number;
    bacterial_confidence: number;
    viral_confidence: number;
    image?: string;
  } | null>(null);
  const [selectedExampleImage, setSelectedExampleImage] = useState<string>("/static/Unknown.png");

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const allowedTypes = ['image/png', 'image/jpg', 'image/jpeg', 'image/webp'];
      if (allowedTypes.includes(file.type)) {
        setSelectedFile(file);
        const reader = new FileReader();
        reader.onloadend = () => {
          setPreview(reader.result as string);
        };
        reader.readAsDataURL(file);
        setResult(null);
      }
    }
  };

  const handleRemoveImage = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    const input = document.getElementById("file-input") as HTMLInputElement;
    if (input) {
      input.value = "";
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      const allowedTypes = ['image/png', 'image/jpg', 'image/jpeg', 'image/webp'];
      if (allowedTypes.includes(file.type)) {
        setSelectedFile(file);
        const reader = new FileReader();
        reader.onloadend = () => {
          setPreview(reader.result as string);
        };
        reader.readAsDataURL(file);
        setResult(null);
      }
    }
  };

  const handleUpload = async () => {
    setLoading(true);
    const body = new FormData();

    try {
      let fileToUpload = selectedFile;

      if (!fileToUpload && preview && preview.startsWith('/static/')) {
        const response = await fetch(preview);
        const blob = await response.blob();
        const fileName = preview.split('/').pop() || 'image.jpg';
        fileToUpload = new File([blob], fileName, { type: blob.type || 'image/jpeg' });
      }

      if (!fileToUpload) {
        console.error('Aucun fichier à uploader');
        setLoading(false);
        return;
      }

      body.append("file", fileToUpload);

      const apiUrl = import.meta.env.RAILWAY_API_URL || '';
      const response = await fetch(`${apiUrl}/api/inference`, {
        method: 'POST',
        body
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Erreur API:', errorData);
        throw new Error(errorData.error || 'Erreur lors de l\'analyse');
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Erreur:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto max-w-5xl w-full pb-24 px-2">
      <CardHeader className="relative z-20 overflow-hidden">
        <CardTitle
          variant="h1"
          className="relative text-center mx-auto max-w-sm flex items-center justify-center gap-3"
        >
          <div
            style={{ fontFamily: "'Gabarito', sans-serif" }}
            className="font-normal bg-gradient-to-br from-[#2b3632] via-[#30574a] to-[#b2f7e1] text-white rounded-xl w-11 h-11 flex items-center justify-center text-5xl pb-1.5"
          >
            +
          </div>
          <span style={{ fontFamily: "'Gabarito', sans-serif" }}>
            PneumoDiag
          </span>
        </CardTitle>
        <CardDescription className="text-center text-2xl font-bold text-black max-w-lg mx-auto leading-7">
          Analysez vos radiographies pour détecter les signes de pneumonie
        </CardDescription>
        <CardDescription className="text-center text-sm">
          <strong>Modèle fine-tuné</strong> avec un modèle faible poids ResNet18
          <br />
          Testé sur 450 radiographies
          <br />
          Précision: 85% (pourtant sans optimisation et avec un modèle simple
          datant de 2015)
        </CardDescription>
      </CardHeader>
      <Card className="border-none shadow-none">
        <CardContent className="p-0 mt-8 min-h-[500px] mx-auto space-y-4 text-muted-foreground grid grid-cols-1 md:grid-cols-2 items-stretch justify-center gap-8">
          <div className="flex flex-col h-full">
            <div
              className={cn(
                "relative w-full flex-1 bg-gray-100 border-2 border-dashed rounded-sm text-center transition-colors flex flex-col",
                "border-upload-border hover:border-primary",
                preview ? "p-1" : "p-4 items-center justify-center"
              )}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              {preview ? (
                <>
                  <img
                    src={preview}
                    alt="Preview"
                    className="w-full h-full object-cover rounded-sm"
                  />
                  <Button
                    variant="secondary"
                    size="icon"
                    className="absolute top-2 right-2 z-10"
                    onClick={handleRemoveImage}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </>
              ) : (
                <>
                  <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-sm text-muted-foreground mb-4">
                    Uploadez une image
                  </p>
                  <input
                    id="file-input"
                    type="file"
                    accept="image/png,image/jpg,image/jpeg,image/webp"
                    className="hidden"
                    onChange={handleFileSelect}
                  />
                  <Button
                    variant="outline"
                    onClick={() =>
                      document.getElementById("file-input")?.click()
                    }
                  >
                    Charger une image
                  </Button>
                </>
              )}
            </div>
            <Button
              onClick={handleUpload}
              className="w-full mt-3"
              size="xl"
              disabled={loading || !preview}
            >
              <Search className="h-4 w-4" />
              {loading ? "Analyse en cours..." : "Analyser la radio"}
            </Button>
          </div>
          <div className="relative -top-4">
            <CardDescription className="mb-4 italic text-sm">
              Exemple de détection (la heatmap indique les zones d'activation)
            </CardDescription>
            <div className="relative rotate-[3deg]">
              <img
                src={
                  result && result.image ? result.image : selectedExampleImage
                }
                alt="Exemple deepfake"
                className="w-full h-auto rounded shadow-lg"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="mt-12 border-none mx-auto shadow-none">
        <CardContent className="p-0 border-none">
          <CardTitle variant="h4">
            Testez avec une de ces images
          </CardTitle>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
            {exampleImages.map((img) => (
              <div
                key={img.value}
                className={cn(
                  "relative h-40 cursor-pointer rounded-md overflow-hidden transition-all hover:opacity-80",
                  selectedExampleImage === img.value
                    ? "border-primary ring-2 ring-primary"
                    : "border-gray-200 hover:border-gray-300"
                )}
                onClick={() => setPreview(img.value)}
              >
                <img
                  src={img.value}
                  alt={img.label}
                  className="w-full h-full object-cover"
                />
                <div className="absolute bottom-0 left-0 w-full">
                  <p className="text-white text-sm bg-black/40 p-0.5 px-1">
                    {img.label}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card
        id="contact-form"
        className="mt-12 border-none max-w-2xl mx-auto shadow-none"
      >
        <CardContent className="p-0 border-none">
          <CardTitle
            variant="h2"
            className="bg-gradient-to-br from-black via-black to-black bg-clip-text text-transparent"
          >
            Étude de cas
          </CardTitle>
          <CardTitle variant="h3-card" className="mb-0 mt-4">
            Le challenge
          </CardTitle>
          <CardTitle variant="h3" className="font-medium">
            Détecter les signes de pneumonie et leur degrès de viralité malgrés
            un jeu de données peu éttoffé en radiographies
          </CardTitle>
          <ul className="list-disc list-inside mb-4 space-y-4">
            <li>
              <strong>Détecter les pneumonies de patients</strong>{" "}
              selon leur degrès de viralité sur les radiographies.
            </li>
            <li>
              <strong>Entraîner un modèle rapidement rapidement à moindre coût.</strong> Réutiliser les
              connaissances pré-existantes d'un modèle de vision et l'adapter à la détection de pneumonies.
            </li>
            <li>
              <strong>Avoir un modèle faible latence.</strong> Doit pouvoir
              fonctionner sur un mobile.
            </li>
            <li>
              <strong>
                Avoir un modèle suffisament précis malgrés avec un jeu de données peu étoffé
              </strong>
            </li>
          </ul>
          <CardTitle variant="h3-card">Résultats et évaluation</CardTitle>
          <ul className="list-inside mb-4 space-y-4">
            <li>
              <strong>
                ⌛ <span>Entraînement en seulement 2 minutes</span> et avec un hardware peu conséquent, seulement mon Mac Book Pro M1
              </strong>. Et juste en 1 seule passe sur le jeu de données!
            </li>
            <li>
              <strong>
                🧠 <span>Méthode de fine-tuning rapide d'un modèlé peu gourmand en ressources</span>, le
                par Transfer Learning
              </strong>. Réseau de neurones dense utile sur les petits jeux de données. Tuning de la dernière couche du modèle, uniquement le classifieur
              pour un entraînement apportant un maximum de résultats rapidement.
            </li>
            <li>
              <strong>
                🎯 Au final facilement et sans optimisation du modèle, précision pour <span>la classe 'Pneumonie bactérienne' :{" "}
                89%</span>
              </strong>
            </li>
            <li>
              <strong>
                🎯 Précision pour <span>la classe 'Normal' : 75%</span>
              </strong>
            </li>
            <li>
              <strong>
                🎯 Précision pour <span>la classe 'Pneumonie virale' :{" "}
                83%</span>
              </strong>
            </li>
            <li>
              <img
                src="/static/langsmith.png"
                className="w-full h-auto rounded mt-3 border border-gray-100 rounded-sm"
              />
              <CardDescription className="italic text-center text-xs">
                Montoring dans Langsmith
              </CardDescription>
            </li>
          </ul>
          <p>Ce POC démontre la puissance de la méthode mais mériterait un modèle plus puissant pour viser une précision plus élevée.</p>
          <CardTitle
            variant="h3"
            className="mt-12 max-w-xl mx-auto text-center"
          >
            On discute de votre projet d'automatisation ou d'application?
          </CardTitle>
          <CardDescription className="text-center mb-4">
            Remplissez le formulaire ci-dessous et je vous recontacte dans les
            24-48 heures.
          </CardDescription>
          <ContactForm />
        </CardContent>
      </Card>
    </main>
  );
}