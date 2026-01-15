import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Upload, X, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
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
  const [result, setResult] = useState<{label: string, confidence: number, real_confidence: number, fake_confidence: number, image?: string} | null>(null);
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

      const response = await fetch('/api/inference', {
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
            className="font-normal bg-gradient-to-br from-[#2b3632] via-[#30574a] to-[#b2f7e1] text-white rounded-xl w-11 h-11 flex items-center justify-center text-5xl"
          >
            +
          </div>
          <span style={{ fontFamily: "'Gabarito', sans-serif" }}>
            Health Assistant
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
            Testez avec une de ces images (correctement détectées)
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

      <Card className="mt-12 border-none max-w-2xl mx-auto shadow-none">
        <CardContent className="p-0 border-none">
          <CardTitle
            variant="h2"
            className="bg-gradient-to-br from-black via-black to-black bg-clip-text text-transparent"
          >
            Étude de cas
          </CardTitle>
          <CardTitle variant="h3">
            Détecter les signes de pneumonie malgrés un entraînement de modèle
            comportant très peu de radiographies
          </CardTitle>
          <CardTitle variant="h3">Le challenge</CardTitle>
          <p className="mb-4">
            Créer un détecteur d'images Nano Banana Pro précis, rapidement et
            peu couteux :
          </p>
          <ul className="list-disc list-inside mb-4 space-y-2">
            <li>
              <strong>
                Pouvoir détecter sur tous les modèles de diffusion
              </strong>{" "}
              : Le modèle devait fonctionner sur Nano Banana Pro mais aussi
              Midjourney, Stable Diffusion, DALL-E (des modèles de diffusion
              aussi).
            </li>
            <li>
              <strong>Entraîner un modèle rapidement</strong> : Réutiliser les
              connaissances pré-existantes d'un modèle de vision, d'abord le
              MobileNetV3 Large sur ImageNet et tester de l'adapter à la
              détection de fakes.
            </li>
            <li>
              <strong>Avoir un modèle faible latence</strong> : Devait pouvoir
              fonctionner en quelque secondes sur un mobile.
            </li>
            <li>
              <strong>Créer un jeu de données</strong> : Combiner des datasets
              scrappés sur Midjourney/DALL-E/SD et Nano Banana Pro pour une
              détection généralisée.
            </li>
          </ul>
          <CardTitle variant="h3">Résultats et évaluation</CardTitle>
          <p className="mb-4">
            Le système utilise une approche de transfer learning avec ResNet18
            pour détecter les images générées par IA :
          </p>
          <ul className="list-inside mb-4 space-y-2">
            <li>
              <strong>
                ⌛ Entraînement en seulement 2 minutes et juste avec un Mac Book
                Pro M1
              </strong>{" "}
              : Juste 1 seule passe sur le jeu de donnée!
            </li>
            <li>
              <strong>
                🧠 Méthode de fine-tuning du ResNet18 par Transfer Learning
              </strong>{" "}
              : Fine-tuning de la dernière couche du modèle, le classifieur
              uniquement (reste du modèle gelé), pour un entraînement ultra
              rapide.
            </li>
            <li>
              <strong>
                🎯 Précision pour la classe 'Pneumonie bactérienne' : 89%
              </strong>
            </li>
            <li>
              <strong>🎯 Précision pour la classe 'Normal' : 75%</strong>
            </li>
            <li>
              <strong>
                🎯 Précision pour la classe 'Pneumonie virale' : 83%
              </strong>
            </li>
            <li>
              <strong>
                ⚡ Le modèle a une <span>très faible latence</span>, quasi
                instannée.
              </strong>
            </li>
          </ul>
          <p>Et voilà.</p>
          <CardTitle variant="h3" className="mt-6 text-center">
            On discute de votre projet?
          </CardTitle>
          <div className="flex justify-center">
            <Button className="mx-auto w-full" size="xl">
              Me contacter
            </Button>
          </div>
        </CardContent>
      </Card>
    </main>
  );
}