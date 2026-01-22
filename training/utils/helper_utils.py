import time
import copy

import numpy as np
try:
    import seaborn as sns
except ImportError:
    sns = None
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision.models import resnet18
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from torchvision.models import ResNet18_Weights
    TV_DEFAULT_WEIGHTS = ResNet18_Weights.DEFAULT
except Exception:
    TV_DEFAULT_WEIGHTS = None

try:
    from torchao.quantization import QuantStub, DeQuantStub
except Exception:
    try:
        from torch.ao.quantization import QuantStub, DeQuantStub
    except Exception:
        from torch.quantization import QuantStub, DeQuantStub


def bench(m, iters=20, shape = (16, 3, 224, 224), device="cpu"):
    torch.manual_seed(17)
    m.eval()
    x = torch.randn(shape).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = m(x)
    return (time.perf_counter() - start) / iters


def compute_accuracy(model, dataloader, device):
    """
    Calcule l'accuracy globale sur un dataloader.
    """
    # Mettre le modèle en mode évaluation
    model.eval()
    # Déplacer le modèle vers le périphérique
    model.to(device)
    # Initialiser les compteurs
    correct = 0
    total = 0
    # Désactiver les calculs de gradient pour l'inférence
    with torch.no_grad():
        # Parcourir les batches du dataloader
        for images, labels in dataloader:
            # Déplacer les images et labels vers le périphérique
            images = images.to(device)
            labels = labels.to(device)
            # Effectuer une passe forward
            outputs = model(images)
            # Obtenir les prédictions (classe avec la plus haute probabilité)
            preds = torch.argmax(outputs, dim=1)
            # Compter les prédictions correctes
            correct += (preds == labels).sum().item()
            # Compter le nombre total d'échantillons
            total += labels.numel()
    # Retourner l'accuracy (fraction de prédictions correctes)
    return (correct / total) if total else 0.0


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png", show=True):
    if plt is None or sns is None:
        print("matplotlib ou seaborn non disponible, impossible d'afficher la matrice de confusion")
        return save_path

    was_interactive = plt.isinteractive()
    if not show:
        plt.ioff()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Étiquettes prédites")
    plt.ylabel("Étiquettes réelles")
    plt.title("Matrice de confusion")

    if show:
        plt.show()
    else:
        plt.close()
        if was_interactive:
            plt.ion()

    return save_path


def per_class_acc_and_conf_matrix(trained_model, data_module):
    """
    Évalue un modèle entraîné sur un dataset de validation et affiche un rapport
    de précision par classe et une matrice de confusion.

    Args:
        trained_model: Le modèle Lightning entraîné à évaluer.
        data_module: Le module de données contenant le dataloader de validation.
    """
    # --- Configuration ---
    # Met le modèle en mode évaluation
    trained_model.eval()
    # Détermine le dispositif à utiliser pour le calcul
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Déplace le modèle vers le dispositif sélectionné
    trained_model = trained_model.to(device)

    # Initialise des listes pour stocker les prédictions et les étiquettes réelles
    all_preds = []
    all_labels = []

    # --- Exécution de l'inférence ---
    # Enveloppe le dataloader avec tqdm pour une barre de progression
    val_loader_with_progress = tqdm(
        data_module.val_dataloader(),
        # Définit une description pour la barre de progression
        desc="Évaluation du modèle",
        # Ne laisse pas la barre de progression après la fin
        leave=False
    )

    # Désactive les calculs de gradient pour l'inférence
    with torch.no_grad():
        # Parcourt les lots dans le dataloader de validation
        for batch in val_loader_with_progress:
            # Dépaquette les images et les étiquettes du lot
            images, labels = batch
            # Déplace les images et les étiquettes vers le dispositif approprié
            images, labels = images.to(device), labels.to(device)

            # Effectue une passe avant pour obtenir les sorties du modèle
            outputs = trained_model(images)
            # Obtient la classe prédite en trouvant l'index avec la valeur la plus élevée
            preds = torch.argmax(outputs, dim=1)

            # Ajoute les prédictions du lot actuel à la liste
            all_preds.append(preds)
            # Ajoute les étiquettes réelles du lot actuel à la liste
            all_labels.append(labels)

    # --- Calcul et affichage des métriques ---
    # Concatène toutes les prédictions en un seul tenseur
    all_preds = torch.cat(all_preds)
    # Concatène toutes les étiquettes réelles en un seul tenseur
    all_labels = torch.cat(all_labels)

    # Initialise la métrique de matrice de confusion
    num_classes = len(data_module.val_dataset.classes)
    confmat = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    # Calcule la matrice de confusion
    cm = confmat(all_preds, all_labels)

    # Calcule la précision par classe à partir de la matrice de confusion
    per_class_acc = cm.diag() / cm.sum(axis=1)
    total = cm.sum().item()
    correct = cm.diag().sum().item()
    acc_global = (correct / total) if total else 0.0
    # Récupère les noms des classes depuis le dataset
    class_names = data_module.val_dataset.classes

    # Affiche un en-tête pour le rapport de précision
    print("--- Rapport de précision par classe ---")
    print(f"Précision globale : {acc_global:.4f}")
    # Parcourt chaque classe et affiche sa précision
    for i, acc in enumerate(per_class_acc):
        # Affiche la précision pour la classe actuelle
        print(f"  - Précision pour la classe '{class_names[i]}' : {acc.item():.4f}")
    # Affiche une nouvelle ligne pour l'espacement
    print()

    # Trace la matrice de confusion
    plot_confusion_matrix(cm.cpu().numpy(), class_names)


class QATBasicBlock(nn.Module):
    """
    Variante ResNet BasicBlock compatible QAT avec noms identiques à torchvision.
    Utilise FloatFunctional pour l'addition résiduelle (compatible quantification).
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # Noms identiques à torchvision ResNet18
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Chemin de downsample comme Sequential (ou None) pour compatibilité torchvision
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

        # Addition FloatFunctional pour que l'addition résiduelle soit consciente de la quantification
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.skip_add.add(out, identity)
        out = self.relu(out)
        return out


class QATResNet18(nn.Module):
    """
    Architecture ResNet18 compatible QAT avec noms identiques à torchvision.
    Inclut QuantStub/DeQuantStub pour que la préparation/conversion QAT fonctionne proprement.
    """
    def __init__(self, num_classes=1000, use_quant_stubs=False, hidden_size=512, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.use_quant_stubs = use_quant_stubs
        if use_quant_stubs:
            self.quant = QuantStub()
        else:
            self.quant = nn.Identity()

        # Noms identiques à torchvision ResNet18
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 étages : [2,2,2,2] blocs, strides : [1,2,2,2]
        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # FC Sequential correspondant aux poids sauvegardés
        num_ftrs = 512 * QATBasicBlock.expansion
        self.fc = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(num_ftrs, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden_size, num_classes)
        )

        if use_quant_stubs:
            self.dequant = DeQuantStub()
        else:
            self.dequant = nn.Identity()

    def _make_layer(self, inplanes, planes, blocks, stride):
        layers = [QATBasicBlock(inplanes, planes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(QATBasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant(x)
        return x



@torch.no_grad()
def load_imagenet_pretrained_into_qat_resnet18(model: nn.Module,
                                              weights=TV_DEFAULT_WEIGHTS,
                                              strict=False):
    """
    Charge les poids ResNet18 de torchvision dans QATResNet18.
    Les noms de couches sont maintenant identiques, seule la FC est ignorée.

    Returns:
        model (nn.Module): même instance avec les poids chargés.
        missing_keys (list[str]), unexpected_keys (list[str])
    """
    # Construire un resnet18 torchvision standard avec les poids ImageNet
    if TV_DEFAULT_WEIGHTS is not None:
        tv = resnet18(weights=weights)
    else:
        tv = resnet18(pretrained=True)  # torchvision plus ancien

    tv_sd = tv.state_dict()

    # Filtrer les clés FC (structure différente: Linear vs Sequential)
    new_sd = {k: v for k, v in tv_sd.items() if not k.startswith("fc.")}

    incompat = model.load_state_dict(new_sd, strict=strict)
    return model, list(incompat.missing_keys), list(incompat.unexpected_keys)


def resnet18_qat_ready_pretrained(num_classes=1000, use_quant_stubs=False,
                                   hidden_size=512, dropout1=0.3, dropout2=0.2):
    """
    Construit votre QATResNet18, charge les poids ImageNet et retourne le modèle FP32.
    Si num_classes != 1000, la couche FC est initialisée aléatoirement.
    """
    model_fp32 = QATResNet18(
        num_classes=num_classes,
        use_quant_stubs=use_quant_stubs,
        hidden_size=hidden_size,
        dropout1=dropout1,
        dropout2=dropout2
    )
    # Charger les poids ImageNet où les formes correspondent
    model_fp32, missing, unexpected = load_imagenet_pretrained_into_qat_resnet18(
        model_fp32, weights=TV_DEFAULT_WEIGHTS, strict=False
    )
    return model_fp32


def load_resnet18_from_checkpoint(weights_path, num_classes=None):
    """
    Charge un ResNet18 standard depuis un checkpoint.
    Gère les préfixes 'model.' et 'module.' automatiquement.
    """
    state = torch.load(weights_path, map_location="cpu")
    state = state.get("state_dict", state)

    state = {
        (k[6:] if k.startswith("model.") else k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
        if not k.startswith("loss_fn.")
    }

    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features

    if num_classes is not None:
        model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif "fc.weight" in state:
        num_classes = state["fc.weight"].shape[0]
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        hidden = state["fc.1.weight"].shape[0]
        num_classes = state["fc.4.weight"].shape[0]
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    model.load_state_dict(state, strict=True)
    return model


def train_model_with_best_checkpoint_and_metrics(
    model, train_loader, val_loader, num_epochs, optimizer, device, save_path=None
):
    """
    Entraîne un modèle et sauvegarde le meilleur checkpoint basé sur la précision de validation.

    Cette fonction exécute une boucle d'entraînement complète avec validation, en suivant
    la précision de validation et en sauvegardant le meilleur modèle. À la fin de l'entraînement,
    le modèle est restauré avec les poids du meilleur checkpoint.

    Args:
        model: Le modèle PyTorch à entraîner
        train_loader: DataLoader pour les données d'entraînement
        val_loader: DataLoader pour les données de validation
        num_epochs: Nombre d'époques d'entraînement
        optimizer: Optimiseur PyTorch (déjà configuré avec les paramètres du modèle)
        device: Périphérique sur lequel entraîner ('cpu', 'cuda', 'mps')
        save_path: Chemin optionnel pour sauvegarder le meilleur checkpoint

    Returns:
        model: Le modèle avec les poids du meilleur checkpoint chargés
    """
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # Variables pour suivre le meilleur modèle
    best_val_acc = 0.0
    best_model_state = None

    # Boucle d'entraînement sur les époques
    for epoch in range(num_epochs):
        # --- Phase d'entraînement ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Époque {epoch+1}/{num_epochs} [Entraînement]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Passe forward, backward et mise à jour
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumuler les métriques d'entraînement
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Phase de validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc=f"Époque {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Accumuler les métriques de validation
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_acc = 100 * val_correct / val_total
                val_pbar.set_postfix(acc=f"{val_acc:.2f}%")

        # Calculer les métriques finales de l'époque
        val_acc = 100 * val_correct / val_total
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Époque {epoch+1}/{num_epochs} - Perte : {avg_train_loss:.4f} - Précision Val : {val_acc:.4f}")

        # Vérifier si c'est le meilleur modèle jusqu'à présent
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_model_state, save_path)
                print(f"Nouvelle meilleure précision : {best_val_acc:.4f}, modèle sauvegardé dans {save_path}")

    # Restaurer le meilleur modèle à la fin de l'entraînement
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nEntraînement terminé :\nMeilleure précision : {best_val_acc:.4f}\nPrécision finale : {val_acc:.4f}")
        if save_path:
            print(f"Modèle final sauvegardé dans {save_path}")

    return model