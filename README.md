# 🚀 Poetry Generator - Fine-Tuning et Évaluation des Modèles LLM

Bienvenue dans ce projet de **génération de poèmes** basé sur le **fine-tuning de modèles de langage**. Ce projet explore différentes architectures de modèles pour produire des poèmes de haute qualité à partir d'instructions données par l'utilisateur.

---

## 📌 Objectifs du projet

- **Fine-tuner plusieurs modèles de langage** pour la génération de poèmes.
- **Comparer les performances** des modèles en utilisant des métriques de qualité (BLEU, ROUGE, Cosine Similarity).
- **Sélectionner le meilleur modèle** en fonction des performances obtenues.
- **Développer une interface utilisateur interactive** permettant aux utilisateurs de générer des poèmes en entrant un titre, un thème et des instructions spécifiques.

---

## 🛠️ 1. Fine-Tuning et Évaluation des Modèles

Nous avons testé et comparé **quatre modèles de langage** pour la génération de poèmes :

1. **GPT-2 Fine-Tuned** 
2. **LLaMA-3-8B Fine-Tuned** 
3. **Mistral 7B Fine-Tuned** 
4. **GPT-NeoX-20B Fine-Tuned**

📌 **Tous les notebooks de fine-tuning sont disponibles dans le dossier `notebooks/` du dépôt.**

### **🔍 Étapes du Fine-Tuning :**

1. **Préparation des données** : Sélection et prétraitement d’un dataset de poèmes (`checkai/instruction-poems`).
2. **Fine-tuning sur GPU** : Entraînement sur Google Colab et stockage sur Google Drive.
3. **Évaluation des performances** : Calcul des scores BLEU, ROUGE et Cosine Similarity.
4. **Comparaison des résultats** : Sélection du modèle le plus performant pour l'application finale.

#### **📊 Résultats de l’évaluation :**

| Modèle | BLEU Score | ROUGE-1 | ROUGE-2 | ROUGE-L | Cosine Similarity |
|--------|-----------|---------|---------|---------|------------------|
| GPT-2 Fine-Tuned | 0.0222 | 0.2000 | 0.0486 | 0.1517 | 0.0029 |
| LLaMA-3-8B Fine-Tuned | 0.0051 | 0.2029 | 0.0228 | 0.1318 | 0.1898 |
| GPT-NeoX-20B (non fine-tuné) | - | - | - | - | - |
| **Mistral 7B Fine-Tuned** | **0.0034** | **0.1684** | **0.0189** | **0.1113** | **0.1649** |

🎯 **Le modèle ..... Fine-Tuned a été sélectionné pour l’application finale.**

📌 **Détails techniques :** Le fine-tuning a été effectué sur Google Colab avec `transformers`, `torch` et `datasets` et GPU `A100` est obligatoire.

---

## 📷 Interface d'Évaluation

Nous avons développé une **interface intuitive avec Streamlit** pour permettre :

- La **génération de poèmes** basés sur un thème et des instructions spécifiques.
- L’**évaluation automatique des résultats** à l’aide de métriques NLP.
- L’**exportation des résultats** au format CSV.
- L’objectif principal de cette interface est d'évaluer et de comparer les modèles afin de sélectionner le plus performant.
- Dans la prochaine étape, nous développerons une application adaptée aux utilisateurs finaux, intégrant le modèle choisi.
- 
Voici quelques images de notre interface ainsi que les résultats obtenus lors de nos tests des modèles.
### **1️⃣ Sélection du modèle et génération d'un poème**
![Interface complète](images/interface_1_complete.PNG)

### **2️⃣ Sélection du modèle et génération de poèmes sur un ensemble de données (un dataset de plusieurs lignes)**
![Interface 2](images/interface_2.PNG)

### **3️⃣ Poème généré avec Mistral 7B**
![Poème généré](images/mistrale_generated_poeme_1.PNG)
![Poème score](images/mistrale_7b_1_poeme_results_scores.PNG)

### **4️⃣ Poème généré avec LLaMA-3-8B**
![Poème généré](images/poeme_llama_1.PNG)
![Poème score](images/1_poeme_score_llama_1.PNG)

### **5️⃣ Poème généré avec GPT-2**
![Poème généré](images/gpt_2_generated_poeme.PNG)

### **6️⃣ Évaluation des scores BLEU et ROUGE pour LLaMA-3-8B sur un petit dataset de 50 lignes**
![Résultats d’évaluation 1](images/resultats_llama_2.PNG)
![Résultats d’évaluation 2](images/resultats_de_llama_1.PNG)

### **7️⃣ Évaluation des scores BLEU et ROUGE pour Mistral 7B sur un petit dataset de 50 lignes**
![Résultats d’évaluation](images/mistral_7b_resultats_2.PNG)
![Résultats d’évaluation 2](images/miostral_7b_resultats_1.PNG)

### **8️⃣  Évaluation des scores BLEU et ROUGE pour GPT-2 sur un petit dataset de 50 lignes**
![Évaluation sur dataset](images/gpt_2_result_2.PNG)
![Évaluation sur dataset](images/gpt2_scores_on_100_cols_1.PNG)

---

## 🚀 2. Accès Public au Modèle Fine-Tuné

💡 **Problème initial :** Les modèles fine-tunés étaient stockés sur Google Drive, ce qui empêchait leur accès public.

✅ **Solution : Hébergement sur Hugging Face Model Hub**

### **📦 Héberger le modèle sur Hugging Face**
1. **Se connecter à Hugging Face et uploader le modèle** :
```python
from huggingface_hub import login, upload_folder

login(token="TON_HF_TOKEN")
model_path = "/content/drive/MyDrive/mistral7b_on_instruction_poems"
repo_id = "ton-utilisateur/mistral7b_finetuned"
upload_folder(repo_id=repo_id, folder_path=model_path)
```

2. **Utiliser le modèle depuis n’importe où** :
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ton-utilisateur/mistral7b_finetuned")
tokenizer = AutoTokenizer.from_pretrained("ton-utilisateur/mistral7b_finetuned")
```

🔗 **Modèle hébergé sur :** [Hugging Face Model Hub](https://huggingface.co/ton-utilisateur/mistral7b_finetuned)

---

## 📌 Prochaines Étapes

✅ **Fine-tuning et évaluation des modèles**<br>
✅ **Sélection du meilleur modèle (Mistral 7B Fine-Tuned)**<br>
✅ **Hébergement du modèle sur Hugging Face pour accès public**<br>
🔜 **Ajout d’une visualisation graphique des résultats de l’évaluation**<br>
🔜 **Déploiement de l’application sur Hugging Face Spaces ou un serveur cloud**<br>
🔜 **Ajout d’un module d’amélioration stylistique des poèmes générés**<br>

---

## 🤝 Contribuer

Si vous souhaitez contribuer à ce projet :

1. **Forkez le dépôt**.
2. **Ajoutez vos améliorations** sur une nouvelle branche.
3. **Soumettez une Pull Request** avec une description détaillée de vos modifications.

N’hésitez pas à poser vos questions ou signaler des problèmes via la section **Issues** sur GitHub.

**🔗 Contact :** Pour toute question, contactez-moi directement via GitHub ou LinkedIn !

---

🔥 **Merci d’avoir suivi ce projet et bonne exploration de la génération de poésie avec l’IA !** 🔥

