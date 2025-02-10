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
## **1️⃣ Sur le dataset d'évaluation (que le modèle a vu durant l'entraînement) :** 

| Modèle | ROUGE-1 | ROUGE-2 | ROUGE-L | Cosine Similarity |
|--------|---------|---------|---------|------------------|
| GPT-2 Fine-Tuned | 0.8345 | 0.7342 | 0.8268 | 0.5457 |
| LLaMA-3-8B Fine-Tuned | 0.7895 | 0.6595 | 0.7802 | 0.6239 |
| GPT-NeoX-20B Fine-Tuned | 0.7002 | 0.4991 | 	0.6892 | 0.5245 |
| Mistral 7B Fine-Tuned | 0.7055 | 	0.5208 | 0.6934 | 0.6023 |
		

## **2️⃣ Sur le dataset de test (des données que le modèle n'a jamais vues) :** 

| Modèle | BLEU Score | ROUGE-1 | ROUGE-2 | ROUGE-L | Cosine Similarity |
|--------|-----------|---------|---------|---------|------------------|
| GPT-2 Fine-Tuned | 0.0222 | 0.2000 | 0.0486 | 0.1517 | 0.0029 |
| LLaMA-3-8B Fine-Tuned | 0.0138 |0.1531 | 0.0290 | 0.0970 | 0.1499 |
| GPT-NeoX-20B Fine-Tuned| - | - | - | - | - |
| Mistral 7B Fine-Tuned | 0.0034 | 0.0906 | 0.0178 | 0.0645 | 0.0866 |


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

## 🚀 2. Accès Public aux Modèles Fine-Tunés

🔗 **Les modèles fine-tunés sont hébergés sur Hugging Face et peuvent être téléchargés ici :**

- [LLaMA-3-8B Fine-Tuned](https://huggingface.co/IAyamina/llama3-8b_on_instruction_poems)
- [Mistral 7B Fine-Tuned](https://huggingface.co/IAyamina/mistral7b_on_instruction_poems)
- [GPT-2 Fine-Tuned](https://huggingface.co/IAyamina/gpt2_on_instruction_poems)
- [GPT-NeoX-20B Fine-Tuned](https://huggingface.co/IAyamina/gptneo20b_on_instruction_poems)

✅ **Solution :**

- Téléchargez le modèle directement depuis Hugging Face.
- Utilisez-le avec **Hugging Face Transformers** dans vos scripts Python.
- Exécutez les notebooks associés pour **réentraîner** le modèle et modifier les hyperparamètres si nécessaire.

💡 **Attention :**

- **Mistral 7B et LLaMA-3-8B nécessitent une clé d’accès**. Vous devez **demander l’accès** sur Hugging Face pour pouvoir les utiliser.

### **🔧 Étapes pour utiliser un modèle fine-tuné :**

1️⃣ **Chargez le modèle dans votre script Python :**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "IAYamina/mistral7b_on_instruction_poems"  # Remplacez par le modèle souhaité
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

2️⃣ **Générez un poème avec le modèle :**

```python
def generate_poem(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate_poem("Un poème sur l'automne"))
```
3️⃣ **Si nécessaire, entraînez à nouveau le modèle avec les notebooks disponibles.**
---

## 🚀 3. Lancer l'Application et Tester les Poèmes

💡 **Attention :**
- Notre code est conçu pour être exécuté sur **Google Colab**.
- Vous devez **posséder une clé d'authentification pour `pyngrok`** si vous souhaitez déployer l'application de la même manière.
- Tous les fichiers nécessaires à l'exécution se trouvent dans `app.py`.

### **🔧 Étapes pour lancer l'application :**

1️⃣ **Téléchargez les modèles et placez-les dans votre Google Drive** en vérifiant bien les chemins d'accès.

2️⃣ **Installez les dépendances nécessaires (`streamlit`, `pyngrok`, etc.)** et assurez-vous de disposer d'un **GPU A100** pour l'exécution optimale.

3️⃣ **Ajoutez vos clés d'authentification et exécutez `app.py` sur Google Colab**.

4️⃣ **Lancez l'application avec Streamlit et accédez-y via un tunnel `ngrok`.**

📌 **Commande pour lancer l'application**
```bash
!streamlit run app.py --server.port 8501
```
---

## 📌 Prochaines Étapes

✅ **Fine-tuning et évaluation des modèles**<br>
✅ **Sélection du meilleur modèle - Mistral 7B Fine-Tuned**<br>


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

