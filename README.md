# 🥦 NutriPredict AI

> Análise preditiva de planos dietéticos com Machine Learning  
> Trabalho Prático — Elementos de Inteligência Artificial e Ciência de Dados 2025/26

---

## 📋 Descrição

O **NutriPredict AI** aplica técnicas de Ciência de Dados e Inteligência Artificial a um conjunto de dados com registos de pacientes, planos dietéticos, nutricionistas e resultados clínicos. O objetivo é identificar padrões associados ao sucesso de dietas e construir modelos preditivos capazes de estimar a probabilidade de sucesso e a perda de peso esperada.

**Resultados obtidos:**
- 🏆 Melhor classificador: **Gradient Boosting** — F1 = 0,872 no teste
- 🏆 Melhor regressor: **Gradient Boosting** — R² = 0,849 no teste
- 📊 Dataset final: **1.902 registos** × **30 features** (após limpeza)

---

## 🗂️ Estrutura do Projeto

```
nutriDataLab/
│
├── data/
│   ├── raw/                    # Dados originais (não modificados)
│   │   ├── patients.csv        # 1.000 pacientes × 11 colunas
│   │   ├── diets.csv           # 10 planos dietéticos × 9 colunas
│   │   ├── nutritionists.csv   # 20 nutricionistas × 5 colunas
│   │   └── outcomes.csv        # 2.523 resultados × 9 colunas
│   └── processed/
│       └── dataset_processed.csv
│
├── models/                     # Modelos treinados (.pkl)
│   ├── classifier_*.pkl
│   └── regressor_*.pkl
│
├── reports/
│   └── figures/                # Gráficos gerados automaticamente
│
├── src/
│   ├── data/
│   │   ├── load_data.py        # Carregamento dos CSVs
│   │   ├── merge_data.py       # Integração das 4 fontes
│   │   └── preprocess.py       # Limpeza, encoding, normalização
│   │
│   ├── features/
│   │   ├── feature_engineering.py  # Criação de variáveis derivadas
│   │   └── feature_selection.py    # Seleção por importância (RF)
│   │
│   ├── clustering/
│   │   └── clustering.py       # K-Means + Hierárquico + PCA
│   │
│   ├── models/
│   │   ├── train.py            # Treino + RandomizedSearchCV
│   │   ├── evaluate.py         # Métricas e ranking de modelos
│   │   └── predict.py          # Inferência para novos pacientes
│   │
│   ├── visualization/
│   │   └── plots.py            # Todos os gráficos da EDA
│   │
│   └── utils/
│       └── helpers.py          # Timer, describe_dataset, save_report
│
└── main.py                     # Pipeline completo (ponto de entrada)
```

---

## ⚙️ Instalação

**Pré-requisitos:** Python 3.10+

```bash
# 1. Clonar o repositório
git clone https://github.com/<grupo>/nutriDataLab.git
cd nutriDataLab

# 2. Criar e ativar ambiente virtual
python -m venv meuambiente
# Windows:
meuambiente\Scripts\activate
# Linux/macOS:
source meuambiente/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt
```

### `requirements.txt`
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 🚀 Execução

```bash
# Executar o pipeline completo
python main.py
```

O pipeline executa automaticamente todas as etapas em sequência:

| Passo | Descrição | Saída |
|-------|-----------|-------|
| 1 | Carregamento dos dados | — |
| 2 | Integração dos datasets | dataset com 2.523 × 29 colunas |
| 3 | Feature engineering | +6 variáveis derivadas → 35 colunas |
| 4 | Análise exploratória | 13 gráficos em `reports/figures/` |
| 5 | Pré-processamento | `data/processed/dataset_processed.csv` |
| 6 | Seleção de features | ranking de importância + comparação |
| 6 | Clustering K-Means | gráficos PCA + perfil dos clusters |
| 8 | Treino de classificadores | 4 modelos + tuning → `models/` |
| 9 | Treino de regressores | 4 modelos + tuning → `models/` |
| 10 | Avaliação e comparação | tabelas de métricas + gráficos |

---

## 📊 Resultados

### Classificação — Sucesso da Dieta (`diet_success`)

| Modelo | CV F1 | Val F1 | **Teste F1** | Accuracy |
|--------|-------|--------|-------------|----------|
| **Gradient Boosting** | 0,8593 | 0,9719 | **0,8720** | 0,888 |
| Random Forest | 0,8461 | 1,0000 | 0,8502 | 0,871 |
| Decision Tree | 0,7985 | 0,8537 | 0,8211 | 0,846 |
| Regressão Logística | 0,8059 | 0,8417 | 0,7712 | 0,811 |

### Regressão — Perda de Peso (`weight_loss_kg`)

| Modelo | CV RMSE | Val R² | **Teste R²** | RMSE (kg) |
|--------|---------|--------|-------------|-----------|
| **Gradient Boosting** | 2,671 | 0,9405 | **0,8492** | 2,706 |
| Random Forest | 3,067 | 0,9724 | 0,8171 | 2,981 |
| Ridge | 3,601 | 0,7440 | 0,7471 | 3,504 |
| Decision Tree | 3,896 | 0,8615 | 0,7114 | 3,744 |

---

## 🔮 Inferência para Novos Pacientes

```python
from src.models.predict import predict_diet_success, predict_weight_loss
import pandas as pd

# Carregar dados processados e amostrar 5 pacientes
df = pd.read_csv("data/processed/dataset_processed.csv")
X = df.drop(columns=["diet_success", "weight_loss_kg"])
sample = X.sample(5, random_state=0)

# Classificação
preds, probas = predict_diet_success(sample, model_name="GradientBoosting")

# Regressão
peso_previsto = predict_weight_loss(sample, model_name="GradientBoosting")
```

Também pode executar diretamente:

```bash
python src/models/predict.py
```

---

## 🧠 Metodologia

### Variável-alvo
`diet_success` é derivada da mediana de perda de peso (−21,60 kg em 6 meses): pacientes com perda ≥ mediana são classificados como **sucesso (1)**, os restantes como **insucesso (0)**.

### Pipeline de dados
1. **Integração:** INNER JOIN (pacientes + resultados) + LEFT JOIN (dietas + nutricionistas)
2. **Limpeza:** imputação por mediana/moda → remoção de outliers por 3×IQR (621 linhas removidas) → remoção de data leakage (`final_bmi`, `satisfaction_score`)
3. **Feature Engineering:** `lifestyle_score`, `bmi_category`, `age_group`, `adherence_x_motivation`, `macro_balance_score`, `kg_per_week_expected`
4. **Encoding:** LabelEncoder para categóricas
5. **Normalização:** StandardScaler nas variáveis contínuas

### Divisão dos dados
```
Treino: 70% (1.331) | Validação: 15% (285) | Teste: 15% (286)
```
Otimização de hiperparâmetros via **RandomizedSearchCV** com 5-fold CV no treino; modelo final retreinado em treino + validação.

---

## 📈 Gráficos Gerados

Todos os gráficos são salvos em `reports/figures/`:

| Ficheiro | Conteúdo |
|----------|----------|
| `target_distribution.png` | Distribuição de `diet_success` e `weight_loss_kg` |
| `correlation_heatmap.png` | Mapa de correlação de todas as variáveis numéricas |
| `correlation_with_targets.png` | Correlação de cada feature com os alvos |
| `gender_vs_success.png` | Taxa de sucesso e perda média por sexo |
| `diet_vs_success.png` | Taxa de sucesso por tipo de dieta |
| `age_vs_weight_loss.png` | Dispersão idade × perda de peso |
| `age_group_vs_success.png` | Sucesso e perda média por faixa etária |
| `adherence_vs_success.png` | Distribuição de aderência por resultado |
| `motivation_vs_success.png` | Boxplot de motivação inicial por resultado |
| `bmi_vs_weight_loss.png` | IMC inicial × perda de peso (por sucesso) |
| `outliers_boxplot.png` | Boxplots das variáveis numéricas (antes da limpeza) |
| `nutritionist_vs_outcome.png` | Experiência do nutricionista × resultados |
| `specialty_vs_success.png` | Taxa de sucesso por especialidade |
| `clustering_elbow_silhouette.png` | Elbow Method + Silhouette Score |
| `clusters_pca.png` | Clusters K-Means projetados em 2D (PCA) |
| `feature_importance_full_*.png` | Importância de features (classificação e regressão) |
| `feature_selection_comparison.png` | Comparação com/sem seleção de features |
| `model_comparison_*.png` | Comparação visual dos modelos |
| `feature_importance_random_forest.png` | Top-10 features do Random Forest |

---

## 👥 Grupo

| Nome | Nº do aluno |
|------|-------------|
| Lara Fernandes Gabry | 55449 |


---

## 📄 Licença

Projeto académico — Universidade da Beira Interior. Todos os direitos reservados © 2026.