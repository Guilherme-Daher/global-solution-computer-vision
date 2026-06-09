# 🛰️ SkyGuard — Classificador de Imagens Satelitais

<div align="center">

![SkyGuard](https://img.shields.io/badge/SkyGuard-Monitoramento%20Orbital-00e676?style=for-the-badge&logo=satellite&logoColor=white)
![FIAP](https://img.shields.io/badge/FIAP-Engenharia%20de%20Software-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=for-the-badge&logo=pytorch)

**Global Solution 2026 · 1º Semestre · FIAP Engenharia de Software · 4º Ano**

**Disciplina:** Applied Computer Vision (ACV)

**Período:** 25/05/2026 – 09/06/2026

**Tema:** Indústria Espacial — O Código que Move o Universo

</div>

---

## 👥 Integrantes

| Nome | RM |
|------|----|
| Guilherme Daher | 98611 |
| Gabriel Freitas | 550187 |
| Heitor Nobre | 551539 |
| Lucca Alexandre | 99700 |
| Vinicius Yamashita | 550908 |

---

## 🌍 Sobre o Projeto

O **SkyGuard** é uma plataforma integrada de monitoramento ambiental que utiliza dados orbitais e inteligência artificial para detectar desastres naturais antes que se tornem catástrofes.

Este repositório contém o módulo de **Visão Computacional (ACV)** do projeto — uma CNN treinada do zero para classificar imagens satelitais em 4 categorias ambientais críticas:

| Classe | Descrição | Relevância Operacional |
|--------|-----------|----------------------|
| 🌊 `enchente` | Superfície de água anormal, solo saturado | Evacuação, roteamento de emergência |
| 🔥 `queimada` | Solo escurecido, vegetação queimada, fumaça | Alertas de incêndio, mapeamento de dano |
| 🏙️ `urbano` | Áreas construídas, telhados, asfalto | Análise de vulnerabilidade de cidades |
| 🌿 `vegetacao` | Cobertura vegetal saudável | Baseline de monitoramento ambiental |

### 🔗 Conexão com a Indústria Espacial

```
Satélite (NASA/ESA/INPE)
       ↓ imagem bruta
  [Módulo ACV — CNN Avançada]
       ↓ classificação automática
  enchente | queimada | urbano | vegetacao
       ↓
  Plataforma SkyGuard → Alertas → Autoridades
```

Este módulo é a **camada de percepção** do sistema integrado. Imagens dos satélites **Sentinel-2 (ESA)**, **Landsat (NASA)** e sistemas do **INPE** são processadas pela CNN para classificação automática, alimentando uma plataforma de alertas em tempo quasi-real.

**ODS Relacionados:** ODS 13 (Ação Climática) · ODS 11 (Cidades Sustentáveis) · ODS 15 (Vida Terrestre)

---

## 🔗 Links Importantes

| Recurso | Link |
|---------|------|
| 🌐 **Site SkyGuard** | [skyguard.azurewebsites.net](https://gs-1sem-dqfxbqehdkbgf9ed.brazilsouth-01.azurewebsites.net) |
| 🤖 **Demo Gradio (Hugging Face)** | [daherzinho/skyguard-satellite-classifier](https://huggingface.co/spaces/daherzinho/skyguard-satellite-classifier) |
| 📓 **Notebook no Colab** | [Abrir no Colab](https://colab.research.google.com/drive/1TLQqS6jcnlhnqnDScpX2ubWenVAQb8Tg) |
| 📓 **Video Youtube** | [Youtube](https://youtu.be/6Wt4rDOFkyU) |

---

## 📦 Dataset

- **Fonte:** [Disaster Classification Dataset — Kaggle (sayankr007)](https://www.kaggle.com/datasets/sayankr007/disaster-classification-dataset)
- **Licença:** MIT
- **Total de imagens:** ~2.000 imagens satelitais reais de desastres

**Mapeamento de classes:**

| Pasta Original | Classe no Projeto |
|---------------|------------------|
| `wild_fire` | queimada |
| `water_disaster` | enchente |
| `infrastructure` | urbano |
| `non_damage_wildlife_forest` | vegetacao |

**Divisão dos dados:**

| Split | Proporção | Uso |
|-------|-----------|-----|
| Treino | 64% | Aprendizado do modelo |
| Validação | 16% | Ajuste de hiperparâmetros |
| Teste | 20% | Avaliação final imparcial |

**Pré-processamento:**
- Redimensionamento para 64×64 px
- Normalização com média `[0.485, 0.456, 0.406]` e desvio `[0.229, 0.224, 0.225]`
- Data Augmentation no treino: flip horizontal/vertical, rotação ±15°, color jitter

---

## 🏗️ Arquiteturas CNN

### Modelo 1 — CNN Baseline

Arquitetura simples com 3 blocos convolucionais, sem regularização. Serve como piso de comparação.

```
Input (3, 64, 64)
    ↓ Conv2d(3→16) + ReLU + MaxPool → (16, 32, 32)
    ↓ Conv2d(16→32) + ReLU + MaxPool → (32, 16, 16)
    ↓ Conv2d(32→64) + ReLU + MaxPool → (64, 8, 8)
    ↓ Flatten → (4096)
    ↓ Linear(4096→256) + ReLU
    ↓ Linear(256→4)
Output: logits para 4 classes
```

### Modelo 2 — CNN Avançada ⭐ (melhor modelo)

Arquitetura profunda com 5 blocos convolucionais e técnicas modernas de regularização.

```
Input (3, 64, 64)
    ↓ [Conv2d + BatchNorm + ReLU + MaxPool] × 4 → (256, 4, 4)
    ↓ [Conv2d + BatchNorm + ReLU] × 1 → (256, 4, 4)
    ↓ Global Average Pooling → (256)
    ↓ Linear(256→512) + ReLU + Dropout(0.4)
    ↓ Linear(512→128) + ReLU + Dropout(0.3)
    ↓ Linear(128→4)
Output: logits para 4 classes
```

**Comparação:**

| | CNN Baseline | CNN Avançada |
|---|---|---|
| Blocos Conv | 3 | 5 |
| Filtros | 16→32→64 | 32→64→128→256→256 |
| BatchNorm | ❌ | ✅ |
| Dropout | ❌ | ✅ (0.4 + 0.3) |
| Global Avg Pool | ❌ | ✅ |
| Parâmetros | ~1.07M | ~1.18M |

---

## 📊 Resultados

### Estratégia de Treinamento

| Componente | Configuração |
|-----------|-------------|
| Otimizador | Adam (lr=1e-3, weight_decay=1e-4) |
| Loss | CrossEntropyLoss |
| LR Scheduler | CosineAnnealingLR |
| Early Stopping | patience=5 épocas |
| Gradient Clipping | max_norm=1.0 |
| Épocas máximas | 30 |

### Métricas no Conjunto de Teste

| Modelo | Acurácia | F1-Score |
|--------|----------|----------|
| CNN Baseline | ~78% | ~78% |
| **CNN Avançada** | **~85%** | **~85%** |
| Meta exigida | 88% | — |

### Por que a CNN Avançada é melhor?

1. **BatchNorm** — estabiliza o treino e acelera a convergência
2. **Dropout duplo** — previne overfitting ao forçar representações redundantes
3. **Global Average Pooling** — reduz parâmetros e melhora generalização
4. **5 blocos vs 3** — captura padrões de maior escala nas imagens satelitais

---

## 🚀 Como Executar

### Opção 1 — Google Colab (recomendado)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TLQqS6jcnlhnqnDScpX2ubWenVAQb8Tg)

1. Clique no badge acima para abrir o notebook no Colab
2. Vá em **Runtime → Change runtime type → T4 GPU**
3. Gere suas credenciais do Kaggle:
   - Acesse [kaggle.com/settings/api](https://www.kaggle.com/settings/api)
   - Clique em **API Tokens → Create Legacy API Key**
   - Um arquivo `kaggle.json` será baixado no seu computador
4. Faça upload do `kaggle.json` no Colab:
   - Clique no ícone de 📁 pasta na barra lateral esquerda
   - Clique no ícone de upload (seta pra cima)
   - Selecione o arquivo `kaggle.json` baixado   
   
6. Faça o upload da pasta `minhas_fotos.zip` dentro do ambiente do Colab
7. Execute **Runtime → Run all**
8. Aguarde o treino (~20 minutos com GPU T4)

### Opção 2 — Demo Online (sem instalação)

Acesse diretamente sem precisar instalar nada:

🔗 [huggingface.co/spaces/daherzinho/skyguard-satellite-classifier](https://huggingface.co/spaces/daherzinho/skyguard-satellite-classifier)

Faça upload de qualquer imagem satelital e veja a classificação em tempo real.

### Opção 3 — Execução Local

```bash
# 1. Clone o repositório
git clone https://github.com/Guilherme-Daher/global-solution-computer-vision.git
cd global-solution-computer-vision

# 2. Crie um ambiente virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o notebook
jupyter notebook GS_CodeFinal.ipynb
```

---

## 📁 Estrutura do Repositório

```
global-solution-computer-vision/
│
├── 📓 GS_CodeFinal.ipynb           # Notebook principal com todo o desenvolvimento
├── 📋 requirements.txt              # Dependências do projeto
├── 📖 README.md                     # Este arquivo
│
├── checkpoints/
│   └── cnn_avancada_best.pt        # Pesos do melhor modelo treinado
│
└── assets/                         # Gráficos e visualizações gerados
    ├── curvas_aprendizado.png      # Evolução de loss e acurácia por época
    ├── matrizes_confusao.png       # Matrizes de confusão dos dois modelos
    ├── acertos_erros.png           # Exemplos visuais de acertos e erros
    └── comparacao_modelos.png      # Comparação final entre arquiteturas
```

---

## 📚 Dependências

```
torch
torchvision
gradio
numpy
Pillow
scikit-learn
matplotlib
seaborn
tqdm
```

Instale com: `pip install -r requirements.txt`

---

## 📚 Referências

- Dataset: [Disaster Classification Dataset — Kaggle](https://www.kaggle.com/datasets/sayankr007/disaster-classification-dataset)
- Ioffe & Szegedy (2015) — Batch Normalization
- Lin et al. (2013) — Network in Network (Global Average Pooling)
- PyTorch Documentation — [pytorch.org](https://pytorch.org)
- ESA Sentinel-2 — [sentinel.esa.int](https://sentinel.esa.int)
- NASA Landsat — [landsat.gsfc.nasa.gov](https://landsat.gsfc.nasa.gov)

---

<div align="center">
<strong>🛰️ SkyGuard · Global Solution 2026 · FIAP Engenharia de Software</strong><br>
<em>Monitoramento Orbital em Tempo Real</em>
</div>
