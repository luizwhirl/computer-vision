# Visão Computacional - Atividades e Projetos

Este repositório contém as resoluções práticas e teóricas de listas de exercícios voltadas para a disciplina de Visão Computacional. O projeto abrange uma jornada completa pela área: começando pelos fundamentos de processamento de imagens e filtros espaciais manuais, avançando pela extração de características e geometria epipolar, até chegar em aplicações modernas de Machine Learning e Deep Learning (Redes Neurais Convolucionais e YOLO).

> [!WARNING]
> **AVISO IMPORTANTE SOBRE A EXECUÇÃO DOS CÓDIGOS** > A resolução e estruturação lógica destas listas foi majoritariamente desenvolvida, montada e validada no **Google Colab**. Embora os códigos tenham sido adaptados e refatorados em scripts `.py` independentes para execução local (via VS Code ou outras IDEs), é **altamente recomendado consultar os Jupyter Notebooks (`.ipynb`) originais**.
> 
> Os Notebooks no Colab contêm as visualizações nativas, plotagens gráficas e o passo a passo com o qual as imagens e *datasets* foram processados, gerando o **resultado mais autêntico e visualmente correto**. Todos os Notebooks foram automatizados (com downloads diretos via `gdown` e `kagglehub`) para facilitar a execução com apenas um clique.

---

## Estrutura do Repositório

O repositório está dividido em quatro listas principais, cada uma focada em conceitos cruciais da Visão Computacional. As pastas contêm o Notebook base (`.ipynb`), os guiões refatorados em Python (`/src/*.py`) e os enunciados/relatórios em `.pdf`.

### Lista 1: Fundamentos de Processamento de Imagens e Filtros
Esta lista foca na base de processamento de imagens, exigindo a compreensão matemática por trás dos algoritmos, frequentemente com a restrição de **não utilizar funções prontas do OpenCV para operações vitais**, implementando a lógica "do zero" (from scratch).
* **Filtros Espaciais Lineares:** Geração e aplicação de filtros de Média e Gaussiano.
* **Filtros Passa-Alta e Deteção de Bordas:** Extração de altas frequências, uso dos operadores de Sobel (derivação de imagens) e detector de Canny.
* **Filtros Não Lineares:** Implementação própria do Filtro Bilateral (sem OpenCV), mantendo a preservação de bordas, em comparação com os filtros Mediana e Non-Local Means.
* **Pirâmides e Redimensionamento:** Criação de Pirâmides Gaussianas (evitando *aliasing*) e comparação de interpolações (Bilinear vs Bicúbica).

### Lista 2: Detectores, Descritores e Identificação de Padrões
O foco desta lista é a extração de pontos-chave (*keypoints*) e características essenciais de imagens (independentes de rotação e escala), essenciais para reconhecimento de objetos.
* **Detecção de Cantos:** Comparação prática entre os métodos de Harris, Shi-Tomasi e FAST.
* **Feature Matching:** Extração e correspondência utilizando algoritmos robustos: SIFT, ORB e BRISK/AKAZE (valiando a distância de Hamming vs L2).
* **Sistema de Identificação de Produtos:** Um classificador primitivo construído utilizando extração SIFT, KNN (*K-Nearest Neighbors*) e o Teste de Razão de Lowe, culminando na avaliação através de uma Matriz de Confusão.
* **Detecção de Blobs:** Algoritmo implementado com múltiplas escalas utilizando a aproximação pelo Laplaciano do Gaussiano (LoG).

### Lista 3: Homografia, Panoramas e Visão Estéreo
Esta lista lida diretamente com transformações geométricas complexas, matrizes e geometria epipolar, unindo múltiplas visões num único plano.
* **Geração de Panoramas (Image Stitching):** O objetivo rigoroso era alinhar imagens consecutivas calculando a Matriz de Homografia utilizando SIFT e RANSAC, fundindo as imagens num *canvas* unificado **sem utilizar a API de `cv2.Stitcher`**.
* **Mapeamento de Perspectiva (Top-Down View):** Conversão da perspetiva da câmara de um jogo de futebol para uma visão 2D superior exata (baseada em marcações de campo em metros) para rastreamento topográfico de jogadores.
* **Mapas de Disparidade e Modo Retrato:** Retificação não-calibrada de câmaras utilizando a Matriz Fundamental. Obtenção do mapa de profundidade (StereoSGBM) e segmentação via algoritmos morfológicos e *GrabCut* para gerar artificialmente um "Modo Retrato" (fundo borrado) fidedigno.

### Lista 4: Reconhecimento Clássico vs Deep Learning
O último segmento aborda o estado da arte na classificação e detecção de objetos, comparando métodos tradicionais com arquiteturas profundas (*Deep Learning*).
* **Machine Learning Clássico:** Classificação de Cães e Gatos utilizando extração de descritores **HOG** (*Histogram of Oriented Gradients*) alinhados a um classificador **SVM** de margem linear.
* **Transfer Learning (CNNs):** O mesmo problema foi abordado usando o Keras/TensorFlow para adaptar redes neurais convolucionais massivas pré-treinadas (`VGG16`, `ResNet50` e `MobileNetV2`), comparando as taxas de acurácia em *holdout*.
* **YOLO e Rastreamento em Vídeo:** Uso do modelo pré-treinado **YOLOv8** para a detecção, rastreamento (tracking) persistente (associando IDs únicos) e contagem veicular seletiva num vídeo real de uma ponte.

---

## Como Executar os Scripts Localmente

Para rodar os arquivos Python refatorados localmente, garanta que possui as bibliotecas necessárias instaladas.

1. Ative o seu ambiente virtual (Python 3.10 ou 3.11 recomendado devido ao TensorFlow e NumPy 1.x):
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

2. Instale as dependências contidas no arquivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

3. Navegue até a pasta da lista e execute os scripts. (Dica: grande parte dos códigos possui inteligência para baixar os arquivos dinamicamente, evite baixar dados volumosos à mão).
```bash
python lista1/src/1.py
```
_Desenvolvido como projeto acadêmico de aprendizado e aplicação técnica em Visão Computacional._