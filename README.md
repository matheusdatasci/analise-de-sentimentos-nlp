
#  pinheiro-roberta-goemotions 
#### 📊 Estatísticas: +40 downloads no [Hugging Face](https://huggingface.co/teu-usuario/teu-modelo) 😎😎😎

<!-- Resumo rápido do modelo -->

Este é um modelo baseado em Roberta, fine-tuned para **classificação multi-label de emoções** em textos usando o dataset **GoEmotions**. Ele classifica 28 categorias de emoção simultaneamente.

### Descrição do Modelo

Modelo **RobertaForSequenceClassification** fine-tuned para classificação multi-label de 28 emoções:

`['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']`

- **Desenvolvido por:** Matheus Pinheiro
- **Tipo de modelo:** RobertaForSequenceClassification (multi-label)  
- **Idioma(s):** Inglês
- **Licença:** MIT
- **Base do modelo:** `roberta-base`

---

### 🛠️ Tecnologias Utilizadas

Este projeto foi desenvolvido utilizando o seguinte stack de tecnologias de *Deep Learning*:

- **Framework de Deep Learning:** [PyTorch](https://pytorch.org/)  — usado para o fine-tuning do modelo pré-treinado
- **Biblioteca de Modelagem:** [Hugging Face Transformers](https://huggingface.co/docs/transformers) — usada para carregar o modelo base, o tokenizador e a *pipeline* de inferência  
- **Modelo Base:** [`roberta-base`](https://huggingface.co/roberta-base)  
- **Linguagem:** Python  
- **Ecossistema:** [Hugging Face Hub](https://huggingface.co/) — utilizado para hospedagem do modelo e do tokenizador  

### Uso Direto

Classificação multi-label de emoções em textos em inglês. Retorna a probabilidade de cada emoção para o texto dado.

## Viés, Riscos e Limitações

- Treinado no dataset GoEmotions (inglês) e pode não generalizar para outros idiomas ou contextos culturais.  
- Algumas emoções podem ter probabilidades baixas mesmo quando presentes.  
- As probabilidades multi-label são relativas, não valores absolutos de emoção.  

### Recomendações

- Interpretar as probabilidades como **scores relativos**, não como rótulos absolutos.

## Como Usar o Modelo

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Carregar tokenizer
tokenizer =  AutoTokenizer.from_pretrained("roberta-base")

# Map de id2label
id2label = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval',
    5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment',
    10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement', 14: 'fear',
    15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness', 20: 'optimism',
    21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
}

# Carregar modelo
modelo = AutoModelForSequenceClassification.from_pretrained("pinheiroxs/pinheiro-roberta-goemotions",id2label=id2label)

# Criar pipeline
classifier = pipeline("text-classification", model=modelo, tokenizer=tokenizer, return_all_scores=True)
```
```python
# Exemplo de uso
texto = "I am very happy today!"
resultado = classifier(texto)[0]

# Formatar resultados
resultado_ordenado = sorted(resultado, key=lambda x: x['score'], reverse=True)
for r in resultado_ordenado:
    print(f"{r['label']:15} : {r['score']*100:.2f}%")
```

### Dados de Treinamento

- **Dataset:** GoEmotions
- **Número de rótulos:** 28 emoções em textos em inglês

#### Pré-processamento

- Tokenização padrão usando `AutoTokenizer`
- Binarização multi-label dos rótulos de emoção

#### Hiperparâmetros de Treinamento

- **Regime de treino:** 4 epochs
- **Batch size:** 16
- **Otimizador:** AdamW
- **Learning rate:** 2e-5
- **Hidden size:** 768
- **Intermediate size:** 3072
- **Número de attention heads:** 12
- **Número de camadas ocultas:** 12
- **Dropout:** 0.1
