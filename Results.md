# Comparaison des Modèles Keras

Ce document présente une comparaison et une évaluation de différents modèles Keras utilisés dans le projet.

## Données

- 3 niveaux de bruit : 0, 0.02, 0.1
- 4 nombres de points : 100, 200, 300, 500
- 3 classes : "M", "Spike" et "Parabole"
- 33 courbes par classe

**3*4*3*33 = 1188 courbes**


## Modèles Évalués

### 1. Modèle 1 : Masquage -10 + LSTM 32
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Masking_Layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Masking</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ LSTM_Layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">4,352</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_Layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">1,056</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">99</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
<span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">5,507</span> (21.51 KB)
</pre>

## 2. Modèle 2 : Masquage -10 + LSTM 64
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Masking_Layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Masking</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ LSTM_Layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">16,896</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_Layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">4,160</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_Layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">99</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
<span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,235</span> (90.76 KB)
</pre>

## 3. Modèle 3 : Fill 0 + Conv1D + MaxPooling1D + Bidirectional LSTM 64
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Conv1D_Layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │           <span style="color: #00af00; text-decoration-color: #00af00">128</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ MaxPooling_Layer (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling1D</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">250</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ bidirectional_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Bidirectional</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │        <span style="color: #00af00; text-decoration-color: #00af00">49,664</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_Layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">8,256</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_Layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">99</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
<span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">60,227</span> (235.26 KB)
</pre>

## 4. Modèle 4 : Masquage -10 + Bidirectional LSTM 64
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Masking_Layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Masking</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Bidirectional_LSTM              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │        <span style="color: #00af00; text-decoration-color: #00af00">33,792</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Bidirectional</span>)                 │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_Layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">8,256</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_Layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">99</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
<span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">44,227</span> (172.76 KB)
</pre>

## 5. Modèle 5 : Fill 0 + Masquage 0 + Bidirectional LSTM 64

## 6. Modèle 6 : Fill 0 + Masquage 0 + LSTM 64
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Masking_Layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Masking</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ LSTM (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">16,896</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_Layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">4,160</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_Layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_Layer_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">99</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
<span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,235</span> (90.76 KB)
</pre>


## Entraînement des modèles


| Modèle   | Accuracy Train | Accuracy Test | Temps d'entraînement | Nombre de paramètres |
| -------- | :------------: | :-----------: | :------------------: | :------------------: |
| Modèle 1 |     66.46%     |    60.83%     |         1m47         |        5,507         |
| Modèle 2 |     67.40%     |    64.17%     |         2m49         |        23,235        |
| Modèle 3 |     66.98%     |    67.92%     |         2m06         |        60,227        |
| Modèle 4 |     70.94%     |    65.83%     |         2m26         |        44,227        |
| Modèle 5 |     82.50%     |    84.17%     |         2m50         |        44,227        |
| Modèle 6 |     92.81%     |    95.83%     |         3m30         |        23,235        |


## Entrainement et test sur des données bruitées

| Modèle   | Accuracy Train | Accuracy Test | Temps d'entraînement | Nombre de paramètres |
| -------- | :------------: | :-----------: | :------------------: | :------------------: |
| Modèle 6 |     57.37%     |    51.68%     |         3m26         |        23,235        |
| Modèle 5 |     72.42%     |    70.59%     |         5m13         |        44,227        |


## Remarques

_Peut-être que le Bidirectional LSTM sera plus adapté avec les CRDL et CLDR qui ne sont pas symétriques._