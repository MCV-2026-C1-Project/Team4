# C1 – Week 2: explicación del trabajo

Este documento explica **qué se pide** en la week 2, **qué he hecho y por qué**, **qué hay en el notebook** `c1-week2.ipynb` y **cómo se entrega**. Todos los números salen de ejecutar el notebook.

---

## 1. Qué se pide

Seguimos con el mismo buscador de cuadros de la week 1. Hay dos novedades: hay que añadir **información espacial** a los descriptores, y hay consultas en las que el cuadro está **colgado en la pared**, así que primero hay que quitar el fondo.

| Task | Qué es | Dónde está en el notebook |
|---|---|---|
| 1 | Implementar histogramas 3D / 2D, por bloques y jerárquicos (pirámide espacial) | §2 (código y visualizaciones) y §3 (experimentos) |
| 2 | Evaluar el mejor descriptor en QSD1-W2 y compararlo con el mejor de la W1 (el dev set es el mismo) | §6.3 |
| 3 | Quitar el fondo de QSD2 **solo con color** y generar una máscara binaria | §4 |
| 4 | Evaluar las máscaras con precision, recall y F1 | §5 |
| 5 | QSD2: quitar el fondo, buscar en la BBDD y evaluar el retrieval | §6 |
| 6 | Entrega a ciegas en QST1 y QST2: top-10 por consulta y, en QST2, también las máscaras | §7 y `run_w2.py` |

**Datos**
- `BBDD/` (en `week1/`): los mismos 287 cuadros.
- `qsd1_w2/qsd1_w1/`: 30 consultas recortadas. **Es exactamente QSD1-W1.**
- `qsd2_w2/qsd2_w1/`: 30 fotos del cuadro en la pared, sus máscaras GT (`.png`) y `gt_corresps.pkl`. Hay **un solo cuadro por imagen** (el GT es `[[100], [219], ...]`), así que el formato de salida sigue siendo una lista de listas.
- QST1 / QST2: se publican el jueves 15 de octubre a las 14:00, sin solución.

**Entrega** (domingo 18 de octubre, 19:00): slides en la plantilla común, código en GitHub y en el Drive:
```
TeamX/week2/QST1/method1/result.pkl
TeamX/week2/QST2/method1/result.pkl
TeamX/week2/QST2/method1/*.png             (las máscaras, mismo nombre que la consulta)
```
Esta semana el enunciado solo pide **un método** (`method1`), a diferencia de la W1. Entregamos solo el method1. El method2 se queda en el notebook como comparación (`python run_w2.py --all-methods` también lo genera si alguna vez hiciera falta).

---

## 2. Cómo he validado (y por qué es distinto de la W1)

En la W1 sacamos 0.90 de mAP@1 en QSD1 pero **0.77 en el test**. Con solo 30 consultas, quedarse con la casilla de mejor número lleva a sobreajustar. Esta semana tenemos 30 consultas anotadas más, así que valido siempre en **tres conjuntos**:

| Conjunto | Qué es | Qué mide |
|---|---|---|
| **QSD1** | las 30 consultas recortadas | lo mismo que la W1 |
| **QSD2-GT** | los cuadros de QSD2 recortados con el bounding box de la **máscara GT** | el descriptor solo, sin errores de segmentación, en fotos tomadas en otras condiciones |
| **QSD2-pred** | los mismos cuadros recortados con **nuestras** máscaras | el pipeline real de la Task 5 |

**Primer hallazgo:** los métodos de la W1 dan 0.90 en QSD1, pero solo **0.57 y 0.70 en QSD2-GT**, aunque ahí los recortes son perfectos. Estaban ajustados a QSD1, y eso explica el 0.77 del test.

Además del mAP@1 y el mAP@5 uso el **MRR** (media de 1/posición del cuadro correcto en el ranking completo). Muchas configuraciones empatan en mAP@1 porque fallan la misma consulta, y el MRR sigue distinguiéndolas. Igual que en la W1, una diferencia de una consulta (0.033) es ruido. Solo me fío de lo que mejora de forma consistente en varios conjuntos, espacios y bins.

---

## 3. Task 1: los descriptores

### 3.1 Implementación
- **Preprocesado:** la normalización de la iluminación de la W1 (cada canal RGB escalado a media 128). Sigue siendo imprescindible: sin ella, el mejor descriptor de bloques baja de 0.96 a 0.66 de mAP@1.
- **Histograma de una región:** primero se calcula, una sola vez por imagen, el bin de cada píxel (con una tabla de 256 valores por canal). Después el histograma de cualquier bloque es un `np.bincount`. Así los bloques salen baratos. He comprobado que da exactamente lo mismo que `cv2.calcHist`.
- **Tipos:** 1D (W1), 2D (los dos canales de color: H–S, a–b, Cr–Cb), 2D + 1D de luminancia, y 3D (los tres canales a la vez).
- **Bloques:** rejilla n×n sin solapamiento, un histograma por bloque, todos concatenados.
- **Pirámide espacial:** varios niveles de bloques (por ejemplo 1×1 + 2×2 + 4×4 + 8×8) concatenados, cada uno multiplicado por un peso.
- **Visualizaciones** (§2.5–2.6), como en las slides del enunciado:
  - la consulta, su cuadro correcto y otro cuadro, descritos con 1D, 2D y 3D. En el 3D cada barra está pintada con el color del centro de su bin, así que se lee como la paleta del cuadro;
  - la imagen con la rejilla y el histograma de cada bloque;
  - los tres niveles de la pirámide.

### 3.2 1D vs 2D vs 3D (§3.2)
**Por qué:** es la primera parte de la Task 1. Probé los 4 tipos en 4 espacios de color con distintos números de bins (116 configuraciones).

| mAP@1 (QSD1 + QSD2-GT) | 1D (W1) | 2D (solo color) | 2D + 1D | **3D** |
|---|---|---|---|---|
| HSV | 0.73 | 0.70 | 0.78 | **0.90** |
| Lab | 0.80 | 0.70 | 0.82 | **0.88** |
| YCrCb | 0.73 | 0.62 | 0.83 | **0.92** |
| RGB | 0.68 | – | – | **0.92** |

- **El 3D es el mejor en todos los espacios.** El 1D pierde la relación entre canales: "muchos píxeles rojos" y "muchos píxeles oscuros" valen lo mismo aunque el rojo sea claro u oscuro.
- **El 2D de solo color es el peor:** una vez normalizada la luz, la luminosidad aporta mucho (lo mismo que vimos en la W1).
- **Maldición de la dimensionalidad:** en 3D, la calidad sube hasta unos 8–16 bins por canal (512–4096 bins) y luego se estanca o baja. Si cada bin recibe pocos píxeles, un cambio pequeño de luz los manda al bin de al lado. Lab es la excepción: sigue mejorando con más bins porque, con bins uniformes, la mayoría quedan vacíos (ver 3.3).

### 3.3 Dónde se ponen los bins (§2.2 y §3.3)
**Por qué:** en la plantilla de slides se ha quedado el feedback del año pasado: *"All teams used a uniform # of bins across channels, even if in some cases the ranges are not the same"*.

En OpenCV (8 bits), los canales de color de **Lab (a, b)** e **YCrCb (Cr, Cb)** solo ocupan más o menos el rango [100, 170] de [0, 255]. Con 8 bins uniformes, casi todos los píxeles caen en 2 o 3 bins. En el ejemplo del notebook, un histograma 3D Lab de 256 bins solo usa **25** con bins uniformes, y **87** si se adaptan al rango real.

Probé tres formas de colocar los bins:
- **uniform:** igual de anchos en [0, 255];
- **range:** igual de anchos en el rango que de verdad usan los datos (percentiles 0.5–99.5 de los píxeles de la BBDD);
- **quantile:** todos los bins reciben el mismo número de píxeles de la BBDD.

| MRR | uniform | **range** | quantile |
|---|---|---|---|
| Lab (8,8,8) | 0.827 | **0.915** | 0.915 |
| Lab (8,16,16) | 0.874 | **0.917** | 0.897 |
| YCrCb (8,16,16) | 0.939 | **0.956** | 0.931 |
| HSV, RGB | sin cambios | | |

**Conclusión:** adaptar los bins al rango mejora mucho Lab y YCrCb, y no cambia nada en HSV y RGB, que ya usan todo el rango. Eso confirma la explicación. Me quedo con `range`, que es más sencillo que `quantile` y da igual o mejor. Además, **poner menos bins para la luminancia que para el color**, por ejemplo (6, 12, 12) u (8, 16, 16), funciona bien. Esto responde directamente al comentario del feedback.

### 3.4 Bloques: cuántos bloques y cuántos bins (§3.4)
**Por qué:** es el núcleo de la Task 1. La dimensión es bloques × bins, y cada bloque tiene menos píxeles que la imagen entera, así que hay que explorar las dos cosas a la vez. Lo muestro en un heatmap.

- **Es la mayor mejora de la semana:** el MRR pasa de 0.84–0.94 (1×1) a 0.97–0.99 (de 4×4 a 8×8), y QSD2-GT llega a 1.00. Dos cuadros con la misma paleta pero distinta composición ya no se confunden.
- Se satura hacia 4×4–6×6 bloques.
- **Bloques y bins se compensan:** HSV 4×4×4 con 5×5 bloques (1600 dimensiones) va igual que 12×8×8 con 4×4 bloques (12 288 dimensiones).

### 3.5 Pirámide espacial (§3.5)
**Por qué:** es la segunda parte de la Task 1. Según la teoría (Lazebnik et al.), los niveles gruesos toleran desalineamientos, los finos discriminan, y conviene dar más peso a los finos.

- Lo que más importa es **el nivel más fino**. Las pirámides sin un nivel fino (1+2, 1+2+4) son peores que 8×8 solo. La pirámide 1+2+4+8 va igual que 8×8, pero no mejor.
- Dar más peso a los niveles finos es igual o mejor en la mayoría de los casos, pero con diferencias pequeñas.
- **¿Por qué no gana aquí?** Las consultas ya están alineadas con la BBDD (recorte contra recorte), así que la tolerancia de los niveles gruesos no hace falta. **En la Task 5, con nuestras máscaras, sí que gana** (ver 6.1).

### 3.6 Distancias (§3.6)
χ², Hellinger y Jensen–Shannon siguen siendo las mejores y casi iguales. La euclídea es mucho peor con histogramas por bloques (MRR 0.72–0.87 frente a 0.97–0.99). Me quedo con **Hellinger**.

---

## 4. Task 3: quitar el fondo solo con color (§4)

**Idea clave:** en todas las fotos el cuadro está en el centro, y **el borde de la imagen es pared** casi siempre. Así que, en cada imagen, aprendo el color de la pared de una franja del borde (4 % del lado) y marco como cuadro todo lo que se aleja de ese color. Cada imagen tiene su propio modelo, así que se adapta a cada pared y a cada luz.

| Método | Modelo del fondo | Decisión |
|---|---|---|
| M1 | color mediano del borde (Lab) | distancia euclídea > umbral fijo T |
| **M2** | **gaussiana: media y covarianza robustas del borde (Lab)** | **distancia de Mahalanobis > umbral de Otsu, automático en cada imagen** |
| M3 | mezcla de 3 gaussianas (GMM) del borde | menos probable que el 99 % de los píxeles del borde |

- **Mahalanobis:** mide la distancia en unidades de lo que varía la pared. Si la pared cambia mucho de brillo (gradientes de luz) pero poco de tono, un cambio de tono cuenta mucho más que uno de brillo.
- **Estimación robusta:** algunos cuadros tocan el borde de la imagen, así que la franja del borde no es 100 % pared. Uso solo el 80 % de píxeles más cercanos al color mediano, que es el color dominante, es decir, la pared.
- **Otsu:** elige el umbral que mejor separa el histograma de distancias en dos clases. No hay ningún umbral fijo que ajustar.

**Postprocesado** (solo morfología sobre la máscara binaria; no hay detectores de contornos):
1. cierre y apertura con un cuadrado pequeño;
2. me quedo con la **componente conexa más grande** (así se eliminan cartelas y objetos de la pared);
3. **relleno de agujeros:** las zonas del cuadro del color de la pared, como un cielo blanco, siguen siendo cuadro;
4. **relleno por filas y columnas:** en cada fila y cada columna relleno entre el primer y el último píxel de cuadro, y me quedo con la intersección. Un cuadro es un rectángulo (algo proyectado), así que esto cierra las muescas de los lados.

En el notebook hay una figura del método paso a paso: franja del borde → mapa de distancias → histograma con el umbral de Otsu → máscara cruda → máscara final → comparación con el GT.

---

## 5. Task 4: evaluación de las máscaras (§5)

Por imagen: precision = TP/(TP+FP), recall = TP/(TP+FN), F1. Doy la **media de las 30 imágenes** y también el F1 mínimo, para ver el peor caso.

| Método | Precision | Recall | **F1** |
|---|---|---|---|
| M1, umbral fijo (T = 15 / 25 / 35) | 0.69–0.84 | 0.92–1.00 | 0.81–0.87 |
| **M2, gaussiana + Mahalanobis + Otsu** | **0.953** | **0.966** | **0.956** |
| M3, GMM | 0.893 | 0.981 | 0.930 |

- **M1:** un umbral fijo no sirve para todas las fotos.
- **M3:** se ajusta demasiado a los píxeles del borde. La pared un poco distinta (sombras, gradiente de luz hacia el centro) queda fuera del modelo y se marca como cuadro. Además es unas 25 veces más lento.

**Ablación de M2:**

| Variante | F1 |
|---|---|
| sin postprocesado | 0.910 |
| + morfología | 0.926 |
| + componente más grande | 0.919 |
| + relleno de agujeros | 0.952 |
| + relleno por filas y columnas (= M2) | **0.956** |
| franja del borde del 2 % / 6 % | 0.950 / 0.957 |
| estimación robusta con el 70 % / 90 % / 100 % | 0.950 / 0.956 / 0.955 |
| + regla de sombras | 0.952 |

- **El paso más útil es rellenar los agujeros.**
- **El método es estable:** con cualquier parámetro razonable da F1 entre 0.95 y 0.957. La estimación robusta casi no cambia la media, pero mejora el peor caso (0.60 → 0.65).
- **Regla de sombras** (un píxel más oscuro que la pared y con su mismo color es sombra, es decir, fondo): quita sombras (la precision sube a 0.959), pero también borra partes de cuadros oscuros de color neutro (el recall baja a 0.954). No compensa, así que **no la uso**.

**Errores que quedan:**
- **Sombras proyectadas** bajo el cuadro.
- **Un cuadro blanco sobre pared blanca** (q20, F1 0.65). Solo con color no se puede separar: es el límite de la regla "just colour".

---

## 6. Task 5: retrieval en QSD2 (§6)

**Pipeline:** máscara M2 → recorte del bounding box de la máscara → descriptor → ranking.

Recorto en vez de usar `calcHist` con máscara porque la rejilla de bloques tiene que quedar alineada con el cuadro, igual que en las imágenes de la BBDD, que también son recortes del cuadro con su marco.

**Problema:** con nuestras máscaras (F1 0.956) el retrieval pierde entre 1 y 5 consultas respecto al recorte GT, según el descriptor. Mirando esos casos, el recorte incluye una franja de pared o sombra arriba o abajo. Esa franja:
1. cambia la **normalización de la iluminación**: en un cuadro oscuro, una franja de pared clara cambia mucho la media;
2. **desplaza la rejilla de bloques.**

### 6.1 Solución: margen interior
Ignoro un margen interior pequeño en **todas** las imágenes, tanto en las consultas como en la BBDD.

| Margen | 0 % | 3 % | **5 %** | 8 % | 12 % |
|---|---|---|---|---|---|
| MRR QSD1 | 0.971 | 0.971 | **0.969** | 0.947 | 0.940 |
| MRR QSD2-GT | 0.991 | 0.997 | **1.000** | 0.988 | 0.975 |
| MRR QSD2-pred | 0.915 | 0.946 | **0.959** | 0.962 | 0.953 |

(media de 6 configuraciones candidatas)

- **El 5 % arregla casi todo** sin empeorar QSD1. Con un 8 % o más empeora, porque el marco forma parte de la firma del cuadro (lo mismo que en la W1).
- **Aquí sí gana la pirámide:** con recortes imperfectos, YCrCb con la pirámide 1+2+4+8 es lo más robusto (mAP@1 de 0.967 en QSD2-pred con un margen del 3–5 %, y 1.000 con un 8 %). Los niveles gruesos toleran el pequeño desplazamiento de la rejilla, tal como dice la teoría.

### 6.2 Métodos finales

**Criterio:**
- arriba en los **tres** conjuntos;
- en una **zona plana** de los experimentos: los vecinos (otros bins, bloques o márgenes) también funcionan, así que no es suerte;
- **espacios de color distintos**, para poder comparar dos descriptores diferentes.

**Solo se entrega el method1**, porque esta semana se pide un único método. El method2 es la alternativa con la que lo comparamos.

| | Descriptor | Dimensión |
|---|---|---|
| **method1** | YCrCb 3D 6×12×12 (bins adaptados al rango), pirámide 1×1 + 2×2 + 4×4 + 8×8 (pesos 1:2:4:8), Hellinger, margen del 5 % | 73 440 |
| method2 (no se entrega) | HSV 3D 8×6×6 (bins uniformes), 5×5 bloques, Hellinger, margen del 5 % | 7 200 |

---

## 7. Resultados finales

| | QSD1 mAP@1 / @5 | QSD2-GT mAP@1 / @5 | QSD2 con nuestras máscaras mAP@1 / @5 |
|---|---|---|---|
| W1 method 1 (HSV 1D) | 0.900 / 0.950 | 0.567 / 0.638 | 0.500 / 0.578 |
| W1 method 2 (Lab 1D) | 0.900 / 0.922 | 0.700 / 0.754 | 0.533 / 0.622 |
| **W2 method1** (YCrCb 3D pirámide) | **0.967 / 0.967** | **1.000 / 1.000** | **0.967 / 0.983** |
| W2 method2 (HSV 3D bloques, no se entrega) | 0.967 / 0.967 | 1.000 / 1.000 | 0.933 / 0.961 |

Máscaras: **P 0.953, R 0.966, F1 0.956**.

**Task 2 (la tabla de la plantilla):** mAP@1 en QSD1: **0.900 (W1) → 0.967 (W2)**.

**Errores que quedan:**
- **QSD1 q26** (los dos métodos): un cuadro casi negro. La normalización multiplica sus canales por un factor enorme, amplifica el ruido y acaba pareciendo un dibujo gris. En una prueba aparte limité esa ganancia, pero el resto de consultas empeoraba mucho, así que mantengo la normalización completa.
- **QSD2 q10** (method1 en posición 2): un plástico negro con reflejos que cambian entre las dos fotos. En la imagen de la BBDD se refleja una persona.
- **QSD2 q14 y q29** (solo method2): pequeños errores del recorte desplazan la rejilla 5×5. La pirámide del method1 los acierta.

---

## 8. Qué hay en el notebook

| Sección | Contenido |
|---|---|
| Intro / 0 | Tareas, datos, cómo ejecutarlo, imports, rutas y estilo de las gráficas (colores de la plantilla de slides) |
| 1 | Carga de datos; QSD2 con el contorno de la máscara GT; los tres conjuntos de validación |
| 2 | Task 1: normalización; espacios de color y colocación de los bins (figura de los rangos); histogramas 1D/2D/3D, bloques y pirámide con sus **visualizaciones**; distancias; métricas |
| 3 | Experimentos: baseline W1, tipos × espacios × bins, colocación de los bins, heatmap bloques × bins, pirámides, distancias. Cada uno con **por qué**, resultado y **conclusión** |
| 4 | Task 3: los 3 modelos de fondo y el postprocesado; figura paso a paso |
| 5 | Task 4: P/R/F1 de los métodos, ablación, resultados por imagen, las 5 peores y las 5 mejores |
| 6 | Task 5: QSD2 con nuestras máscaras, casos perdidos, margen interior, métodos finales con intervalo de confianza, **Task 2** (W1 vs W2) y errores que quedan |
| 7 | Task 6: genera `results/QST1/...` y `results/QST2/...` (pkl + máscaras). Si las carpetas de test no existen, no hace nada |
| 8 | Resumen listo para las slides |

**Cómo usarlo:** el notebook tiene que estar en `Project/week2/`, junto a `qsd1_w2/` y `qsd2_w2/`. La BBDD se lee de `../week1/BBDD`. Hace falta `opencv-python`, `numpy`, `pandas` y `matplotlib` (y `scikit-learn` solo para el método GMM, que se salta si no está). Ejecutado entero tarda unos 20 minutos, porque repite todos los experimentos. Las figuras para las slides se guardan en `figures/`.

**Para el test:**
1. Cuando salgan QST1 y QST2, ponlas como `qst1_w2/` y `qst2_w2/` dentro de `week2/`.
2. Ejecuta `python run_w2.py`. Tarda menos de un minuto, hace lo mismo que la §7 del notebook y solo genera el method1. Con `python run_w2.py --check`, antes evalúa QSD1 y QSD2 y tiene que dar los números de la tabla de arriba.

---

## 9. Qué contar en las slides (Task 1: 3 slides, Task 3: 2, Task 5: 1)

- **Task 1, slide 1:** "3D > 2D > 1D, y los bins tienen que seguir el rango de los datos". Usar las figuras `kinds_bins.png` y `lab_uniform_vs_range.png` o `bin_scheme.png`. Mencionar el feedback del año pasado.
- **Task 1, slide 2:** "Los bloques son la mayor mejora". Figura `blocks_heatmap.png`, con el compromiso bloques × bins.
- **Task 1, slide 3:** "Pirámide: igual que el nivel fino con recortes perfectos, pero más robusta con nuestras máscaras". Figuras `pyramid_results.png` y `margin.png`. Añadir la configuración final y el salto W1 → W2 (Task 2): `task2_w1_vs_w2.png`.
- **Task 3, slide 1:** el método paso a paso, `bg_steps.png`.
- **Task 3, slide 2:** la comparación de métodos y la ablación (tablas de la sección 5), más `bg_best_worst.png` con los errores (sombras y cuadro blanco).
- **Task 5:** el pipeline, la tabla GT vs nuestras máscaras, el problema de la franja (`t5_lost.png`) y la solución con el margen.
- **Fila de la tabla de la Task 2:** 0.900 → 0.967. **Task 4:** P 0.953 / R 0.966 / F1 0.956. **Task 5:** mAP@1 0.967 (method1).

**Aviso:**
- El margen interior y la normalización son preprocesado, y el descriptor sigue siendo un histograma 3D por bloques, así que creo que cumple el enunciado.
- La máscara usa solo color más morfología sobre la máscara binaria (componentes conexas, relleno de agujeros), sin detectores de contornos ni de objetos. Si queréis ir sobre seguro, confirmadlo con los profesores.
