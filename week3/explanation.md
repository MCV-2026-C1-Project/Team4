# C1 – Week 3: explicación del trabajo

Este documento explica **qué se pide** en la week 3, **qué hemos hecho y por qué**, **qué hay en el notebook** `c1-week3.ipynb` y
**cómo se entrega**. Todos los números salen de ejecutar el notebook (también están en `C1_W3_all_tests.xlsx`).

---

## 1. Qué se pide

Seguimos con el buscador de cuadros. Esta semana hay tres novedades en las consultas: **ruido** en algunas fotos, **cambios de
color** (de tono / hue) en otras y, en QSD2, **hasta dos cuadros por foto**.

| Task | Qué es | Dónde está en el notebook |
|---|---|---|
| 1 | Filtrar el ruido con filtros lineales o no lineales (modelo de ruido desconocido, solo algunas imágenes tienen ruido). Mostrar ejemplos de QSD1-W3 | §2 |
| 2 | Implementar descriptores de textura (LBP, DCT, wavelets…) y evaluar QSD1-W3 **solo con textura** | §3 |
| 3 | QSD2-W3: detectar **todos** los cuadros (máx. 2), quitar el fondo y crear la máscara binaria | §4 |
| 4 | QSD2-W3: detectar, quitar fondo, buscar y devolver las correspondencias con **un nivel más de lista** | §5 |
| 5 | Entrega a ciegas en QST1 / QST2 (un solo método; en QST2 también las máscaras) | §6 y `run_w3.py` |

**Datos**
- `BBDD/`: los mismos 287 cuadros.
- `qsd1_w3/`: 30 consultas recortadas. `augmentations.pkl` dice qué se le hizo a cada una (`None`, `None-MultiplyHue` = cambio de
  color, `UnnamedImpulseNoise` = ruido): 9 / 14 / 7. La carpeta `non_augmented/` tiene **las mismas fotos sin modificar**, lo que
  permite medir el ruido y evaluar los filtros de forma objetiva (PSNR / SSIM).
- `qsd2_w3/`: 30 fotos de pared con **39 cuadros** (9 fotos tienen dos), máscaras GT, `frames.pkl` (las 4 esquinas de cada cuadro) y
  el GT con un id por cuadro: `[[236], [107], [280, 285], …]`. En el GT los cuadros van **de izquierda a derecha**.
- QST1 (50) y QST2 (30) se publican el jueves 22 de octubre a las 14:00.

**Entrega** (domingo 25 de octubre, 19:00): slides (Tasks 1–4), código en GitHub y en el Drive
`TeamX/week3/QST1/method1/result.pkl`, `TeamX/week3/QST2/method1/result.pkl` y `TeamX/week3/QST2/method1/*.png`.

---

## 2. Punto de partida: qué hicieron los demás grupos

La diapositiva 3 del enunciado de la W3 tiene los resultados de test de la W2:
- **Retrieval QST2:** el mejor fue **nuestro grupo (mAP@1 0.90)**; además sacamos un 10 en la W1 y en la W2.
- **Máscaras QST2:** las mejores fueron las de los **Teams 2, 8 y 3 (F1 0.98)**; las nuestras sacaron 0.96. El código del Team 8 es
  público (los de los Teams 2 y 3 no), así que esta semana **partimos de su modelo de pared** y lo adaptamos a dos cuadros.

Así que: **descriptor de color = el nuestro de la W2** (el mejor del test) y **máscaras = el modelo del Team 8** (de los mejores del test).

**Cómo validamos.** Igual que en la W2, para no sobreajustar a 30 consultas evaluamos todo en **cuatro conjuntos (129 cuadros)**:
QSD1-W3, los cuadros de QSD2-W3 recortados con el GT, y los dos conjuntos de la W2 (QSD1-W2 y QSD2-W2 recortado con el GT). Además, en
la Task 4, QSD2-W3 y QSD2-W2 recortados con **nuestras** máscaras. Solo nos quedamos con configuraciones que están en una **meseta**
(las vecinas también funcionan).

---

## 3. Task 1: ruido (§2)

### 3.1 Qué ruido es
Restando cada consulta con ruido de su original: un ~3–4 % de los píxeles cambia más de 40 niveles y el resto no cambia nada; el
perfil de una fila muestra **picos aislados hacia arriba y hacia abajo**. Es **ruido impulsivo** (tipo sal y pimienta con valores
aleatorios, algo extendido por el JPEG). Las imágenes con cambio de color solo cambian 1–2 niveles de media (salvo q26).

Teoría (L6): para ruido impulsivo los filtros lineales (media, gaussiano) solo reparten cada pico entre sus vecinos; el **filtro de
mediana** (no lineal) lo elimina y conserva los bordes.

### 3.2 Qué imágenes tienen ruido: detector de impulsos aislados
Un impulso es un píxel **más claro (o más oscuro) que sus 8 vecinos por un margen grande**. La textura natural casi nunca hace eso (los
bordes los comparten varios píxeles). Medimos, a **resolución original**, la fracción de píxeles que se separan más de 40 niveles de
todos sus vecinos (dilatación / erosión con un anillo 3×3, morfología L4).

| | valor del detector r |
|---|---|
| 16 fotos con ruido | mínimo **0.0098** |
| 391 imágenes limpias (BBDD, W2, W3 sin ruido) | máximo **0.0025** |

Con el umbral **0.005** el detector acierta el **100 %** (margen ×2 a cada lado). Después de reducir a 512 px ya no separa: el
reescalado promedia los impulsos. Por eso el ruido se trata antes de reescalar.

### 3.3 Qué filtro
Comparamos 12 filtros con PSNR y SSIM contra la foto original, en las 16 fotos con ruido y en las 14 limpias:

| Filtro | SSIM fotos con ruido | SSIM fotos limpias |
|---|---|---|
| sin filtro | 0.476 | 1.000 |
| media 3×3 / gaussiano σ=1 (lineales) | 0.767 / 0.801 | 0.919 / 0.916 |
| mediana 3×3 | 0.855 | 0.938 |
| **mediana 3×3 dos veces** | **0.877** | 0.923 |
| mediana 5×5 / 7×7 | 0.842 / 0.801 | 0.862 / 0.800 |
| bilateral / non-local means | 0.598 / 0.743 | 0.933 / 0.889 |

- **Los lineales pierden**: convierten cada pico en una mancha.
- **Mediana 3×3 aplicada dos veces** es la mejor: la primera pasada quita los impulsos aislados y la segunda los que el JPEG había
  extendido a 2 píxeles. Una mediana más grande también quita el ruido pero borra detalle.
- **Filtrar una imagen limpia siempre empeora** (SSIM 0.92–0.94), así que solo filtramos las que marca el detector.

**Método Task 1:** `si r > 0.005: mediana3×3(mediana3×3(foto))`, a resolución original y antes de todo lo demás.

**Efecto en el retrieval (§3.5):** sin filtro, en las 7 consultas con ruido LBP baja a 0.57 y HOG a 0.86 de mAP@1; con el filtro los dos
vuelven a 1.00. DCT, wavelets y color no notan el ruido (resumen zonas grandes).

---

## 4. Task 2: descriptores de textura (§3)

Todos se calculan sobre la imagen **sin ruido**, en **gris** (así el cambio de color no les afecta), **reescalada a un cuadrado fijo**
(256×256 o 128×128: el bloque k de la consulta cubre la misma parte del cuadro que el bloque k de la BBDD) y **por bloques** (la lección
de la W2: dónde están las cosas importa).

| Familia | Qué describe | Qué probamos | Mejor configuración | mAP@1 QSD1-W3 | media 4 sets |
|---|---|---|---|---|---|
| **LBP** | patrón de signos alrededor de cada píxel | (P, R), bloques, uniforme vs 256 códigos, multiescala, distancia | P=8, R=2, **los 256 códigos**, 10×10 bloques, χ² | 1.000 | ≈0.98 |
| **DCT por bloques** | estructura gruesa de cada bloque (baja frecuencia) | bloques, nº de coeficientes, quitar DC, normalizar bloque | 4×4 bloques, 36 coef. zig-zag, **sin DC + bloque normalizado**, L2 | 1.000 | 0.971 |
| **Wavelets (Haar)** | energía de bordes por orientación y escala (L7) | niveles, bloques, aproximación, estadísticos, Haar vs db2 | 3 niveles, 8×8 bloques, mean\|c\| + std, L1 | 1.000 | 0.985 |
| **HOG** | histogramas de orientación del gradiente | tamaño, celda, bloque, orientaciones | 128×128, celdas 16 px, bloques 2×2, 8 orientaciones, L2 | 1.000 | **1.000** |
| color W2 (referencia) | colores por bloque | – | YCrCb 3D pirámide | 0.967 | 0.958 |

Conclusiones:
- **Los bloques son imprescindibles** también en textura: un LBP global no distingue cuadros (≈0.1–0.2); con bloques llega a ~0.95.
- **LBP:** con nuestra resolución un radio mayor (R = 2–4) es mejor que el clásico R = 1. **Sorpresa:** esperábamos que el LBP uniforme
  ganara (10 bins en vez de 256), pero con 8×8–12×12 bloques **los 256 códigos llegan a ≈0.98** y el mejor uniforme (multiescala) se queda
  en ≈0.95: a R = 2–4 hay muchos patrones *no uniformes* (trazos cruzados, grano) y el mapeo uniforme los junta en un solo bin.
- **DCT:** con 6–36 coeficientes por bloque guardamos la **composición** del cuadro (una versión borrosa), sin detalle ni ruido. Quitar el
  DC (el brillo del bloque) sube QSD2 de 0.90 a 0.95; normalizar cada bloque sube la media de 0.955 a 0.971.
- **Wavelets:** la familia más precisa después de HOG; 3 niveles y 8×8 bloques están en una meseta (2–4 niveles y 8–12 bloques igual).
- **HOG** es el más robusto: muchas configuraciones aciertan **los 129 cuadros** en la posición 1. Necesita celdas de ~1/8 de la imagen y
  normalización por bloques (justo eso lo hace robusto a la luz del museo).
- **Cambio de color:** solo falla el descriptor de color (q26, el cuadro rojo que se vuelve naranja). Todos los de textura son inmunes.

**Respuesta a la Task 2 (tabla de la plantilla):** con **solo textura**, QSD1-W3 da **mAP@1 = mAP@5 = 1.000** (HOG; wavelets y DCT empatan).

---

## 5. Task 3: detectar hasta dos cuadros y quitar el fondo (§4)

### 5.1 Del método del Team 8 a dos cuadros
**Modelo de pared del Team 8:** el color (a, b en Lab) de la pared se modela como **función cuadrática de su luminosidad L**, ajustada de
forma robusta en una franja del borde. Los focos y las sombras cambian la luminosidad de la pared pero no su color, así que para cada píxel
el modelo predice *el color que tendría la pared con ese brillo* y la distancia de Mahalanobis a esa predicción decide (≤ 5.5 → parece pared).
Pared = píxeles que parecen pared **y están conectados con el borde**; los muy oscuros nunca son pared.

Ablación (cada fila añade un cambio; media de las 30 fotos de QSD2-W3):

| Paso | P | R | F1 | nº de cuadros correcto |
|---|---|---|---|---|
| 0 · nuestro método de la W2 (un cuadro) | 0.958 | 0.806 | 0.855 | 70 % |
| 1 · reglas del Team 8 en la W2 (franja 4 %, regla global de filas/columnas al 50 %) | 0.933 | 0.817 | 0.825 | 80 % |
| 2 · modelo del Team 8 + hasta 2 componentes | 0.970 | 0.766 | 0.801 | 80 % |
| 3 · + **franja del borde del 1 %** | 0.983 | 0.962 | 0.970 | 97 % |
| 4 · + por cuadro: rellenar agujeros + relleno por filas/columnas | 0.982 | 0.968 | 0.973 | 97 % |
| 5 · + recortar filas/columnas con poca máscara (30 %) | 0.984 | 0.966 | 0.973 | 97 % |
| 6 · + ajuste a un cuadrilátero | 0.990 | 0.958 | 0.971 | 97 % |
| 7 · + **recuperar por bordes cuadros del color de la pared (FINAL)** | **0.988** | **0.965** | **0.975** | **100 %** |

- **0 → 1:** nuestro método de la W2 se queda con la componente más grande y pierde un cuadro en las fotos con dos. El del Team 8, tal cual,
  también falla: su última regla borra las filas/columnas con menos de un 50 % de cuadro **en toda la imagen**, y eso borra los cuadros que
  ocupan menos de media foto (dos cuadros uno al lado del otro).
- **2 → 3, el cambio clave:** en la W3 muchos cuadros están a solo un 1–2 % del borde de la foto, así que con una franja del 4 % el modelo
  "aprende" el cuadro como si fuera pared. Con el 1 % sigue habiendo miles de píxeles de pared.
- **6 → 7:** en q2 hay un dibujo sobre papel blanco en una pared blanca; el color solo ve los trazos. Su **marco** sí se ve como bordes: si
  hemos encontrado menos de 2 cuadros, las regiones cerradas por bordes que no tocan ninguna detección de color y son casi rectangulares se
  añaden. El recuento pasa a ser **perfecto (30/30)**.
- En **QSD2-W2** (un cuadro por foto) el método da **F1 0.977** (nuestro de la W2: 0.951 con este código; el Team 8 dio 0.976 en la W2).

### 5.2 Robustez
- Franja 0.5–2 %, σ 4.5–6.5, recorte al 50 %: el F1 cambia menos de 0.01 → meseta, no valor de suerte (una franja del 2 % ya empeora un poco).
- **Sin el filtro de ruido** las máscaras salen iguales: la segmentación trabaja sobre una copia de 512 px y el reescalado ya disuelve los
  impulsos. Lo dejamos igualmente (no cuesta nada y protege si el test tiene más ruido).
- Cuadrilátero vs caja alineada: mismo F1 (0.975 vs 0.976); el cuadrilátero es más preciso (0.988 vs 0.982) y es el modelo correcto para
  cuadros girados o en perspectiva.
- **Bordes como barrera en todas partes (descartado):** recupera los cuadros del color de la pared, pero la **sombra** de debajo de cada marco
  queda encerrada entre el borde del marco y el de la sombra y pasa a ser "cuadro": baja la precisión (0.971 en W3, 0.948 en W2). Por eso los
  bordes solo se usan para *añadir* cuadros que faltan, nunca para cambiar las detecciones de color.

**Errores que quedan:** q21 (un lienzo cuya mitad de abajo es del mismo blanco que la pared y no tiene marco: ni color ni bordes pueden
separarlo), las sombras bajo los marcos y los márgenes de papel del color de la pared.

---

## 6. Task 4: retrieval en QSD2-W3 (§5)

**Pipeline:** foto → filtro de ruido → máscaras → para cada cuadro, recorte de su caja → descriptores → ranking. La salida tiene un nivel más
de lista: `[[[7, 25, …], [44, 81, …]], [[107, …]], …]`, con los cuadros **de izquierda a derecha** y todos los ids como `int`.
Evaluación por cuadro: la lista k se compara con el cuadro k del GT; un cuadro no encontrado cuenta 0.

### Por qué combinamos descriptores (late fusion)
Con los recortes del GT, HOG solo ya es perfecto. Pero con **nuestras** máscaras los recortes tienen pequeños errores (una tira de pared o
sombra, un borde perdido, una pequeña rotación) y **cada descriptor falla en cuadros distintos**:

| cuadro | color | wavelet | HOG | fusión |
|---|---|---|---|---|
| q2/1 | 1 | 1 | 27 | **1** |
| q8/1 | 16 | 1 | 1 | **1** |
| q19/2 | 85 | 1 | 1 | **1** |
| q21/1 | 1 | 1 | 11 | **1** |
| q29/1 | 1 | 1 | 57 | **1** |
| (y 4 más con rango 2–3) | | | | **1** |

(posición del cuadro correcto en el ranking). Para combinarlos, en cada consulta normalizamos las distancias de cada descriptor a los 287
cuadros (**z-score**: menos su media, entre su desviación) y las sumamos. Como los fallos no coinciden, los otros dos descriptores
siempre corrigen al que falla.

| Descriptores (QSD2-W3 con nuestras máscaras) | mAP@1 | mAP@5 |
|---|---|---|
| color W2 | 0.897 | 0.923 |
| HOG | 0.897 | 0.906 |
| wavelet | 0.974 | 0.987 |
| HOG + color | 0.974 | 0.981 |
| **HOG + wavelet + color** | **1.000** | **1.000** |

- **HOG + wavelet + color** (pesos iguales) es la única combinación **perfecta en los seis conjuntos** (los cuatro con recorte GT y QSD2-W3 /
  QSD2-W2 con nuestras máscaras). Añadir DCT o LBP no aporta nada y pierde una consulta de QSD1-W2.
- El mapa de pesos muestra una **meseta** alrededor de pesos iguales: no es una elección de suerte.
- Rectificar la perspectiva de cada cuadro ayuda a algunos descriptores y perjudica a otros; con la fusión los dos dan 1.000, así que nos
  quedamos con el recorte simple.

**Resultado Task 4:** QSD2-W3 con nuestras máscaras **mAP@1 = mAP@5 = 1.000** (39 cuadros), igual que con los recortes del GT.

---

## 7. Resultados finales (method1)

| | mAP@1 | mAP@5 |
|---|---|---|
| QSD1-W3 (método final) | **1.000** | **1.000** |
| QSD1-W3 solo textura (Task 2) | **1.000** | **1.000** |
| QSD2-W3 con nuestras máscaras (Task 4) | **1.000** | **1.000** |

Máscaras QSD2-W3 (Task 3): **P 0.988 · R 0.965 · F1 0.975**, número de cuadros correcto en 30/30 fotos.

---

## 8. Qué hay en el notebook

| Sección | Contenido |
|---|---|
| Intro / 0 | Tareas, qué cambia, lecciones de W1–W2 y de los otros grupos, pipeline final, setup y estilo de las gráficas |
| 1 | Carga de datos, figuras de las aumentaciones y de QSD2 con los marcos GT, protocolo de validación |
| 2 | Task 1: qué ruido es, detector, 12 filtros con PSNR/SSIM, ejemplos de QSD1-W3 |
| 3 | Task 2: LBP, DCT, wavelets y HOG (explicación + visualización + barrido de parámetros + conclusión de cada uno); comparación con el color y por tipo de aumentación, con y sin filtro |
| 4 | Task 3: del método del Team 8 a dos cuadros (ablación paso a paso), robustez, figura paso a paso, F1 por foto y peores casos |
| 5 | Task 4: fusión de descriptores, rangos por cuadro, mapa de pesos, resultado final, formato de salida y ejemplos |
| 6 | Task 5: genera los `result.pkl` y las máscaras si están `qst1_w3/` y `qst2_w3/` |
| 7 | Resumen para las slides y exportación del Excel |

**Cómo usarlo:** el notebook va en `Project/week3/` junto a `qsd1_w3/` y `qsd2_w3/`. La BBDD se lee de `week3/BBDD` o, si no está, de
`../week2/week2_submission/BBDD` o `../week1/BBDD` (para que la carpeta sea autocontenida, copiad la BBDD en `week3/`). Los conjuntos de la W2
(`qsd1_w2`, `qsd2_w2`) se usan como validación extra y se buscan en `week3/` o en `../week2/…`. Hace falta `numpy`, `opencv-python`,
`scikit-image`, `PyWavelets`, `pandas`, `matplotlib` y `openpyxl` (`pip install -r requirements.txt`). Tarda unos 20 minutos porque
repite todos los experimentos. Las figuras para las slides se guardan en `figures/`.

**Para el test (jueves):**
1. Poned `qst1_w3/` y `qst2_w3/` dentro de `week3/`.
2. `python run_w3.py --check` (evalúa QSD1/QSD2 y tiene que dar los números de arriba) o `python run_w3.py` (solo test, < 2 min).
3. Subid `results/QST1/method1/result.pkl`, `results/QST2/method1/result.pkl` y `results/QST2/method1/*.png` al Drive como
   `TeamX/week3/...` (y a `/home/dlcv0X/m1-results/week3/...` si os lo piden).

**Aviso sobre el formato de QST1:** la diapositiva de la Task 5 dice *"add extra list level to allow for the possibility of more than one
painting per image"* para QST1 y QST2, así que en QST1 también usamos tres niveles: `[[[ids]], [[ids]], …]`. Si los profesores prefieren
la lista de listas de siempre para QST1, basta con `[p[0] for p in preds]`.

---

## 9. Qué contar en las slides

- **Task 1 (1–2 slides):** "Conoce tu ruido antes de filtrar": `t1_noise_type.png` (impulsos) + `t1_detector.png` (detector, 100 %) +
  `t1_filters.png` (la mediana gana a los lineales; filtrar una imagen limpia la empeora) + `t1_examples.png` (ejemplos de QSD1-W3).
- **Task 2 (2–3 slides):** una con los cuatro descriptores explicados (`t2_lbp_explained.png`, `t2_dct_explained.png`,
  `t2_wavelet_explained.png`, `t2_hog_explained.png`); otra con los barridos (`t2_lbp_heatmap.png`, `t2_dct_results.png`) y la tabla de
  los mejores; otra con `t2_per_augmentation.png` ("la textura es inmune al cambio de color; el filtro salva a LBP y HOG").
  Fila de la tabla: **mAP@1 QSD1-W3 solo textura = 1.000**.
- **Task 3 (2 slides):** `t3_steps.png` (el método paso a paso, mencionando que viene del Team 8) y `t3_ablation.png` + `t3_worst.png`
  (qué aporta cada paso, errores). Fila de la tabla: **P 0.988 / R 0.965 / F1 0.975**.
- **Task 4 (1 slide):** `t4_complementary.png` (cada descriptor falla en cuadros distintos → fusión) + `t4_examples.png`.
  Fila de la tabla: **mAP@1 QSD2-W3 = 1.000**.
