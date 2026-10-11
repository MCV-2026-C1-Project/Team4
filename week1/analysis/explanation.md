# C1 – Week 1: explicación del trabajo

Este documento explica **qué se pide** en la week 1, **qué he hecho y por qué**, **qué hay en el notebook** `week1.ipynb` y **qué significa cada métrica**. Todos los números salen de ejecutar el notebook sobre QSD1 (30 consultas).

---

## 1. Qué se pide

Tenemos un **museo (BBDD)** con 287 fotos de cuadros y un conjunto de **fotos de consulta** de algunos de esos cuadros. Para cada consulta, el sistema tiene que devolver **los K cuadros del museo que más se le parecen, ordenados** de más a menos parecido. Es un buscador por imagen (*query by example*).

La restricción de esta semana es que la imagen solo se puede describir con **histogramas 1D**: de nivel de gris, o un histograma por canal de color puestos uno detrás de otro (concatenados). No se permiten histogramas 2D ni 3D.

| Task | Qué es | Dónde está en el notebook |
|---|---|---|
| 1 | Calcular el descriptor (histograma) de cada imagen. Hasta 2 métodos | §2 |
| 2 | Implementar medidas para comparar histogramas | §3 |
| 3 | Para cada consulta de QSD1, ordenar el museo y evaluar con mAP@1 y mAP@5 | §4 (y §5, §6) |
| 4 | Generar las predicciones para QST1 (top-10 por consulta) en un `.pkl` | §7 |

**Datos**
- `BBDD/`: 287 cuadros `bbdd_XXXXX.jpg`. El id es el número del nombre.
- `qsd1_w1/`: 30 consultas de desarrollo más `gt_corresps.pkl`, la solución: `[[120], [170], ...]` significa que la consulta 0 es el cuadro 120, la consulta 1 el 170, etc.
- `qst1_w1/`: consultas de test. Se publican el jueves 1 de octubre a las 12:00 y **no traen solución**: entregamos las predicciones y los profesores las puntúan.

**Entrega** (domingo 4 de octubre, 19:00): código, slides (problema, método, resultados y discusión) y `TeamX/week1/QST1/method1/result.pkl` y `method2/result.pkl` en el Drive.

**Esquema del sistema**

```
imagen → (preprocesado) → espacio de color → histograma 1D por canal → normalizar → concatenar
       → distancia contra los 287 cuadros → ordenar → top-K
```

---

## 2. Las métricas: qué miden y para qué sirven

### mAP@K (la métrica oficial)
Para cada consulta, el sistema devuelve un ranking. Cada consulta tiene **un único cuadro correcto**, así que:

- **AP@K de una consulta** = `1 / posición` si el cuadro correcto aparece entre los K primeros, y `0` si no aparece.
  - Primera posición → 1; segunda → 0.5; tercera → 0.33; quinta → 0.2; fuera del top-K → 0.
- **mAP@K** = la media del AP@K sobre todas las consultas.

Qué nos dice cada una:
- **mAP@1**: la proporción de consultas en las que **el primer resultado es el correcto**. Es la más exigente y la más fácil de interpretar: 0.90 significa que acierta a la primera en el 90% de las consultas.
- **mAP@5**: también da puntos si el correcto sale en las posiciones 2 a 5, aunque menos cuanto más abajo. Si es bastante mayor que el mAP@1, el sistema "casi acierta": el cuadro correcto está cerca, pero algo se le cuela delante.

Son las que piden en la tabla de la Task 3 y las que calcularán los profesores en QST1. He comprobado que mi cálculo coincide con la implementación oficial (`apk`/`mapk` de benhamner/Metrics).

### Top-10
Es la proporción de consultas cuyo cuadro correcto está entre los 10 primeros. No es oficial, pero es útil porque **en QST1 entregamos 10 resultados por consulta**: si el cuadro correcto no está en el top-10, esa consulta está perdida.

### Posición (rank) del cuadro correcto
Para cada consulta, en qué posición del ranking completo (1 a 287) aparece el cuadro correcto. Sirve para ver **cómo de mal** falla un método: no es lo mismo quedar segundo que quedar en la posición 173. Es lo que se dibuja en la gráfica de §6.1.

### Intervalo de confianza (bootstrap)
Con solo 30 consultas, **cada consulta vale 1/30 ≈ 0.033 de mAP@1**. Pasar de 0.50 a 0.53 es acertar **una consulta más**, y eso puede ser pura casualidad. Para saber si una mejora es real:
- se generan 2.000 conjuntos de 30 consultas elegidas al azar, con repetición, a partir de las nuestras;
- se calcula el mAP@5 de cada conjunto;
- el intervalo entre los percentiles 2.5 y 97.5 es el **intervalo de confianza del 95%**.

Si los intervalos de dos métodos **no se solapan**, la diferencia entre ellos es real y no suerte.

### Criterio para decidir
No me he quedado con "la casilla con el número más alto", porque con 30 consultas eso lleva a sobreajustar. Solo me he fiado de un cambio si **mejora de forma consistente** con varios espacios de color, números de bins y medidas.

---

## 3. Las medidas de distancia (Task 2)

Todas comparan el histograma de la consulta con los 287 del museo. Para poder ordenar todo de la misma forma, **todas devuelven una distancia (menor = más parecido)**. Las que miden similitud (intersección, Hellinger y correlación) se cambian de signo.

| Medida | Idea | Resultado en nuestros datos |
|---|---|---|
| Euclídea | raíz de la suma de diferencias al cuadrado; la dominan los bins grandes | de las peores |
| L1 | suma de diferencias absolutas; todos los bins cuentan igual | buena |
| χ² | como la euclídea, pero dividiendo por el tamaño del bin: las diferencias en bins pequeños pesan más | de las mejores |
| Intersección | cuánto se solapan los dos histogramas | buena; **con histogramas normalizados ordena exactamente igual que L1** |
| Hellinger | compara las raíces cuadradas, así que la dominan menos los bins grandes | de las mejores |
| Jensen–Shannon | distancia entre dos distribuciones de probabilidad | de las mejores |
| Correlación | parecido en la forma (la `HISTCMP_CORREL` de OpenCV) | de las peores |

---

## 4. Qué he hecho, paso a paso, y por qué

### 4.1 Mirar primero los datos (§1 del notebook)
Antes de probar descriptores, miré **en qué se diferencia una consulta de su cuadro correcto**:
- **Encuadre:** casi idéntico. En las dos fotos sale el mismo marco y un poco de pared.
- **Iluminación:** cambia mucho. Algunas consultas están más oscuras, otras más claras, y otras tienen una luz más cálida o más fría.

Para cuantificarlo, calculé la media de cada canal de Lab en la consulta y en su cuadro. La diferencia media es de **15.9 en la luminosidad L**, frente a solo **1.2 en a** y **2.3 en b**, que son los canales de color. 17 de las 30 consultas son más oscuras que su cuadro, y el resto más claras.

**Conclusión:** el brillo de la foto no es fiable; el color sí. Esto guió los experimentos.

### 4.2 Espacio de color × bins × medida, sin preprocesado (§5.1)
**Por qué:** son las tres decisiones básicas de las Tasks 1 y 2. Probé 6 espacios de color (gris, RGB, HSV, Lab, YCrCb y rg normalizado) × 5 números de bins (8 a 128) × 7 medidas, es decir, 210 combinaciones.

**Resultados:**
- **Gris:** mAP@5 ≈ 0.30. **RGB:** ≈ 0.35–0.44. **HSV, Lab, YCrCb y rg:** ≈ 0.5–0.63. El color importa, y funcionan mejor los espacios que **separan el brillo del color**. En RGB, los tres canales cambian cuando cambia la luz.
- La euclídea y la correlación son siempre las peores; L1, χ², Hellinger y Jensen–Shannon quedan muy parecidas.
- Con demasiados bins empeora: un cambio pequeño de luz mueve los píxeles al bin de al lado. Lo mejor está entre 16 y 64.
- El techo sin preprocesado es de **mAP@5 ≈ 0.55–0.63**. El problema no es la medida, es la iluminación.

### 4.3 Normalizar la iluminación: el paso clave (§5.2)
**Por qué:** si la diferencia principal es la luz, hay que cancelarla **antes** de calcular el histograma. Probé tres correcciones punto a punto, que son transformaciones de rango como las de la clase L02:

| Nombre | Qué hace |
|---|---|
| `grayworld` | escala R, G y B para que tengan la misma media; quita el **tono** de la luz y mantiene el brillo |
| `contrast_stretch` | estira linealmente cada canal, llevando los percentiles 1 y 99 a 0 y 255; es el *contrast mapping* de L02 |
| `channel_mean_128` | escala cada canal R, G y B para que su media sea 128; quita el tono **y** la exposición |

**Resultados** (mAP@5 medio sobre 3 números de bins × 4 medidas):

| Espacio | Sin preprocesado | grayworld | contrast_stretch | **channel_mean_128** |
|---|---|---|---|---|
| Gris | 0.30 | 0.29 | 0.50 | **0.56** |
| RGB | 0.37 | 0.36 | 0.65 | **0.79** |
| HSV | 0.53 | 0.60 | 0.61 | **0.89** |
| Lab | 0.53 | 0.63 | 0.68 | **0.90** |
| YCrCb | 0.48 | 0.58 | 0.72 | **0.90** |

**Por qué funciona:** un cambio de luz multiplica, aproximadamente, cada canal por una constante (R·kR, G·kG, B·kB). Si divides cada canal por su media, esas constantes se cancelan. Es el modelo clásico de iluminación "diagonal" o de von Kries.

**Comprobación de que el 128 no es un número mágico:** con 100, 128 o 160 como valor objetivo los resultados son parecidos (HSV: 0.91, 0.95 y 0.89; Lab: 0.91, 0.91 y 0.94).

En el notebook hay una figura con una consulta oscura y su cuadro, **antes y después** de normalizar, con sus histogramas superpuestos: después de normalizar, H y S casi coinciden y V se acerca mucho.

### 4.4 Con la normalización: qué espacio, bins y medida (§5.3)
**Por qué:** una vez resuelto el problema de la luz, la mejor configuración puede cambiar, así que repetí la exploración completa.
- HSV, Lab y YCrCb quedan todos alrededor de **0.9**; RGB y gris se quedan por detrás.
- **HSV funciona mejor con pocos bins (16)**; Lab y YCrCb, con 32.
- χ², Hellinger y Jensen–Shannon quedan ligeramente por encima de L1. La euclídea y la correlación siguen siendo las peores (~0.7).

### 4.5 ¿La luminosidad sigue molestando? (§5.4)
**Por qué:** antes de normalizar, el brillo era la parte poco fiable. Probé a multiplicar el histograma de luminosidad (V, L o Y) por un peso entre 0 y 1.5.

**Resultado:** una vez normalizada la imagen, la luminosidad **aporta mucha información**. Quitarla (peso 0) hace bajar el mAP@5 de ≈ 0.92 a ≈ 0.74. Con cualquier peso a partir de 0.5 se obtiene lo mismo, así que dejo pesos iguales, que es lo más sencillo.

### 4.6 ¿Recortar el marco ayuda? (§5.5)
**Por qué:** los marcos, a menudo dorados u oscuros, se repiten en muchos cuadros y podrían confundir al sistema. Probé a quitar entre el 5% y el 30% de la imagen por cada lado.

**Resultado:** **no ayuda.** Los recortes pequeños cambian el resultado en ±1 consulta, que es ruido, y los grandes empeoran claramente (HSV baja de 0.94 a 0.79 con un 30%). En este dataset, la consulta y el cuadro del museo muestran el **mismo marco y la misma pared**, así que el marco forma parte de la "firma" de la foto.

### 4.7 ¿Concatenar varios espacios de color? (§5.6)
**Por qué:** cada espacio describe el color de una forma distinta, y concatenarlos sigue siendo un histograma 1D, así que está permitido.

**Resultado:** la mejor combinación (HSV + Lab) solo supera a HSV solo en ≈ 0.02 de mAP@5 (0.954 frente a 0.936), **menos de una consulta**, y añadir RGB no supera al mejor espacio por sí solo. No compensa la complejidad, así que me quedo con espacios individuales.

---

## 5. Métodos finales (§6)

| | Preprocesado | Espacio | Bins por canal | Medida |
|---|---|---|---|---|
| **Método 1** | media de cada canal → 128 | HSV | 16 (vector de 48) | χ² |
| **Método 2** | media de cada canal → 128 | CIELab | 32 (vector de 96) | Hellinger |

**Resultados en QSD1**

| | mAP@1 | mAP@5 | Top-10 | IC 95% del mAP@5 |
|---|---|---|---|---|
| Punto de partida (HSV 16, L1, sin normalizar) | 0.50 | 0.59 | 87% | [0.43, 0.74] |
| **Método 1** | **0.90** | **0.95** | 100% | [0.90, 1.00] |
| **Método 2** | **0.90** | **0.92** | 100% | [0.82, 1.00] |

Los intervalos del punto de partida y del Método 1 **no se solapan**, así que la mejora es real.

**Por qué estos dos métodos**
- Son las mejores configuraciones y además son **robustas**: sus vecinas (otros bins, otras medidas) también funcionan bien.
- Usan **dos espacios de color distintos**, los dos citados en el enunciado.
- **Fallan en consultas distintas**, lo que da juego en la discusión:
  - Método 1: q1, q5 y q10, todas en la posición 2.
  - Método 2: q14 y q15 en la posición 3, y q26 en la 9.
- Ninguno deja nunca el cuadro correcto fuera del top-10.

**Consultas que arregla la normalización:** q14, q17, q25, q26 y q28. Con el punto de partida estaban en las posiciones 13, 19, 9, 173 y 10; con el Método 1, todas salen en la posición 1.

**Por qué fallan los errores que quedan:** en la consulta 1, el Método 1 devuelve una escena de calle que no tiene nada que ver, pero cuyos histogramas H, S y V son casi idénticos a los de la consulta. Además, la consulta tiene un reflejo en el cristal. Un histograma **global** no sabe *dónde* está cada color, así que dos cuadros distintos con la misma paleta son indistinguibles. Esto motiva los **histogramas por bloques de la week 2**.

---

## 6. Qué hay en el notebook, sección por sección

| Sección | Contenido |
|---|---|
| **Intro** | Resumen del proyecto, las tareas, los datos, las entregas y el esquema del sistema |
| **0. Setup** | Imports, rutas relativas, parámetros y estilo de las gráficas |
| **1. Data** | Carga de imágenes y ground truth; parejas consulta–cuadro; gráfica de cuánto cambia cada canal de Lab entre consulta y cuadro |
| **2. Task 1** | Espacios de color, cálculo del histograma 1D por canal (normalizado y concatenado). **Visualización como en la slide del enunciado**: la imagen al lado de un histograma de barras por canal (en H, cada barra va pintada con su tono). Comparación de los histogramas de una consulta y de su cuadro, superpuestos |
| **3. Task 2** | Las 7 medidas con su fórmula e intuición, y una comprobación rápida |
| **4. Task 3** | `retrieve` (top-K), `apk`/`mapk` (la métrica oficial), la posición del cuadro correcto, y una función `evaluate` que prueba cualquier configuración con caché. Resultado del punto de partida |
| **5. Experimentos** | 5.1 Espacio × bins × medida · 5.2 Normalización de la iluminación · 5.3 Exploración con la imagen normalizada · 5.4 Peso de la luminosidad · 5.5 Recorte del marco · 5.6 Concatenar espacios de color. Cada uno con **por qué**, gráfica y **conclusión** |
| **6. Métodos finales** | Tabla de resultados con intervalos de confianza, gráfica de barras, posición del cuadro correcto por consulta, casos arreglados, errores que quedan con sus imágenes, y un análisis con histogramas de por qué falla un caso |
| **7. Task 4** | Genera `results/QST1/method1/result.pkl` y `method2/result.pkl` (top-10). Si la carpeta `qst1_w1` aún no existe, la celda se salta sin error |
| **8. Resumen** | Conclusiones listas para pasar a las slides |

**Cómo usarlo:** el notebook tiene que estar en `Project/week1/`, junto a `BBDD/` y `qsd1_w1/`. Hace falta `opencv-python`, `numpy`, `pandas` y `matplotlib`. Ejecutado entero tarda unos 4 minutos, porque repite todos los experimentos. Cuando salga QST1, poned la carpeta `qst1_w1` junto a las otras y volved a ejecutarlo. Si la carpeta se llama de otra forma, cambiad `QST1_DIR` en la sección 0.

---

## 7. Qué contar en las slides

- **Problema:** las fotos de consulta se diferencian del museo sobre todo en la **iluminación** (exposición y tono de la luz); el encuadre es casi igual.
- **Método:** histogramas 1D por canal, normalizados y concatenados, con una **normalización de la iluminación** previa (cada canal RGB escalado para que su media sea 128).
- **Resultados:** la tabla de la sección 5 y la gráfica de posiciones por consulta.
- **Discusión:**
  - Los espacios que separan brillo y color son mejores que RGB, y RGB es mejor que gris.
  - La euclídea y la correlación son las peores medidas.
  - Con demasiados bins empeora.
  - Recortar el marco no ayuda.
  - La normalización es la clave: pasa de 0.59 a 0.95.
  - Límite del método: un histograma global ignora dónde está cada color (el ejemplo de la consulta 1), lo que lleva a la week 2.
- **Aviso:** la normalización es un preprocesado y el descriptor sigue siendo un histograma 1D, así que creo que cumple las reglas. Si queréis ir sobre seguro, confirmadlo con los profesores.
