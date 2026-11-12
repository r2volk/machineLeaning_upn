# Guía para defender Ánimo

Guía de repaso para la exposición. La explicación extensa y la práctica para entrenar otra versión están en `explicacion-completa-del-proyecto-animo.md`. Esta guía refleja el código comentado y formateado actual.

### La idea en un minuto

Ánimo recibe un mensaje en español, comprueba reglas de precaución y, si no se activan, convierte palabras en números para clasificar una señal emocional. El clasificador aprende de ejemplos sintéticos etiquetados. La elección del modelo se hace con validación, y su resultado final se mide una sola vez con test. Una puntuación débil produce una petición de contexto. La respuesta conversacional se elige de textos redactados, no de un generador de lenguaje. Todo funciona localmente.
No digas «detecta con certeza cómo se siente una persona», «es un psicólogo», «aprende de cada conversación», «los datos son reales», «está validado clínicamente» ni «conserva una precisión del 80%». Ninguna de esas afirmaciones describe este proyecto.

### Orden de estudio

1. Propósito, datos y protocolo: README, mapa del curso y ficha del corpus.
2. Código de ML: módulos p00 a p06, en orden, con sus comentarios.
3. Evidencia: informe, métricas por clase y errores de test.
4. Aplicación y límites: seguridad, conversación, servidor y pruebas.

### Archivo por archivo

| Archivo o carpeta | Qué contiene y por qué existe |
| --- | --- |
| app.py | Servidor y contrato HTTP. Importa la inferencia, no implementa algoritmos de ML. Reserva un puerto y anuncia la URL realmente utilizada. |
| index.html | Envía mensajes a Python y muestra respuestas y resultados. No entrena el modelo. |
| seguridad.py | Reglas explícitas de precaución y mensaje de apoyo. No está entrenado ni produce una probabilidad de riesgo. |
| conversacion.py | Tres respuestas redactadas por etiqueta, incluida incertidumbre; alterna según turno. |
| entrenamiento/__init__.py | Marca el directorio como paquete importable. No ejecuta entrenamiento. |
| entrenamiento/__main__.py | Hace posible python -m entrenamiento etapa; importa y ejecuta la etapa solicitada. |
| p00_configuracion.py | Rutas basadas en el archivo, clases, semilla y reglas fijas; lectura JSON y hashes. |
| p01_datos.py | Normalización, lectura del corpus, control de duplicados, división por grupos y lectura de particiones. |
| p02_modelos.py | Constructor de cinco candidatos, métricas y lógica de abstención. No ejecuta un experimento al importarse. |
| p03_comparacion.py | Ajusta candidatos solo en train, compara en validación, selecciona modelo/umbral, curva y congela artefacto. |
| p04_evaluacion.py | Verifica evidencia, reserva test, calcula un único lote de predicciones y su incertidumbre exploratoria. |
| p05_inferencia.py | Carga el modelo y predice mensajes nuevos sin modificarlo; explica contribuciones. |
| p06_informe.py | Genera tres gráficos y el CSV de errores desde resultados guardados. No escribe el informe Markdown. |
| datos/escenarios.txt | Corpus fuente completo legible: encabezado de clase y dos paráfrasis por línea. |
| datos/train.csv, validacion.csv, test.csv | Particiones materializadas con ids, grupos, texto, clase y origen. Solo texto entra al clasificador. |
| datos/manifiesto.json | Asignación de escenarios, conteos y hashes. Permite auditar la división. |
| artefactos/modelo.joblib | Pipeline entrenado, nombre y umbral. Se carga solo el artefacto local comprobado. |
| resultados/seleccion.json | Todos los resultados de validación, curva, umbrales, versiones y hashes previos al test. |
| resultados/test_reservado.json | Constancia de apertura exclusiva y estado de evaluación. No borrar para volver a probar. |
| resultados/test_final.json | Métricas y todas las predicciones del único test; no oculta fallos. |
| resultados/errores_test.csv | Errores del test derivados del JSON, con indicación de abstención. |
| resultados/figuras/ | Gráficos PNG y SVG exportables. Se generan con matplotlib. |
| entrenamiento/pruebas/test_metodologia.py | Tests de ML en la misma carpeta exigida: fuga, negación, abstención, métricas y no repetición de test. |
| tests/test_api.py, tests/casos_seguridad.json | Casos de desarrollo de la API y reglas. No son el test estadístico del clasificador. |
| pytest.ini | Indica a pytest dónde encontrar los tests. |
| requirements.txt, requirements-lock.txt | Dependencias directas y captura exacta del entorno verificado. |
| .gitignore | Excluye cachés, entorno local y archivos del sistema del control de versiones. |
| docs/ | Instalación, reproducibilidad, mapa del curso, demostración y evidencia de verificación. |

### Cifras que debes poder explicar

| Dato | Resultado del experimento original |
| --- | --- |
| Corpus | 400 mensajes sintéticos en 200 escenarios y cinco clases |
| Particiones | 240 train, 80 validación y 80 test, separados por escenario |
| Ganador | Regresión logística con TF-IDF de unigramas y bigramas |
| Macro-F1 de validación | 0,7866 |
| Accuracy de test | 53 de 80, equivalente a 66,25 por ciento |
| Macro-F1 de test | 0,6531 |
| Abstención en test | 60 aceptados de 80; 44 aciertos entre esos 60 |
| Debilidad principal | Alegría: 4 aciertos entre 16 ejemplos reales |

Los scores no son intensidad emocional ni probabilidades clínicas calibradas. El resultado vale para este corpus sintético, no para una población real.

### Qué cambió en la versión actual

- Los 14 archivos Python tienen comentarios breves en minúscula y un formato más espaciado. No se cambió la lógica del clasificador.
- `p06_informe.py` solo genera gráficos y el CSV de errores. `INFORME_RESULTADOS.md` es el informe escrito conservado; no hay plantilla ni función `tabla` en p06.
- Los números de línea anteriores dejaron de ser válidos; busca las funciones por nombre.
- Los hashes de código del experimento original no coinciden con los archivos ahora comentados y formateados. No se actualizaron los registros históricos. El modelo conserva su huella y las cifras originales; la evaluación existente se reutiliza. Si falta ese resultado, la comprobación de código puede bloquear otra evaluación.

### Cómo explicar las fórmulas

**X e y.** X es una lista de textos, luego una matriz dispersa: cada fila es un mensaje y cada columna una palabra o pareja de palabras. y contiene nombres de clases, sin orden numérico significativo. Scikit-learn gestiona las etiquetas internamente.

**TF-IDF.** tf = 1 + log(conteo) para un término presente; idf = log((1+n)/(1+df)) + 1. n y df proceden exclusivamente de train. Después se normaliza cada vector por su longitud L2. No todas las palabras raras son importantes emocionalmente: el clasificador debe aprender sus pesos.

**Logística multinomial.** Para cada clase k, z_k = b_k + suma(w_kj * x_j). Softmax transforma los logits: p_k = exp(z_k) / suma(exp(z_l)). Se elige la clase con mayor score, sujeto a abstención. El ajuste minimiza una pérdida de clasificación con regularización; C=4 fija la fuerza inversa de esa regularización. No significa cuatro clases ni cuatro iteraciones. max_iter es un límite del optimizador.

**Naive Bayes.** Combina una probabilidad previa de clase con asociaciones de términos y clase. Supone independencia condicional de características, una aproximación fuerte en lenguaje. Se usa alpha=1 para suavizar. En este experimento se compara una variante práctica con TF-IDF, por lo que no debe describirse como conteos multinomiales puros.

**Explicación.** Para la candidata k y segunda clase l, la contribución de una palabra j es x_j*(w_kj-w_lj). La suma de todas las contribuciones más b_k-b_l coincide con la diferencia de logits. La interfaz solo muestra las cinco mayores contribuciones positivas, sin intercepto ni aportes negativos; no es una explicación completa ni causal. Una palabra frecuente como «mi» puede aparecer: es un sesgo aprendido visible.

**Métricas.** Accuracy = aciertos/total. Precision indica cuántos de los predichos como una clase pertenecían a ella; recall cuántos de los reales fueron recuperados. F1 es 2PR/(P+R). Macro-F1 promedia F1 de cada clase con el mismo peso. Soporte es el número de ejemplos reales de la clase. En la matriz las filas son reales y las columnas predicciones.

**Abstención.** Score máximo suficiente, margen suficiente y vocabulario no vacío. Una etiqueta incierta no es una sexta clase entrenada. Cobertura = aceptados/total. Accuracy selectiva = aciertos entre aceptados/aceptados. Puede subir simplemente al rechazar más mensajes, por eso ambas se informan juntas.

### Preguntas probables y respuestas sugeridas

**¿Por qué regresión logística si clasificas emociones?** La logística resuelve clasificación. Aquí distingue cinco clases y permite inspeccionar los pesos aprendidos; se relaciona con las semanas 11 y 12.

**¿El chatbot realmente aprende? ¿Qué se aprende?** Aprende vocabulario, IDF y pesos durante el entrenamiento. Al conversar permanecen fijos. Las respuestas y reglas de precaución están programadas.

**¿Por qué no una red neuronal o un modelo de lenguaje?** Los modelos clásicos permiten estudiar el método con un corpus pequeño y defender sus decisiones. El proyecto excluye modelos preentrenados y APIs.

**¿De dónde vienen los datos?** Son 400 frases sintéticas redactadas con asistencia de IA. Las etiquetas expresan la intención de redacción, no emociones medidas en participantes.

**¿Usar IA para redactar datos incumple no usar modelos preentrenados?** La aplicación y su entrenamiento no cargan ni consultan modelos preentrenados. La autoría asistida se declara; si el curso también la prohíbe, hace falta otro corpus autorizado y un estudio nuevo.

**¿Cómo evitas fuga de datos?** Se dividen escenarios antes de vectorizar, manteniendo juntas sus paráfrasis. El Pipeline aprende solo de train; selección usa validación y la evaluación final usa test. Los ids no entran en X.

**¿Por qué tres conjuntos?** Train ajusta parámetros; validación elige modelo y umbral; test mide al seleccionado. Elegir con test sesgaría la evaluación.

**¿Cómo sé que el test se usó una vez?** Hay una reserva exclusiva previa a abrirlo y predicciones guardadas que se reutilizan. Es protección contra repeticiones accidentales, no prueba contra manipulación deliberada.

**¿El bootstrap usa test mil veces?** El bootstrap remuestrea escenarios y sus predicciones existentes. No vuelve a entrenar ni a llamar al modelo; su intervalo es exploratorio.

**¿Por qué el test salió peor que validación?** Puede reflejar variación muestral y selección favorecida por validación. No se identificó una causa única ni se cambió el modelo para mejorar test.

**¿Son suficientes los resultados para usarlo con personas?** La evidencia corresponde a un prototipo con datos sintéticos. No demuestra eficacia clínica ni rendimiento con personas reales.

**¿Por qué no consigues un 95%?** No se persiguió una cifra usando test repetidamente. Mejorar exige datos adecuados y otra evaluación independiente.

**¿Qué pasa con sarcasmo o «no estoy feliz»?** Conservar negaciones y usar bigramas ayuda a representar orden local, pero no garantiza comprensión de negación o ironía.

**¿Qué ocurre con una palabra nueva o emojis?** Las palabras desconocidas se ignoran y los emojis se eliminan. Sin características conocidas se pide contexto; con algunas conocidas todavía puede haber errores.

**¿Un score del 60% significa que estoy 60% triste?** Es una puntuación relativa entre clases, sin calibración clínica. No representa intensidad emocional.

**¿Por qué la meta de 80% en validación no se cumple en test?** Era un criterio para elegir el umbral en validación, no una garantía. En test se informa la accuracy selectiva real junto con la cobertura.

**¿Qué hace con riesgo de suicidio?** Las reglas priorizan apoyo humano y el recurso peruano configurado. Se omite el análisis emocional; el prototipo no diagnostica, evalúa seguridad ni contacta servicios.

**¿Y si el mensaje dice «no quiero suicidarme» o habla de un artículo?** Puede activarse precaución por contexto. Las reglas son conservadoras y pueden producir falsas alarmas o pasar por alto otras expresiones.

**¿Qué pasa si falla el modelo?** Las reglas de precaución siguen funcionando. Para el chat normal, la API devuelve un error 503 y no inventa una clasificación.

**¿Hay memoria de la conversación?** El navegador conserva mensajes y turno durante la sesión, pero ML analiza cada mensaje por separado. El indicador de precaución es estado explícito, no memoria semántica.

**¿Cómo se protege la privacidad?** No hay persistencia de conversaciones, cuentas ni APIs externas. Los logs no incluyen el cuerpo del mensaje; exportar guarda un archivo local por decisión del usuario.

**¿Por qué Flask y un HTML?** Un proceso sirve página y API. Se abre la dirección impresa por Flask.

**¿Qué pruebas hay?** Hay 57 casos de software sobre metodología, API y precaución. No equivalen al test estadístico ni demuestran seguridad clínica.

**¿Qué mejorarías a continuación?** Datos autorizados y diversos, anotación independiente, revisión de seguridad y un test externo nuevo. Después comprobaría si calibración o validación cruzada aportan valor.

**¿Qué hizo cada integrante y cuánto tardaron?** No hay registro para afirmar cuatro autores humanos ni dos meses de trabajo. La asistencia de IA debe declararse según las reglas del curso.

**¿Cómo añado palabras o frases para mejorar el modelo?** En otra copia añado escenarios, regenero los datos derivados y entreno de nuevo. Solo los términos de train entran al vocabulario. El capítulo 32 del manual explica los pasos.

**¿Puedo borrar el marcador de test y probar hasta obtener una cifra mejor?** No. Tras observar sus errores, ese test ya influyó en el desarrollo. Una nueva evaluación independiente requiere escenarios nuevos reservados; repartir las mismas frases no la garantiza.

**¿Qué hace exactamente p06 y por qué conserva ese nombre?** Genera gráficos y CSV de errores desde evidencia guardada. Conserva el nombre para el comando existente, pero no escribe el Markdown ni repite la inferencia de test.

### Función por función

El inventario siguiente se extrae del código real. Cada entrada explica su responsabilidad; los módulos se estudian de arriba a abajo. Las funciones internas de Flask aparecen como crear_app.nombre. Las pruebas documentan contratos, no entrenamiento adicional.

#### JavaScript de index.html

JavaScript conecta la página con `/api/chat` y muestra resultados. El aprendizaje automático se ejecuta en Python.

#### `entrenamiento/__main__.py`

- `main()`: Selecciona una etapa; repetir informe no vuelve a ejecutar el test. En app.py reserva un socket loopback; en entrenamiento/__main__.py selecciona una etapa mediante argparse.

#### `entrenamiento/p00_configuracion.py`

- `huella(ruta)`: SHA-256 de un archivo: permite detectar cambios, no prueba autoría.
- `leer_json(ruta)`: Lee un archivo UTF-8 con datos JSON.
- `guardar_json(ruta, contenido)`: Escribe JSON legible mediante reemplazo atómico en el mismo disco.

#### `entrenamiento/p01_datos.py`

- `normalizar(texto)`: Minúsculas, tildes y signos; conserva negaciones y no usa diccionarios externos. No aprende parámetros; la misma función se aplica a entrenamiento y mensajes nuevos. Quita tildes, signos, números y emojis; conserva palabras como no.
- `leer_escenarios(ruta=DATOS / 'escenarios.txt')`: Convierte cada línea curada en dos filas con la misma etiqueta y grupo. Dos frases por escenario, una etiqueta. Rechaza duplicados normalizados antes de dividir. No aumenta artificialmente el corpus con plantillas.
- `preparar()`: Congela una división estratificada por grupos 60/20/20, sin ajustar modelos. Ordena y baraja grupos por clase; asigna 60/20/20 y materializa archivos. Si ya existen, solo verifica. No cambia la partición según resultados.
- `verificar_datos()`: Comprueba bytes congelados; no lee etiquetas de test para modelar ni evaluar. Lee bytes para SHA-256. Comparar hashes de test no supone usar sus etiquetas o predecirlo.
- `cargar(particion)`: Lee una partición explícita. Comparación e inferencia nunca solicitan test.
- `xy(filas)`: Separa los únicos predictores (texto) de las etiquetas; excluye grupo e id.

#### `entrenamiento/p02_modelos.py`

- `crear_modelo(nombre)`: Un Pipeline aprende vocabulario/IDF solo durante fit(train). C=4 es fijo. Devuelve un estimador SIN entrenar. Unigrams o bigrams y conteos o TF-IDF dependen del nombre; parámetros fijos antes del experimento.
- `metricas(reales, predichas)`: Accuracy, macro-F1 y detalle de todas las clases en un orden fijo. Fija el orden de clases y zero_division=0. La matriz y los soportes permiten interpretar errores que accuracy oculta.
- `mascara_aceptacion(probabilidades, umbral, conocidos=None)`: Abstención heurística por score máximo, margen y vocabulario conocido.
- `selectivas(reales, predichas, aceptar)`: Siempre informa cobertura junto a accuracy selectiva: abstenerse tiene costo. Si nadie es aceptado, accuracy selectiva es None, no cero ni uno. No compara silenciosamente denominadores distintos.

#### `entrenamiento/p03_comparacion.py`

- `comparar()`: Entrena cinco candidatos en train; desempata por el orden predeclarado. La lista NOMBRES fija el desempate. Guarda todos los resultados y ajusta el umbral en validación. El artefacto final sigue ajustado solo a train.
- `curva_aprendizaje(nombre, train, x_val, y_val)`: Ablación de tamaño con subconjuntos anidados de grupos, usando solo validación. Entrena versiones auxiliares del ganador con cada tamaño; se evalúan solo en validación. No sustituyen el artefacto seleccionado.

#### `entrenamiento/p04_evaluacion.py`

- `intervalo_por_grupos(filas, predichas, repeticiones=1000)`: Bootstrap de predicciones guardadas, remuestreando escenarios, no frases sueltas. Se remuestrea escenario con reemplazo y se incluyen ambas paráfrasis. El intervalo percentil es descriptivo y no elimina sesgo sintético.
- `evaluar_una_vez()`: Reserva test con creación exclusiva ANTES de cargarlo. Una segunda llamada no predice. La existencia del resultado evita cargar modelo o test; una reserva sin resultado bloquea otra apertura. predict_proba se llama una vez por lote y argmax obtiene etiquetas.

#### `entrenamiento/p05_inferencia.py`

- `cargar_modelo()`: Carga el artefacto local verificado; si falta, lo reconstruye solo con train. Joblib se usa solo con el archivo local verificado. Si falta, reconstruye en memoria desde train usando la selección fija, sin tocar resultados.
- `explicar(pipeline, vector, indice_clase, indice_alternativa)`: Aportes x_j*(peso_clase-peso_alternativa); asociaciones, nunca causas psicológicas. Escoge coeficientes de logística o log-probabilidades NB. Devuelve hasta cinco aportes positivos de términos presentes; omite interceptos y contribuciones negativas.
- `predecir(artefacto, texto)`: Predicción de texto independiente con scores, abstención y explicación local. No llama fit. Transform reutiliza vocabulario/IDF. Clasifica con predict_proba; añade candidata, alternativa, motivo, scores y evidencia.

#### `entrenamiento/p06_informe.py`

- `graficar(seleccion, final)`: Exporta figuras estáticas a partir de métricas congeladas (PNG y SVG).
- `guardar_figura(figura, destino)`: Guarda el mismo gráfico vectorial y rasterizado para informe o diapositivas.
- `generar_informe()`: Regenera comparación, matriz de confusión y curva en PNG y SVG, y lista los errores en CSV. Lee evidencia existente; no escribe INFORME_RESULTADOS.md, no entrena y no vuelve a ejecutar test.

#### `seguridad.py`

- `detectar_riesgo(texto)`: Devuelve precaución si encuentra una señal; ausencia de coincidencia NO significa seguridad. No está entrenada. La ausencia de coincidencia no prueba ausencia de riesgo; también activa ante menciones informativas o negadas.

#### `conversacion.py`

- `responder(etiqueta, turno)`: Alterna preguntas sin guardar conversaciones; no finge recordar mensajes previos. El turno solo alterna textos preescritos; no es memoria ni personalización aprendida.

#### `app.py`

- `crear_app()`: Configura Flask, carga el modelo una vez y registra endpoints sin estado persistente.
- `crear_app.preparar_peticion()`: Autoriza los scripts y estilos propios.
- `crear_app.proteger_respuesta(respuesta)`: Añade cabeceras de protección y evita caché.
- `crear_app.inicio()`: Sirve `index.html` desde Flask.
- `crear_app.salud()`: Permite comprobar si el modelo está cargado sin exponer mensajes.
- `crear_app.chat()`: Valida entrada; prioriza seguridad; ejecuta inferencia y respuesta redactada. Primero valida JSON y texto, luego aplica precaución y contexto de alerta; después limita longitud y turno y usa ML. Los mensajes no se escriben a disco.
- `crear_app.resultados()`: Devuelve métricas guardadas; visitar Laboratorio no entrena ni evalúa.
- `crear_app.documento(nombre)`: Descarga solo documentos permitidos; ninguna ruta arbitraria ni artefactos pickle.
- `crear_app.demasiado_grande(error)`: Evita procesar cuerpos excesivos y devuelve un error legible por la interfaz.
- `main()`: Escoge un puerto local libre y anuncia la dirección después de reservarlo. En app.py reserva un socket loopback; en entrenamiento/__main__.py selecciona una etapa mediante argparse.

#### `entrenamiento/pruebas/test_metodologia.py`

- `test_grupos_y_proporciones_congelados()`: Prueba grupos y proporciones congelados.
- `test_vectorizador_no_aprende_validacion()`: Prueba vectorizador no aprende validacion.
- `test_negaciones_no_se_eliminan()`: Prueba negaciones no se eliminan.
- `test_metricas_con_resultado_conocido()`: Prueba metricas con resultado conocido.
- `test_abstencion_no_es_una_emocion()`: Prueba abstencion no es una emocion.
- `test_no_hay_accuracy_si_nadie_es_aceptado()`: Prueba no hay accuracy si nadie es aceptado.
- `test_texto_desconocido_no_fuerza_neutral()`: Prueba texto desconocido no fuerza neutral.
- `test_explicacion_tiene_aportes_del_modelo()`: Prueba explicacion tiene aportes del modelo.
- `test_evaluacion_existente_no_vuelve_a_cargar_modelo(monkeypatch, tmp_path)`: Prueba evaluacion existente no vuelve a cargar modelo.
- `test_evaluacion_existente_no_vuelve_a_cargar_modelo.prohibido(*args, **kwargs)`: Función auxiliar de la prueba o fixture que prepara dependencias.
- `test_recuperacion_sin_artefacto_solo_usa_train(monkeypatch, tmp_path)`: Simula un artefacto ausente, registra las particiones consultadas y comprueba que solo se usa train. Contrasta scores sobre un mensaje de desarrollo y verifica que no se escriba un archivo nuevo.
- `test_recuperacion_sin_artefacto_solo_usa_train.registrar(particion)`: Envoltorio de prueba que registra el nombre de la partición y delega la lectura real; permite detectar una apertura indebida de validación o test.
- `test_artefacto_alterado_no_se_deserializa(monkeypatch, tmp_path)`: Crea bytes falsos y verifica que el hash impida deserializarlos.

#### `tests/test_api.py`

- `app()`: Función auxiliar de la prueba o fixture que prepara dependencias.
- `client(app)`: Función auxiliar de la prueba o fixture que prepara dependencias.
- `test_pagina_desde_flask_y_csp(client)`: Prueba pagina desde flask y csp.
- `test_entradas_invalidas(client, data)`: Prueba entradas invalidas.
- `test_json_malformado_y_tipo_incorrecto(client)`: Prueba json malformado y tipo incorrecto.
- `test_limites_y_turno(client)`: Prueba limites y turno.
- `test_respuesta_normal(client)`: Prueba respuesta normal.
- `test_casos_de_precaucion(caso)`: Prueba casos de precaucion.
- `test_riesgo_salta_modelo_aun_no_disponible(app, monkeypatch)`: Prueba riesgo salta modelo aun no disponible.
- `test_riesgo_salta_modelo_aun_no_disponible.prohibido(*args)`: Función auxiliar de la prueba o fixture que prepara dependencias.
- `test_riesgo_en_mensaje_largo_tiene_prioridad(client)`: Prueba riesgo en mensaje largo tiene prioridad.
- `test_continuacion_de_seguridad(client)`: Prueba continuacion de seguridad.
- `test_variantes_de_riesgo(texto)`: Prueba variantes de riesgo.
- `test_modelo_degradado_no_inventa_respuestas(app)`: Prueba modelo degradado no inventa respuestas.
- `test_no_hay_acceso_a_archivos_arbitrarios(client)`: Prueba no hay acceso a archivos arbitrarios.
- `test_html_del_usuario_permanece_dato(client)`: Prueba html del usuario permanece dato.
### Recorrido breve para presentar

1. Explica que se clasifican señales del texto y declara el origen sintético del corpus.
2. Abre p01 y muestra por qué las dos paráfrasis permanecen juntas.
3. Abre p02 y p03: vectorización, ajuste con train y selección con validación.
4. Explica p04 y muestra el resultado final, incluyendo los errores de alegría.
5. Envía un mensaje de desarrollo en el chat y explica p05: transformación, scores, abstención y aportes.
6. Muestra las reglas de precaución y sus límites; termina con una mejora que requeriría datos y test nuevos.

Para arrancar, activa el entorno, ejecuta `python app.py` y abre la dirección impresa. Laboratorio muestra los resultados guardados; visitarlo no entrena ni evalúa el modelo.
