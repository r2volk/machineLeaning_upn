import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
import { createCanvas } from '@napi-rs/canvas';
const ROOT=process.cwd(), B=path.join(ROOT,'.ppt-academica');
const FONT='Helvetica Neue', MONO='Menlo';
const C={bg:'#F6F8F4',ink:'#1D3027',muted:'#617369',green:'#315D47',light:'#E4EEE3',deep:'#183B2C',white:'#FFFFFF',line:'#CAD8CB',amber:'#956338',red:'#9B543C',soft:'#EEF3EC'};
const deck=Presentation.create({slideSize:{width:1280,height:720}}), check=[],records=[];
const ctx=createCanvas(1,1).getContext('2d');
function text(s,value,x,y,w,h,size=28,color=C.ink,o={}){
 const sh=s.shapes.add({geometry:'textbox',name:o.name||`text-${records.length}-${s.shapes.items.length}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=value;sh.text.style={fontSize:size,typeface:o.mono?MONO:FONT,color,bold:o.bold||false,insets:0,autoFit:'none',wrap:'none',verticalAlignment:'top',alignment:o.align||'left'};
 ctx.font=`${o.bold?'bold ':''}${size}px "${o.mono?MONO:FONT}"`;
 for(const line of String(value).split('\n')){const m=ctx.measureText(line).width;if(m>w+2)check.push({slide:records.length,text:line,width:Math.round(m),available:w});}
 return sh;
}
function rect(s,x,y,w,h,fill,stroke='none',name='panel'){
 return s.shapes.add({geometry:'roundRect',name,position:{left:x,top:y,width:w,height:h},fill,line:{style:'solid',fill:stroke,width:stroke==='none'?0:1},borderRadius:'rounded-xl'});
}
function page(title,notes){
 const s=deck.slides.add(); s.background.fill=C.bg;records.push({slide:records.length+1,title});
 if(records.length>1){const categorias=['','Antecedentes','Planteamiento del problema','Objetivos','Materiales y Métodos utilizados','Materiales y Métodos utilizados','Algoritmos','Algoritmos','Algoritmos','Algoritmos','Algoritmos','Materiales y Métodos utilizados','Algoritmos','Algoritmos','Herramientas de Ingeniería','Materiales y Métodos utilizados']; text(s,categorias[records.length-1],64,46,1080,62,42,C.ink,{bold:true,name:'slide-title'});text(s,title,65,112,1100,32,22,C.green,{name:'slide-subtitle'});text(s,`${String(records.length).padStart(2,'0')} / 16`,1100,677,116,20,16,C.muted,{align:'right',name:'page-number'});}
 s.speakerNotes.text=notes+'\n\nFuentes locales: entrenamiento/p01_datos.py, p02_modelos.py, p03_comparacion.py, p04_evaluacion.py, p05_inferencia.py; app.py; seguridad.py; entrenamiento/resultados/seleccion.json y test_final.json; INFORME_RESULTADOS.md; explicacion-completa-del-proyecto-animo.md. En clase: semanas 9, 11 y 12. Las cifras corresponden al experimento de la entrega, con datos 100 % sintéticos.';
 return s;
}
function lines(s,arr,x,y,w=500,size=27,color=C.muted,step=47,name='body'){
 arr.forEach((v,i)=>text(s,v,x,y+i*step,w,step,size,color,{name:`reveal-${name}-${i}`}));
}
function code(s,arr,x,y,w=700,size=25,step=42,filename=''){
 if(filename)text(s,filename,x,y-42,w,27,18,C.green,{mono:true});
 rect(s,x-20,y-12,w+36,arr.length*step+26,C.soft,'none','code-surface');
 arr.forEach((line,i)=>text(s,line,x,y+i*step,w,step,size,C.ink,{mono:true,name:`reveal-code-${i}`}));
}
function foot(s,v){text(s,v,64,632,1148,28,18,C.muted,{name:'footer-note'});}
function table(s,values,x,y,w,h,widths,config={}){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});t.borders.assign({fill:C.line,width:1,style:'solid'});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  let cell=t.getCell(r,c);cell.fill=r===0?C.deep:C.white;
  cell.text.style={fontSize:config.size||22,typeface:FONT,color:r===0?C.white:C.ink,bold:r===0,insets:7,alignment:c===0?'left':'center',verticalAlignment:'middle'};
  if(config.matrix&&r>0&&c>0&&r===c){cell.fill=C.light;cell.text.style={fontSize:config.size||22,typeface:FONT,color:C.green,bold:true,alignment:'center',verticalAlignment:'middle'};}
  if(config.highlight===r){cell.fill=C.light;cell.text.style={fontSize:config.size||22,typeface:FONT,color:C.ink,bold:true,alignment:c===0?'left':'center',verticalAlignment:'middle'};}
 } return t;
}
// 01 · Portada
{
 const s=page('Ánimo','Objetivo de la exposición: explicar el problema, los datos sintéticos, el flujo de entrenamiento y la evaluación. Ánimo interpreta señales en texto; no diagnostica ni reemplaza ayuda profesional. Ilustración de portada generada para esta presentación, no representa una persona real.');
 const bytes=await fs.readFile(path.join(B,'assets/portada.png'));
 s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:'Ilustración editorial abstracta de conversación',position:{left:0,top:0,width:1280,height:720},fit:'cover'});
 text(s,'PROYECTO FINAL DE MACHINE LEARNING',70,83,800,35,20,C.green,{bold:true});
 text(s,'Ánimo',65,215,650,120,102,C.ink,{bold:true,name:'reveal-cover-title'});
 text(s,'Un chatbot que interpreta\nseñales emocionales en texto',70,360,710,120,42,C.ink,{name:'reveal-cover-subtitle'});
 text(s,'Python · scikit-learn · Flask',72,603,600,36,24,C.muted);
}
// 02 · Objetivo
{
 const s=page('El proyecto analiza señales en el texto','Es una tarea de clasificación supervisada multiclase. La entrada es un mensaje breve en español y la salida es una de cinco etiquetas o incertidumbre. El modelo predice señales del mensaje, no el estado real de la persona. Las respuestas son plantillas redactadas. La emoción ansiedad se refiere a preocupación textual.');
 text(s,'«Estoy nervioso por mi examen»',64,175,1152,60,38,C.ink,{name:'reveal-main'});
 text(s,'ENTRADA',64,275,300,30,19,C.green,{bold:true});text(s,'Mensaje en español',64,318,510,45,30,C.ink);
 text(s,'SALIDA',650,275,300,30,19,C.green,{bold:true});text(s,'Una señal o incertidumbre',650,318,555,45,30,C.ink);
 text(s,'Alegría   Ansiedad   Enojo   Neutral   Tristeza',64,461,1152,53,30,C.green,{bold:true,name:'reveal-labels'});
 foot(s,'La interpretación puede fallar. Es un prototipo académico, no una herramienta clínica.');
}
// problema y objetivos
{
 const s=page('Clasificar señales emocionales en texto','El problema es identificar patrones en mensajes breves, con cinco etiquetas y una opción de incertidumbre. No se mide directamente el estado emocional ni se realiza diagnóstico.');
 text(s,'¿Cómo reconocer señales emocionales',64,212,1140,55,39,C.ink,{bold:true,name:'reveal-problema'});
 text(s,'en mensajes breves en español?',64,270,1140,55,39,C.ink,{bold:true});
 lines(s,['El lenguaje puede ser ambiguo.','Una misma palabra aparece en distintos contextos.','El modelo puede equivocarse o pedir más contexto.'],67,390,1110,27,C.muted,52,'problema');
 foot(s,'La etiqueta describe el texto y no determina el estado real de la persona.');
}
{
 const s=page('Objetivo general y objetivos específicos','El objetivo general es desarrollar un chatbot local que clasifique señales emocionales en mensajes breves. Los objetivos específicos son preparar datos agrupados, comparar modelos usando validación y evaluar el ganador con un test reservado.');
 text(s,'Desarrollar un chatbot local',64,190,1100,55,41,C.green,{bold:true,name:'reveal-objetivo'});
 text(s,'que clasifique señales emocionales en texto',64,245,1100,53,34,C.ink);
 lines(s,['Preparar datos y separar escenarios.','Entrenar y comparar modelos clásicos.','Evaluar los errores y explicar los límites.'],68,360,1100,29,C.ink,65,'objetivos');
 foot(s,'Las respuestas son redactadas y el sistema no es una herramienta clínica.');
}
// 03 · dataset
{
 const s=page('Los datos son sintéticos y están agrupados','Todo el corpus es sintético y fue redactado con asistencia de IA. Hay 400 mensajes de 200 escenarios: dos paráfrasis por escenario. Cinco clases balanceadas. No hay testimonios, anotadores independientes ni validación clínica. Las etiquetas describen la intención de redacción, no una emoción medida de una persona.');
 text(s,'400',64,155,330,120,100,C.green,{bold:true,name:'reveal-main'});text(s,'mensajes sintéticos',69,275,380,45,31,C.ink);
 lines(s,['200 escenarios','2 frases por escenario','5 clases balanceadas'],69,354,370,26,C.muted,49,'datos');
 rect(s,500,190,710,310,C.white,C.line,'example');text(s,'UN ESCENARIO, DOS PARÁFRASIS',535,225,620,30,19,C.green,{bold:true});
 text(s,'«Estoy feliz porque aprobé\nel examen»',535,282,640,85,27,C.ink,{name:'reveal-example-a'});
 text(s,'«Pasé la evaluación y estoy\nmuy contenta»',535,389,640,85,27,C.muted,{name:'reveal-example-b'});
 foot(s,'Una paráfrasis nunca se separa de su escenario.');
}
// 04 · grupos
{
 const s=page('Separar por escenario reduce la fuga','La función preparar agrupa por clase, ordena, baraja con Random(2026), divide los escenarios por clase en 60/20/20 y asigna cada frase a la partición de su grupo. Dos paráfrasis no cruzan particiones. El código se presenta abreviado de forma fiel para que se pueda leer; los identificadores y etiquetas nunca son predictores. La agrupación no elimina similitudes de vocabulario entre distintos escenarios.');
 code(s,["rng = random.Random(SEMILLA)","for clase in CLASES:","    grupos = sorted({...})","    rng.shuffle(grupos)","    # 60 % train, 20 % validación,","    # 20 % test, siempre por grupo"],64,205,690,25,44,'entrenamiento/p01_datos.py · preparar');
 text(s,'240',850,199,300,80,67,C.green,{bold:true,name:'reveal-train'});text(s,'train',855,277,220,34,25,C.muted);
 text(s,'80',850,340,300,77,61,C.ink,{bold:true,name:'reveal-val'});text(s,'validación',855,414,250,34,25,C.muted);
 text(s,'80',850,473,300,77,61,C.ink,{bold:true,name:'reveal-test'});text(s,'test reservado',855,548,300,34,25,C.muted);
 foot(s,'Fragmento resumido del código. La semilla fija la división para hacerla reproducible.');
}
// 05 · Normalizar
{
 const s=page('El preprocesamiento conserva las negaciones','normalizar usa NFKC y casefold, NFD, elimina marcas Unicode Mn y extrae secuencias a-z. El vectorizador después toma tokens de dos o más letras. Se retiran tildes, signos, números y emojis; ñ se convierte en n. No se eliminan no, sin o nunca. Conservarlas no garantiza que el modelo comprenda la negación.');
 code(s,["texto = unicodedata.normalize(","    'NFKC', texto).casefold()","texto = ''.join(c for c in","    unicodedata.normalize('NFD', texto)","    if unicodedata.category(c) != 'Mn')","return ' '.join(re.findall(r'[a-z]+', texto))"],64,203,1040,24,43,'entrenamiento/p01_datos.py · normalizar');
 text(s,'«¡NO estoy FELÍZ!»',70,520,530,48,32,C.ink,{name:'reveal-before'});text(s,'«no estoy feliz»',700,520,480,48,32,C.green,{bold:true,name:'reveal-after'});
 foot(s,'Ejemplo de normalización. El orden y el contexto siguen siendo difíciles para el modelo.');
}
// 06 · TF-IDF
{
 const s=page('TF-IDF transforma palabras en números','Se conserva el Pipeline que aprende IDF y vocabulario únicamente de train. El candidato ganador usa unigramas y bigramas; token_pattern acepta palabras de dos o más letras; sublinear_tf=True y normalización L2 por defecto. Ejemplo pedagógico no corresponde a una predicción real. Hay 1651 características aprendidas en el ganador. No usa embeddings ni modelos preentrenados.');
 code(s,["TfidfVectorizer(","    preprocessor=normalizar,","    ngram_range=(1, 2),","    sublinear_tf=True)"] ,64,216,715,26,46,'entrenamiento/p02_modelos.py · extracto');
 text(s,'«no estoy feliz»',867,207,350,48,27,C.ink,{name:'reveal-sentence'});
 lines(s,['no','estoy','feliz','no estoy','estoy feliz'],869,282,340,24,C.green,42,'tokens');
 text(s,'1651',64,479,340,76,65,C.green,{bold:true,name:'reveal-vocab'});text(s,'características del ganador',380,507,500,38,25,C.muted);
 foot(s,'TF-IDF pondera la frecuencia del término y su frecuencia en los documentos de entrenamiento.');
}
// 07 · Pipeline
{
 const s=page('Pipeline une TF-IDF y regresión logística','crear_modelo devuelve Pipeline con dos pasos; fit en p03_comparacion ajusta ambos exclusivamente con train. El clasificador es logística multinomial con C=4 y max_iter=2000. C regula de forma inversa la fuerza de regularización, no es el número de clases. No se usa test para ajustar el vocabulario, el IDF ni los pesos. Código condensado y equivalente.');
 code(s,["clasificador = LogisticRegression(","    C=4.0, max_iter=2000,","    random_state=SEMILLA)","modelo = Pipeline([", "    ('vectorizador', vectorizador),", "    ('clasificador', clasificador)","])"],64,191,1000,25,43,'entrenamiento/p02_modelos.py · extracto');
 text(s,'modelo.fit(x_train, y_train)',70,546,900,43,29,C.green,{mono:true,bold:true,name:'reveal-fit'});
 foot(s,'fit aprende la transformación y los pesos. Al predecir, transform reutiliza lo aprendido.');
}
// 08 · Comparacion
{
 const s=page('La validación elige el modelo ganador','Tabla de seleccion.json, F1 macro de validación. Todos los candidatos se ajustan con las 240 frases de train y se comparan sobre las mismas 80 frases de validación. Orden de desempate predeclarado. Gana logística TF-IDF de unigramas y bigramas, con 0.786632 sin redondear. La referencia de clase más frecuente obtiene macro-F1 0.0667. Comparación de una sola partición; no prueba superioridad universal.');
 table(s,[['Candidato','Macro-F1 validación'],['Logística · TF-IDF', '0,773'],['Logística · conteos','0,745'],['Logística · TF-IDF + bigramas','0,787'],['Naive Bayes · TF-IDF','0,698'],['Referencia · clase frecuente','0,067']],64,160,1152,374,[800,352],{size:25,highlight:3});
 text(s,'Ganador: TF-IDF con unigramas y bigramas',64,556,1140,48,32,C.green,{bold:true,name:'reveal-winner'});
 foot(s,'Criterio de selección: macro-F1. El conjunto test sigue cerrado durante esta comparación.');
}
// 09 · Abstencion
{
 const s=page('El sistema puede pedir más contexto','El umbral de score es 0.30, elegido en validación por la regla de máxima cobertura con accuracy selectiva de al menos 0.80 y cobertura al menos 0.50. También se requiere margen mínimo de 0.10 y una característica conocida. En validación a 0.30 cobertura 58/80=0.725 y exactitud de aceptados 49/58=0.845. Los scores no están calibrados. Incierto no es una clase de entrenamiento.');
 code(s,["aceptar = (score_max >= umbral)","aceptar &= (score_max - segundo >= 0.10)","aceptar &= vector_tiene_palabras"],64,214,1080,25,47,'entrenamiento/p02_modelos.py · lógica equivalente');
 text(s,'Umbral elegido: 0,30',64,439,660,60,39,C.green,{bold:true,name:'reveal-umbral'});
 text(s,'Si la señal es débil, el chat pregunta.',64,518,1120,47,31,C.ink,{name:'reveal-contexto'});
 foot(s,'Regla heurística elegida en validación. Las puntuaciones no son probabilidades clínicas calibradas.');
}
// 10 · Resultados
{
 const s=page('El test revela aciertos y fallos por clase','Test único de 80 mensajes sintéticos: accuracy 0.6625, macro-F1 0.6530819642. Matriz en orden alegria, ansiedad, enojo, neutral, tristeza. Cada clase tiene 16 ejemplos. Alegría solo 4 de 16: recall 0.25. Se guardaron 27 errores, 16 aceptados por el filtro. La cobertura de test fue 60/80 = 0.75 y 44/60 = 0.733 entre aceptados. El test no debe usarse para modificar el modelo y volver a medir.');
 const l=['Alegría','Ansiedad','Enojo','Neutral','Tristeza'],m=[[4,3,4,1,4],[1,12,2,0,1],[1,0,14,0,1],[3,0,0,12,1],[3,2,0,0,11]];
 table(s,[['Real / Pred.',...l],...m.map((row,i)=>[l[i],...row.map(String)])],64,192,810,318,[210,120,120,120,120,120],{matrix:true,size:21});
 text(s,'66,25 %',921,206,295,75,54,C.green,{bold:true,name:'reveal-accuracy'});text(s,'accuracy de test',922,280,295,37,23,C.muted);
 text(s,'0,653',921,349,295,75,54,C.ink,{bold:true,name:'reveal-f1'});text(s,'macro-F1 de test',922,423,295,37,23,C.muted);
 text(s,'4 / 16',921,487,295,56,41,C.red,{bold:true,name:'reveal-error'});text(s,'aciertos en alegría',922,544,295,37,22,C.muted);
 foot(s,'Filas: clase real. Columnas: predicción. Corpus sintético; no mide rendimiento clínico.');
}
// 11 · Inferencia
{
 const s=page('El chat reutiliza el modelo entrenado','predecir transforma el texto con el vectorizador entrenado y usa predict_proba en el clasificador. Extrae orden de puntuaciones, aplica mascara_aceptacion, calcula contribuciones y construye JSON. No llama fit ni modifica parámetros. Las contribuciones son x_j*(peso candidata - peso alternativa), solo se muestran hasta cinco positivas; no son causas ni explicación completa.');
 code(s,["vector = (pipeline['vectorizador']", "    .transform([texto]))", "scores = (pipeline['clasificador']", "    .predict_proba(vector)[0])", "aceptada = mascara_aceptacion(","    scores.reshape(1, -1),", "    artefacto['umbral'], [vector.nnz > 0])"],64,200,1000,25,43,'entrenamiento/p05_inferencia.py · extracto');
 text(s,'Sin volver a entrenar',64,550,1100,47,33,C.green,{bold:true,name:'reveal-main'});
 foot(s,'Fragmento reformateado para lectura. Cada mensaje se analiza de manera independiente.');
}
// 12 · Seguridad
{
 const s=page('Las reglas de precaución tienen prioridad','Código de app.py: si detectar_riesgo o safety_context es verdadero, se devuelve la respuesta de apoyo y analysis=None antes de consultar ML. El cuerpo HTTP tiene límite de 32768 bytes, por encima se rechaza antes de analizar. Hay falsos positivos por negaciones o referencias informativas, y falsos negativos por eufemismos o errores. La línea 113 opción 5 fue requisito del proyecto; servicio de apoyo de Perú. No es validación clínica ni certificación de seguridad.');
 code(s,["if (detectar_riesgo(texto) or", "        contexto_seguridad):","    return jsonify(","        reply=RESPUESTA_RIESGO,", "        safety=True, analysis=None)"],64,210,1055,25,46,'app.py · chat');
 text(s,'Línea 113, opción 5 · Perú',64,500,1110,48,33,C.green,{bold:true,name:'reveal-main'});
 text(s,'Las reglas pueden fallar y no evalúan seguridad clínica.',64,558,1110,38,25,C.muted);
 foot(s,'La precaución se procesa antes de la clasificación emocional.');
}
// 13 · pruebas y ejecucion
{
 const s=page('La aplicación sirve la interfaz desde Flask','La página index.html se sirve desde Flask en /. La interfaz llama a /api/chat con fetch y JSON. app.py usa make_server y prueba puertos 5000 a 5010, imprime dirección reservada. En la máquina del usuario, ControlCenter compartía el 5000 y se usó temporalmente 5001 para demostración; abrir la dirección exacta impresa. La última batería registrada incluye 57 tests de software aprobados; los tests no son el test estadístico. No se ejecuta evaluación del modelo al entrar a la aplicación.');
 code(s,["source .venv/bin/activate", "python app.py", "", "# Abrir la dirección impresa"],64,220,1050,27,52,'Ejecución local');
 text(s,'57',67,478,215,85,70,C.green,{bold:true,name:'reveal-tests'});text(s,'pruebas de software aprobadas',290,512,800,45,28,C.ink);
 foot(s,'Flask sirve index.html en /. El chat consulta la API mediante rutas relativas.');
}
// 14 · conclusion
{
 const s=page('Conclusiones y siguiente paso','Se cumplió el flujo académico completo: datos sintéticos, división por grupos, Pipeline, comparación con validación, test reservado y análisis de errores. No se afirma eficacia con población real. Limitación clave: alegría solo 4/16 en test y sensibilidad a negaciones. Siguiente estudio: mensajes de personas con consentimiento, anotadores independientes, cobertura dialectal y un test externo nuevo, reservado antes de cualquier ajuste. La maestra puede pedir explicar la diferencia fit/transform, el agrupamiento y la matriz.');
 text(s,'Un flujo completo y explicable',64,170,1130,60,43,C.green,{bold:true,name:'reveal-conclusion'});
 lines(s,['El modelo aprende patrones de palabras.','La evaluación final muestra límites reales.','Las reglas de precaución requieren revisión.'],68,270,1120,30,C.ink,74,'conclusion');
 text(s,'Siguiente paso',64,524,300,37,20,C.green,{bold:true});text(s,'datos humanos consentidos y test externo nuevo',350,520,855,40,26,C.ink);
 foot(s,'Resultados del experimento sintético. No representan diagnóstico ni rendimiento poblacional.');
}
await fs.writeFile(path.join(B,'plan.json'),JSON.stringify(records,null,2));await fs.writeFile(path.join(B,'text-width-check.json'),JSON.stringify(check,null,2));
if(check.length)console.log('REVISAR_ANCHO',JSON.stringify(check));
for(let i=0;i<deck.slides.items.length;i++){const png=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(B,'render',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));console.log('Render',i+1);}
await (await PresentationFile.exportPptx(deck)).save(path.join(B,'draft.pptx'));
console.log('Exportado',deck.slides.items.length,'diapositivas');
