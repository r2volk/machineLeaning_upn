import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
import { createCanvas } from '@napi-rs/canvas';
const ROOT=process.cwd(), B=path.join(ROOT,'.ppt-academica-v2');
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
 if(records.length>1){const categorias=['','Antecedentes','Planteamiento del problema','Objetivos','Herramientas de Ingeniería',...Array(7).fill('Algoritmos'),...Array(4).fill('Materiales y Métodos utilizados')]; text(s,categorias[records.length-1],64,46,1080,62,42,C.ink,{bold:true,name:'slide-title'});text(s,title,65,112,1100,32,22,C.green,{name:'slide-subtitle'});text(s,`${String(records.length).padStart(2,'0')} / 16`,1100,677,116,20,16,C.muted,{align:'right',name:'page-number'});}
 s.speakerNotes.text=notes+'\n\nFuentes locales: entrenamiento/p01_datos.py, p02_modelos.py, p03_comparacion.py, p04_evaluacion.py, p05_inferencia.py; app.py; seguridad.py; entrenamiento/resultados/seleccion.json y test_final.json; INFORME_RESULTADOS.md; explicacion-completa-del-proyecto-animo.md. En clase: semanas 9, 11 y 12. Las cifras corresponden al experimento de la entrega, con datos 100 % sintéticos.';
 return s;
}
function lines(s,arr,x,y,w=500,size=27,color=C.muted,step=47,name='body'){
 arr.forEach((v,i)=>text(s,v,x,y+i*step,w,step,size,color,{name:`reveal-${name}-${i}`}));
}
function code(s,arr,x,y,w=700,size=23,step=39,filename=''){
 if(filename)text(s,filename,x,y-42,w,27,18,C.green,{mono:true});
 rect(s,x-20,y-14,w+38,arr.length*step+26,'#282C34','none','code-surface');
 const keys=new Set(['if','else','elif','return','for','in','and','or','not','from','import','as','True','False','None']);
 arr.forEach((line,i)=>{
  const tokens=line.match(/#[^\n]*|"[^"\n]*"|'[^'\n]*'|\b[A-Za-z_][A-Za-z_0-9]*\b|\b\d+(?:\.\d+)?\b|[^A-Za-z_0-9]/g)||[];
  let offset=0;
  for(let j=0;j<tokens.length;j++){
   const t=tokens[j];let color='#ABB2BF';
   if(t.startsWith('#'))color='#7F848E';
   else if(t.startsWith('"')||t.startsWith("'"))color='#98C379';
   else if(keys.has(t))color='#C678DD';
   else if(/^\d/.test(t))color='#D19A66';
   else if(/^[A-Z]/.test(t))color='#E5C07B';
   else if(/^[a-z_]/.test(t)&&tokens[j+1]==='(')color='#61AFEF';
   ctx.font=`${size}px "${MONO}"`;const tw=ctx.measureText(t).width;
   if(t.trim())text(s,t,x+offset,y+i*step,tw+3,step,size,color,{mono:true,name:`code-${i}-${j}`});
   offset+=tw;
  }
  if(offset>w)check.push({slide:records.length,text:line,width:offset,available:w});
 });
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
function aside(s,title,body){text(s,title,820,202,390,45,28,C.green,{bold:true});lines(s,body,820,267,390,23,C.muted,43,'explain');}
// 1 portada
{
 const s=page('Ánimo','Proyecto académico de clasificación de señales emocionales. Integrantes proporcionados por el equipo. La ilustración es una imagen generada y no representa una persona real.');
 const bytes=await fs.readFile(path.join(B,'assets/portada.png'));
 s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:'Ilustración de conversación',position:{left:0,top:0,width:1280,height:720},fit:'cover'});
 text(s,'PROYECTO DE MACHINE LEARNING',66,65,900,32,20,C.green,{bold:true});
 text(s,'Ánimo',60,123,720,115,88,C.ink,{bold:true,name:'reveal-cover'});
 text(s,'Clasificación de señales emocionales\nen mensajes escritos',65,250,900,100,35,C.ink);
 text(s,'INTEGRANTES',67,390,750,28,18,C.green,{bold:true});
 lines(s,['Sanchez Laurencio, Ricardo Markos','Becerra Llamo, Jaime David','Gutierrez De La Cruz, Joaquin Elias','Stammer Tello, Percy Aaron','Mayo Samaritano, Diego Valentino'],67,433,850,25,C.ink,39,'equipo');
}
// 2 antecedentes
{
 const s=page('Estudios previos sobre emociones y estrés en texto','GoEmotions: Demszky et al. (2020), https://aclanthology.org/2020.acl-main.372/ . Presenta un corpus de comentarios de Reddit en inglés con anotaciones emocionales. Dreaddit: Turcan y McKeown (2019), https://aclanthology.org/D19-6213/ . Estudia clasificación de estrés en textos de redes sociales. Son antecedentes metodológicos; Ánimo no usa sus datos ni sus modelos y estos trabajos no validan rendimiento en universitarios peruanos.');
 text(s,'GoEmotions',65,191,530,48,36,C.green,{bold:true});
 lines(s,['Demszky y colaboradores, 2020','Etiquetas emocionales en comentarios','de Reddit escritos en inglés.','Aporte: clasificar emociones desde texto.'],65,256,570,25,C.ink,44,'goemotions');
 text(s,'Dreaddit',685,191,530,48,36,C.green,{bold:true});
 lines(s,['Turcan y McKeown, 2019','Análisis de señales de estrés','en textos de redes sociales.','Aporte: estudiar lenguaje y estrés.'],685,256,530,25,C.ink,44,'dreaddit');
 text(s,'Relación con Ánimo',65,494,1100,42,28,C.green,{bold:true});
 text(s,'Estos estudios motivan el enfoque. Ánimo usa su propio corpus sintético en español.',65,549,1130,39,25,C.ink);
 foot(s,'No se reutilizan sus datos ni modelos. No prueban eficacia en estudiantes universitarios.');
}
// 3 problema
{
 const s=page('Una dificultad de identificación y respuesta en el contexto universitario','Planteamiento propuesto por el equipo. La dificultad para identificar oportunamente emociones negativas en estudiantes universitarios a través de mensajes escritos limita la posibilidad de ofrecer respuestas de apoyo emocional adecuadas. Es la motivación de diseño, no una prevalencia medida ni un problema cuya solución real se haya validado con este corpus sintético.');
 lines(s,['La dificultad para identificar oportunamente emociones negativas','en estudiantes universitarios a través de mensajes escritos','limita la posibilidad de ofrecer respuestas','de apoyo emocional adecuadas.'],66,196,1140,32,C.ink,54,'problema');
 text(s,'Propuesta del proyecto',66,451,1120,42,28,C.green,{bold:true});
 lines(s,['Clasificar señales del mensaje y ofrecer una respuesta inicial redactada.','Pedir más contexto cuando la predicción sea incierta.'],66,506,1140,25,C.muted,45,'propuesta');
 foot(s,'Motivación del prototipo. No se ha medido detección oportuna ni beneficio en estudiantes reales.');
}
// 4 objetivos
{
 const s=page('Objetivo general y objetivos específicos','Objetivo general: desarrollar un prototipo local para clasificar señales emocionales en mensajes escritos en español y brindar respuestas iniciales redactadas. El contexto de uso propuesto es universitario, pero la evaluación se limita al corpus sintético. Objetivos específicos: preparar y dividir datos por escenarios, comparar clasificadores con validación y evaluar honestamente al ganador.');
 text(s,'Objetivo general',65,180,1120,40,29,C.green,{bold:true});
 lines(s,['Desarrollar un chatbot que clasifique señales emocionales','en mensajes en español y ofrezca respuestas iniciales de apoyo.'],65,233,1140,28,C.ink,46,'general');
 text(s,'Objetivos específicos',65,368,1120,40,29,C.green,{bold:true});
 lines(s,['Preparar datos etiquetados y evitar fuga entre escenarios.','Comparar modelos clásicos y seleccionar con validación.','Evaluar errores, abstención y límites del prototipo.'],65,423,1140,27,C.ink,55,'especificos');
 foot(s,'La seguridad se atiende con reglas de precaución independientes del clasificador.');
}
// 5 herramientas
{
 const s=page('Software de desarrollo, aprendizaje y verificación','Python es el lenguaje; scikit-learn implementa Pipeline, vectorizadores, clasificadores y métricas. NumPy y SciPy apoyan operaciones numéricas y matrices dispersas. Matplotlib genera figuras; joblib guarda el artefacto. Flask sirve la aplicación; pytest verifica software. VS Code y Git apoyan el desarrollo; GitHub aloja el repositorio. No se consultan APIs de modelos externos.');
 table(s,[['Herramienta','Uso en el proyecto'],['Python y VS Code','Lenguaje y edición del código'],['scikit-learn','Vectorización, modelos, Pipeline y métricas'],['NumPy y SciPy','Arreglos y representación numérica dispersa'],['Matplotlib y joblib','Gráficos y guardado del modelo'],['Flask y pytest','Aplicación local y pruebas de software'],['Git y GitHub','Versiones y repositorio del proyecto']],64,184,1152,380,[360,792],{size:23});
 foot(s,'Entorno aislado con .venv y versiones fijadas en requirements.txt.');
}
// 6 normalizacion
{
 const s=page('Normalización del texto','Fragmento resumido y equivalente de p01_datos.normalizar. Unicode normaliza caracteres, casefold reduce mayúsculas, NFD separa acentos y la expresión regular extrae letras. Se conservan negaciones, pero se pierden emojis y la distinción entre ñ y n.');
 code(s,["texto = unicodedata.normalize(","    'NFKC', texto).casefold()","texto = ''.join(c for c in","    unicodedata.normalize('NFD', texto)","    if unicodedata.category(c) != 'Mn')","return ' '.join(re.findall(r'[a-z]+', texto))"],64,205,718,22,43,'p01_datos.py  normalizar');
 aside(s,'Una entrada consistente',['Unifica mayúsculas.','Retira acentos y signos.','Conserva la palabra no.','También pierde matices.']);
 text(s,'¡NO estoy FELÍZ!   →   no estoy feliz',65,531,1120,51,31,C.green,{bold:true});
 foot(s,'La misma limpieza se aplica al entrenamiento y a los mensajes nuevos.');
}
// 7 tfidf
{
 const s=page('TF-IDF con unigramas y bigramas','Extracto del candidato ganador en p02_modelos. Las opciones compartidas incluyen normalizar y tokenización de al menos dos letras. sublinear_tf suaviza frecuencias y el vectorizador aplica IDF suavizado y normalización L2. Solo fit de train aprende las 1651 características.');
 code(s,["vectorizador = TfidfVectorizer(","    preprocessor=normalizar,","    lowercase=False,","    ngram_range=(1, 2),","    sublinear_tf=True)","# fit aprende solo con train"],64,204,710,24,43,'p02_modelos.py  extracto del ganador');
 aside(s,'Palabras a números',['TF pondera la frecuencia.','IDF usa documentos de train.','Bigramas añaden orden local.','Se aprende el vocabulario.']);
 text(s,'1651 características aprendidas en el ganador',65,532,1140,52,32,C.green,{bold:true});
 foot(s,'Un mensaje nuevo se transforma con ese vocabulario, sin añadir nuevas columnas.');
}
// 8 logistica
{
 const s=page('Regresión logística y Pipeline','Regresión logística multinomial para cinco clases, con regularización C=4 y máximo de 2000 iteraciones. El Pipeline aprende representación y pesos usando solo train. Fit es ajuste; predict usa el ajuste existente. Los parámetros mostrados se fijaron antes de test.');
 code(s,["clasificador = LogisticRegression(","    C=4.0, max_iter=2000,","    random_state=SEMILLA)","modelo = Pipeline([","    ('vectorizador', vectorizador),","    ('clasificador', clasificador)])","modelo.fit(x_train, y_train)"],64,204,710,23,42,'p02_modelos.py y p03_comparacion.py');
 aside(s,'Qué aprende',['Un peso por característica','para distinguir cada clase.','C controla regularización.','Pipeline evita fuga del IDF.']);
 foot(s,'El modelo servido conserva el ajuste de train. No se reajusta después del test.');
}
// 9 bayes
{
 const s=page('Naive Bayes y referencia constante','Extractos de p02_modelos. MultinomialNB usa alpha=1.0 y entradas TF-IDF no negativas. La independencia condicional es una aproximación; no se trata de conteos multinomiales puros. DummyClassifier most_frequent establece una referencia constante. Se comparan sobre las mismas particiones que la logística.');
 code(s,["# otro clasificador aprendido","clasificador = MultinomialNB(","    alpha=1.0)","","# referencia sin analizar palabras","clasificador = DummyClassifier(","    strategy='most_frequent')"],64,204,710,23,42,'p02_modelos.py  candidatos alternativos');
 aside(s,'Por qué compararlos',['Bayes ofrece otra familia','de clasificación sencilla.','La referencia comprueba','si aprender texto aporta.']);
 foot(s,'La comparación incluye tres variantes de logística, Bayes y una referencia constante.');
}
// 10 comparacion
{
 const s=page('Selección del modelo con validación','Cifras guardadas en seleccion.json. Se entrenan cinco candidatos en las mismas 240 frases de train y se comparan en 80 frases de validación. El criterio es macro-F1 sin redondear; en empate se conserva el primero del orden predefinido. El test no decide el ganador.');
 table(s,[['Candidato','Macro-F1 validación'],['Logística con TF-IDF','0,773'],['Logística con conteos','0,745'],['Logística con TF-IDF y bigramas','0,787'],['Naive Bayes con TF-IDF','0,698'],['Referencia constante','0,067']],64,179,1152,353,[800,352],{size:24,highlight:3});
 text(s,'Ganador: logística con unigramas y bigramas',65,558,1130,48,31,C.green,{bold:true});
 foot(s,'Macro-F1 da el mismo peso a las cinco clases. Resultados de una sola partición.');
}
// 11 abstencion
{
 const s=page('Abstención ante señales insuficientes','Lógica equivalente a mascara_aceptacion. Umbral 0.30 seleccionado en validación, diferencia mínima 0.10 y al menos una característica conocida. El criterio de selección busca cobertura al menos .50 y exactitud entre aceptados al menos .80 en validación. No garantiza esas cifras en test; incierto no es una sexta clase.');
 code(s,["aceptar = (score_max >= umbral)","aceptar &= (score_max - segundo >= 0.10)","aceptar &= vector_tiene_palabras","","# si falla alguna condicion","etiqueta = clase if aceptar else 'incierto'"],64,208,712,22,44,'p02_modelos.py  lógica equivalente');
 aside(s,'Pedir más contexto',['Umbral elegido: 0,30.','Margen mínimo: 0,10.','Exige vocabulario conocido.','Puede rechazar aciertos.']);
 foot(s,'Los scores son puntuaciones relativas del modelo, sin calibración clínica.');
}
// 12 inferencia seguridad
{
 const s=page('Inferencia y prioridad de precaución','Resumen equivalente del flujo de app.py y p05_inferencia.py. Primero se atienden las reglas o el indicador de una alerta previa, sin consultar ML. De lo contrario se transforma el mensaje y se obtienen scores; después se aplican abstención y aportes. Las reglas pueden fallar y no determinan si una persona está a salvo.');
 code(s,["if detectar_riesgo(texto) or contexto_seguridad:","    return jsonify(reply=RESPUESTA_RIESGO,","                   safety=True, analysis=None)","vector = pipeline['vectorizador'].transform([texto])","scores = pipeline['clasificador'].predict_proba(","    vector)[0]"],64,205,716,21,45,'app.py y p05_inferencia.py  extractos');
 aside(s,'Usar lo aprendido',['No vuelve a entrenar.','Analiza un mensaje a la vez.','Las reglas van antes de ML.','El apoyo declara límites.']);
 foot(s,'La respuesta de precaución incluye apoyo humano y el recurso peruano del proyecto.');
}
// 13 materiales
{
 const s=page('Equipo y corpus de trabajo','Materiales: equipo local macOS utilizado para ejecutar Python, corpus sintético, archivos de particiones y artefacto guardado. No se inventan modelo de laptop ni capacidad de RAM. Corpus de 400 mensajes en 200 escenarios con dos paráfrasis por escenario, sin participantes reales.');
 text(s,'Material de datos',65,190,545,46,33,C.green,{bold:true});
 lines(s,['400 mensajes sintéticos en español.','200 escenarios, dos frases por grupo.','Cinco categorías balanceadas.','Etiquetas por intención de redacción.'],65,254,565,26,C.ink,51,'material');
 text(s,'Equipo y archivos',685,190,530,46,33,C.green,{bold:true});
 lines(s,['Laptop del equipo y ejecución local.','Entorno Python y dependencias.','CSV, manifiesto y modelo.joblib.','Sin GPU ni APIs de modelos externos.'],685,254,530,26,C.ink,51,'equipo');
 foot(s,'Corpus redactado con asistencia de IA. Sin conversaciones reales ni etiquetas de especialistas.');
}
// 14 metodo
{
 const s=page('Diseño experimental y procedimiento','El método es clasificación supervisada con división por escenario y clase, antes de ajustar representación. Train contiene 240 frases, validación 80 y test 80. Dos paráfrasis nunca cruzan particiones. El marcador de test limita aperturas accidentales, pero no es una prueba contra manipulación. Las figuras reutilizan predicciones.');
 table(s,[['Etapa','Procedimiento'],['Preparación','Normalizar, validar y agrupar escenarios'],['División','240 train, 80 validación y 80 test'],['Entrenamiento','Ajustar cinco candidatos únicamente con train'],['Selección','Elegir modelo y umbral con validación'],['Evaluación','Congelar y ejecutar un lote final de test'],['Análisis','Métricas por clase, errores y cobertura']],64,180,1152,382,[310,842],{size:23});
 foot(s,'Semilla fija. Los grupos evitan fuga entre las dos paráfrasis de un mismo escenario.');
}
// 15 resultados
{
 const s=page('Resultados y análisis de errores','Matriz exacta de test_final.json. Test de 80 mensajes, accuracy 53/80=.6625 y macro-F1=.6530819642. Alegría tuvo 4 aciertos de 16 ejemplos. Hay 27 errores totales, 16 aceptados por el filtro. Cobertura selectiva 60/80=.75 y accuracy de aceptados 44/60=.7333. Estas cifras corresponden al corpus sintético.');
 const l=['Alegría','Ansiedad','Enojo','Neutral','Tristeza'],m=[[4,3,4,1,4],[1,12,2,0,1],[1,0,14,0,1],[3,0,0,12,1],[3,2,0,0,11]];
 table(s,[['Real / Pred.',...l],...m.map((row,i)=>[l[i],...row.map(String)])],64,184,810,331,[210,120,120,120,120,120],{matrix:true,size:21});
 text(s,'66,25 %',920,210,290,74,52,C.green,{bold:true});text(s,'accuracy de test',921,284,290,37,23,C.muted);
 text(s,'0,653',920,359,290,74,52,C.ink,{bold:true});text(s,'macro-F1 de test',921,433,290,37,23,C.muted);
 text(s,'Alegría: 4 aciertos de 16 casos reales',65,554,1140,47,31,C.red,{bold:true});
 foot(s,'Filas reales y columnas predichas. No mide eficacia de apoyo emocional en estudiantes.');
}
// 16 limites
{
 const s=page('Verificación, limitaciones y mejoras','La batería tiene 57 pruebas de software aprobadas y no equivale al test estadístico. Los datos sintéticos limitan la validez externa; no hay estudio con estudiantes reales ni medida del efecto del apoyo. Para mejorar: datos autorizados, anotación independiente, revisión profesional de respuestas y test externo reservado antes de ajustar. No se debe reutilizar el test observado para sostener una nueva evaluación independiente.');
 text(s,'Verificación',65,180,1120,41,31,C.green,{bold:true});
 text(s,'57 pruebas de software. El test del modelo se evalúa por separado.',65,235,1140,41,27,C.ink);
 text(s,'Limitaciones',65,326,1120,41,31,C.green,{bold:true});
 lines(s,['Datos sintéticos, corpus pequeño y errores visibles.','Sin evaluación con estudiantes ni validación clínica.'],65,381,1140,26,C.ink,44,'limites');
 text(s,'Siguiente estudio',65,500,1120,41,31,C.green,{bold:true});
 text(s,'Datos autorizados, revisión de etiquetas y un test externo nuevo.',65,553,1140,42,27,C.ink);
 foot(s,'El prototipo demuestra el método de aprendizaje automático, con sus límites declarados.');
}
await fs.writeFile(path.join(B,'plan.json'),JSON.stringify(records,null,2));await fs.writeFile(path.join(B,'text-width-check.json'),JSON.stringify(check,null,2));
if(check.length)console.log('REVISAR_ANCHO',JSON.stringify(check));
for(let i=0;i<deck.slides.items.length;i++){const png=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(B,'render',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));console.log('Render',i+1);}
await (await PresentationFile.exportPptx(deck)).save(path.join(B,'draft.pptx'));
