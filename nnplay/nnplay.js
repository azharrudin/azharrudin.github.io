const $ = s => document.querySelector(s);
const sleep = ms => new Promise(r => setTimeout(r, ms));
let dataset = null;
let nn = null;
let stopTraining = false;

function setStatus(msg){ $("#status").textContent = msg; }

const fmt = v => Number(v).toFixed(6);
window.MathJax = {
  tex: {inlineMath: [['\\(', '\\)']]},
  chtml: {displayAlign: 'left'}
};
function clearMath(message=""){
  const panel=$("#mathSteps");
  window.MathJax?.typesetClear?.([panel]);
  panel.replaceChildren();
  panel.textContent=message;
}
function mathTex(line){
  return line
    .replace(/∂L\/∂w\[(\d+),(\d+)\]/g, String.raw`\frac{\partial L}{\partial w_{$1,$2}}`)
    .replace(/∂L\/∂w/g, String.raw`\frac{\partial L}{\partial w}`)
    .replace(/∂L\/∂b/g, String.raw`\frac{\partial L}{\partial b}`)
    .replace(/w_next/g, 'w_{\\mathrm{next}}').replace(/δ_next/g, '\\delta_{\\mathrm{next}}')
    .replace(/a_prev/g, 'a_{\\mathrm{prev}}').replace(/w_new/g, 'w_{\\mathrm{new}}').replace(/b_new/g, 'b_{\\mathrm{new}}')
    .replace(/ŷ/g, '\\hat{y}').replace(/δ/g, '\\delta').replace(/η/g, '\\eta')
    .replace(/Σ/g, '\\sum').replace(/×/g, '\\cdot ').replace(/−/g, '-').replace(/²/g, '^{2}')
    .replace(/½/g, '\\frac{1}{2}').replace(/f′/g, "f'")
    .replace(/\bsigmoid\b/g, '\\sigma').replace(/\btanh\b/g, '\\tanh')
    .replace(/\brelu\b/g, '\\operatorname{ReLU}').replace(/\bmax\b/g, '\\max')
    .replace(/\bexp\b/g, '\\exp').replace(/\bb\b/g, 'b');
}

function valueNumber(value){
  return value!==null && value!==undefined && Number.isFinite(Number(value)) ? fmt(value) : "—";
}
function valueVector(values,labels=[]){
  if(!Array.isArray(values)) return "—";
  const limit=10;
  const items=values.slice(0,limit).map((value,index)=>
    `${labels[index] || index+1}=${valueNumber(value)}`);
  if(values.length>limit) items.push(`… +${values.length-limit} nilai`);
  return items.join("\n");
}
function valueMatrix(matrix){
  if(!Array.isArray(matrix)) return "—";
  const rowLimit=6, columnLimit=8;
  const rows=matrix.slice(0,rowLimit).map((row,index)=>{
    const values=row.slice(0,columnLimit).map(valueNumber).join(", ");
    return `neuron ${index+1}: [${values}${row.length>columnLimit?", …":""}]`;
  });
  if(matrix.length>rowLimit) rows.push(`… +${matrix.length-rowLimit} neuron`);
  return rows.join("\n");
}
function valueLayers(layers){
  if(!Array.isArray(layers)) return "—";
  return layers.map((layer,index)=>`Layer ${index+1}\n${Array.isArray(layer[0])?valueMatrix(layer):valueVector(layer)}`).join("\n");
}
function weightRows(matrix,layer,previousLabels=[],source="Parameter network saat ini",variant=""){
  if(!Array.isArray(matrix)) return [];
  return matrix.flatMap((row,j)=>row.map((value,i)=>{
    const from=previousLabels[i]
      ? `fitur “${previousLabels[i]}” (a${i+1} layer ${layer-1})`
      : `neuron ${i+1} layer ${layer-1}`;
    const symbol=variant
      ? String.raw`\(w_{\mathrm{${variant}},${j+1},${i+1}}^{(${layer})}\)`
      : String.raw`\(w_{${j+1},${i+1}}^{(${layer})}\)`;
    return [symbol,valueNumber(value),`${source}; bobot koneksi dari ${from} menuju neuron ${j+1} layer ${layer}.`];
  }));
}
function biasRows(values,layer,source="Parameter network saat ini",variant=""){
  if(!Array.isArray(values)) return [];
  return values.map((value,j)=>{
    const symbol=variant
      ? String.raw`\(b_{\mathrm{${variant}},${j+1}}^{(${layer})}\)`
      : String.raw`\(b_{${j+1}}^{(${layer})}\)`;
    return [symbol,valueNumber(value),`${source}; bias milik neuron ${j+1} layer ${layer}, bukan berasal dari neuron layer sebelumnya.`];
  });
}
function networkParameterRows(weights,biases,featureNames,source,variant=""){
  if(!Array.isArray(weights) || !Array.isArray(biases)) return [];
  return weights.flatMap((matrix,index)=>[
    ...weightRows(matrix,index+1,index===0?featureNames:[],source,variant),
    ...biasRows(biases[index],index+1,source,variant)
  ]);
}

function variableGuide(title,context={}){
  if(title.startsWith("Training · Row") || title==="Predict · Input User"){
    const source=context.inputSource || (title.startsWith("Training") ? "baris CSV yang dipilih" : "form Predict · Input User");
    return {
      intro:"Tahap ini mengubah input asli ke skala 0–1 sebelum masuk ke network.",
      variables:[
        [String.raw`\(x_i\)`, valueVector(context.raw,context.featureNames), `Nilai fitur ke-i dari ${source}.`],
        [String.raw`\(\min_i\)`, valueVector(context.mins,context.minLabels || context.featureNames), "Nilai minimum tiap fitur dari seluruh baris CSV training saat data diparse."],
        [String.raw`\(\max_i\)`, valueVector(context.maxs,context.maxLabels || context.featureNames), "Nilai maksimum tiap fitur dari seluruh baris CSV training saat data diparse."],
        [String.raw`\(a_i^{(0)}\)`, valueVector(context.normalized,context.featureNames), "Hasil normalisasi min–max dari xᵢ. Nilai ini menjadi input layer pertama."],
        [String.raw`\(i\)`, (context.featureNames || []).map((name,index)=>`${index+1} = ${name}`).join("\n") || "—", "Indeks fitur mengikuti urutan kolom CSV selain kolom target."]
      ]
    };
  }

  if(title.startsWith("Forward · Layer")){
    const layer=Number(title.match(/Layer (\d+)/)?.[1] || 1);
    return {
      intro:`Tahap forward layer ${layer} menghitung weighted sum, lalu melewatkannya ke fungsi aktivasi.`,
      variables:[
        [String.raw`\(a_i^{(${layer-1})}\)`, valueVector(context.aPrev,context.previousLabels), layer===1 ? "Input hasil normalisasi CSV/input user." : `Nilai a dari hasil forward layer ${layer-1}.`],
        ...weightRows(context.weights,layer,context.previousLabels,context.parameterSource),
        ...biasRows(context.biases,layer,context.parameterSource),
        [String.raw`\(z_j^{(${layer})}\)`, valueVector(context.z), "Hasil Σ(w × a sebelumnya) + b pada proses ini."],
        [String.raw`\(f\)`, context.activation || "—", layer===context.outputLayer ? "Pilihan Activation output di sidebar." : "Pilihan Activation hidden di sidebar."],
        [String.raw`\(a_j^{(${layer})}\)`, valueVector(context.a), layer===context.outputLayer ? "Hasil f(z) dan menjadi prediksi akhir network." : `Hasil f(z); nilai ini menjadi input layer ${layer+1}.`]
      ]
    };
  }

  if(title==="Loss"){
    return {
      intro:"Loss mengukur jarak antara prediksi network dan target pada sample yang dipilih.",
      variables:[
        [String.raw`\(\hat y\)`, valueVector(context.pred), "Aktivasi neuron output dari forward propagation."],
        [String.raw`\(y\)`, valueVector(context.target), `Nilai kolom “${context.targetName || "target"}” dari ${context.targetSource || "baris CSV yang dipilih"}.`],
        [String.raw`\(L\)`, valueNumber(context.loss), "Hasil ½(ŷ − y)² untuk sample ini."]
      ]
    };
  }

  if(title.startsWith("Backward · Loss awal")){
    return {
      intro:"Nilai awal ini menjadi dasar perhitungan gradien sebelum bobot diperbarui.",
      variables:[
        [String.raw`\(\hat y\)`, valueVector(context.pred), "Prediksi terbaru dari forward propagation untuk sample yang dipilih."],
        [String.raw`\(y\)`, valueVector(context.target), `Nilai kolom “${context.targetName || "target"}” dari ${context.targetSource || "baris CSV yang dipilih"}.`],
        [String.raw`\(L\)`, valueNumber(context.loss), "Loss sample sebelum update bobot dan bias."],
        [String.raw`\(\eta\)`, valueNumber(context.lr), "Nilai kontrol Learning rate di sidebar."]
      ]
    };
  }

  if(title.startsWith("Backward · Layer")){
    const layer=Number(title.match(/Layer (\d+)/)?.[1] || 1);
    const outputLayer=layer===context.outputLayer;
    const variables=[
      [String.raw`\(f'(z)\)`, valueVector(context.derivatives), `Turunan aktivasi ${context.activation || ""} menggunakan z dan a yang tersimpan saat forward.`],
      [String.raw`\(\delta_j^{(${layer})}\)`, valueVector(context.delta), outputLayer ? "Dari (ŷ − y) × f′(z)." : "Dari jumlah w layer berikutnya × delta layer berikutnya, lalu dikali f′(z)."],
      [String.raw`\(a_i^{(${layer-1})}\)`, valueVector(context.aPrev,context.previousLabels), layer===1 ? "Input network hasil normalisasi." : `Aktivasi hasil forward layer ${layer-1}.`],
      [String.raw`\(\partial L/\partial w\)`, valueMatrix(context.gradsW), "Dari δ × aktivasi layer sebelumnya."],
      [String.raw`\(\partial L/\partial b\)`, valueVector(context.gradsB), "Nilainya sama dengan delta neuron pada layer ini."]
    ];
    if(outputLayer){
      variables.splice(1,0,
        [String.raw`\(\hat y\)`, valueVector(context.pred), "Aktivasi output yang tersimpan dari forward propagation."],
        [String.raw`\(y\)`, valueVector(context.target), `Nilai kolom “${context.targetName || "target"}” dari ${context.targetSource || "baris CSV yang dipilih"}.`]);
    }else{
      variables.splice(1,0,
        ...weightRows(context.nextWeights,layer+1,[],`${context.parameterSource}; dipakai untuk membawa error kembali ke layer ${layer}`),
        [String.raw`\(\delta_k^{(${layer+1})}\)`, valueVector(context.nextDelta), `Delta yang sebelumnya sudah dihitung pada layer ${layer+1}.`]);
    }
    if(title.includes("update setelah animasi")){
      variables.push(
        ...weightRows(context.weights,layer,context.previousLabels,`${context.parameterSource}; nilai sebelum update ini`,"lama"),
        ...biasRows(context.biases,layer,`${context.parameterSource}; nilai sebelum update ini`,"lama"),
        [String.raw`\(\eta\)`, valueNumber(context.lr), "Nilai kontrol Learning rate di sidebar."],
        ...weightRows(context.updatedWeights,layer,context.previousLabels,"Hasil w_lama pada koneksi yang sama − η × gradiennya","baru"),
        ...biasRows(context.updatedBiases,layer,"Hasil b_lama neuron yang sama − η × gradiennya","baru")
      );
    }
    return {intro:`Tahap backward layer ${layer} menerapkan chain rule dari output menuju input.`,variables};
  }

  if(title==="Hasil Predict"){
    return {
      intro:"Hasil ini hanya berasal dari normalisasi dan forward propagation; bobot tidak diubah.",
      variables:[
        [String.raw`\(\hat y\)`, valueVector(context.pred), "Aktivasi neuron output terakhir dari forward propagation."],
        ...networkParameterRows(context.weights,context.biases,context.featureNames,context.parameterSource)
      ]
    };
  }

  if(title==="Gradien selesai"){
    return {
      intro:"Backward selesai menghitung gradien untuk diperiksa, tetapi bobot dan bias belum diperbarui.",
      variables:[
        [String.raw`\(\partial L/\partial w\)`, context.gradientSummary || "Lihat detail substitusi", "Gradien bobot hasil chain rule dari output menuju input."],
        [String.raw`\(\partial L/\partial b\)`, context.biasGradientSummary || "Lihat detail substitusi", "Gradien bias hasil chain rule."],
        ...networkParameterRows(context.weights,context.biases,context.featureNames,"Tidak berubah karena tombol Backward hanya menampilkan gradien"),
        [String.raw`\(L\)`, valueNumber(context.loss), "Loss dihitung dengan parameter yang belum diubah."]
      ]
    };
  }

  return {
    intro:"Update selesai menerapkan gradien ke parameter network, lalu menghitung ulang loss.",
    variables:[
      ...networkParameterRows(context.weights,context.biases,context.featureNames,"Hasil parameter lama − η × gradien masing-masing","baru"),
      [String.raw`\(\eta\)`, valueNumber(context.lr), "Nilai kontrol Learning rate di sidebar."],
      [String.raw`\(L\)`, valueNumber(context.loss), "Loss yang dihitung ulang sesudah update parameter."]
    ]
  };
}

function createMathLine(line){
  const row=document.createElement("div");
  row.className="math-line";
  let prefix="";
  if(line.startsWith("Neuron ")){
    const colon=line.indexOf(":");
    prefix=line.slice(0,colon+1)+" ";
    line=line.slice(colon+1).trim();
  }
  if(line.startsWith("Learning rate ")){
    prefix="Learning rate ";
    line=line.slice(prefix.length);
  }
  // Descriptions remain readable text; only equations are sent to TeX.
  row.textContent=prefix+(line.includes("=")&&!line.startsWith("Bobot")&&!line.startsWith("x")
    ? `\\(${mathTex(line)}\\)` : line);
  return row;
}

async function openVariableDialog(processTitle, processLines, processContext){
  const dialog=$("#variableDialog");
  const dialogBody=dialog.querySelector(".dialog-body");
  const sources=$("#variableSources");
  const values=$("#variableValues");
  const guide=variableGuide(processTitle,processContext);
  window.MathJax?.typesetClear?.([dialogBody]);
  $("#variableDialogTitle").textContent=`Detail Variabel · ${processTitle}`;
  $("#variableDialogIntro").textContent=guide.intro;
  sources.replaceChildren();
  values.replaceChildren();
  ["Variabel","Nilai","Asal nilai"].forEach(label=>{
    const heading=document.createElement("div");
    heading.className="variable-heading";
    heading.textContent=label;
    sources.appendChild(heading);
  });
  guide.variables.forEach(([symbol,value,origin])=>{
    const symbolEl=document.createElement("div");
    const valueEl=document.createElement("div");
    const originEl=document.createElement("div");
    symbolEl.className="variable-symbol";
    valueEl.className="variable-value";
    originEl.className="variable-origin";
    symbolEl.textContent=symbol;
    valueEl.textContent=value;
    originEl.textContent=origin;
    sources.append(symbolEl,valueEl,originEl);
  });
  processLines.filter(Boolean).forEach(line=>values.appendChild(createMathLine(line)));
  if(typeof dialog.showModal==="function") dialog.showModal();
  else dialog.setAttribute("open","");
  try{
    if(window.MathJax?.typesetPromise){
      await window.MathJax.startup.promise;
      if(dialog.open) await window.MathJax.typesetPromise([dialogBody]);
    }
  }catch(error){console.warn("MathJax popup rendering failed:",error)}
}

async function mathStep(title, lines, backward=false, context={}){
  const processContext=typeof structuredClone==="function"
    ? structuredClone(context)
    : JSON.parse(JSON.stringify(context));
  const card=document.createElement("div");
  card.className="math-step interactive"+(backward?" backward":"");
  card.tabIndex=0;
  card.setAttribute("role","button");
  card.setAttribute("aria-haspopup","dialog");
  card.setAttribute("aria-label",`${title}. Buka penjelasan asal dan nilai variabel.`);
  const heading=document.createElement("strong"), body=document.createElement("div");
  const hint=document.createElement("span");
  body.className="math-body";
  hint.className="math-step-hint";
  heading.textContent=title;
  hint.textContent="Klik untuk melihat asal dan nilai variabel";
  heading.appendChild(hint);
  lines.filter(Boolean).forEach(line=>body.appendChild(createMathLine(line)));
  card.append(heading,body);
  const processLines=lines.slice();
  card.addEventListener("click",()=>openVariableDialog(title,processLines,processContext));
  card.addEventListener("keydown",event=>{
    if(event.key==="Enter" || event.key===" "){
      event.preventDefault();
      openVariableDialog(title,processLines,processContext);
    }
  });
  const panel=$("#mathSteps");
  panel.appendChild(card);
  try{
    if(window.MathJax?.typesetPromise){
      await window.MathJax.startup.promise;
      if(card.isConnected) await window.MathJax.typesetPromise([card]);
    }
  }catch(error){console.warn("MathJax rendering failed:",error)}
  panel.scrollTop=panel.scrollHeight;
}

$("#variableDialogClose").addEventListener("click",()=>$("#variableDialog").close());
$("#variableDialog").addEventListener("click",event=>{
  if(event.target===$("#variableDialog")) $("#variableDialog").close();
});
function activationMath(type,z,a,derivative=false){
  if(derivative){
    if(type==="sigmoid") return `${fmt(a)} × (1 − ${fmt(a)})`;
    if(type==="tanh") return `1 − (${fmt(a)})²`;
    if(type==="relu") return String.raw`\begin{cases}1 & z>0\\0 & z\leq 0\end{cases}\quad(z=${fmt(z)})`;
    return "1";
  }
  if(type==="sigmoid") return String.raw`\frac{1}{1 + \exp(-(${fmt(z)}))}`;
  if(type==="tanh") return `tanh(${fmt(z)})`;
  if(type==="relu") return `max(0, ${fmt(z)})`;
  return fmt(z);
}
function parameterSource(){
  return nn.updateCount===0
    ? "Inisialisasi acak saat Build Network/Randomize"
    : `Hasil update SGD sebelumnya (${nn.updateCount} langkah update)`;
}
function forwardMath(l){
  const {A,Z}=nn.cache;
  const type=l===nn.W.length-1?nn.outputAct:nn.hiddenAct;
  const lines=[];
  nn.W[l].forEach((row,j)=>{
    lines.push(`Neuron ${j+1}: z = Σ(w × a) + b`,
      `z = ${row.map((w,i)=>`(${fmt(w)} × ${fmt(A[l][i])})`).join(" + ")} + (${fmt(nn.B[l][j])}) = ${fmt(Z[l][j])}`,
      `a = ${type}(z) = ${activationMath(type,Z[l][j],A[l+1][j])} = ${fmt(A[l+1][j])}`, "");
  });
  return mathStep(`Forward · Layer ${l+1}`,lines,false,{
    aPrev:A[l],
    previousLabels:l===0?dataset.featureNames:[],
    weights:nn.W[l],
    biases:nn.B[l],
    z:Z[l],
    a:A[l+1],
    activation:type,
    parameterSource:parameterSource(),
    outputLayer:nn.W.length
  });
}
function backwardMath(l,y,grads,apply){
  const {A,Z}=nn.cache;
  const output=l===nn.W.length-1;
  const type=output?nn.outputAct:nn.hiddenAct;
  const lines=[];
  nn.W[l].forEach((row,j)=>{
    const derivative=dAct(type,Z[l][j],A[l+1][j]);
    const upstream=output ? `(${fmt(A[l+1][j])} − ${fmt(y[j])})` :
      `(${nn.W[l+1].map((r,k)=>`(${fmt(r[j])} × ${fmt(grads.deltas[l+1][k])})`).join(" + ")})`;
    lines.push(`Neuron ${j+1}: f′(z) = ${activationMath(type,Z[l][j],A[l+1][j],true)} = ${fmt(derivative)}`,
      output?"δ = (ŷ − y) × f′(z)":"δ = Σ(w_next × δ_next) × f′(z)",
      `δ = ${upstream} × ${fmt(derivative)} = ${fmt(grads.deltas[l][j])}`);
    row.forEach((w,i)=>{
      const gradient=grads.gradsW[l][j][i];
      lines.push(`∂L/∂w[${j+1},${i+1}] = δ × a_prev = ${fmt(grads.deltas[l][j])} × ${fmt(A[l][i])} = ${fmt(gradient)}`);
      if(apply) lines.push(`w_new = w − η × ∂L/∂w = ${fmt(w)} − ${fmt(nn.lr)} × (${fmt(gradient)}) = ${fmt(w-nn.lr*gradient)}`);
    });
    lines.push(`∂L/∂b = δ = ${fmt(grads.gradsB[l][j])}`);
    if(apply) lines.push(`b_new = ${fmt(nn.B[l][j])} − ${fmt(nn.lr)} × (${fmt(grads.gradsB[l][j])}) = ${fmt(nn.B[l][j]-nn.lr*grads.gradsB[l][j])}`);
    lines.push("");
  });
  return mathStep(`Backward · Layer ${l+1}${apply?" · update setelah animasi":" · tanpa update"}`,lines,true,{
    z:Z[l],
    a:A[l+1],
    aPrev:A[l],
    previousLabels:l===0?dataset.featureNames:[],
    derivatives:Z[l].map((z,j)=>dAct(type,z,A[l+1][j])),
    delta:grads.deltas[l],
    gradsW:grads.gradsW[l],
    gradsB:grads.gradsB[l],
    weights:nn.W[l],
    biases:nn.B[l],
    updatedWeights:nn.W[l].map((row,j)=>row.map((weight,i)=>weight-nn.lr*grads.gradsW[l][j][i])),
    updatedBiases:nn.B[l].map((bias,j)=>bias-nn.lr*grads.gradsB[l][j]),
    nextWeights:output?null:nn.W[l+1],
    nextDelta:output?null:grads.deltas[l+1],
    pred:A.at(-1),
    target:y,
    targetName:dataset.targetName,
    targetSource:`CSV row ${Number($("#sampleSel").value || 0)+1}`,
    activation:type,
    parameterSource:parameterSource(),
    lr:nn.lr,
    outputLayer:nn.W.length
  });
}

async function runSimulation(action){
  const controls=[...document.querySelectorAll(".sidebar button, .sidebar input, .sidebar select, .sidebar textarea")].filter(el=>el.id!=="stopBtn");
  controls.forEach(el=>el.disabled=true);
  try{await action()}catch(e){setStatus("ERROR: "+e.message)}
  finally{controls.forEach(el=>el.disabled=false)}
}

function parseCSVText(text){
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if(lines.length < 2) throw new Error("CSV minimal harus punya header + 1 data row.");
  const headers = lines[0].split(",").map(s=>s.trim());
  const rows = lines.slice(1).map((line,ri)=>{
    const vals = line.split(",").map(v=>v.trim());
    if(vals.length !== headers.length) throw new Error(`Row ${ri+2}: jumlah kolom tidak sama.`);
    const nums = vals.map(Number);
    if(nums.some(v=>Number.isNaN(v))) throw new Error(`Row ${ri+2}: semua nilai harus numerik.`);
    return nums;
  });
  return {headers, rows};
}

function refreshTargetOptions(headers){
  const sel = $("#targetCol");
  const prev = sel.value;
  sel.innerHTML = "";
  headers.forEach(h=>{
    const o = document.createElement("option");
    o.value=h; o.textContent=h; sel.appendChild(o);
  });
  if(headers.includes("target")) sel.value="target";
  else if(headers.includes(prev)) sel.value=prev;
  else sel.selectedIndex=headers.length-1;
}

function normalizeFeatures(X){
  const n = X[0].length;
  const mins = Array(n).fill(Infinity), maxs = Array(n).fill(-Infinity);
  X.forEach(r=>r.forEach((v,j)=>{ mins[j]=Math.min(mins[j],v); maxs[j]=Math.max(maxs[j],v); }));
  return {
    X:X.map(r=>r.map((v,j)=>maxs[j]===mins[j] ? 0 : (v-mins[j])/(maxs[j]-mins[j]))),
    mins,maxs
  };
}

function loadDataset(){
  const raw = parseCSVText($("#csv").value);
  refreshTargetOptions(raw.headers);
  const targetName = $("#targetCol").value || raw.headers[raw.headers.length-1];
  const t = raw.headers.indexOf(targetName);
  const inputIdx = raw.headers.map((_,i)=>i).filter(i=>i!==t);
  const Xraw = raw.rows.map(r=>inputIdx.map(i=>r[i]));
  const y = raw.rows.map(r=>[r[t]]);
  const norm = normalizeFeatures(Xraw);
  dataset = {
    headers: raw.headers,
    featureNames: inputIdx.map(i=>raw.headers[i]),
    targetName, Xraw, X:norm.X, y, rows: raw.rows,
    mins:norm.mins, maxs:norm.maxs
  };
  // A changed dataset requires matching weights and normalization statistics.
  nn=null;
  $("#network").replaceChildren();
  $("#mLoss").textContent="-";
  $("#mPred").textContent="-";
  clearMath("Data berubah. Klik Build Network untuk memulai simulasi.");
  fillSamples();
  fillPredictInputs();
  renderPreview();
  $("#dataInfo").textContent =
    `${raw.rows.length} rows · ${inputIdx.length} input feature · target: ${targetName}`;
  setStatus("CSV berhasil diparse. Klik Build Network.");
  return dataset;
}

function fillSamples(){
  const s = $("#sampleSel");
  s.innerHTML="";
  dataset.Xraw.forEach((row,i)=>{
    const o=document.createElement("option");
    o.value=i;
    o.textContent=`Row ${i+1}: [${row.join(", ")}] → ${dataset.y[i][0]}`;
    s.appendChild(o);
  });
}

function resetPrediction(){
  $("#predictResult").textContent="Belum ada prediksi input user dengan bobot saat ini.";
}
function fillPredictInputs(){
  const container=$("#predictInputs");
  container.replaceChildren();
  dataset.featureNames.forEach((name,i)=>{
    const label=document.createElement("label"), input=document.createElement("input");
    input.id=`predict-input-${i}`;
    input.type="number";
    input.step="any";
    input.value=dataset.Xraw[0][i];
    label.htmlFor=input.id;
    label.textContent=`${name} (training: ${dataset.mins[i]} … ${dataset.maxs[i]})`;
    input.oninput=resetPrediction;
    container.append(label,input);
  });
  resetPrediction();
}
function normalizePrediction(raw){
  if(raw.length!==dataset.featureNames.length || raw.some(v=>!Number.isFinite(v)))
    throw new Error("Semua fitur prediksi harus berupa angka valid.");
  return raw.map((v,i)=>dataset.maxs[i]===dataset.mins[i]?0:
    (v-dataset.mins[i])/(dataset.maxs[i]-dataset.mins[i]));
}
async function predictUser(){
  const raw=dataset.featureNames.map((name,i)=>{
    const value=$(`#predict-input-${i}`).value.trim();
    if(!value || !Number.isFinite(Number(value))) throw new Error(`Isi fitur ${name} dengan angka valid.`);
    return Number(value);
  });
  if(!nn) buildNetwork();
  const pred=await animateForward(raw);
  $("#predictResult").textContent=`Input: [${raw.join(", ")}]\n${dataset.targetName} (prediksi): ${fmt(pred[0])}`;
}

function act(name,x){
  if(name==="sigmoid") return 1/(1+Math.exp(-x));
  if(name==="tanh") return Math.tanh(x);
  if(name==="relu") return Math.max(0,x);
  return x;
}
function dAct(name,z,a){
  if(name==="sigmoid") return a*(1-a);
  if(name==="tanh") return 1-a*a;
  if(name==="relu") return z>0?1:0;
  return 1;
}
const rand = () => (Math.random()*2-1)*0.8;

class NeuralNetwork{
  constructor(sizes, hiddenAct, outputAct, lr){
    this.sizes=sizes;
    this.hiddenAct=hiddenAct;
    this.outputAct=outputAct;
    this.lr=lr;
    this.W=[];
    this.B=[];
    this.updateCount=0;
    for(let l=0;l<sizes.length-1;l++){
      this.W.push(Array.from({length:sizes[l+1]},()=>Array.from({length:sizes[l]},rand)));
      this.B.push(Array.from({length:sizes[l+1]},()=>rand()));
    }
    this.cache=null;
  }

  forward(x){
    const A=[x.slice()], Z=[];
    for(let l=0;l<this.W.length;l++){
      const z = this.W[l].map((row,j)=>row.reduce((s,w,i)=>s+w*A[l][i], this.B[l][j]));
      const type = l===this.W.length-1 ? this.outputAct : this.hiddenAct;
      const a = z.map(v=>act(type,v));
      Z.push(z); A.push(a);
    }
    this.cache={A,Z};
    return A[A.length-1];
  }

  loss(pred,target){
    return pred.reduce((s,p,i)=>s+0.5*(p-target[i])**2,0);
  }

  backward(target, apply=true){
    if(!this.cache) throw new Error("Jalankan forward dahulu.");
    const {A,Z}=this.cache;
    const L=this.W.length;
    const deltas=Array(L);
    deltas[L-1]=A[L].map((a,j)=>{
      const type=this.outputAct;
      return (a-target[j])*dAct(type,Z[L-1][j],a);
    });
    for(let l=L-2;l>=0;l--){
      deltas[l]=A[l+1].map((a,i)=>{
        let sum=0;
        for(let j=0;j<this.sizes[l+2];j++) sum += this.W[l+1][j][i]*deltas[l+1][j];
        return sum*dAct(this.hiddenAct,Z[l][i],a);
      });
    }

    const gradsW=this.W.map((layer,l)=>
      layer.map((row,j)=>row.map((_,i)=>deltas[l][j]*A[l][i]))
    );
    const gradsB=deltas.map(d=>d.slice());

    if(apply){
      for(let l=0;l<this.W.length;l++){
        for(let j=0;j<this.W[l].length;j++){
          for(let i=0;i<this.W[l][j].length;i++) this.W[l][j][i]-=this.lr*gradsW[l][j][i];
          this.B[l][j]-=this.lr*gradsB[l][j];
        }
      }
      this.updateCount++;
    }
    return {deltas,gradsW,gradsB};
  }

  trainOne(x,y){
    const p=this.forward(x);
    const loss=this.loss(p,y);
    const grads=this.backward(y,true);
    return {p,loss,grads};
  }
}

function parseHidden(){
  const txt=$("#hiddenLayers").value.trim();
  if(!txt) return [];
  const arr=txt.split(",").map(x=>Number(x.trim()));
  if(arr.some(x=>!Number.isInteger(x)||x<1||x>30)) throw new Error("Hidden layers harus integer 1..30, contoh 4,4.");
  if(arr.length>5) throw new Error("Maksimum 5 hidden layer untuk visualisasi.");
  return arr;
}

function buildNetwork(){
  if(!dataset) loadDataset();
  const hidden=parseHidden();
  const sizes=[dataset.featureNames.length,...hidden,1];
  nn=new NeuralNetwork(sizes,$("#hiddenAct").value,$("#outputAct").value,Number($("#lr").value));
  drawNetwork();
  updateMetrics();
  resetPrediction();
  clearMath("Klik Forward, Backward, atau Train 1 Step untuk melihat proses matematika.");
  setStatus(`Network dibuat:\n${sizes.join(" → ")}\nActivation: ${nn.hiddenAct} / ${nn.outputAct}`);
}

function nodeId(l,n){return `node-${l}-${n}`}
function edgeId(l,from,to){return `edge-${l}-${from}-${to}`}

function nodePositions(){
  const sizes=nn.sizes;
  const width=1000, height=560, padX=80, padY=48;
  const gapX=(width-padX*2)/(sizes.length-1 || 1);
  const pos=[];
  sizes.forEach((count,l)=>{
    const usable=height-padY*2;
    const gapY=count===1?0:usable/(count-1);
    const x=padX+l*gapX;
    pos[l]=[];
    for(let n=0;n<count;n++){
      const y=count===1?height/2:padY+n*gapY;
      pos[l][n]={x,y};
    }
  });
  return pos;
}

function drawNetwork(){
  const svg=$("#network"); svg.innerHTML="";
  const pos=nodePositions();

  for(let l=0;l<nn.sizes.length-1;l++){
    for(let i=0;i<nn.sizes[l];i++){
      for(let j=0;j<nn.sizes[l+1];j++){
        const w=nn.W[l][j][i];
        const e=document.createElementNS("http://www.w3.org/2000/svg","line");
        e.id=edgeId(l,i,j);
        e.setAttribute("x1",pos[l][i].x);e.setAttribute("y1",pos[l][i].y);
        e.setAttribute("x2",pos[l+1][j].x);e.setAttribute("y2",pos[l+1][j].y);
        e.setAttribute("class",`edge ${w>=0?"positive":"negative"}`);
        e.setAttribute("stroke-width",Math.min(4,0.8+Math.abs(w)*1.6));
        svg.appendChild(e);
      }
    }
  }

  nn.sizes.forEach((count,l)=>{
    for(let n=0;n<count;n++){
      const g=document.createElementNS("http://www.w3.org/2000/svg","g");
      g.id=nodeId(l,n);
      g.setAttribute("class",`node ${l===0?"input":l===nn.sizes.length-1?"output":""}`);
      const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
      c.setAttribute("cx",pos[l][n].x);c.setAttribute("cy",pos[l][n].y);c.setAttribute("r",23);
      const t=document.createElementNS("http://www.w3.org/2000/svg","text");
      t.setAttribute("x",pos[l][n].x);t.setAttribute("y",pos[l][n].y-2);
      t.textContent=l===0?(dataset.featureNames[n]||`x${n+1}`):(l===nn.sizes.length-1?"ŷ":`h${l}.${n+1}`);
      const v=document.createElementNS("http://www.w3.org/2000/svg","text");
      v.setAttribute("class","val"); v.setAttribute("x",pos[l][n].x);v.setAttribute("y",pos[l][n].y+14);
      v.textContent="";
      g.append(c,t,v);svg.appendChild(g);
    }
  });
}

function clearActive(){
  document.querySelectorAll(".active-forward,.active-backward").forEach(el=>{
    el.classList.remove("active-forward","active-backward");
  });
}
function setNodeValue(l,n,val){
  const g=document.getElementById(nodeId(l,n));
  if(g) g.querySelector(".val").textContent=Number(val).toFixed(3);
}

async function animateForward(userInput=null){
  if(!nn) buildNetwork();
  clearActive();
  const idx=Number($("#sampleSel").value||0);
  const predicting=userInput!==null;
  const raw=predicting?userInput:dataset.Xraw[idx];
  const x=predicting?normalizePrediction(raw):dataset.X[idx], y=predicting?null:dataset.y[idx];
  const pred=nn.forward(x);
  const speed=Number($("#speed").value);
  clearMath();
  await mathStep(predicting?"Predict · Input User":`Training · Row ${idx+1}`, x.map((v,i)=>
    String.raw`a_{${i+1}}^{(0)} = ${dataset.maxs[i]===dataset.mins[i]?String.raw`0\;\text{(fitur konstan)}`:String.raw`\frac{${fmt(raw[i])} - ${fmt(dataset.mins[i])}}{${fmt(dataset.maxs[i])} - ${fmt(dataset.mins[i])}}`} = ${fmt(v)}`),false,{
      raw,
      normalized:x,
      mins:dataset.mins,
      maxs:dataset.maxs,
      featureNames:dataset.featureNames,
      minLabels:dataset.featureNames.map((name,i)=>`${name} (CSV row ${dataset.Xraw.findIndex(row=>row[i]===dataset.mins[i])+1})`),
      maxLabels:dataset.featureNames.map((name,i)=>`${name} (CSV row ${dataset.Xraw.findIndex(row=>row[i]===dataset.maxs[i])+1})`),
      inputSource:predicting ? "form Predict · Input User" : `CSV row ${idx+1}`
    });

  nn.cache.A[0].forEach((v,n)=>setNodeValue(0,n,v));

  for(let l=0;l<nn.sizes.length-1;l++){
    await forwardMath(l);
    document.querySelectorAll(`[id^="edge-${l}-"]`).forEach(e=>e.classList.add("active-forward"));
    for(let j=0;j<nn.sizes[l+1];j++){
      const g=document.getElementById(nodeId(l+1,j));
      g.classList.add("active-forward");
      setNodeValue(l+1,j,nn.cache.A[l+1][j]);
    }
    await sleep(speed);
    document.querySelectorAll(`[id^="edge-${l}-"]`).forEach(e=>e.classList.remove("active-forward"));
    document.querySelectorAll(`#network .node.active-forward`).forEach(e=>e.classList.remove("active-forward"));
  }
  if(predicting){
    await mathStep("Hasil Predict",[`ŷ = ${fmt(pred[0])}`,"Forward selesai. Bobot dan bias tetap."],false,{
      pred,
      weights:nn.W,
      biases:nn.B,
      featureNames:dataset.featureNames,
      parameterSource:parameterSource()
    });
    $("#mLoss").textContent="-";
    $("#mPred").textContent=pred[0].toFixed(4);
    setStatus(`PREDICT INPUT USER\nInput asli: [${raw.join(", ")}]\nInput normalisasi: [${x.map(fmt).join(", ")}]\nPrediction: ${fmt(pred[0])}`);
    return pred;
  }
  const loss=nn.loss(pred,y);
  await mathStep("Loss",[`L = ½ × (ŷ − y)² = ½ × (${fmt(pred[0])} − ${fmt(y[0])})² = ${fmt(loss)}`],false,{
    pred,
    target:y,
    targetName:dataset.targetName,
    targetSource:`CSV row ${idx+1}`,
    loss
  });
  $("#mLoss").textContent=loss.toFixed(6);
  $("#mPred").textContent=pred[0].toFixed(4);
  setStatus(`FORWARD\nInput: [${x.map(v=>v.toFixed(3)).join(", ")}]\nPrediction: ${pred[0].toFixed(6)}\nTarget: ${y[0]}\nLoss: ${loss.toFixed(6)}`);
}

async function animateBackward(applyUpdate=true, keepMath=false){
  if(!nn) buildNetwork();
  const idx=Number($("#sampleSel").value||0);
  const x=dataset.X[idx], y=dataset.y[idx];
  nn.forward(x);
  nn.lr=Number($("#lr").value);
  const grads=nn.backward(y,false);
  if(!keepMath) clearMath();
  const initialPred=nn.cache.A.at(-1);
  const initialLoss=nn.loss(initialPred,y);
  await mathStep("Backward · Loss awal",[`L = ½ × (${fmt(initialPred[0])} − ${fmt(y[0])})² = ${fmt(initialLoss)}`,`Learning rate η = ${fmt(nn.lr)}`],true,{
    pred:initialPred,
    target:y,
    targetName:dataset.targetName,
    targetSource:`CSV row ${idx+1}`,
    loss:initialLoss,
    lr:nn.lr
  });
  const speed=Number($("#speed").value);
  clearActive();

  for(let l=nn.W.length-1;l>=0;l--){
    await backwardMath(l,y,grads,applyUpdate);
    document.querySelectorAll(`[id^="edge-${l}-"]`).forEach(e=>e.classList.add("active-backward"));
    for(let i=0;i<nn.sizes[l+1];i++){
      const g=document.getElementById(nodeId(l+1,i));
      if(g) g.classList.add("active-backward");
      setNodeValue(l+1,i,grads.deltas[l][i]);
    }
    await sleep(speed);
    document.querySelectorAll(`[id^="edge-${l}-"]`).forEach(e=>e.classList.remove("active-backward"));
    document.querySelectorAll(`#network .node.active-backward`).forEach(e=>e.classList.remove("active-backward"));
  }
  if(applyUpdate){nn.backward(y,true);resetPrediction()}
  drawNetwork();
  nn.forward(x);
  nn.cache.A.forEach((layer,l)=>layer.forEach((v,n)=>setNodeValue(l,n,v)));
  const p=nn.cache.A.at(-1);
  const loss=nn.loss(p,y);
  await mathStep(applyUpdate?"Update selesai":"Gradien selesai",[applyUpdate?`Bobot dan bias diperbarui. Loss setelah update = ${fmt(loss)}`:"Bobot dan bias tetap. Klik Train 1 Step untuk menerapkan gradien."],true,{
    gradientSummary:valueLayers(grads.gradsW),
    biasGradientSummary:valueLayers(grads.gradsB),
    weightSummary:valueLayers(nn.W),
    biasSummary:valueLayers(nn.B),
    weights:nn.W,
    biases:nn.B,
    featureNames:dataset.featureNames,
    loss,
    lr:nn.lr
  });
  $("#mLoss").textContent=loss.toFixed(6);
  $("#mPred").textContent=p[0].toFixed(4);
  setStatus(`BACKPROPAGATION\nGradient dihitung dari output → input.\nWeights ${applyUpdate?"sudah":"belum"} diperbarui.\nLoss sekarang: ${loss.toFixed(6)}`);
}

function updateMetrics(){
  if(!nn)return;
  $("#mInputs").textContent=nn.sizes[0];
  $("#mHidden").textContent=nn.sizes.slice(1,-1).reduce((a,b)=>a+b,0);
  $("#mLoss").textContent="-";
  $("#mPred").textContent="-";
}

function averageLoss(){
  let total=0;
  dataset.X.forEach((x,i)=> total += nn.loss(nn.forward(x),dataset.y[i]));
  return total/dataset.X.length;
}

async function trainAll(){
  if(!nn) buildNetwork();
  resetPrediction();
  clearMath("Training seluruh CSV sedang berjalan. Gunakan Forward / Backward untuk melihat matematika per sample.");
  stopTraining=false;
  const epochs=Number($("#epochs").value);
  nn.lr=Number($("#lr").value);
  let lastLoss=0;
  for(let e=1;e<=epochs;e++){
    for(let i=0;i<dataset.X.length;i++) nn.trainOne(dataset.X[i],dataset.y[i]);
    if(e===1 || e%50===0 || e===epochs){
      lastLoss=averageLoss();
      $("#mLoss").textContent=lastLoss.toFixed(6);
      setStatus(`TRAINING\nEpoch ${e}/${epochs}\nAverage loss: ${lastLoss.toFixed(6)}`);
      if(e%200===0){ drawNetwork(); renderPreview(); await sleep(0); }
    }
    if(stopTraining) break;
  }
  drawNetwork(); renderPreview();
  const idx=Number($("#sampleSel").value||0);
  const p=nn.forward(dataset.X[idx]);
  $("#mPred").textContent=p[0].toFixed(4);
  setStatus(`${stopTraining?"Training dihentikan":"Training selesai"}.\nAverage loss: ${averageLoss().toFixed(6)}`);
}

function renderPreview(){
  if(!dataset)return;
  let h="<table><thead><tr><th>#</th>";
  dataset.featureNames.forEach(n=>h+=`<th>${n}</th>`);
  h+=`<th>${dataset.targetName}</th><th>Prediction</th></tr></thead><tbody>`;
  dataset.Xraw.forEach((row,i)=>{
    const p=nn?nn.forward(dataset.X[i])[0]:null;
    h+=`<tr><td>${i+1}</td>${row.map(v=>`<td>${v}</td>`).join("")}<td>${dataset.y[i][0]}</td><td>${p==null?"-":p.toFixed(4)}</td></tr>`;
  });
  h+="</tbody></table>";
  $("#preview").innerHTML=h;
}

$("#parseBtn").onclick=()=>{try{loadDataset()}catch(e){setStatus("ERROR: "+e.message)}};
$("#buildBtn").onclick=()=>{try{loadDataset();buildNetwork()}catch(e){setStatus("ERROR: "+e.message)}};
$("#randomBtn").onclick=()=>{try{if(!dataset)loadDataset();buildNetwork()}catch(e){setStatus("ERROR: "+e.message)}};
$("#forwardBtn").onclick=()=>runSimulation(()=>animateForward());
$("#backwardBtn").onclick=()=>runSimulation(()=>animateBackward(false));
$("#stepBtn").onclick=()=>runSimulation(async()=>{
  if(!nn)buildNetwork();
  await animateForward();
  await animateBackward(true,true);
  renderPreview();
});
$("#trainBtn").onclick=()=>runSimulation(()=>trainAll());
$("#predictBtn").onclick=()=>runSimulation(async()=>{
  try{await predictUser()}catch(e){$("#predictResult").textContent="ERROR: "+e.message;throw e}
});
$("#stopBtn").onclick=()=>{stopTraining=true};
$("#speed").oninput=e=>$("#speedVal").value=e.target.value+" ms";
$("#targetCol").onchange=()=>{try{loadDataset()}catch(e){setStatus("ERROR: "+e.message)}};

try{
  const raw=parseCSVText($("#csv").value);
  refreshTargetOptions(raw.headers);
  loadDataset();
  buildNetwork();
}catch(e){setStatus("ERROR: "+e.message)}
