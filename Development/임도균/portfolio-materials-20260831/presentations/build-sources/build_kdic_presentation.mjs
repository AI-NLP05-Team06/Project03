import fs from 'node:fs/promises';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const OUT = 'C:/Users/임도균/Documents/Codex/2026-07-22/d/outputs/KDIC_RAG_프로젝트3_발표초안.pptx';
const W = 1280, H = 720;
const C = { navy:'#102A43', blue:'#2B6CB0', cyan:'#4EA8DE', teal:'#0FAF8F', amber:'#E69A15', red:'#D94B58', ink:'#132238', muted:'#52677D', pale:'#F4F7FA', panel:'#E8EEF4', rule:'#C8D3DE', white:'#FFFFFF' };
const FONT = 'Malgun Gothic';

async function saveBlob(path, blob){ await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer())); }
function shape(slide, geometry, position, fill='none', lineFill='none', lineWidth=0){ return slide.shapes.add({ geometry, position, fill, line:{style:'solid', fill:lineFill, width:lineWidth} }); }
function text(slide, value, x, y, w, h, size=20, color=C.ink, bold=false, opts={}){
  const box=shape(slide,'textbox',{left:x,top:y,width:w,height:h});
  box.text=value;
  box.text.style={fontSize:size,typeface:FONT,color,bold,alignment:opts.align||'left',verticalAlignment:opts.valign||'top',autoFit:'shrinkText',wrap:'square',insets:{top:0,right:0,bottom:0,left:0}};
  return box;
}
function rule(slide,x,y,w,color=C.rule,height=2){ shape(slide,'rect',{left:x,top:y,width:w,height},color,color,0); }
function footer(slide,n){ text(slide,'KDIC 금융정보 RAG 챗봇 · 6조',64,674,370,18,11,C.muted); text(slide,String(n).padStart(2,'0'),1160,674,56,18,11,C.muted,true,{align:'right'}); }
function title(slide,n,kicker,headline,sub=''){
  text(slide,kicker.toUpperCase(),64,45,500,22,12,C.blue,true);
  text(slide,headline,64,78,1120,58,34,C.ink,true);
  if(sub) text(slide,sub,64,145,1070,34,17,C.muted);
  rule(slide,64,196,1152);
  footer(slide,n);
}
function bullet(slide, value, x, y, w, color=C.ink, size=18){ text(slide,'•',x,y,w?18:18,24,size,C.blue,true); text(slide,value,x+24,y,w-24,30,size,color,false); }
function screenshotPlaceholder(slide,x,y,w,h,tag,need){
  shape(slide,'rect',{left:x,top:y,width:w,height:h},C.pale,C.rule,1);
  rule(slide,x,y,w,C.cyan,5);
  text(slide,'SCREENSHOT REQUIRED',x+28,y+38,w-56,20,12,C.blue,true);
  text(slide,tag,x+28,y+74,w-56,38,24,C.ink,true);
  text(slide,need,x+28,y+126,w-56,h-148,16,C.muted,false);
}
function label(slide, value,x,y,w,color=C.blue){ text(slide,value.toUpperCase(),x,y,w,18,11,color,true); }
function step(slide,n,label,x,y,w,accent=C.blue){
  shape(slide,'rect',{left:x,top:y,width:w,height:106},C.white,C.rule,1);
  text(slide,String(n).padStart(2,'0'),x+18,y+16,42,24,14,accent,true);
  text(slide,label,x+18,y+50,w-36,34,20,C.ink,true);
}

function cover(p){ const s=p.slides.add(); s.background.fill=C.navy; shape(s,'rect',{left:0,top:0,width:W,height:H},C.navy,C.navy,0); shape(s,'rect',{left:760,top:0,width:520,height:H},C.blue,C.blue,0); shape(s,'rect',{left:820,top:0,width:460,height:H},C.cyan,C.cyan,0); text(s,'PROJECT 03 · TEAM 06',70,70,420,24,14,'#B9D9F5',true); text(s,'금융정보 RAG 챗봇',70,188,680,70,48,C.white,true); text(s,'많은 시도를\n운영 가능한 결정으로',70,272,650,164,58,C.white,true); text(s,'유민규 (팀장) · 임도균 · 박주영',72,535,540,30,20,'#C7D7E7'); text(s,'강사·멘토 발표',72,582,260,24,15,'#B9D9F5'); text(s,'정확성 · 근거성 · 사용성 · 운영성',838,525,345,72,22,C.navy,true); text(s,'6',1120,615,100,70,42,C.navy,true,{align:'right'}); }
function agenda(p){ const s=p.slides.add(); title(s,2,'AGENDA','문제를 풀기 전에, 무엇을 고민해야 했는가','기술 선택의 결과보다 선택 기준과 운영 구조를 중심으로 설명합니다.'); const items=[['01','문제 정의','금융 챗봇에 필요한 네 가지 조건'],['02','실험과 선택','질의분석·검색·평가에서의 판단'],['03','사용자 경험','질문 유형과 근거 중심 답변'],['04','운영 가능한 구조','관리자 UI와 안전한 반영 흐름'],['05','결론','학습을 지속하는 RAG 운영 원칙']]; items.forEach((it,i)=>{const y=235+i*76; text(s,it[0],72,y,62,30,16,C.blue,true); text(s,it[1],168,y-3,260,30,22,C.ink,true); text(s,it[2],455,y,630,30,16,C.muted); rule(s,72,y+49,1080);}); }
function problem(p){ const s=p.slides.add(); title(s,3,'STARTING QUESTION','금융 챗봇은 “답변이 나온다”만으로 충분하지 않습니다.','잘못된 정보, 근거 없는 안내, 운영 중 임의 변경은 모두 서비스 신뢰를 해칠 수 있습니다.'); const cols=[['정확성','질문에 맞는\n정보를 찾는가?',C.blue],['근거성','답변의 출처를\n확인할 수 있는가?',C.teal],['사용성','사용자가 다음 행동을\n알 수 있는가?',C.amber],['운영성','변경을 안전하게\n검증·복구할 수 있는가?',C.red]]; cols.forEach((d,i)=>{const x=64+i*288; if(i) rule(s,x-28,270,1,C.rule,260); text(s,`0${i+1}`,x,262,60,22,14,d[2],true); text(s,d[0],x,305,230,36,25,C.ink,true); text(s,d[1],x,365,220,78,18,C.muted); }); text(s,'이 네 가지를 동시에 만족시키는 것이 프로젝트의 출발점이었습니다.',64,590,1010,34,23,C.navy,true); }
function experimentation(p){ const s=p.slides.add(); title(s,4,'OUR APPROACH','많은 시도는 “기법의 나열”이 아니라 선택 기준을 만드는 과정이었습니다.','각 단계에서 한 가지 정답을 가정하지 않고, 질문과 운영 상황에 맞는 판단 기준을 만들었습니다.'); const stages=[['질의 분석','업무·의도·모호성\n후속 질문 필요성'],['검색 실험','BM25-Nori · Dense\nSparse · Hybrid'],['근거 정렬','Structured · Parent-Child\nReranker'],['답변·운영','근거 기반 답변\n검증·로그·롤백']]; stages.forEach((d,i)=>{const x=64+i*286; step(s,i+1,d[0],x,275,230,[C.blue,C.teal,C.amber,C.red][i]); text(s,d[1],x+18,397,190,60,16,C.muted); if(i<3){ text(s,'→',x+240,319,34,34,26,C.muted,true,{align:'center'}); }}); text(s,'핵심 질문: “가장 높은 단일 점수”가 아니라 “어떤 질문에 어떤 방식이 적절한가?”',64,570,1090,32,22,C.navy,true); }
function retrieval(p){ const s=p.slides.add(); title(s,5,'RETRIEVAL TRADE-OFF','검색에서는 범위와 순도를 함께 봐야 했습니다.','더 많은 청크를 찾는 것과, 답변에 쓸 근거를 정확히 고르는 것은 같은 문제가 아닙니다.'); shape(s,'rect',{left:64,top:250,width:520,height:250},C.pale,C.rule,1); shape(s,'rect',{left:696,top:250,width:520,height:250},C.pale,C.rule,1); label(s,'COVERAGE',92,280,200,C.teal); text(s,'더 넓게 찾기',92,316,310,34,28,C.ink,true); text(s,'Recall · Complete\nGold 청크 포착',92,375,300,64,18,C.muted); label(s,'PURITY & RANKING',724,280,260,C.red); text(s,'더 정확히 고르기',724,316,350,34,28,C.ink,true); text(s,'Precision · F1 · nDCG\n상위 근거의 순도와 순위',724,375,330,64,18,C.muted); text(s,'Dense · BM25-Nori · Hybrid · Reranker를 비교하며, “검색 후보 확보 → 근거 재정렬”의 역할을 분리했습니다.',64,555,1110,38,20,C.navy,true); }
function evaluation(p){ const s=p.slides.add(); title(s,6,'EVALUATION INSIGHT','평가는 숫자를 고르는 작업이 아니라, 숫자를 해석하는 작업이었습니다.','Top-K와 지표 K가 다르면 검색 설정 효과와 평가 범위 효과가 섞일 수 있습니다.'); screenshotPlaceholder(s,64,244,620,360,'검색 비교 평가 결과','필요 화면: 현재안·개선안 지표표와 Gold 청크 기준 리포트가 함께 보이는 관리자 UI 캡처'); shape(s,'rect',{left:736,top:244,width:480,height:360},C.pale,C.rule,1); label(s,'WHAT WE LEARNED',770,277,260,C.blue); bullet(s,'Gold 청크를 정답 근거로 사용',770,320,390,C.ink,17); bullet(s,'Recall 상승과 Precision 하락을 분리 해석',770,372,390,C.ink,17); bullet(s,'질문별 결과까지 확인',770,424,390,C.ink,17); bullet(s,'동일 K 기준의 공정 비교 필요',770,476,390,C.ink,17); }
function userExperience(p){ const s=p.slides.add(); title(s,7,'USER PERSPECTIVE','사용자 질문은 모두 같은 방식으로 처리할 수 없습니다.','질문의 성격에 따라 바로 안내할지, 검색할지, 추가 정보를 물을지 판단해야 합니다.'); const items=[['단순·명확','정확한 용어\n짧은 정보 요청','바로 안내 또는\n정밀 검색',C.teal],['모호·후속','“어떻게 해?”\n문맥 부족','선택형 추가 질문\n문맥 결합',C.amber],['복합·고위험','여러 조건\n다중 근거 필요','검색 후보 확장\n근거 재정렬',C.red]]; items.forEach((d,i)=>{const x=64+i*384; text(s,`0${i+1}`,x,260,55,22,13,d[3],true); text(s,d[0],x,302,260,30,24,C.ink,true); text(s,d[1],x,358,260,56,17,C.muted); rule(s,x,444,254,d[3],4); text(s,d[2],x,470,270,56,18,C.navy,true); if(i<2) text(s,'→',x+306,365,52,30,24,C.muted,true,{align:'center'}); }); }
function answerSystem(p){ const s=p.slides.add(); title(s,8,'ANSWER SYSTEM','좋은 답변은 길이가 아니라, 근거와 다음 행동이 명확한 답변입니다.','답변 생성 단계에서도 검색 결과를 그대로 나열하지 않고, 질문 의도에 맞게 근거를 구성합니다.'); shape(s,'rect',{left:64,top:248,width:530,height:330},C.pale,C.rule,1); label(s,'ANSWER POLICY',96,282,250,C.blue); text(s,'근거 기반 안내',96,326,330,30,26,C.ink,true); bullet(s,'출처 URL·신청 경로 제시',96,385,380,C.muted,17); bullet(s,'근거 부족 시 제한 답변',96,433,380,C.muted,17); bullet(s,'쉬운 안내와 전문가 설명의 분리',96,481,410,C.muted,17); screenshotPlaceholder(s,654,248,562,330,'최종 챗봇 답변 화면','필요 화면: 답변 본문, 출처 링크, 후속 질문 또는 제한 답변이 보이는 사용자 챗봇 캡처'); }
function workflow(p){ const s=p.slides.add(); title(s,9,'OPERATING PRINCIPLE','검증되지 않은 변경은 실제 챗봇에 바로 반영하지 않습니다.','개발자에게는 안전한 변경 절차를, 운영자에게는 이해 가능한 의사결정 흐름을 제공했습니다.'); const names=['초안','검색 평가','답변 비교','승인 반영','롤백']; const desc=['파라미터·청크\n프롬프트 변경','Gold 청크 기반\n검색 품질 확인','현재안·개선안\n최종 답변 확인','실제 챗봇에\n다음 질문부터 적용','이전 상태로\n언제든 복구']; names.forEach((name,i)=>{const x=64+i*232; step(s,i+1,name,x,290,180,[C.blue,C.teal,C.amber,C.navy,C.red][i]); text(s,desc[i],x+18,414,150,54,15,C.muted); if(i<4) text(s,'→',x+184,327,36,28,22,C.muted,true,{align:'center'}); }); text(s,'Draft → Evaluate → Apply → Rollback',64,570,700,30,23,C.navy,true); }
function adminUi(p){ const s=p.slides.add(); title(s,10,'ADMIN UI','관리자 UI는 설정 화면이 아니라, 운영 의사결정을 돕는 도구입니다.','복잡한 RAG 파라미터를 코드 대신 흐름·이름·비교 결과로 관리하도록 구성했습니다.'); screenshotPlaceholder(s,64,238,735,390,'관리자 UI · 파이프라인/파라미터 화면','필요 화면: 파이프라인 블록, 빠른 파라미터 수정 패널, 변경된 블록 표시가 함께 보이는 캡처'); shape(s,'rect',{left:844,top:238,width:372,height:390},C.pale,C.rule,1); label(s,'DESIGN DECISIONS',876,274,250,C.blue); bullet(s,'저장 가능한\n파라미터 설정',876,320,280,C.ink,17); bullet(s,'현재안·개선안\n비교',876,402,280,C.ink,17); bullet(s,'승인 전\n초안 유지',876,484,280,C.ink,17); bullet(s,'반영 이력과\n롤백',876,566,280,C.ink,17); }
function operations(p){ const s=p.slides.add(); title(s,11,'OPERATIONS','운영 개선은 배포 이후의 관찰과 재검증까지 포함합니다.','속도·비용·보안·사용 패턴을 함께 보며 다음 개선 실험을 결정합니다.'); screenshotPlaceholder(s,64,244,548,350,'운영 모니터링','필요 화면: 질의 수, 응답시간, 토큰, 캐시 HIT/MISS가 보이는 운영 모니터링 캡처'); screenshotPlaceholder(s,668,244,548,350,'보안·반영 관리','필요 화면: 가드레일 또는 반영센터·롤백 이력이 보이는 관리자 UI 캡처'); text(s,'운영 로그 → 문제 발견 → 개선안 저장 → 재평가 → 안전 반영',64,620,1110,30,21,C.navy,true); }
function closing(p){ const s=p.slides.add(); s.background.fill=C.navy; text(s,'CLOSING',72,68,220,20,12,'#A7D8F5',true); text(s,'많은 시도를\n하나의 운영 원칙으로',72,158,760,142,50,C.white,true); rule(s,72,343,360,C.cyan,4); text(s,'정확한 답변을 만드는 것에서 멈추지 않고,\n검증하고 안전하게 개선할 수 있는 RAG 운영 구조를 설계했습니다.',72,385,760,76,23,'#D5E3EF'); const words=[['학습','여러 기법과 질문을 실험'],['검증','Gold 청크·답변 비교'],['운영','승인·로그·롤백']]; words.forEach((d,i)=>{const x=760+i*150; shape(s,'rect',{left:x,top:408,width:125,height:125},i===1?C.cyan:'#1C3D5D','#1C3D5D',0); text(s,d[0],x+16,435,92,30,20,i===1?C.navy:C.white,true,{align:'center'}); text(s,d[1],x+14,477,96,36,11,i===1?C.navy:'#C8D9E8',false,{align:'center'}); }); text(s,'6조 · 유민규(팀장) · 임도균 · 박주영',72,648,540,22,15,'#A7D8F5'); }

async function main(){
  const p=Presentation.create({slideSize:{width:W,height:H}});
  [cover,agenda,problem,experimentation,retrieval,evaluation,userExperience,answerSystem,workflow,adminUi,operations,closing].forEach(build=>build(p));
  await fs.mkdir('C:/Users/임도균/Documents/Codex/2026-07-22/d/outputs', {recursive:true});
  for(const [i,slide] of p.slides.items.entries()) await saveBlob(`C:/Users/임도균/Documents/Codex/2026-07-22/d/ppt_build/slide-${String(i+1).padStart(2,'0')}.png`, await p.export({slide,format:'png',scale:1}));
  await saveBlob('C:/Users/임도균/Documents/Codex/2026-07-22/d/ppt_build/montage.webp', await p.export({format:'webp',montage:true,scale:1}));
  const pptx=await PresentationFile.exportPptx(p);
  await pptx.save(OUT);
}
main().catch(err=>{console.error(err);process.exitCode=1;});
