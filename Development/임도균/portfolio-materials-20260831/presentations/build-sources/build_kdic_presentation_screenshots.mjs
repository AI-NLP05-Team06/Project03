import fs from 'node:fs/promises';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const OUT = 'C:/Users/임도균/Documents/Codex/2026-07-22/d/outputs/KDIC_RAG_프로젝트3_발표자료_easy_질의분석흐름수정.pptx';
const W = 1280, H = 720;
const C = { navy:'#102A43', blue:'#1769E0', cyan:'#5DA9E9', teal:'#0AA77B', amber:'#D58C09', red:'#D9485F', ink:'#14243A', muted:'#52677D', pale:'#F4F7FA', panel:'#E8EEF4', lightBlue:'#EAF3FF', rule:'#C8D3DE', white:'#FFFFFF' };
const FONT = 'Malgun Gothic';
const A = 'C:/Users/임도균/AppData/Local/Temp/';
const IMG = {
  progress: A+'codex-clipboard-d5892676-8bed-43cb-8248-92889de0ef3a.png',
  answer: A+'codex-clipboard-5edd71b3-50c5-43e7-b300-789a1ac6f651.png',
  summary: A+'codex-clipboard-2997b021-a26b-4588-b8a6-a72485bf902a.png',
  cache: A+'codex-clipboard-8191d891-075a-4a14-b64a-89973528d924.png',
  dashboard: A+'codex-clipboard-92d226fd-d30a-4188-9c3d-c60261b745e0.png',
  navMonitoring: A+'codex-clipboard-04591eb3-4495-4d7d-be09-1cbf3cbce525.png',
  monitoring: A+'codex-clipboard-bae9e4a9-68ba-44c3-b779-e1143d26ae64.png',
  logs: A+'codex-clipboard-b6ec1c47-27e8-4203-a4ff-2556a1e88203.png',
  navEnhance: A+'codex-clipboard-9435b821-072c-4e17-b362-e9cee5d39cf6.png',
  pipeline: A+'codex-clipboard-0b14a9c9-458a-485c-976e-30343e01775e.png',
  parameters: A+'codex-clipboard-b4cd08bc-3b3f-4ff0-9a7f-ce16fc7e3864.png',
  quick: A+'codex-clipboard-a65ec302-b972-4860-a8ab-2bc9dfb64d28.png',
  chatbotTest: A+'codex-clipboard-3922af00-1b9c-4956-aeab-e842ff295027.png',
  prompts: A+'codex-clipboard-17c01f10-cbc5-4c2d-9881-f47801d9d114.png',
  approval: A+'codex-clipboard-644e44b7-21d1-44c9-b923-050844d94c03.png',
};

async function saveBlob(path, blob){ await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer())); }
async function imageBytes(path){ const b=await fs.readFile(path); return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength); }
function shape(s, geometry, position, fill='none', lineFill='none', lineWidth=0){ return s.shapes.add({geometry,position,fill,line:{style:'solid',fill:lineFill,width:lineWidth}}); }
function text(s,value,x,y,w,h,size=20,color=C.ink,bold=false,opts={}){ const b=shape(s,'textbox',{left:x,top:y,width:w,height:h}); b.text=value; b.text.style={fontSize:size,typeface:FONT,color,bold,alignment:opts.align||'left',verticalAlignment:opts.valign||'top',autoFit:'shrinkText',wrap:'square',insets:{top:0,right:0,bottom:0,left:0}}; return b; }
function rule(s,x,y,w,color=C.rule,height=2){ shape(s,'rect',{left:x,top:y,width:w,height},color,color,0); }
function vrule(s,x,y,h,color=C.rule,width=2){ shape(s,'rect',{left:x,top:y,width,height:h},color,color,0); }
function footer(s,n){ text(s,'KDIC 금융정보 RAG 챗봇 · 6조',56,678,380,16,10,C.muted); text(s,String(n).padStart(2,'0'),1160,678,62,16,10,C.muted,true,{align:'right'}); }
let autoPageNumber = 1;
function title(s,n,kicker,headline,sub=''){ text(s,kicker.toUpperCase(),56,34,560,20,11,C.blue,true); text(s,headline,56,62,1145,47,33,C.ink,true); if(sub) text(s,sub,56,117,1110,27,16,C.muted); rule(s,56,160,1168); footer(s,++autoPageNumber); }
function reheader(s,kicker,headline,sub=''){ shape(s,'rect',{left:0,top:0,width:1280,height:170},C.white,C.white,0); text(s,kicker.toUpperCase(),56,34,560,20,11,C.blue,true); text(s,headline,56,62,1145,47,33,C.ink,true); if(sub) text(s,sub,56,117,1110,27,16,C.muted); rule(s,56,160,1168); }
function tag(s,value,x,y,w,color=C.blue){ shape(s,'roundRect',{left:x,top:y,width:w,height:28},'#EAF3FF','#EAF3FF',0); text(s,value,x+12,y+6,w-24,16,11,color,true,{align:'center'}); }
function bullet(s,value,x,y,w,color=C.ink,size=18){ text(s,'•',x,y,18,25,size,C.blue,true); text(s,value,x+22,y,w-22,34,size,color,false); }
function commonGateStrip(s,y){
  shape(s,'roundRect',{left:74,top:y,width:1132,height:32},'#EAF7F2',C.teal,1);
  text(s,'공통 Hard Gate 통과',94,y+7,170,17,14,C.teal,true);
  text(s,'실행·질의 유효 100%  |  OOS/DIRECT 오탐 0  |  Hard Filter Gold 제외 0  |  CLARIFY Precision 100%',286,y+8,875,16,12,C.ink,true);
}
async function screen(s,path,x,y,w,h,alt,fit='contain'){
  shape(s,'roundRect',{left:x-4,top:y-4,width:w+8,height:h+8},C.white,C.rule,1);
  s.images.add({blob:await imageBytes(path),contentType:'image/png',alt,fit,position:{left:x,top:y,width:w,height:h},geometry:'roundRect',borderRadius:'rounded-lg'});
}
function callout(s,number,label,x,y,w,color=C.blue){ shape(s,'roundRect',{left:x,top:y,width:w,height:56},C.white,color,2); shape(s,'ellipse',{left:x+10,top:y+11,width:34,height:34},color,color,0); text(s,String(number),x+10,y+17,34,18,14,C.white,true,{align:'center'}); text(s,label,x+55,y+14,w-66,25,16,C.ink,true); }

function cover(p){ const s=p.slides.add(); s.background.fill=C.navy; shape(s,'rect',{left:0,top:0,width:W,height:H},C.navy,C.navy,0); shape(s,'rect',{left:760,top:0,width:520,height:H},C.blue,C.blue,0); shape(s,'rect',{left:850,top:0,width:430,height:H},C.cyan,C.cyan,0); text(s,'PROJECT 03 · TEAM 06',68,66,420,20,13,'#B9D9F5',true); text(s,'금융정보 RAG 챗봇',68,180,670,58,44,C.white,true); text(s,'많은 시도를\n운영 가능한 결정으로',68,255,690,152,54,C.white,true); rule(s,68,448,250,'#B9D9F5',3); text(s,'유민규 (팀장) · 임도균 · 박주영',70,486,560,28,20,'#D7E7F5'); text(s,'강사·멘토 발표 · 15분',70,538,300,22,15,'#B9D9F5'); text(s,'정확성\n근거성\n사용성\n운영성',887,240,260,172,28,C.navy,true); text(s,'6',1120,610,90,55,40,C.navy,true,{align:'right'}); }
function agenda(p){ const s=p.slides.add(); title(s,2,'CONTENTS','발표 목차','예금보험공사 RAG 챗봇을 사용자 경험과 운영 관점에서 설명합니다.'); const items=[['01','프로젝트 개요','금융 챗봇에 필요한 신뢰 조건'],['02','질의 분석','질문 유형에 따른 응답 경로'],['03','검색 기법 및 평가','근거 탐색과 품질 해석'],['04','답변 시스템','근거 기반 답변과 질의 캐시'],['05','관리자 UI 및 운영','안전한 변경·관찰·복구'],['06','결론','운영 가능한 RAG의 기준']]; items.forEach((d,i)=>{const y=190+i*66;text(s,d[0],72,y,60,25,15,C.blue,true);text(s,d[1],163,y-3,300,28,21,C.ink,true);text(s,d[2],490,y,620,28,16,C.muted);rule(s,72,y+45,1080);}); }
function concerns(p){ const s=p.slides.add(); title(s,3,'01 PROJECT OVERVIEW','프로젝트 개요','금융 챗봇은 정확한 검색, 근거 있는 답변, 이해 가능한 안내, 안전한 운영을 함께 요구합니다.'); const data=[['정확성','질문에 맞는\n근거를 찾는가?',C.blue],['근거성','출처와 신청 경로를\n확인할 수 있는가?',C.teal],['사용성','사용자가 다음 행동을\n이해하는가?',C.amber],['운영성','변경을 검증하고\n되돌릴 수 있는가?',C.red]]; data.forEach((d,i)=>{const x=58+i*295;shape(s,'roundRect',{left:x,top:246,width:250,height:245},C.pale,C.rule,1);text(s,`0${i+1}`,x+22,270,50,22,14,d[2],true);text(s,d[0],x+22,315,180,31,24,C.ink,true);text(s,d[1],x+22,372,192,68,18,C.muted);}); text(s,'검색 평가는 Gold 청크와 질문별 검색 결과를 함께 검토해, 지표 하나로 결론 내리지 않도록 설계했습니다.',58,568,1090,30,19,C.navy,true); }
function queryCandidateDecision(p){ const s=p.slides.add(); title(s,4,'02 QUERY ANALYSIS','질의 분석','질의분석은 많이 구조화하는 모델보다, 검색에 필요한 정보를 안전하게 넘기는 모델을 선택했습니다.'); const data=[['후보','V1.5 간편 라우터','규칙 중심\n교차 업무 복합질의만 LLM','31 / 400\nLLM 호출 7.75%',C.blue],['후보','V2.2 최소 경량화','규칙 우선\n애매한 질문은 LLM','139 / 270\nLLM 호출 51.48%',C.amber],['후보','V3.1 전체 질의분석','대부분 LLM 분석\n의도·역할까지 구조화','240 / 270\nLLM 호출 88.89%',C.red]]; data.forEach((d,i)=>{const x=74+i*377;shape(s,'roundRect',{left:x,top:220,width:322,height:250},C.pale,C.rule,1);text(s,d[0],x+22,245,60,18,13,d[4],true);text(s,d[1],x+22,284,252,28,22,C.ink,true);rule(s,x+22,329,246,d[4],3);text(s,d[2],x+22,354,252,49,17,C.muted);text(s,d[3],x+22,423,252,30,17,C.navy,true);}); shape(s,'roundRect',{left:74,top:520,width:1132,height:88},C.navy,C.navy,0);text(s,'결정  |  V1.5 간편 라우터',102,540,420,24,21,C.white,true);text(s,'답변 Top-5 기준 Recall@5·NeedCoverage@5가 가장 높고, V1.4와 동일한 지연 조건에서 동작',102,570,1010,22,16,'#D6E5F2'); }
function queryFinalPipeline(p){ const s=p.slides.add(); title(s,5,'02 QUERY ANALYSIS','질의 분석','선정안 V1.5는 원문을 보존하고, 교차 업무 복합질의에서만 LLM 구조화 분해를 사용합니다.'); rule(s,278,261,57,C.muted,2); rule(s,595,261,75,C.muted,2); rule(s,880,397,65,C.muted,2); rule(s,880,531,65,C.muted,2); text(s,'›',314,248,20,24,22,C.muted,true,{align:'center'}); text(s,'›',649,248,20,24,22,C.muted,true,{align:'center'}); text(s,'↓',766,320,20,28,22,C.muted,true,{align:'center'}); text(s,'›',922,367,20,24,22,C.muted,true,{align:'center'}); text(s,'↓',766,449,20,28,22,C.muted,true,{align:'center'}); text(s,'›',922,500,20,24,22,C.muted,true,{align:'center'}); text(s,'↓',1052,544,20,28,22,C.muted,true,{align:'center'}); const nodes=[['원문 보존\n최소 정리',68,225,210,72,C.blue],['대화 상태·문맥 판정',335,225,260,72,C.blue],['경로 결정',670,225,210,72,C.blue],['RETRIEVE',670,365,210,64,C.teal],['단일·동일 업무 복합\n원문 검색 1.0',945,335,235,90,C.teal],['교차 업무 복합',670,500,210,62,C.amber],['HCX-007 구조화 분해\n→ 결과 검증',945,475,235,70,C.amber],['승인: 원문 0.4 + 분해질의 0.6\n검증 실패: 원문 1.0 fallback',945,572,235,54,C.teal]]; nodes.forEach(d=>{shape(s,'roundRect',{left:d[1],top:d[2],width:d[3],height:d[4]},C.pale,d[5],1);text(s,d[0],d[1]+16,d[2]+14,d[3]-32,d[4]-22,17,C.ink,true,{align:'center',valign:'middle'});}); tag(s,'DIRECT_RESPONSE · CLARIFY · 명백한 OUT_OF_SCOPE는 검색 전 종료',68,638,500,C.blue); tag(s,'업무 필터는 NONE: 질의분석 단계에서 정답 업무를 Hard Filter로 제외하지 않음',594,638,560,C.teal); }
function queryCandidateDecisionFinal(p){
  const s=p.slides.add();
  title(s,4,'02 QUERY ANALYSIS','질의 분석','질의분석은 많이 구조화하는 모델보다, 검색에 필요한 정보를 안전하게 넘기는 모델을 선택했습니다.');
  const data=[['후보','V1.5 간편 라우터','규칙 중심\n교차 업무 복합질의만 LLM','31 / 400 · 7.75%',C.blue],['후보','V2.2 최소 경량화','규칙 우선\n애매한 질문은 LLM','139 / 270 · 51.48%',C.amber],['후보','V3.1 전체 질의분석','대부분 LLM 분석\n의도·역할까지 구조화','240 / 270 · 88.89%',C.red]];
  data.forEach((d,i)=>{ const x=74+i*377; shape(s,'roundRect',{left:x,top:220,width:322,height:250},C.pale,C.rule,1); text(s,d[0],x+22,245,60,18,13,d[4],true); text(s,d[1],x+22,284,252,28,22,C.ink,true); rule(s,x+22,329,246,d[4],3); text(s,d[2],x+22,354,252,49,17,C.muted); text(s,`LLM 호출  ${d[3]}`,x+22,423,252,22,16,C.navy,true); });
  shape(s,'roundRect',{left:74,top:505,width:1132,height:108},C.navy,C.navy,0);
  text(s,'평가 순서  |  Hard Gate 통과 → Recall@5 · NeedCoverage@5 → 분석 지연시간 · LLM 호출량',102,526,1020,22,18,'#D6E5F2',true);
  text(s,'결정  |  V1.5 간편 라우터 — Hard Gate 통과 · Top-5 근거 회수 성과 확보 · LLM 호출 7.75%',102,569,1030,22,18,C.white,true);
}

function hardGate(p){
  const s=p.slides.add();
  title(s,5,'02 QUERY ANALYSIS','질의 분석','Hard Gate는 지표가 높더라도 검색을 막거나 정답 근거를 잃게 만드는 후보를 먼저 탈락시키는 안전 기준입니다.');
  tag(s,'모든 비교 후보 공통 통과',76,195,230,C.teal);
  const data=[['실행 완전성','실행 오류 0건',C.blue],['검색 질의 유효성','빈 검색 질의 0건',C.blue],['정상 질문 보호','잘못된 OOS·DIRECT 종료 0건',C.teal],['Gold 근거 보호','Hard Filter 정답 업무 제외 0건',C.teal],['불필요한 중단 방지','검색 전 CLARIFY Precision 100%',C.amber]];
  data.forEach((d,i)=>{const row=i<3?0:1;const col=row===0?i:i-3;const x=row===0?76+col*375:264+col*375;const y=row===0?258:428;shape(s,'roundRect',{left:x,top:y,width:320,height:126},C.pale,C.rule,1);text(s,`0${i+1}`,x+20,y+18,42,18,13,d[2],true);text(s,d[0],x+20,y+49,255,24,19,C.ink,true);text(s,d[1],x+20,y+84,270,22,16,C.muted);});
  shape(s,'roundRect',{left:76,top:600,width:1130,height:42},C.navy,C.navy,0);text(s,'평가 순서  |  Hard Gate 통과 → Recall@5·NeedCoverage@5 → 분석 지연시간(평균·P95)·LLM 호출량',100,611,1080,18,17,C.white,true,{align:'center'});
  reheader(s,'02 QUERY ANALYSIS','질의 분석','Hard Gate는 지표가 높더라도 검색을 막거나 정답 근거를 잃게 만드는 후보를 먼저 탈락시키는 안전 기준입니다.');
}

function queryV15Profile(p){
  const s=p.slides.add();
  title(s,8,'02 QUERY ANALYSIS','질의 분석 (V1.5)','V1.5는 원문을 보존한 뒤, 교차 업무 복합질의에만 선택적으로 구조화 분해를 적용합니다.');
  tag(s,'V1.5 · 최종 선택',74,194,180,C.blue);
  // 공통 전처리: 실제 V1.5 순서대로 표현한다. 경로가 갈라지는 지점은 한 줄 흐름으로 연결하지 않는다.
  const top=[['사용자 질문',64,205,170],['원문 보존 · 최소 정리',274,205,210],['대화 상태·문맥 관계 판정',524,205,230],['최종 경로 결정',794,205,190]];
  top.forEach((d,i)=>{shape(s,'roundRect',{left:d[1],top:d[2],width:d[3],height:62},C.pale,C.blue,1);text(s,d[0],d[1]+12,d[2]+20,d[3]-24,22,16,C.ink,true,{align:'center'});if(i<top.length-1){text(s,'→',d[1]+d[3]+7,d[2]+18,26,24,21,C.muted,true,{align:'center'});}});
  shape(s,'roundRect',{left:1020,top:198,width:176,height:78},'#F6F8FA',C.rule,1);text(s,'DIRECT · OOS · CLARIFY\n검색 전 종료 / 추가질문',1032,216,152,39,14,C.muted,true,{align:'center',valign:'middle'});text(s,'↗',987,218,25,24,20,C.muted,true,{align:'center'});
  text(s,'↓',868,281,38,24,21,C.muted,true,{align:'center'});
  shape(s,'roundRect',{left:794,top:318,width:190,height:54},C.pale,C.teal,1);text(s,'RETRIEVE',810,335,158,20,16,C.teal,true,{align:'center'});
  text(s,'↓',868,376,38,24,21,C.muted,true,{align:'center'});
  shape(s,'roundRect',{left:672,top:408,width:434,height:58},C.pale,C.teal,1);text(s,'업무 탐지 + 질의 유형 판정',688,427,402,21,16,C.ink,true,{align:'center'});
  // 유형 판정 결과를 두 검색계획으로 분기한다.
  vrule(s,889,466,17,C.muted,2); rule(s,254,483,635,C.muted,2); vrule(s,254,483,19,C.muted,2); vrule(s,650,483,19,C.muted,2);
  text(s,'↓',242,485,24,18,16,C.muted,true,{align:'center'}); text(s,'↓',638,485,24,18,16,C.muted,true,{align:'center'});
  const branches=[
    ['SINGLE · SAME_BUSINESS_MULTI','원문 또는 문맥 결합 독립질의\n원문 1.0으로 검색',74,502,360,C.blue],
    ['CROSS_BUSINESS 후보','업무 2개 이상일 때만 HCX-007 호출\n구조화 분해 → 결과 검증',470,502,360,C.amber],
    ['분해 승인','원문 앵커 0.4 + 분해질의 총합 0.6\n가중치 합계 1.0',866,502,340,C.teal]
  ];
  branches.forEach(d=>{shape(s,'roundRect',{left:d[2],top:d[3],width:d[4],height:104},C.pale,d[5],1);text(s,d[0],d[2]+18,d[3]+17,d[4]-36,21,16,d[5],true,{align:'center'});text(s,d[1],d[2]+20,d[3]+51,d[4]-40,37,14,C.muted,false,{align:'center',valign:'middle'});});
  text(s,'→',834,541,26,24,21,C.muted,true,{align:'center'});
  text(s,'분해 거부·실패·API 오류 → 원문 1.0 fallback',470,620,470,19,14,C.muted,true,{align:'center'});
  shape(s,'roundRect',{left:74,top:642,width:1132,height:26},C.navy,C.navy,0);text(s,'선정 이유  |  단일·동일 업무 질문은 원문을 보존하고, 교차 업무 질문만 분해해 검색 범위와 지연시간을 함께 관리',98,647,1084,15,14,C.white,true,{align:'center'});
  text(s,'02 QUERY ANALYSIS',56,34,560,20,11,C.blue,true); text(s,'질의 분석 (V1.5)',56,62,1145,47,33,C.ink,true); text(s,'V1.5는 원문을 보존한 뒤, 교차 업무 복합질의에만 선택적으로 구조화 분해를 적용합니다.',56,117,1110,27,16,C.muted); rule(s,56,160,1168,C.rule,1);
}

function queryV22Profile(p){
  const s=p.slides.add();
  title(s,7,'02 QUERY ANALYSIS','질의 분석 (V2.2)','V2.2는 규칙 Fast Path와 선택적 LLM 분석을 결합해, 검색 전 안전성을 가장 촘촘하게 검증한 후보입니다.');
  tag(s,'V2.2 · 최소 경량화',74,194,190,C.amber);
  const nodes=[['원문·문맥 복원',74,C.blue],['DIRECT · OOS\n규칙',253,C.blue],['업무 없는 질문\nCLARIFY',446,C.blue],['명확한 질문\nFast Retrieve',639,C.teal],['필요 시 HCX\n최소 분석',832,C.amber],['Fail-open +\n검색 후 검사',1025,C.teal]];
  nodes.forEach((d,i)=>{shape(s,'roundRect',{left:d[1],top:252,width:155,height:86},C.pale,d[2],1);text(s,d[0],d[1]+12,275,131,40,16,C.ink,true,{align:'center',valign:'middle'});if(i<5)text(s,'→',d[1]+157,279,34,24,21,C.muted,true,{align:'center'});});
  commonGateStrip(s,358);
  const facts=[['규칙 Fast Path','약 48.5%\nLLM 호출 없이 처리',C.blue],['OOS Fail-open','애매한 범위 밖 질문은\n검색 후 근거로 재검사',C.teal],['검색 후 추가질문','2단계 Coverage 90%\n불필요 후검사 0건',C.teal],['비용·시간','LLM 139 / 270 · 51.48%\n평균 0.85초 · P95 1.93초',C.amber]];
  facts.forEach((d,i)=>{const x=74+i*280;shape(s,'roundRect',{left:x,top:405,width:245,height:108},C.pale,C.rule,1);text(s,d[0],x+18,426,170,20,16,d[2],true);text(s,d[1],x+18,459,208,36,17,C.ink,true);});
  shape(s,'roundRect',{left:74,top:570,width:1132,height:52},'#FFF8E7',C.amber,1);text(s,'해석  |  검색을 막지 않는 안전성은 검증됐지만, 애매한 질문마다 LLM을 쓰는 비율이 V1.5보다 높아 실시간 최종안으로는 선택하지 않았습니다.',102,586,1060,22,17,C.ink,true,{align:'center'});
  shape(s,'rect',{left:0,top:240,width:72,height:120},C.white,C.white,0);
  reheader(s,'02 QUERY ANALYSIS','질의 분석 (V2.2)','V2.2는 규칙 Fast Path와 선택적 LLM 분석을 결합해, 검색 전 안전성을 가장 촘촘하게 검증한 후보입니다.');
  text(s,'02 QUERY ANALYSIS',56,34,560,20,11,C.blue,true);
  text(s,'질의 분석 (V2.2)',56,62,1145,47,33,C.ink,true);
  text(s,'V2.2는 규칙 Fast Path와 선택적 LLM 분석을 결합해, 검색 전 안전성을 가장 촘촘하게 검증한 후보입니다.',56,117,1110,27,16,C.muted);
}

function queryV31Profile(p){
  const s=p.slides.add();
  title(s,6,'02 QUERY ANALYSIS','질의 분석 (V3.1)','V3.1은 질문 의미를 가장 풍부하게 구조화했지만, 실시간 검색 앞단으로는 비용과 재작성 위험이 컸습니다.');
  tag(s,'V3.1 · 전체 질의분석',74,194,190,C.red);
  const nodes=[['LLM 의미 분석',88,C.red],['의도·역할·Need\n구조화',310,C.red],['검색 질의\n재작성·분해',532,C.red],['HARD · SOFT · NONE\n필터 정책',754,C.amber],['검색 이관',976,C.teal]];
  nodes.forEach((d,i)=>{shape(s,'roundRect',{left:d[1],top:252,width:180,height:92},C.pale,d[2],1);text(s,d[0],d[1]+14,278,152,42,18,C.ink,true,{align:'center',valign:'middle'});if(i<4)text(s,'→',d[1]+184,280,34,24,22,C.muted,true,{align:'center'});});
  commonGateStrip(s,365);
  shape(s,'roundRect',{left:74,top:415,width:525,height:115},C.pale,C.rule,1);text(s,'얻은 정보',100,434,220,24,19,C.teal,true);text(s,'의도 분류 · 사용자 역할 · 누락 슬롯 · Need 단위 검색 질의\n→ 복잡한 질문을 가장 자세히 구조화',100,470,450,39,17,C.muted);
  shape(s,'roundRect',{left:680,top:415,width:525,height:115},C.pale,C.rule,1);text(s,'실시간 적용 시 부담',706,434,270,24,19,C.red,true);text(s,'LLM 240 / 270 · 88.89% · 평균 분석 2.00초\n재작성·분해가 좋은 원문 키워드를 약화할 위험',706,470,450,39,17,C.muted);
  shape(s,'roundRect',{left:74,top:575,width:1132,height:46},C.navy,C.navy,0);text(s,'비교 결과  |  구조화 범위는 넓지만, LLM 호출 88.89%·평균 2초·재작성 위험 때문에 최종안으로는 제외',100,588,1080,20,17,C.white,true,{align:'center'});
}

function queryTestEvidence(p){
  const s=p.slides.add();
  title(s,5,'02 QUERY ANALYSIS','질의분석 검증','질의분석은 안전성 Gate를 통과한 뒤, 복합질의 분해가 실제 검색 성과를 높이는지로 검증했습니다.');
  const cols=[
    ['안전성 Gate','실행 성공·검색 질의 유효','100%',C.blue],
    ['안전성 Gate','정상 KDIC 질문\n잘못된 OOS·DIRECT 종료','0건',C.blue],
    ['안전성 Gate','Hard Filter로\nGold 업무 제외','0건',C.blue],
  ];
  cols.forEach((d,i)=>{ const x=76+i*376, y=218; shape(s,'roundRect',{left:x,top:y,width:320,height:112},C.pale,C.rule,1); text(s,d[0],x+20,y+17,110,18,12,d[3],true); text(s,d[1],x+20,y+46,182,40,16,C.muted); text(s,d[2],x+214,y+52,86,28,21,C.ink,true,{align:'right'}); });
  shape(s,'roundRect',{left:76,top:370,width:500,height:182},C.pale,C.rule,1); tag(s,'복합질의 비교 조건',100,394,160,C.amber); text(s,'원문 검색  vs  분해 질의 검색',100,437,350,25,20,C.ink,true); text(s,'동일 검색기·Top-K·Gold 청크 기준으로 비교\n복합질의의 운영 경로에는 원문 혼합을 적용하지 않음',100,480,420,46,16,C.muted);
  shape(s,'roundRect',{left:620,top:370,width:580,height:182},C.pale,C.rule,1); tag(s,'검색 성과 확인 지표',646,394,160,C.teal); const metrics=[['Hit@3','상위 근거 진입'],['Recall@5','Top-5 근거 회수'],['nDCG@5','정답 근거 순위'],['NeedCoverage@5','복합 Need 충족']]; metrics.forEach((d,i)=>{const x=646+(i%2)*270,y=437+Math.floor(i/2)*50;text(s,d[0],x,y,244,20,16,i===0?C.blue:C.teal,true);text(s,d[1],x,y+23,244,18,14,C.muted);});
  shape(s,'roundRect',{left:76,top:588,width:1124,height:48},C.navy,C.navy,0); text(s,'결론  |  복합질의는 분해 질의 적용 시 검색 성과가 더 높아, V1.5의 복합질의 경로로 반영',98,602,1080,20,18,C.white,true,{align:'center'});
  tag(s,'세부 지표 수치 정리 후 반영',76,650,210,C.amber);
  reheader(s,'02 QUERY ANALYSIS','질의분석 검증','질의분석은 안전성 Gate를 통과한 뒤, 복합질의 분해가 실제 검색 성과를 높이는지로 검증했습니다.');
}

function queryFinalPipelineFinal(p){
  const s=p.slides.add();
  title(s,5,'02 QUERY ANALYSIS','V1.5 최종 질의분석 파이프라인','대화 문맥을 필요한 경우에만 결합하고, 교차 업무 복합질의만 원문 앵커와 분해질의로 검색합니다.');
  // 1) 공통 입력 처리
  const intake=[['사용자 질문',56,202,150],['원문 보존\n최소 정규화',242,202,178],['대화 상태 확인',456,202,160],['문맥 관계 판정',652,202,170],['최종 경로 결정',858,202,160]];
  intake.forEach((d,i)=>{shape(s,'roundRect',{left:d[1],top:d[2],width:d[3],height:62},C.pale,C.blue,1);text(s,d[0],d[1]+12,d[2]+17,d[3]-24,30,15,C.ink,true,{align:'center',valign:'middle'});if(i<intake.length-1)text(s,'→',d[1]+d[3]+5,d[2]+18,26,24,20,C.muted,true,{align:'center'});});
  shape(s,'roundRect',{left:1054,top:192,width:166,height:82},'#F6F8FA',C.rule,1);text(s,'DIRECT_RESPONSE\nOUT_OF_SCOPE · CLARIFY\n검색 전 종료·추가질문',1066,205,142,53,13,C.muted,true,{align:'center',valign:'middle'});text(s,'↗',1025,216,24,24,20,C.muted,true,{align:'center'});
  text(s,'↓',925,276,26,24,20,C.muted,true,{align:'center'});
  // 2) 검색 경로에서만 업무와 유형을 판정한다.
  shape(s,'roundRect',{left:858,top:305,width:160,height:48},C.pale,C.teal,1);text(s,'RETRIEVE',874,320,128,20,15,C.teal,true,{align:'center'});
  text(s,'↓',925,355,26,24,20,C.muted,true,{align:'center'});
  shape(s,'roundRect',{left:700,top:386,width:476,height:56},C.pale,C.teal,1);text(s,'업무 탐지 + 질의 유형 판정',716,404,444,22,16,C.ink,true,{align:'center'});
  // 3) 하나의 검색계획으로 끝나는 경로와 교차 업무 분해 경로를 병렬로 표현한다.
  vrule(s,938,442,15,C.muted,2); rule(s,245,457,693,C.muted,2); vrule(s,245,457,17,C.muted,2); vrule(s,640,457,17,C.muted,2);
  text(s,'↓',233,459,24,17,16,C.muted,true,{align:'center'}); text(s,'↓',628,459,24,17,16,C.muted,true,{align:'center'});
  shape(s,'roundRect',{left:70,top:462,width:350,height:90},C.pale,C.blue,1);text(s,'SINGLE · SAME_BUSINESS_MULTI',88,477,314,21,15,C.blue,true,{align:'center'});text(s,'원문 또는 문맥 결합 독립질의 1개\n검색계획: ORIGINAL 1.0',92,509,306,32,14,C.muted,false,{align:'center',valign:'middle'});
  shape(s,'roundRect',{left:465,top:462,width:350,height:90},C.pale,C.amber,1);text(s,'CROSS_BUSINESS 후보',483,477,314,21,15,C.amber,true,{align:'center'});text(s,'RETRIEVE + 복합 후보 + 업무 2개 이상일 때만\nHCX-007이 검색용 하위질의를 JSON으로 생성',487,509,306,33,13,C.muted,false,{align:'center',valign:'middle'});
  shape(s,'roundRect',{left:860,top:462,width:350,height:90},C.pale,C.amber,1);text(s,'분해 결과 검증',878,477,314,21,15,C.amber,true,{align:'center'});text(s,'하위질의 수·업무 포함·원문 조건 보존·\n용어 공유·confidence를 검사',882,509,306,32,13,C.muted,false,{align:'center',valign:'middle'});
  // 4) 승인과 fallback을 분명히 나눠, “분해 질의만 1.0”이라는 잘못된 읽기를 막는다.
  shape(s,'roundRect',{left:70,top:568,width:350,height:68},C.pale,C.teal,1);text(s,'단일·동일 업무 경로',86,580,318,18,14,C.teal,true,{align:'center'});text(s,'ORIGINAL 1.0 → 검색 시스템',86,607,318,18,14,C.muted,true,{align:'center'});
  shape(s,'roundRect',{left:465,top:568,width:350,height:68},C.pale,C.teal,1);text(s,'분해 승인',481,580,318,18,14,C.teal,true,{align:'center'});text(s,'원문 앵커 0.4 + 분해질의 총합 0.6',481,607,318,18,14,C.muted,true,{align:'center'});
  shape(s,'roundRect',{left:860,top:568,width:350,height:68},'#FFF8E7',C.amber,1);text(s,'분해 거부·실패·API 오류',876,580,318,18,14,C.amber,true,{align:'center'});text(s,'원문 1.0 fallback → 검색 중단 없음',876,607,318,18,14,C.muted,true,{align:'center'});
  text(s,'→',823,500,28,24,20,C.muted,true,{align:'center'});
  tag(s,'검색 직전 안전장치: 문맥 업무 보존 · 질의 존재 · 가중치 합계 1.0 · 업무 Hard Filter NONE',56,648,710,C.teal);
  tag(s,'유효한 검색계획만 검색 시스템으로 전달',786,648,390,C.blue);
}

function retrievalCandidateTest(p){
  const s=p.slides.add();
  title(s,7,'03 RETRIEVAL & EVALUATION','검색기법 후보와 베이스라인','BM25-Nori부터 Dense·Structured·Hybrid까지 차례로 확장하며, Gold 청크 회수 성과와 운영 비용을 비교했습니다.');
  const steps=[['후보','BM25-Nori\nBGE-M3 Dense·Sparse\n일반 Hybrid',C.blue],['구조 개선','제목·소제목·본문을\n함께 임베딩한\nStructured Dense',C.teal],['후보 재정렬','Structured Hybrid\n+ BGE Reranker\n+ Parent-Child',C.amber]];
  steps.forEach((d,i)=>{ const x=76+i*375; if(i<2){rule(s,x+320,372,55,C.muted,2); text(s,'›',x+355,359,18,24,22,C.muted,true,{align:'center'});} shape(s,'roundRect',{left:x,top:258,width:320,height:225},C.pale,C.rule,1); text(s,`0${i+1}`,x+22,282,42,20,14,d[3],true); text(s,d[0],x+22,326,220,28,23,C.ink,true); rule(s,x+22,369,242,d[3],3); text(s,d[1],x+22,399,245,61,18,C.muted); });
  shape(s,'roundRect',{left:76,top:547,width:1130,height:62},C.navy,C.navy,0); text(s,'테스트 기준  |  Gold 청크 기반 Hit·Recall·MRR·MAP·nDCG·Precision·F1  +  질문별 결과  +  검색 지연시간',102,567,1060,22,18,C.white,true);
}

function retrievalResultDecision(p){
  const s=p.slides.add();
  title(s,8,'03 RETRIEVAL & EVALUATION','검색 기법 및 평가','평가 결과는 “Hybrid이면 자동 개선”이 아니라, Structured 입력과 Reranker가 실제 개선 요인이었음을 보여줬습니다.');
  const rows=[['BM25-Nori',39.02,C.muted],['BGE-M3 Sparse',39.97,C.muted],['일반 Dense',49.31,C.blue],['Structured Dense',54.17,C.teal],['Structured Hybrid',54.52,C.teal],['Hybrid + Reranker',59.15,C.amber]];
  text(s,'7개 핵심 지표 단순평균(%)',76,208,390,23,18,C.ink,true);
  text(s,'Hit@3 · Recall@5 · MRR@10 · MAP@10 · nDCG@5 · Precision@5 · F1@5',76,234,630,18,13,C.muted);
  rows.forEach((d,i)=>{ const y=265+i*43; text(s,d[0],76,y,230,22,17,C.ink,i>=3); shape(s,'roundRect',{left:310,top:y+3,width:d[1]*6.0,height:17},d[2],d[2],0); text(s,`${d[1].toFixed(2)}%`,680,y,100,22,17,C.ink,true,{align:'right'}); });
  text(s,'※ Complete@5는 계산식 검증 이슈로 종합점수에 포함하지 않았습니다.',76,535,620,18,13,C.muted);
  shape(s,'roundRect',{left:832,top:225,width:350,height:297},C.pale,C.rule,1); tag(s,'TEST → DECISION',862,250,170,C.amber); text(s,'확인한 사실',862,304,220,28,22,C.ink,true); bullet(s,'Structured Dense\n일반 Dense 대비 +4.86%p',862,352,285,C.muted,16); bullet(s,'Structured Hybrid\nStructured Dense 대비 +0.35%p',862,420,285,C.muted,16); bullet(s,'Reranker\nStructured Hybrid 대비 +4.63%p',862,488,285,C.navy,16);
  shape(s,'roundRect',{left:76,top:570,width:1106,height:49},C.navy,C.navy,0); text(s,'결정  |  Structured Hybrid로 후보를 넓히고, BGE Reranker로 Top-20을 다시 정렬해 최종 Child Top-5를 선택',100,585,1050,20,17,C.white,true);
}

function retrievalResultDecisionFinal(p){
  const s=p.slides.add();
  title(s,8,'03 RETRIEVAL & EVALUATION','검색 기법 및 평가','평가 결과는 “Hybrid이면 자동 개선”이 아니라, Structured 입력과 Reranker가 실제 개선 요인이었음을 보여줬습니다.');
  const rows=[['BM25-Nori',39.02,C.muted],['BGE-M3 Sparse',39.97,C.muted],['일반 Dense',49.31,C.blue],['Structured Dense',54.17,C.teal],['Structured Hybrid',54.52,C.teal],['Hybrid + Reranker',59.15,C.amber]];
  text(s,'7개 핵심 지표 단순평균(%)',76,208,390,23,18,C.ink,true);
  text(s,'Hit@3 · Recall@5 · MRR@10 · MAP@10 · nDCG@5 · Precision@5 · F1@5',76,234,630,18,13,C.muted);
  rows.forEach((d,i)=>{ const y=265+i*43; text(s,d[0],76,y,230,22,17,C.ink,i>=3); shape(s,'roundRect',{left:310,top:y+3,width:d[1]*6.0,height:17},d[2],d[2],0); text(s,`${d[1].toFixed(2)}%`,680,y,100,22,17,C.ink,true,{align:'right'}); });
  text(s,'※ Complete@5는 계산식 검증 이슈로 종합점수에 포함하지 않았습니다.',76,535,620,18,13,C.muted);
  shape(s,'roundRect',{left:832,top:225,width:350,height:315},C.pale,C.rule,1); tag(s,'TEST → DECISION',862,250,170,C.amber); text(s,'확인한 사실',862,304,220,28,22,C.ink,true);
  text(s,'Structured 입력',862,355,140,22,18,C.ink,true); text(s,'일반 Dense 대비  +4.86%p',862,381,260,20,16,C.muted);
  text(s,'BM25 추가',862,426,140,22,18,C.ink,true); text(s,'Structured Dense 대비  +0.35%p',862,452,270,20,16,C.muted);
  text(s,'Reranker 적용',862,478,140,22,18,C.navy,true); text(s,'Structured Hybrid 대비  +4.63%p',862,504,270,20,16,C.navy,true);
  shape(s,'roundRect',{left:76,top:570,width:1106,height:49},C.navy,C.navy,0); text(s,'결정  |  Structured Hybrid로 후보를 넓히고, BGE Reranker로 Top-20을 다시 정렬해 최종 Child Top-5를 선택',100,585,1050,20,17,C.white,true);
  text(s,'03 RETRIEVAL & EVALUATION',56,34,560,20,11,C.blue,true); text(s,'검색 기법 및 평가',56,62,1145,47,33,C.ink,true); text(s,'평가 결과는 “Hybrid이면 자동 개선”이 아니라, Structured 입력과 Reranker가 실제 개선 요인이었음을 보여줬습니다.',56,117,1110,27,16,C.muted);
}

function retrievalMetricDecision(p){
  const s=p.slides.add();
  title(s,11,'03 RETRIEVAL & EVALUATION','베이스라인 비교 결과','후보를 단순평균으로 줄 세우지 않고, 실제 답변 근거에 직접 영향을 주는 핵심 지표로 Reranker 효과를 확인했습니다.');
  tag(s,'선정 지표',76,204,130,C.blue);
  const metrics=[['Hit@3','상위 3개 안에 Gold 근거가\n하나라도 있는가?',C.blue],['Recall@5','답변에 전달할 Top-5가\nGold를 얼마나 회수했는가?',C.teal],['nDCG@5','정답 근거가 상위 순위에\n배치됐는가?',C.amber],['F1@5','근거 회수와 불필요한 청크\n사이의 균형은 어떤가?',C.red]];
  metrics.forEach((d,i)=>{const x=76+i*278;shape(s,'roundRect',{left:x,top:244,width:244,height:145},C.pale,C.rule,1);text(s,d[0],x+18,269,180,26,22,d[2],true);rule(s,x+18,309,206,d[2],3);text(s,d[1],x+18,331,208,42,16,C.muted);});
  shape(s,'roundRect',{left:76,top:415,width:1130,height:178},C.pale,C.rule,1);tag(s,'RERANKER 재정렬 효과  |  Structured Hybrid 대비',102,435,315,C.amber);
  const tx=[112,372,657,939];
  const tw=[235,260,255,215];
  const headers=['핵심 지표','Structured Hybrid','Hybrid + Reranker','변화'];
  shape(s,'roundRect',{left:100,top:468,width:1082,height:25},C.lightBlue,C.lightBlue,0);
  headers.forEach((h,i)=>text(s,h,tx[i],474,tw[i],15,13,C.navy,true,{align:i===0?'left':'center'}));
  const rows=[['Hit@3','72.50%','80.83%','+8.33%p'],['Recall@5','72.64%','80.42%','+7.78%p'],['nDCG@5','62.99%','68.00%','+5.01%p']];
  rows.forEach((r,i)=>{const y=501+i*27;rule(s,100,y,1082,C.rule,1);text(s,r[0],tx[0],y+7,tw[0],16,15,C.ink,true);text(s,r[1],tx[1],y+7,tw[1],16,16,C.muted,true,{align:'center'});text(s,r[2],tx[2],y+7,tw[2],16,16,C.blue,true,{align:'center'});text(s,r[3],tx[3],y+7,tw[3],16,16,C.teal,true,{align:'center'});});
  shape(s,'roundRect',{left:76,top:608,width:1130,height:40},C.navy,C.navy,0);text(s,'결정  |  Structured Hybrid만의 변화는 작았고, Reranker 적용 후 핵심 지표가 함께 개선돼 최종 검색기로 채택',100,619,1080,20,16,C.white,true,{align:'center'});
}

function retrievalFinalPipeline(p){
  const s=p.slides.add();
  title(s,9,'03 RETRIEVAL & EVALUATION','최종 검색 파이프라인','베이스라인 비교에서 확인한 Structured 입력·Reranker 효과를 반영해, 최종 검색기를 구성했습니다.');
  const nodes=[['BGE-M3\nStructured Dense\nTop-20',70,245,185,96,C.blue],['BM25 Nori-none\nTop-20',70,390,185,72,C.teal],['Min-Max 정규화\nDense : BM25 = 0.7 : 0.3',330,285,250,82,C.blue],['복수 질의면\n가중 RRF K=10\n통합 후보 Top-20',650,275,230,100,C.amber],['BAAI Reranker\n입력 20 · 배치 8\n최대 512 토큰',955,275,220,100,C.red],['최종 Child Top-5\n→ Parent 문맥 확장\nParent당 최대 8,192자',955,455,220,100,C.teal]];
  rule(s,255,293,45,C.muted,2); vrule(s,300,293,33,C.muted,2); rule(s,255,426,45,C.muted,2); vrule(s,300,326,100,C.muted,2); rule(s,300,326,30,C.muted,2); rule(s,580,326,70,C.muted,2); rule(s,880,326,75,C.muted,2); text(s,'›',310,313,18,24,22,C.muted,true,{align:'center'}); text(s,'›',632,313,18,24,22,C.muted,true,{align:'center'}); text(s,'›',937,313,18,24,22,C.muted,true,{align:'center'}); text(s,'↓',1055,394,20,28,22,C.muted,true,{align:'center'});
  nodes.forEach(d=>{shape(s,'roundRect',{left:d[1],top:d[2],width:d[3],height:d[4]},C.pale,d[5],1);text(s,d[0],d[1]+14,d[2]+16,d[3]-28,d[4]-24,17,C.ink,true,{align:'center',valign:'middle'});});
  shape(s,'roundRect',{left:70,top:590,width:1105,height:48},C.navy,C.navy,0); text(s,'운영 정책  |  업무 필터 NONE · Hard Filter 미사용 · 매칭 Child는 유지하고 Parent는 답변 근거만 확장',94,605,1058,20,17,C.white,true);
  reheader(s,'03 RETRIEVAL & EVALUATION','최종 검색 파이프라인','베이스라인 비교에서 확인한 Structured 입력·Reranker 효과를 반영해, 최종 검색기를 구성했습니다.');
  text(s,'03 RETRIEVAL & EVALUATION',56,34,560,20,11,C.blue,true); text(s,'최종 검색 파이프라인',56,62,1145,47,33,C.ink,true); text(s,'베이스라인 비교에서 확인한 Structured 입력·Reranker 효과를 반영해, 최종 검색기를 구성했습니다.',56,117,1110,27,16,C.muted);
}

function answerCandidateTest(p){
  const s=p.slides.add();
  title(s,7,'04 ANSWER SYSTEM','답변 시스템','같은 검색 근거를 사용한 상태에서, 답변 전 중간 구조만 바꿔 B·C·D안을 비교했습니다.');
  tag(s,'공통 비교 조건  |  동일 질문 · 동일 검색 Top-K·순서 · HCX-007 · Temperature 0 · 출력 형식·출처 UI 동일',76,190,1125,C.blue);
  const cards=[
    ['B안','Basic Evidence Pack','Top-5를 제목·소제목·본문·URL로\n정리한 뒤 기본 답변 1회 생성','장점  입력 경계·출처가 명확\n한계  혼동 사실을 별도 보강하지 못함','비교 기준선\n단독 운영안 미채택',C.blue],
    ['C안','Fact Index 보강','Basic Evidence Pack +\n사전 검수 Fact Index → 답변 1회','장점  역할·조건·예외 혼동을\n추가 LLM 호출 없이 보강','최종 기본 답변안',C.teal],
    ['D안','Answer Skeleton','LLM이 원본 Top-5에서\nAnswer Skeleton 생성 → 기본 답변','장점  답변 항목·Claim·근거를\n동적으로 구조화','선택 적용\nLLM 2회 호출',C.amber],
  ];
  cards.forEach((d,i)=>{const x=70+i*390;shape(s,'roundRect',{left:x,top:246,width:350,height:300},C.pale,d[5],1);tag(s,d[0],x+20,268,65,d[5]);text(s,d[1],x+20,315,292,30,22,C.ink,true);rule(s,x+20,358,302,d[5],3);text(s,d[2],x+20,382,305,48,16,C.navy,true);rule(s,x+20,448,302,C.rule,1);text(s,d[3],x+20,470,306,50,15,C.muted);text(s,d[4],x+20,501,306,30,14,d[5],true,{align:'right'});});
  shape(s,'roundRect',{left:70,top:584,width:1140,height:48},C.navy,C.navy,0);
  text(s,'결정 기준  |  답변 정확성 · 필수 사실 누락 · 근거 충실성 · 위험 오류 · 응답시간 · LLM 호출·검수 부담',92,598,1095,20,16,C.white,true,{align:'center'});
}

function answerFactIndexPipeline(p){
  const s=p.slides.add();
  title(s,8,'04 ANSWER SYSTEM','답변 시스템','C안은 검색 근거를 대체하지 않고, 사람이 검수한 혼동 방지 정보를 필요한 경우에만 덧붙입니다.');
  const nodes=[
    ['Top-5 검색·리랭킹 청크',58,277,180,76,C.blue],
    ['Basic Evidence Pack\n제목·본문·URL 정리',278,277,190,76,C.blue],
    ['Fact Index 연결\nchunk_id·업무·혼동유형',508,277,198,76,C.teal],
    ['보강 Evidence Pack\n원본 근거 + 검증 정보',746,277,198,76,C.teal],
    ['기본 답변 생성\nLLM 1회',984,277,180,76,C.amber],
  ];
  nodes.forEach((d,i)=>{shape(s,'roundRect',{left:d[1],top:d[2],width:d[3],height:d[4]},C.pale,d[5],1);text(s,d[0],d[1]+12,d[2]+17,d[3]-24,d[4]-28,16,C.ink,true,{align:'center',valign:'middle'});if(i<nodes.length-1){rule(s,d[1]+d[3],314,34,C.muted,2);text(s,'›',d[1]+d[3]+9,300,16,24,22,C.muted,true,{align:'center'});}});
  shape(s,'roundRect',{left:76,top:415,width:485,height:163},C.pale,C.rule,1);tag(s,'Fact Index 예시',100,438,125,C.teal);text(s,'VS-0011 · BI-001_chunk_003 · eligibility',100,480,400,20,15,C.ink,true);text(s,'“미성년자 예금보험금은 친권자 또는\n후견인이 수령할 수 있다.”',100,511,390,42,17,C.navy,true);text(s,'Guardrail  |  친권자와 일반 부모를 동일하게 표현하지 않음',100,558,420,16,12,C.muted);
  shape(s,'roundRect',{left:603,top:415,width:565,height:163},'#FFF8E7',C.amber,1);tag(s,'적용 규칙',628,438,100,C.amber);bullet(s,'승인(APPROVED)된 Fact Index만 연결',628,479,470,C.ink,16);bullet(s,'원본 Evidence를 삭제·덮어쓰지 않음',628,517,470,C.ink,16);bullet(s,'관련 Index가 없으면 Basic Evidence Pack과 동일하게 처리',628,555,490,C.ink,16);
  shape(s,'roundRect',{left:76,top:610,width:1092,height:35},C.navy,C.navy,0);text(s,'효과  |  Fact Sheet를 매 질문마다 새로 만들지 않아도, 검수된 사실·금지 주장을 Evidence Pack에서 바로 활용',96,619,1050,18,15,C.white,true,{align:'center'});
  text(s,'04 ANSWER SYSTEM',56,34,560,20,11,C.blue,true); text(s,'답변 시스템',56,62,1145,47,33,C.ink,true); text(s,'C안은 검색 근거를 대체하지 않고, 사람이 검수한 혼동 방지 정보를 필요한 경우에만 덧붙입니다.',56,117,1110,27,16,C.muted); rule(s,56,160,1168);
}

function answerRoutingDecision(p){
  const s=p.slides.add();
  title(s,9,'04 ANSWER SYSTEM','답변 시스템','한 가지 답변 구조를 모든 질문에 강제하지 않고, 질문의 관계 복잡도에 따라 C와 D를 구분했습니다.');
  const rows=[
    ['기본·단일 사실 질문','C안','Fact Index 보강 Evidence Pack','1회','검수된 혼동 규칙을 적용하면서 빠르게 답변',C.teal],
    ['복합질문 · 같은 업무 안의 조건·절차','C안','Fact Index 보강 Evidence Pack','1회','필수 조건·예외를 근거와 함께 유지',C.teal],
    ['서로 다른 업무를 비교하는 질문','D안','LLM Answer Skeleton → 기본 답변','2회','LLM이 업무별 Claim·근거를 분리해 비교 구조를 구성',C.amber],
  ];
  const xs=[82,380,520,842,950], ws=[270,110,292,80,210];
  shape(s,'roundRect',{left:70,top:220,width:1140,height:46},C.lightBlue,C.lightBlue,0);
  ['질문 유형','운영안','답변 전 중간 구조','LLM','선택 이유'].forEach((h,i)=>text(s,h,xs[i],235,ws[i],16,14,C.navy,true,{align:i===0||i===4?'left':'center'}));
  rows.forEach((r,i)=>{const y=267+i*94;shape(s,'roundRect',{left:70,top:y,width:1140,height:88},i===2?'#FFF8E7':C.pale,r[5],1);text(s,r[0],xs[0],y+25,ws[0],38,17,C.ink,true,{valign:'middle'});tag(s,r[1],xs[1]+12,y+28,76,r[5]);text(s,r[2],xs[2],y+23,ws[2],42,16,C.navy,true,{align:'center',valign:'middle'});text(s,r[3],xs[3],y+32,ws[3],22,17,r[5],true,{align:'center'});text(s,r[4],xs[4],y+23,ws[4],42,15,C.muted,false,{valign:'middle'});});
  shape(s,'roundRect',{left:70,top:564,width:1140,height:67},C.navy,C.navy,0);text(s,'최종 선정  |  B안은 비교 기준선으로 유지 · C안은 기본 답변안 · D안은 서로 다른 업무의 조건·절차를 비교하는 질문에만 선택 적용',94,578,1090,22,17,C.white,true,{align:'center'});text(s,'D안은 Basic Evidence Pack을 선행하지 않고, LLM이 원본 Top-5에서 Skeleton을 생성해 그 효과를 분리 평가',94,606,1090,16,13,'#D6E5F2',false,{align:'center'});
  reheader(s,'04 ANSWER SYSTEM','답변 시스템: C·D 적용 기준','한 가지 답변 구조를 모든 질문에 강제하지 않고, 질문의 관계 복잡도에 따라 C와 D를 구분했습니다.');
  text(s,'04 ANSWER SYSTEM',56,34,560,20,11,C.blue,true); text(s,'답변 시스템: C·D 적용 기준',56,62,1145,47,33,C.ink,true); text(s,'한 가지 답변 구조를 모든 질문에 강제하지 않고, 질문의 관계 복잡도에 따라 C와 D를 구분했습니다.',56,117,1110,27,16,C.muted);
}

function answerExperienceAndCache(p){
  const s=p.slides.add();
  title(s,10,'04 ANSWER SYSTEM','답변 시스템','사용자에게는 결론과 신청 행동을 먼저 보여주고, 판단 근거는 저장된 데이터로 다시 확인할 수 있게 했습니다.');
  screen(s,IMG.answer,68,205,458,365,'근거 기반 기본 답변','contain');
  screen(s,IMG.cache,736,241,390,300,'질의 캐시 적용 화면','cover');
  shape(s,'roundRect',{left:565,top:205,width:135,height:365},C.pale,C.rule,1);tag(s,'근거 보기',580,228,104,C.blue);text(s,'새 LLM\n호출 없음',580,278,102,48,17,C.ink,true,{align:'center'});rule(s,580,344,102,C.rule,1);text(s,'사용 청크\n핵심 판단\nFact Index\n누락 정보\n공식 출처',580,370,104,136,15,C.muted,true,{align:'center'});
  shape(s,'roundRect',{left:68,top:600,width:1058,height:38},C.navy,C.navy,0);text(s,'질의 캐시  |  검증된 정형 질문만 재사용 · Cache HIT는 LLM 토큰 0 · 실제 응답시간과 절감량은 운영 로그로 분리 기록',90,610,1016,18,15,C.white,true,{align:'center'});
  shape(s,'rect',{left:0,top:0,width:1280,height:170},C.white,C.white,0); text(s,'04 ANSWER SYSTEM',56,34,560,20,11,C.blue,true); text(s,'답변 시스템',56,62,1145,47,33,C.ink,true); text(s,'사용자에게는 결론과 신청 행동을 먼저 보여주고, 판단 근거는 저장된 데이터로 다시 확인할 수 있게 했습니다.',56,117,1110,27,16,C.muted); rule(s,56,160,1168);
}
function dashboard(p){ const s=p.slides.add(); title(s,10,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','운영 현황과 반영 흐름을 한 화면에 배치해 현재 상태와 다음 행동을 함께 확인합니다.'); screen(s,IMG.dashboard,68,188,1144,438,'관리자 UI 대시보드','contain'); text(s,'초안 → 비교 → 승인 → Rollback',68,648,520,24,18,C.navy,true); }
function adminDesign(p){ const s=p.slides.add(); title(s,11,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','운영자가 코드 없이도 변경을 이해하고, 검증하고, 되돌릴 수 있어야 합니다.'); const rows=[['운영 서버에서 코드 직접 수정','배포 위험과 책임 범위 확대','코드는 로컬·Git에서 관리\n관리자 화면에는 검증된 조정 항목만 제공',C.red],['여러 화면에서 값 각각 수정','설정 불일치와 재현성 저하','이름 붙인 파라미터 설정 저장\n모든 비교 화면에서 같은 설정 호출',C.blue],['실험값 즉시 반영','사용자 답변 품질 변동','초안 → 평가 → 답변 비교 → 승인 → 롤백',C.teal]]; rows.forEach((d,i)=>{const y=218+i*125;text(s,`0${i+1}`,74,y+9,42,20,14,d[3],true);text(s,'문제',126,y,54,22,15,C.muted,true);text(s,d[0],126,y+28,330,25,19,C.ink,true);text(s,d[1],126,y+59,330,24,16,C.muted);rule(s,520,y+6,2,d[3],85);text(s,'설계 결정',550,y,90,22,15,d[3],true);text(s,d[2],550,y+31,520,52,18,C.navy,true);rule(s,74,y+103,1010,C.rule,1);}); reheader(s,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','운영자가 코드 없이도 변경을 이해하고, 검증하고, 되돌릴 수 있어야 합니다.'); }
function safeChange(p){ const s=p.slides.add(); title(s,12,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','파라미터 변경은 초안에서 시작해 평가와 답변 비교를 거친 뒤에만 반영합니다.'); const stages=[['01','파이프라인','변경 가능한 블록에서\n파라미터 초안 작성',C.blue],['02','설정 창고','이름을 붙여 저장\n같은 설정 재사용',C.teal],['03','검색·답변 비교','테스트셋과 실제 질문으로\n현재안·개선안 확인',C.amber],['04','반영 센터','승인 후 적용\n이력 보관·롤백',C.red]]; stages.forEach((d,i)=>{const x=68+i*284;shape(s,'roundRect',{left:x,top:285,width:224,height:190},C.pale,C.rule,1);text(s,d[0],x+22,307,50,20,13,d[3],true);text(s,d[1],x+22,351,176,28,22,C.ink,true);text(s,d[2],x+22,399,180,52,17,C.muted);if(i<3) text(s,'→',x+228,357,44,28,24,C.muted,true,{align:'center'});}); shape(s,'roundRect',{left:68,top:545,width:1100,height:54},C.navy,C.navy,0); text(s,'운영 화면에서는 코드 자체를 수정하지 않는다. 검증 가능한 값만 초안으로 관리한다.',94,561,1040,22,18,C.white,true,{align:'center'}); }
function candidateTestDecision(p){ const s=p.slides.add(); title(s,6,'03 RETRIEVAL & EVALUATION','검색 기법 및 평가','후보를 넓게 비교한 뒤, 동일 조건의 테스트를 거쳐 최종 검색 구성을 결정했습니다.'); const data=[['후보','BM25-Nori\nBGE-M3 Dense·Sparse\nHybrid·Reranker','검색 방식별 강점과\n비용·지연시간을 함께 검토',C.blue],['테스트','Gold 청크 기반\nHit·Recall·MRR·MAP\nPrecision·F1','문항별 검색 결과까지\n확인해 평균 지표를 해석',C.amber],['결정','Structured Hybrid\n+ BGE Reranker\n+ Parent-Child','근거 회수와 정렬 품질을\n균형 있게 확보',C.teal]]; data.forEach((d,i)=>{const x=76+i*375;if(i<2) text(s,'→',x+326,365,40,36,28,C.muted,true,{align:'center'});shape(s,'roundRect',{left:x,top:248,width:320,height:304},C.pale,C.rule,1);text(s,`0${i+1}`,x+24,273,40,20,14,d[3],true);text(s,d[0],x+24,318,230,30,25,C.ink,true);rule(s,x+24,362,246,d[3],3);text(s,d[1],x+24,391,252,74,18,C.navy,true);text(s,d[2],x+24,489,254,45,17,C.muted);}); text(s,'결정 기준: 지표 상승만 보지 않고, 질문별 Gold 일치·최종 답변 근거·운영 비용을 함께 확인',76,604,1080,27,19,C.navy,true); }
function evaluationPrinciple(p){ const s=p.slides.add(); title(s,7,'03 RETRIEVAL & EVALUATION','검색 기법 및 평가','검색 품질은 “얼마나 찾았는가”와 “상위 결과가 얼마나 정확한가”를 함께 봅니다.'); const cols=[['비교 조건','현재안과 개선안의\nTop-K·평가 K를 명확히 표시','K가 다르면 순위 범위 효과도\n함께 발생',C.blue],['해석 기준','Recall·Complete\n근거를 놓치지 않는 정도','Precision·F1\n불필요한 청크를 줄이는 정도',C.teal],['질문별 확인','Gold 청크와 검색 결과를\n문항 단위로 대조','다중 청크·답변 불가 문항은\n별도 판단',C.amber]]; cols.forEach((d,i)=>{const x=76+i*375;shape(s,'roundRect',{left:x,top:250,width:320,height:285},C.pale,C.rule,1);text(s,`0${i+1}`,x+24,274,42,20,14,d[3],true);text(s,d[0],x+24,317,220,28,22,C.ink,true);rule(s,x+24,359,240,d[3],3);text(s,d[1],x+24,387,248,52,17,C.muted);text(s,d[2],x+24,460,254,52,17,C.navy,true);}); text(s,'지표 평균 하나로 반영하지 않고, 동일 조건의 수치·질문별 사례·최종 답변을 함께 확인합니다.',76,603,1080,28,19,C.navy,true); /* header repeated last to avoid renderer z-order omission */ text(s,'03 RETRIEVAL & EVALUATION',56,34,560,20,11,C.blue,true); text(s,'검색 기법 및 평가',56,62,1145,47,33,C.ink,true); text(s,'검색 품질은 “얼마나 찾았는가”와 “상위 결과가 얼마나 정확한가”를 함께 봅니다.',56,117,1110,27,16,C.muted); }
function observability(p){ const s=p.slides.add(); title(s,13,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','캐시 응답과 LLM 생성 응답을 나눠 기록해, 속도와 비용을 정확히 해석합니다.'); const data=[['CACHE HIT','LLM 토큰 0\n캐시 조회 시간','최초 생성 대비 절감 시간·토큰은\n별도 정보로 보관',C.teal],['CACHE MISS','실제 LLM 토큰\n실제 생성 응답시간','비용·성능 원인 분석의\n기준 데이터',C.blue],['모니터링','P95 응답시간\n질의량·업무 분포','시간대·업무별 병목을\n운영 개선 대상으로 전환',C.amber]]; data.forEach((d,i)=>{const x=76+i*375;shape(s,'roundRect',{left:x,top:250,width:320,height:280},C.pale,C.rule,1);tag(s,d[0],x+24,276,144,d[3]);text(s,d[1],x+24,334,248,52,19,C.ink,true);rule(s,x+24,412,245,C.rule,1);text(s,d[2],x+24,440,252,50,17,C.muted);}); text(s,'트러블슈팅  |  캐시 HIT에도 최초 생성 시간이 표시되던 문제를 요청별 실제 응답시간 기준으로 바로잡았습니다.',76,603,1080,28,19,C.navy,true); reheader(s,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','캐시 응답과 LLM 생성 응답을 나눠 기록해, 속도와 비용을 정확히 해석합니다.'); }
function monitoring(p){ const s=p.slides.add(); title(s,14,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','운영 현황에서는 추세를 보고, 챗봇 로그에서는 개별 요청의 실행 기록을 확인합니다.'); screen(s,IMG.navMonitoring,72,245,230,245,'운영 모니터링 메뉴','contain'); screen(s,IMG.monitoring,350,206,820,405,'운영 모니터링 대시보드','contain'); tag(s,'운영 현황 · 챗봇 로그',72,548,230,C.blue); text(s,'왼쪽 메뉴에서 전체 현황과 요청별 로그를 오가며 확인',72,592,235,44,16,C.muted); }
function logs(p){ const s=p.slides.add(); title(s,15,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','요청 단위의 캐시 HIT/MISS·토큰·응답시간을 기록해 운영 지표의 근거를 남깁니다.'); screen(s,IMG.logs,54,190,1172,418,'챗봇 로그 테이블','contain'); tag(s,'CACHE HIT: 이번 요청에서 LLM 토큰 0 · 캐시 조회 시간',62,628,480,C.teal); tag(s,'CACHE MISS: 실제 LLM 토큰 · 실제 응답시간',558,628,410,C.blue); reheader(s,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','요청 단위의 캐시 HIT/MISS·토큰·응답시간을 기록해 운영 지표의 근거를 남깁니다.'); text(s,'05 ADMIN UI & OPERATIONS',56,34,560,20,11,C.blue,true); text(s,'관리자 UI 및 운영',56,62,1145,47,33,C.ink,true); text(s,'요청 단위의 캐시 HIT/MISS·토큰·응답시간을 기록해 운영 지표의 근거를 남깁니다.',56,117,1110,27,16,C.muted); }
function enhancement(p){ const s=p.slides.add(); title(s,14,'ENHANCEMENT MENU','검색·답변 개선을 하나의 작업 흐름으로','코드 세부 구현을 직접 바꾸지 않아도, 검증 가능한 설정을 선택할 수 있습니다.'); screen(s,IMG.navEnhance,82,222,254,386,'챗봇 고도화 메뉴','contain'); const items=[['파이프라인','현재 런타임 흐름과 변경 가능한 블록 확인'],['파라미터','저장된 검색 설정을 선택·비교'],['검색 비교 평가','테스트셋 기반의 품질 검증'],['챗봇 테스트','현재안과 개선안의 최종 답변 비교'],['프롬프트','답변 정책 변경과 버전 관리']]; items.forEach((d,i)=>{const y=212+i*78; text(s,`0${i+1}`,410,y,42,20,13,C.blue,true);text(s,d[0],460,y-3,200,24,20,C.ink,true);text(s,d[1],460,y+28,570,23,16,C.muted);rule(s,410,y+61,690);}); }
function pipeline(p){ const s=p.slides.add(); title(s,5,'03 RETRIEVAL & EVALUATION','검색 기법 및 평가','Structured Hybrid 검색을 기반으로 후보를 확보하고, 재정렬·근거 확장으로 답변 근거를 구성합니다.'); screen(s,IMG.pipeline,44,174,1192,438,'운영 파이프라인 설정','contain'); tag(s,'질의·대화 → 분석 → 검색·근거 → 답변 생성 → 검증·출력',58,632,600,C.blue); }
function parameters(p){ const s=p.slides.add(); title(s,16,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','검색 비중·후보 깊이·최소 관련성 점수·Parent-Child를 하나의 이름 있는 설정으로 관리합니다.'); screen(s,IMG.parameters,56,205,720,410,'파이프라인 파라미터 설정','contain'); shape(s,'roundRect',{left:824,top:220,width:340,height:370},C.pale,C.rule,1); tag(s,'빠른 수정',850,246,105,C.amber); text(s,'저장 전, 한 번에\n값을 조정합니다.',850,297,270,58,24,C.ink,true); rule(s,850,376,284,C.rule,1); const items=[['01','키보드 이동','숫자를 연속 입력해 빠르게 조정'],['02','가중치 자동 보정','Dense·BM25 비중 합계는 1로 유지'],['03','이름 저장','비교·답변 테스트·반영에서 같은 설정 재사용']]; items.forEach((d,i)=>{const y=404+i*56;text(s,d[0],850,y,34,16,12,C.amber,true);text(s,d[1],892,y-2,130,18,15,C.ink,true);text(s,d[2],892,y+20,230,18,12,C.muted);}); shape(s,'roundRect',{left:56,top:630,width:1108,height:34},C.navy,C.navy,0);text(s,'설정 흐름  |  이름 있는 파라미터 저장 → 테스트셋 검색 평가 → 최종 답변 비교 → 반영 센터 승인',80,639,1060,18,15,C.white,true,{align:'center'}); reheader(s,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','검색 비중·후보 깊이·최소 관련성 점수·Parent-Child를 하나의 이름 있는 설정으로 관리합니다.'); }
function chatbotTest(p){ const s=p.slides.add(); title(s,17,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','같은 질문을 현재안과 개선안으로 실행해 최종 사용자 답변까지 확인합니다.'); screen(s,IMG.chatbotTest,44,185,1192,420,'최종 챗봇 답변 비교','contain'); }
function promptApproval(p){ const s=p.slides.add(); title(s,18,'05 ADMIN UI & OPERATIONS','관리자 UI 및 운영','프롬프트·파라미터·청크 변경은 반영 전 검토하고, 승인 이후 이력과 함께 운영합니다.'); screen(s,IMG.prompts,40,183,580,420,'프롬프트 관리','contain'); screen(s,IMG.approval,660,183,580,420,'통합 반영 센터','contain'); tag(s,'가드레일: 개인정보·민감정보·금칙어 대응',52,633,350,C.red); tag(s,'API 키: 서버 비밀로 관리, 관리자 화면에는 노출하지 않음',430,633,500,C.blue); tag(s,'청크 관리: JSON/JSONL 초안 추가 → 승인 후 반영',952,633,290,C.teal); }
function closing(p){ const s=p.slides.add(); s.background.fill=C.navy; text(s,'CLOSING',72,70,220,20,12,'#A7D8F5',true); text(s,'많은 시도를\n운영 가능한 원칙으로',72,160,740,145,51,C.white,true); rule(s,72,345,360,C.cyan,4); text(s,'정확한 검색과 답변을 넘어\n검증·관찰·안전한 변경까지 설계했습니다.',72,390,700,65,23,'#D6E5F2'); const words=[['학습','질문과 기법을 실험'],['검증','근거·답변을 비교'],['운영','승인·로그·롤백']]; words.forEach((d,i)=>{const x=810+i*135;shape(s,'roundRect',{left:x,top:410,width:112,height:112},i===1?C.cyan:'#1C3D5D','#1C3D5D',0);text(s,d[0],x+12,433,88,26,19,i===1?C.navy:C.white,true,{align:'center'});text(s,d[1],x+10,474,92,28,11,i===1?C.navy:'#C8D9E8',false,{align:'center'});}); text(s,'6조 · 유민규(팀장) · 임도균 · 박주영',72,650,540,22,15,'#A7D8F5'); }

async function main(){
  const p=Presentation.create({slideSize:{width:W,height:H}});
  const builders=[cover,agenda,concerns,queryCandidateDecisionFinal,hardGate,queryV31Profile,queryV22Profile,queryV15Profile,queryTestEvidence,queryFinalPipelineFinal,retrievalCandidateTest,retrievalMetricDecision,retrievalFinalPipeline,answerCandidateTest,answerFactIndexPipeline,answerRoutingDecision,answerExperienceAndCache,dashboard,adminDesign,safeChange,observability,monitoring,logs,parameters,chatbotTest,promptApproval,closing];
  for(const build of builders) await build(p);
  await fs.mkdir('C:/Users/임도균/Documents/Codex/2026-07-22/d/outputs',{recursive:true});
  for(const [i,slide] of p.slides.items.entries()) await saveBlob(`C:/Users/임도균/Documents/Codex/2026-07-22/d/ppt_build/final-slide-${String(i+1).padStart(2,'0')}.png`,await p.export({slide,format:'png',scale:1}));
  await saveBlob('C:/Users/임도균/Documents/Codex/2026-07-22/d/ppt_build/final-montage.webp',await p.export({format:'webp',montage:true,scale:1}));
  const pptx=await PresentationFile.exportPptx(p); await pptx.save(OUT);
}
main().catch(err=>{console.error(err);process.exitCode=1;});
