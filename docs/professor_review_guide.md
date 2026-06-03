# 교수님 검토 가이드

대상 원고:

Fast Surface-Mesh Projected-Density Simulation for Low-q Single-Shot XFEL
Diffraction: Validation, Benchmarking, and Public-Data Template Retrieval

이 문서는 교수님께서 과학적 판단과 제출 전 승인에 집중하실 수 있도록, 검토가 필요한 항목을 정리한 가이드입니다. 형식 정리, figure/table asset 정리, validation 결과 파일 정리는 현재 private GitHub repository에 함께 정리해 두었습니다.

Private GitHub repository:

https://github.com/OkKakao/SAXS-simulation

## 논문 1-page 요약

이 원고는 low-q single-shot XFEL diffraction pattern을 빠르게 계산하기 위한 surface-mesh projected-density simulation framework를 정리한 논문입니다. 핵심 아이디어는 droplet 또는 particle의 3D 형상을 surface mesh로 표현하고, 각 orientation에서 detector plane으로 projected density를 만든 뒤, low-q 영역의 diffraction pattern을 빠르게 계산하는 것입니다. 이를 통해 많은 orientation과 parameter set을 반복 계산해야 하는 synthetic template generation 또는 retrieval screening 상황에서 계산 시간을 줄이는 것을 목표로 합니다.

현재 논문은 세 단계의 검증 흐름으로 구성되어 있습니다. 첫째, sphere, ellipsoid, box/polyhedron 같은 analytical shape에서 계산 결과와 보존량을 비교해 framework의 기본 정확성을 확인합니다. 둘째, mesh size, q-grid, projection backend, orientation library size에 따른 runtime과 memory scaling을 benchmark하여 계산 효율을 정리합니다. 셋째, CXIDB ID57 public nanoparticle data에 대해 synthetic template retrieval과 radial-control 비교를 수행해 public experimental data에서 template-based matching이 의미 있는 separation을 보이는지 확인합니다.

GroEL/CXIDB ID187 관련 내용은 main validation claim으로 두지 않고 supplementary demonstration으로 제한했습니다. GroEL density model, z-projection, mesh decimation, real-vs-control comparison을 제시하지만, experimental image energy가 radial structure에 크게 지배되는 한계가 있어 high-confidence orientation retrieval claim으로 쓰지 않았습니다.

따라서 현재 원고의 핵심 claim은 "완성된 구조복원 알고리즘"이라기보다, low-q XFEL/SPI-style simulation과 template screening을 빠르게 수행할 수 있는 검증된 computational framework를 제시한다는 쪽입니다.

## 교수님께서 반드시 결정해 주셔야 하는 부분

아래 항목은 학생이 형식적으로 정리할 수는 있지만, 최종 판단은 교수님 확인이 필요합니다.

1. Journal of Synchrotron Radiation에 투고하는 방향이 적절한지
2. 현재 제목의 길이와 표현을 유지할지, 더 짧게 줄일지
3. 논문의 novelty를 "fast surface-mesh projected-density simulation"으로 잡는 것이 적절한지
4. ID57 결과를 main public-data validation으로 제시하는 것이 충분한지
5. GroEL 결과를 supplementary demonstration으로 제한한 현재 방향이 적절한지
6. 추가 validation, benchmark, 또는 control analysis가 필요한지
7. main figure/table과 supplementary figure/table 구성에 빠진 것이 있는지
8. 반드시 추가해야 할 핵심 reference가 있는지
9. 최종 author list, author order, corresponding author, ORCID 정보를 어떻게 정리할지
10. acknowledgements, funding, conflict of interest, data availability 문구를 어떻게 확정할지
11. AI-tool use disclosure를 acknowledgements 또는 methods에 어느 정도로 적을지
12. open-access fee 또는 waiver 관련 제출 정보를 어떻게 처리할지

## 함께 보낼 파일과 위치

- Main manuscript draft: `manuscript/Fast Surface-Mesh Projected-Density Simulation for Low-q Single-Shot XFEL Diffraction Validation, Benchmarking, and Public-Data Template Retrieval.docx`
- Main text section drafts: `manuscript/00_Abstract.txt` to `manuscript/07_Conclusion.txt`
- Supplementary information draft text: `manuscript/08_Supplementary_information.txt`
- Figure and table captions: `manuscript/09_Figure_table_captions.txt`
- Reference and citation planning file: `manuscript/10_References_and_citation_map.txt`
- Main figure/table assets: `assets/main_manuscript/`
- Supplementary figure/table assets: `assets/supplementary_information/`
- Validation and benchmark package: `validation_package/`

## 가장 먼저 확인 부탁드릴 부분

### 1. 논문 제목과 저널 방향

- 현재 제목이 연구 내용을 충분히 대표하는지 확인이 필요합니다.
- Journal of Synchrotron Radiation 투고 방향이 적절한지 확인이 필요합니다.
- 제목이 다소 길기 때문에, 정보량을 유지할지 더 짧게 줄일지 결정이 필요합니다.
- `Low-q`, `single-shot XFEL diffraction`, `projected-density simulation`, `template retrieval` 중 어떤 표현을 제목에서 반드시 유지할지 판단이 필요합니다.

### 2. 논문의 핵심 주장 강도

현재 원고의 중심 주장은 다음과 같이 정리되어 있습니다.

- surface mesh와 projected density를 이용해 low-q single-shot XFEL diffraction을 빠르게 계산할 수 있다.
- analytical shape validation, mass/volume conservation, performance benchmark로 계산 framework를 검증했다.
- CXIDB ID57 public nanoparticle data에서 template retrieval과 radial-control 비교를 통해 public-data validation을 제시했다.
- GroEL/CXIDB ID187은 main validation claim이 아니라, protein-density modelling과 supplementary demonstration 성격으로 정리했다.

확인 부탁드릴 점:

- 이 주장 수준이 충분히 보수적인지 확인이 필요합니다.
- ID57 결과를 main public-data validation으로 내세우는 것이 적절한지 확인이 필요합니다.
- GroEL 결과를 supplementary demonstration으로 제한한 현재 방향이 적절한지 확인이 필요합니다.
- "retrieval"이라는 표현이 실제 성과보다 과하게 읽히는 부분이 없는지 확인이 필요합니다.

### 3. Validation 설계의 충분성

현재 validation은 다음 흐름으로 구성되어 있습니다.

- Section 3: sphere, ellipsoid, box/polyhedron analytical validation
- Section 3: resolution convergence and orientation-dependent mass conservation
- Section 4: runtime and memory benchmark
- Section 5: CXIDB ID57 public-data retrieval/control comparison
- Supplementary: ID57 self-retrieval, GroEL modelling, GroEL mesh/projection checks

확인 부탁드릴 점:

- analytical validation과 public-data validation의 조합이 논문 설득력에 충분한지 확인이 필요합니다.
- volume conservation을 강조하는 현재 설명이 적절한지 확인이 필요합니다.
- 추가 validation이 필요하다면 어떤 결과가 가장 설득력 있는지 의견이 필요합니다.
- Section 4 benchmark가 논문의 중심 주장에 충분히 연결되는지 확인이 필요합니다.

### 4. GroEL 관련 해석

현재 GroEL은 main claim이 아니라 supplementary 쪽에 두었습니다.

확인 부탁드릴 점:

- GroEL structure preview를 supplementary에 넣는 것이 적절한지 확인이 필요합니다.
- GroEL real-vs-control 차이가 작다는 점을 limitation으로 정리한 방향이 적절한지 확인이 필요합니다.
- GroEL을 main text에서 어느 정도까지 언급할지 판단이 필요합니다.
- GroEL 관련 figure/table을 supplementary로 이동한 현재 구성이 적절한지 확인이 필요합니다.

### 5. Figure와 table 구성

현재 main manuscript 구성:

- Main figures: Figure 1 to Figure 8
- Main tables: Table 1 to Table 3
- Supplementary figures: Figure S1 to Figure S4
- Supplementary tables: Table S1 to Table S3

확인 부탁드릴 점:

- main text에 남긴 figure/table 수가 적절한지 확인이 필요합니다.
- ID57 관련 결과를 Figure 6, Figure 7, Table 3 중심으로 보여 주는 구성이 적절한지 확인이 필요합니다.
- GroEL 관련 결과를 supplementary로 보낸 것이 적절한지 확인이 필요합니다.
- IUCr 제출 시 graphical abstract 또는 thumbnail 후보로 어떤 이미지를 추천할지 의견이 필요합니다.

## 저자와 제출 정보

다음 항목은 교수님 확인 없이는 확정하기 어렵습니다.

- 최종 author list
- author order
- corresponding author
- affiliation 표기
- ORCID 정보
- acknowledgements
- funding information and grant numbers
- conflict of interest statement
- data availability statement
- AI-tool use disclosure 필요 여부
- open-access fee 또는 waiver 관련 제출 정보

## AI-tool use disclosure 초안

IUCr Notes for Authors는 AI tools를 사용할 수 있지만 author로 기재할 수 없고, AI tool 사용 사실을 methods 또는 acknowledgements에 문서화해야 한다고 안내하고 있습니다. 또한 AI tool로 인해 생길 수 있는 오류는 co-author가 확인해야 합니다.

단순히 "code refinement and data processing"이라고 쓰면 AI가 scientific data analysis나 interpretation을 직접 수행한 것처럼 읽힐 수 있습니다. 더 보수적으로는 다음 표현이 적절해 보입니다.

Suggested wording:

`AI-assisted tools were used for code refinement, organization of data-processing scripts and workflows, and editorial drafting support. All numerical results, scientific interpretations, and manuscript content were reviewed and verified by the authors.`

조금 더 짧은 문구:

`AI-assisted tools were used for code refinement, data-processing workflow organization, and editorial support. All scientific content and numerical results were reviewed and verified by the authors.`

확인 부탁드릴 점:

- AI-tool use disclosure를 acknowledgements에 넣을지, methods에 넣을지 결정이 필요합니다.
- "data processing"이라고 직접 쓸지, "data-processing workflow/script organization"처럼 좁혀 쓸지 결정이 필요합니다.
- 저자 검토 문장을 반드시 함께 넣는 것이 좋을지 확인이 필요합니다.

## Reference 관련 확인

현재 reference list와 citation 위치는 `manuscript/10_References_and_citation_map.txt`에 정리되어 있습니다.

반드시 포함하려고 정리한 reference:

- VTK Flying Edges 관련 reference
- CXIDB ID57 관련 reference
- CXIDB ID187 관련 reference
- PDB 1SS8/GroEL 관련 reference
- XFEL/SPI/SAXS simulation 및 phase-retrieval 관련 background reference

확인 부탁드릴 점:

- 빠진 핵심 reference가 있는지 확인이 필요합니다.
- simulation method와 retrieval/template matching을 설명하는 데 더 적절한 reference가 있는지 확인이 필요합니다.
- reference 수가 과하거나 부족한지 확인이 필요합니다.
- final manuscript에서는 IUCr author-year style로 정리할 예정입니다.

## IUCr 형식 관련 현재 체크

IUCr Journal of Synchrotron Radiation Notes for Authors 기준으로 확인한 주요 항목입니다. 확인일은 2026-06-03입니다.

- Research Paper는 보통 약 10000 words를 넘지 않는 것이 권장됩니다.
- Abstract는 약 250 words 수준이 권장됩니다.
- Synopsis가 필요합니다.
- Keywords는 최소 5개가 필요합니다.
- Word 제출 시 table은 Word table editor 사용이 권장됩니다.
- Word 제출 시 displayed equation은 Word equation editor 사용이 권장됩니다.
- Figure는 high-resolution graphics file로 제출해야 하며, bitmap figure는 minimum 600 dpi가 요구됩니다.
- Reference는 author-year citation과 alphabetic reference list 형식입니다.
- Supporting information citation은 main article 또는 related literature section에 반영되어야 합니다.
- AI tool을 사용한 경우 methods 또는 acknowledgements에 사용 사실을 문서화해야 합니다.

현재 keyword 후보:

`single-particle imaging; low-q XFEL diffraction; projected-density simulation; surface-mesh methods; template retrieval`

## 교수님께 드릴 구체 질문

1. Journal of Synchrotron Radiation 투고 방향이 적절할까요?
2. 현재 제목은 유지하는 것이 좋을까요, 아니면 더 짧게 줄이는 것이 좋을까요?
3. 논문의 핵심 novelty를 "fast surface-mesh projected-density simulation"으로 잡은 것이 적절할까요?
4. ID57을 main public-data validation으로 제시하는 것이 충분히 설득력 있을까요?
5. GroEL을 supplementary demonstration으로 제한한 현재 방향이 적절할까요?
6. Figure 1 to Figure 8, Table 1 to Table 3 구성이 적절할까요?
7. Supplementary Figure S1 to S4, Table S1 to S3 구성에서 빼거나 추가할 것이 있을까요?
8. 추가해야 할 validation 또는 benchmark가 있을까요?
9. 빠진 핵심 reference가 있을까요?
10. 최종 author list, author order, corresponding author는 어떻게 정리하면 될까요?
11. Funding, acknowledgements, conflict of interest, data availability 문구는 어떻게 쓰면 될까요?
12. AI-tool use disclosure는 acknowledgements에 넣는 것이 좋을까요, methods에 넣는 것이 좋을까요?
13. AI-tool use 범위는 "code refinement and data processing"으로 써도 될까요, 아니면 "code refinement and data-processing workflow organization"처럼 좁혀 쓰는 것이 좋을까요?

## 교수님 검토 후 우리가 바로 할 수 있는 작업

- 제목, abstract, synopsis, keywords 수정
- main text와 supplementary text의 claim 강도 조정
- figure/table placement와 caption 수정
- reference 추가, 삭제, 형식 정리
- end matter 초안 작성
- submission package checklist 작성
- high-resolution figure export와 editable table 정리
- GitHub repository README와 data policy 문서 보강

## 짧은 전달 문구 초안

교수님, 논문 초안과 validation/figure/table 정리 파일을 private GitHub repository에 정리해 두었습니다. 교수님께서 전체 내용을 빠르게 파악하실 수 있도록 `docs/professor_review_guide.md`에 논문 요약, 제가 판단하기 어려운 claim 강도, 저자 및 end matter, journal submission 관련 질문을 정리했습니다. 우선 main manuscript draft와 이 review guide를 중심으로 봐 주시면, 피드백에 맞춰 본문, supplementary, figure/table, reference를 다시 정리하겠습니다.

## 참고한 공식 문서

- IUCr Journal of Synchrotron Radiation Notes for Authors: https://journals.iucr.org/s/services/notesforauthors.html
- IUCr Journal of Synchrotron Radiation Author Services: https://journals.iucr.org/s/services/authorservices.html
