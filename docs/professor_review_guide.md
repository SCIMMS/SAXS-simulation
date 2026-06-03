# 교수님 검토 가이드

대상 원고:

Fast Surface-Mesh Projected-Density Simulation for Low-q Single-Shot XFEL
Diffraction: Validation, Benchmarking, and Public-Data Template Retrieval

이 문서는 교수님께서 과학적 판단과 제출 전 승인에 집중하실 수 있도록, 검토가 필요한 항목을 정리한 가이드입니다. 형식 정리, figure/table asset 정리, validation 결과 파일 정리는 현재 private GitHub repository에 함께 정리해 두었습니다.

Private GitHub repository:

https://github.com/OkKakao/SAXS-simulation

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
12. AI-tool use disclosure가 필요하다면 어느 정도로 쓰는 것이 적절할까요?

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

교수님, 논문 초안과 validation/figure/table 정리 파일을 private GitHub repository에 정리해 두었습니다. 제가 판단하기 어려운 과학적 claim 강도, 저자 및 end matter, journal submission 관련 항목은 `docs/professor_review_guide.md`에 질문 형태로 정리했습니다. 우선 main manuscript draft와 이 review guide를 중심으로 봐 주시면, 피드백에 맞춰 본문, supplementary, figure/table, reference를 다시 정리하겠습니다.

## 참고한 공식 문서

- IUCr Journal of Synchrotron Radiation Notes for Authors: https://journals.iucr.org/s/services/notesforauthors.html
- IUCr Journal of Synchrotron Radiation Author Services: https://journals.iucr.org/s/services/authorservices.html
