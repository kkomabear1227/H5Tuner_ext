---
name: autotuner-context
description: H5Tuner/TunIO 병렬 I/O 오토튜닝 작업을 시작할 때 읽는다. 문서 7개의 읽는 순서, 누리온·뉴론 클러스터 환경, 그리고 실기에서만 드러난 함정들을 안내한다. 튜닝 캠페인 실행, shim 빌드, sweep, 프레임워크 비교, 클러스터 이식, HDF5/MPI-IO 파라미터 관련 작업이면 먼저 이것부터 본다.
---

# H5Tuner / TunIO 오토튜닝 — 세션 시작 안내

## 0. 이 프로젝트가 무엇인가

**신규 다계층 병렬 I/O 오토튜닝 프레임워크를 만들기 위한 베이스라인 확보**가 목적이다.

H5Tuner(Behzad et al.)와 TunIO(IPDPS'24)를 **같은 파라미터 공간·같은 목적함수·같은 측정 방식·같은 세대 예산** 위에 다시 구현했다. 논문의 숫자를 인용해서는 비교가 성립하지 않기 때문이다.

무언가를 공유할지 분리할지 판단할 때 기준은 **"이게 다르면 비교 결과를 알고리즘 차이로 설명할 수 있는가"**다.

## 1. 문서를 먼저 읽는다

**문서 7개가 저장소 루트에 읽는 순서대로 번호를 달고 있다.** 2026-09-14 부터 공개 저장소에 함께 올린다. 그 전까지는 `.gitignore`에 있었다.

| 파일 | 내용 |
|---|---|
| `02-설계-결정.md` | **설계 결정과 이유.** 새 세션은 §0부터 |
| `04-누리온-환경.md` | 누리온(PBS) 환경, 실기 결과, 함정 |
| `05-뉴론-이식.md` | 뉴론(Slurm) 이식 조사 |
| `03-원본-코드-분석.md` | 저장소 구조, 기존 버그, TunIO 논문 요약(§11) |
| `07-후속-연구-계획.md` | 확장 계획, 후속 연구 가능성 9건(§8) |
| `06-진행-상황.md` | 진행 추적, 남은 작업, 결정 기록 |

**읽는 순서.** `02-설계-결정` §0 → §1(목적) → §8(검증 상태) → §14(누리온 실행법). 실기 작업이면 `04-누리온-환경.md`를 통째로 읽는다.

`02-설계-결정` §7에 **논문과 다른 점 22건**이 표로 있다. "TunIO를 재현했다"고 말할 때 근거가 되는 목록이다.

## 2. 지금 어디까지 왔나

**세 프레임워크가 실기에서 돈다.** 누리온에서 H5Tuner가 h5bench write를 260 → 575 MB/s로 **121% 개선**했고, TunIO가 sweep 순위를 받아 같은 성능을 **32% 적은 시간**에 얻고 조기 종료했다(RoTI 137 대 173).

**TunIO 세 컴포넌트가 모두 구현됐다.** I/O Discovery(`autotuner/discovery/`), 부분집합 선택(`subset.py`, `rl/picker.py`), 조기 종료(`early_stopping.py`, `rl/stopper.py`)다. Darshan도 붙어 있다(`darshan.py`).

**남은 것은 `02-설계-결정` §10과 `06-진행-상황` §5에 있다.** 논문 충실도 셋(PCA, 2단 신경망, 세대 50)과 뉴론 이식이다.

## 3. 반드시 알아야 할 함정

전부 **에러를 내지 않고 조용히 잘못된 결과를 낸다.** 실기에서만 드러났다.

**shim이 로드조차 안 될 수 있다.** `LD_PRELOAD`는 의존성을 못 찾으면 조용히 무시한다. 그러면 모든 후보가 튜닝 없이 실행되고 캠페인은 정상으로 보인다. **shim에 rpath를 박고 mxml을 정적 링크해서 해결했다.** 확인 방법은 없는 config를 주고 `H5Tuner:` 메시지가 나오는지 세는 것이다.

**h5bench는 async API를 부른다.** `H5Fcreate`가 아니라 `H5Fcreate_async`다. 동기 API만 후킹하면 훅이 하나도 발동하지 않는다. shim에 async 훅 3개가 있고, HDF5 헤더가 이 이름들을 매크로로 정의하므로 `#undef`가 필요하다.

**MPI-IO 힌트는 Intel MPI에서만 작동한다.** OpenMPI 3.1.0의 ROMIO에는 Lustre 드라이버가 없어 `striping_factor`, `cb_nodes` 등이 전부 무시된다. Intel MPI에서 `I_MPI_EXTRA_FILESYSTEM=1`과 `I_MPI_EXTRA_FILESYSTEM_LIST=lustre`를 켜야 한다. 없으면 ROMIO가 Lustre를 NFS로 오인한다. 상세는 `04-누리온-환경.md` §7c.

**h5bench의 크기 지정에 함정이 둘 있다.** 쓰기 연산에서 `NUM_PARTICLES`는 무시되고 `DIM_1×DIM_2×DIM_3`가 쓰인다(기본값 전부 1). 그리고 크기 접미사는 앞에 공백이 필요하다(`512K`는 512, `512 K`라야 524288). 어댑터가 정수로 전개해 두 함정을 피한다.

**계산 노드의 Python은 3.6이다.** 로그인 노드는 3.9다. 잡 스크립트에서 `/apps/applications/PYTHON/3.9.5/bin/python3`을 절대 경로로 쓴다.

## 4. 클러스터 작업 규칙

**SSH는 `BatchMode=yes`로, 한 번만 시도한다.** 실패하면 즉시 멈추고 사용자에게 재인증을 요청한다. **폴링 루프를 돌릴 때는 반드시 `rc=255`에서 중단한다.** 그러지 않으면 인증 실패가 누적되어 계정이 잠길 수 있다(실제로 8회까지 쌓인 적이 있다).

`~/.ssh/config`에 `nurion`과 `neuron`이 있고 `ControlPersist 12h`다. 만료되면 사용자가 터미널에서 `ssh nurion`을 한 번 해야 한다.

**잡 개수를 아낀다.** 진단은 한 잡에 여러 조건을 넣는다. 캠페인은 `--launcher local`로 allocation 안에서 돌리면 큐 대기가 한 번뿐이다. 세대 단위 배치가 구현돼 있어 40세대가 41잡이다.

**walltime을 짧게 잡으면 backfill로 먼저 실행된다.** 누리온 `normal`이 88~100% 혼잡이라 40분짜리는 18분 만에 시작하고 60분짜리는 몇 시간 걸렸다.

**출력을 자르지 않는다.** `tail -20` 때문에 HDF5 assertion 메시지를 놓쳐 원인 규명이 하루 늦어진 적이 있다.

## 5. 실행 명령

로컬에서 확인하는 것들이다. MPI도 HDF5도 필요 없다.

```bash
python3 -m autotuner info
python3 -m autotuner run --framework h5tuner --dry-run --seed 1 --generations 8 --population 10
python3 -m autotuner sweep --space minimal --levels 2 --dry-run --record traces/g.jsonl
python3 -m autotuner rank traces/g.jsonl --space minimal
```

누리온 실기는 `02-설계-결정.md` §14에 확정된 경로와 함께 있다. 요약하면 Intel 툴체인이다.

```
module load intel/19.1.2 impi/19.1.2
HDF5  = ~/00-dependency/installation-home/hdf5        (Intel 빌드, parallel ON)
shim  = /scratch/$USER/12-autotuner/lib-icc-impi/libautotuner.so
h5bench = ~/01-benchmark/h5bench-icc-impi/h5bench_write
```

I/O Discovery는 libclang이 필요하고 `.venv`에 있다. 코어는 libclang 없이 돈다.

## 6. 작업 방식

**미명시 지점에서는 멈추고 물어본다.** 논문이 명시하지 않은 핵심 구조를 만나면 후보와 손익을 제시한다. 추측해서 구현하면 "논문을 재현했다"고 방어할 수 없다.

**정한 것은 근거와 함께 기록한다.** `02-설계-결정` §7과 `06-진행-상황` §3이 그 자리다. 논문에서 벗어났으면 특히 명확히 적는다.

**문서는 줄글·단문·두괄식으로 쓴다.** 표는 파라미터 목록이나 수치 비교처럼 진짜 목록인 곳에만 쓴다. 추론을 표로 압축하지 않는다. 문서는 한국어, 코드 주석은 영어다.

**소스만 읽고 단정하지 않는다.** h5bench 코드를 읽고 `chunk_cache`와 `sieve_buf_size`가 무효라고 예측했는데 실측에서 4위와 10위로 살아 있었다. sweep이 답을 준다.

## 7. 보류된 것

**조기 종료기의 보상 설계 결함을 2026-08-12에 보류했다.** 합성 곡선에서 휴리스틱이 강화학습을 이긴다. 버그가 아니라 벤치마크 설계 문제이고 해법도 찾아뒀다(`06-진행-상황` §4.1). **버그로 오인해 다시 조사하지 말 것.**

**MACSio는 누리온에서 SIF 모드가 HDF5 assertion으로 깨진다.** shim과 무관하며(대조군도 죽는다) 2026-08-31에 보류했다. `04-누리온-환경.md` §7.

**`test/` 프로그램이 링크되지 않는다.** OpenMPI의 `libmpi.la`가 없는 `libpbs.la`를 참조한다. shim 자체는 영향받지 않는다. `06-진행-상황` §4.2.
