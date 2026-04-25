# Pack Architecture — nature의 확장 모델

**Status**: M1–M3 implemented; later milestones (e.g.\ §17 scatter-gather)
remain design-only.

**관계**:
- 기존 `nature/tools/` / `nature/context/footer/` / `nature/frame/manager.py`는 마이그레이션 대상이지만 **big-bang 재작성 없이** legacy shim으로 공존한다.
- 본 문서가 다루는 내용 중 일부 (per-agent hint toggle 등)는 `ARCHITECTURE.md`의 정식 설명으로 이전됨.

**목표 독자**: nature 코드베이스에 Phase 2+ 기능을 추가하려는 개발자. 이 문서를 읽고 나면 "새 기능은 어떤 이름의 파일/객체로 어디에 놓는가"에 답할 수 있어야 한다.

---

## 0. Why this document

2026-04-15 벤치마크 이후 Phase 2~5의 새 기능들(Edit 피드백, 코드 레벨 캡, text parser 확장, per-agent hint toggle, Memory Ledger, runtime model swap)을 그대로 `frame/manager.py`와 `tools/` 안에 쌓으면 **반드시** 엉킨다. 이유:

- 새 기능마다 manager.py, 개별 tool 구현, footer rule, 이벤트 emit 지점이 여러 곳에서 동시에 바뀜.
- 각 기능의 영향 범위(어떤 agent에 적용되는지, 어떤 이벤트를 emit하는지, 어떤 상태를 읽는지)가 코드 전반에 흩어져 있어 감사하기 어렵다.
- 플러그인 설치형 확장은 이 구조 위에선 불가능.

이 문서는 **"하나의 feature = 하나의 설치 가능한 단위"** 원칙 위에서 코어를 재구조화하는 설계를 제안한다. 실제 구현은 Phase 2를 pilot으로 삼아 점진 마이그레이션한다.

---

## 1. The observation: A/B split

nature 안에서 "에이전트와 상호작용하는 것"은 근본적으로 두 가지 모드로 나뉜다.

- **(A) Agent가 호출하는 것** — 에이전트가 tool_use 블록으로 능동 호출. 기존 "tool"과 같은 의미.
- **(B) Framework가 에이전트에 개입하는 것** — 프레임워크가 특정 조건에서 능동적으로 실행. 에이전트 호출 없음.

하나의 feature는 (A), (B), 또는 (A) + (B) 조합 중 어떤 모양이든 가능하다.

| Feature | (A) Tool? | (B) Intervention? |
|---|---|---|
| `Read` / `Write` / `Grep` / `Glob` | O | — |
| `Bash` | O | O (11 safety gate) |
| `TodoWrite` | O | O (사용법 prompt + 리마인드 footer + 이벤트 emit) |
| `Agent` (sub-frame spawn) | O | O (self-delegation guard) |
| Receptionist synthesis gate | — | O (frame resolve 차단) |
| 기존 footer rules (`synthesis_nudge` 등) | — | O |
| Phase 2.1 Edit fuzzy suggest | — | O |
| Phase 2.2 Edit loop detection | — | O |
| Phase 2.3 Edit re-read hint | — | O |
| Phase 3 budget cap | — | O (Gate + Counter) |
| Phase 4 text tool parser | — | O (별도 계층) |
| v2-P6 Memory Ledger | — | O |
| v2-P7 runtime model swap | — | O |

즉 앞으로 추가할 거의 모든 기능이 (B) — Intervention이다. 현재 구조는 (A)만 일급 citizen이고 (B)는 manager.py/footer/rules 등에 흩어져 있음. 이 불균형을 바로잡는 것이 이 문서의 출발점.

---

## 2. Three-layer taxonomy

```
Pack (설치 단위)
 └── Capability (응집 feature — 이름 있는 번들)
      ├── Tool          — (A) agent-invoked
      └── Intervention  — (B) framework-initiated
```

각 레이어의 **존재 이유**:

- **Pack**: 설치/제거/버전 관리의 경계. 의존성 명세. `~/.nature/packs/` 아래에서 스캔되는 디렉토리 하나 = Pack 하나.
- **Capability**: 사람이 "이 feature"라고 부르는 단위. 한 Pack은 여러 Capability를 가질 수 있고, 한 Capability는 여러 Tool과 Intervention을 가질 수 있다. Capability가 있는 이유는 **응집**: "Edit 도구와 그 가드 3개"는 하나의 Capability다. 별개가 아님.
- **Tool / Intervention**: 실제 동작 단위. 아래에서 상세히.

---

## 3. Tool (the A side)

Tool은 지금 nature의 `Tool`과 **같은 것**이다. 이 문서는 Tool의 의미를 바꾸지 않는다. 다만 Tool이 이제 Capability의 자식이 되고, Capability가 Pack의 자식이 된다는 상속 구조만 추가.

```python
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    async def execute(
        self,
        tool_input: dict[str, Any],
        ctx: ToolExecutionContext,
    ) -> ToolResult: ...
```

기존 Bash/Read/Write/Edit/... 구현체는 **내부 로직 변경 없이** 새 Capability의 `tools=[...]` 리스트에 등록된다. 이것이 마이그레이션 1단계에서 유일하게 필요한 변경.

---

## 4. Intervention (the B side)

Intervention은 프레임워크가 "어떤 조건에서 무엇을 할지"를 선언하는 객체. 실행 주체는 프레임워크이고, 효과는 명시적이고 열거 가능하다.

```python
@dataclass
class Intervention:
    id: str                      # 전역 유일. 예: "edit_guards.fuzzy_suggest"
    kind: InterventionKind       # "gate" | "listener" | "contributor"
    trigger: TriggerSpec         # 언제 실행될지
    action: InterventionAction   # 실행 함수 (순수함수, 부작용은 Effect 반환으로만)
    description: str = ""
    default_enabled: bool = True
```

### 4.1 Three kinds

| kind | 실행 시점 | 대표 목적 | 할 수 있는 Effect |
|---|---|---|---|
| **Gate** | 특정 액션 **직전** | 허용/차단, 입력 변형 | `Block`, `ModifyToolInput` |
| **Listener** | 이벤트/후속 훅 **이후** | 관찰, 반응, 파생 상태 업데이트 | 대부분의 Effect (`EmitEvent`, `ModifyToolResult`, `InjectUserMessage`, `SwapModel`, `UpdateFrameField`) |
| **Contributor** | 컨텍스트 **빌드 시** | 정적/조건부 텍스트 기여 | `AppendFooter`, `AppendInstructions` |

**Gate vs Listener의 차이**는 "액션을 블록할 수 있는가"다. Gate는 블록 권한이 있고, 실행 전에 호출된다. Listener는 블록할 수 없고, 실행 이후에 호출된다. 이 구분은 이후 effect application 순서를 단순하게 만든다.

**Contributor는 왜 별도인가?** Gate/Listener는 이벤트 발생에 반응하는 **반응형**이지만, Contributor는 "매 턴 footer를 쌓을 때 내 몫을 내놓는" **선언형**이다. 호출 패턴이 근본적으로 다르고, 반환 타입도 텍스트 기여로 제한된다. 구분하면 dispatch 로직이 깔끔해진다.

**Listener에는 추가로 phase가 있다** — §4.5 참고. Gate와 Contributor는 phase 개념 없음 (단일 패스).

### 4.2 Trigger space

Intervention이 구독할 수 있는 조건은 유한하게 열거한다. 임의의 hook 포인트를 허용하면 감사가 불가능해지므로, 의도적으로 닫힌 집합:

```python
@dataclass(frozen=True)
class OnToolCall:
    tool_name: str | None = None       # None = 모든 tool
    phase: ToolPhase = ToolPhase.POST  # "pre" | "post"
    where: Callable[[ToolCallInfo], bool] | None = None  # 추가 필터 (예: is_error만)

@dataclass(frozen=True)
class OnLLMCall:
    phase: LLMPhase                    # "pre" | "post"

@dataclass(frozen=True)
class OnEvent:
    event_type: EventType              # Phase 1의 EventType 그대로 사용

@dataclass(frozen=True)
class OnTurn:
    phase: TurnPhase                   # "before_llm" | "after_llm"

@dataclass(frozen=True)
class OnFrame:
    phase: FramePhase                  # "opened" | "resolving" | "resolved" | "errored" | "closed"

@dataclass(frozen=True)
class OnCondition:
    predicate: Callable[[FrameSnapshot], bool]  # 매 관찰 포인트마다 평가

TriggerSpec = (
    OnToolCall | OnLLMCall | OnEvent | OnTurn | OnFrame | OnCondition
)
```

**설계 원칙**:
- 트리거는 **정적 메타데이터**. Intervention 객체가 로드될 때 registry가 인덱스를 구축한다.
- 런타임에 바뀌지 않는다. Intervention이 동적으로 구독을 바꾸고 싶으면 두 개를 등록하고 default_enabled로 토글한다.
- `where` predicate는 "이 Intervention이 반응할지 말지" 필터링 용도. 정적 트리거 + 동적 필터의 조합.

### 4.3 Effect space

Intervention의 `action`은 **Effect 리스트를 반환**한다. 직접 상태를 수정하지 않는다. 이 계약이 핵심이다:

```python
@dataclass
class Block:
    reason: str
    trace_event: EventType | None = None

@dataclass
class ModifyToolInput:
    patch: dict[str, Any]              # partial override

@dataclass
class ModifyToolResult:
    output: str | None = None
    is_error: bool | None = None
    append_hint: str | None = None     # error 메시지에 추가 텍스트

@dataclass
class AppendFooter:
    text: str
    source_id: str                     # source intervention id

@dataclass
class AppendInstructions:
    text: str
    source_id: str

@dataclass
class InjectUserMessage:
    text: str
    source_id: str
    ttl: int = 1                        # 다음 N 턴 동안 유효

@dataclass
class SwapModel:
    new_model: str
    reason: str = ""

@dataclass
class UpdateFrameField:
    path: str                           # dot-path: "ledger.files_confirmed"
    value: Any
    mode: Literal["set", "append", "merge"] = "set"

@dataclass
class EmitEvent:
    event_type: EventType
    payload: Any                        # Pydantic 모델 인스턴스

Effect = (
    Block | ModifyToolInput | ModifyToolResult
    | AppendFooter | AppendInstructions | InjectUserMessage
    | SwapModel | UpdateFrameField | EmitEvent
)
```

**왜 Effect 리스트를 반환하는가?** 이유 세 개:

1. **순수성**: action은 입력(Context)과 출력(Effect list)만 있는 순수함수. 테스트가 trivial함 — mock 없이 "이 입력 넣으면 이 Effect 나와야 함"으로 끝.
2. **감사성**: 턴의 로그를 "어떤 intervention이 어떤 effect를 냈는지"로 정확히 재구성 가능. 대시보드에서 "왜 이 Edit이 차단됐는가" 같은 질문에 즉답.
3. **Replay 안전성**: action을 동일 입력으로 다시 호출하면 동일 Effect가 나와야 한다. 상태 mutation이 내부적으로 일어나면 이 불변성이 깨진다.

**Effect 적용은 registry가 한다**. Intervention은 "내가 하고 싶은 것"을 선언만 한다. 실제 `frame.model = ...`, `event_store.append(...)` 같은 부작용은 registry의 effect applier가 수행.

### 4.4 Action contract

```python
InterventionAction = Callable[
    [InterventionContext],
    Awaitable[list[Effect]] | list[Effect],
]
```

sync/async 둘 다 허용. dispatch 시점에 `inspect.iscoroutine`으로 분기.

```python
@dataclass
class InterventionContext:
    """Intervention action이 읽을 수 있는 모든 것 (read-only)."""
    frame: Frame | None              # deepcopy at dispatch — read-only 약속
    event: Event | None              # on_event/on_tool_call/on_llm_call 트리거인 경우
    tool_call: ToolCallInfo | None
    session_id: str
    registry: PackRegistry           # 다른 capability 조회용
    now: float                       # 타임스탬프 (테스트 주입 가능)
    # POST_EFFECT phase listener에게만 주어진다 (§4.5).
    # PRIMARY phase listener에게는 빈 리스트.
    primary_effects: list[Effect] = field(default_factory=list)
    # compose-time dispatch 전용 (Frame 미노출 케이스).
    body: ContextBody | None = None
    header: ContextHeader | None = None
    self_actor: str = ""
```

**read-only 약속**이 핵심이다. Context에 mutator가 있으면 Effect 계약이 무너진다. `frame` 필드는 `Frame`의 deepcopy (전체 복사) — Python은 불변성을 강제할 수 없으므로, intervention이 실수로 mutation해도 원본엔 영향 없도록 복사본을 넘긴다. 성능 우려는 §13 참고.

### 4.5 Listener phases — PRIMARY vs POST_EFFECT

같은 트리거에 여러 Listener가 붙을 때, 그들끼리 cross-reference하는 건 빠르게 복잡해진다 (의존 그래프, cycle 위험, depth limit). 그 복잡도를 피하기 위해 Listener에는 **2개의 명시적 phase**가 있다:

```python
class InterventionPhase(int, Enum):
    PRIMARY = 0       # 기본값 — 트리거에 직접 반응
    POST_EFFECT = 1   # PRIMARY가 만든 effect 리스트를 보고 추가 반응
```

`Intervention.phase` 필드로 선언:

```python
@dataclass
class Intervention:
    id: str
    kind: InterventionKind
    trigger: TriggerSpec
    action: InterventionAction
    phase: InterventionPhase = InterventionPhase.PRIMARY  # NEW
    description: str = ""
    default_enabled: bool = True
```

**Dispatch 흐름** (한 트리거 발화 → registry):

1. 해당 트리거의 모든 listener 수집
2. PRIMARY phase listener 전부 실행 → effects 수집
3. POST_EFFECT phase listener 실행 — `ctx.primary_effects`에 PRIMARY effect 리스트가 들어있음
4. PRIMARY effects + POST_EFFECT effects 를 caller에게 반환
5. **끝**. 더 깊은 phase 없음. PRIMARY 재진입 불가능.

**규칙**:
- 같은 phase 내에서 listener끼리 서로의 결과를 못 본다 (순서는 declaration order).
- POST_EFFECT는 PRIMARY effect 리스트만 본다 (frame state는 caller가 아직 안 mutate).
- **이벤트 cascade는 같은 dispatch 안에서 일어나지 않는다**. Listener A가 emit한 event는 다음 dispatch cycle (다른 trigger 발화 시점)에서 처리됨. 같은 턴에 영향을 주려면 POST_EFFECT phase로 명시 declare.

**예시 — edit_guards 4개 intervention의 phase 분포**:

| Intervention | trigger | phase | 이유 |
|---|---|---|---|
| `fuzzy_suggest` | OnToolCall(Edit, POST, error) | PRIMARY | Edit 실패에 직접 반응 |
| `reread_hint` | OnToolCall(Edit, POST, error) | PRIMARY | 동일 |
| `loop_detector` | OnToolCall(Edit, POST, error) | POST_EFFECT | 동일 트리거지만 fuzzy_suggest가 emit한 EDIT_MISS effect를 본 뒤 카운트 |
| `loop_block` | OnToolCall(Edit, PRE) | n/a (Gate) | 별도 경로, 다음 턴 시작 시 frame state 읽음 |

Gate와 Contributor는 phase 개념이 없다 — 단일 패스로 끝남. 필요해지면 향후 추가.

---

## 5. Capability

```python
@dataclass
class Capability:
    name: str                    # "todo_list", "edit_guards", "bash_exec"
    description: str
    tools: list[Tool] = field(default_factory=list)
    interventions: list[Intervention] = field(default_factory=list)
    event_types: list[EventType] = field(default_factory=list)
```

**Capability의 의미**: 한 사람이 "아, 그 edit 가드들"이라고 묶어서 부를 수 있는 단위. Tool 1개 + Intervention N개가 같은 Capability 안에 살 수도 있고, Intervention만 있는 Capability도 가능.

**왜 Pack과 별도 레이어인가?** 한 Pack이 여러 Capability를 담을 수 있어야 하기 때문. 예:

- `nature-core` Pack → `read_file`, `write_file`, `search_code`, `delegate_agent` 여러 Capability
- `nature-edit-guards` Pack → `edit_guards` Capability 하나
- `nature-budget` Pack → `reads_budget`, `turns_budget`, `tools_budget` 세 Capability

단일 Capability만 있는 Pack이 흔하겠지만, 구조적으로는 1:N 관계를 유지.

**`event_types` 필드**: 이 Capability가 **새로 도입하는** 이벤트 타입 목록. 예를 들어 `edit_guards`는 `EDIT_MISS`, `LOOP_DETECTED`, `LOOP_BLOCKED`를 "내가 발행하는 이벤트"로 선언한다. Registry가 install 시 중복 체크 및 의존성 검증에 사용.

---

## 6. Pack

### 6.1 Manifest

```json
{
  "name": "nature-edit-guards",
  "version": "0.1.0",
  "description": "Fuzzy match, loop detection, re-read hint for Edit tool",
  "entry_point": "pack:pack",
  "depends_on": ["nature-core>=0.1.0"],
  "provides_events": [
    "edit.miss",
    "loop.detected",
    "loop.blocked"
  ]
}
```

- `entry_point`: `module:attribute` 포맷. 해당 모듈의 해당 속성이 `Pack` 객체여야 함.
- `depends_on`: 다른 Pack 이름 + semver 제약. registry가 토폴로지 정렬 후 install.
- `provides_events`: 이 Pack이 처음 도입하는 이벤트 타입 이름. 이미 선언된 이벤트를 중복 선언하면 conflict.

### 6.2 Install/uninstall protocol

```python
class Pack(Protocol):
    meta: PackMeta
    capabilities: list[Capability]

    def on_install(self, registry: PackRegistry) -> None: ...
    def on_uninstall(self, registry: PackRegistry) -> None: ...
```

**Install 순서**:

1. Registry가 pack.json 읽기
2. 의존성 해결 (토폴로지 정렬)
3. 각 Pack에 대해 `on_install(registry)` 호출
4. Pack은 자기 capabilities를 `registry.register_capability(cap)`로 등록
5. Registry가 Tool/Intervention을 인덱스에 꽂음
6. `provides_events`가 EventType enum에 런타임 등록 (또는 사전 선언)

**Uninstall**: 반대 순서. 단, 이미 세션에 사용 중인 이벤트 타입이 있으면 런타임 언로드 대신 "이 세션 종료 후 언로드" 마킹만.

### 6.3 Dependency resolution

- 의존성은 **Pack 단위**. 한 Capability가 다른 Capability의 Tool을 참조하면 그 두 Pack은 의존 관계.
- semver로 제약. 충돌 시 명확한 에러 메시지 + 실행 거부 (자동 다운그레이드 없음).
- 현재 nature의 기본 동작을 유지하는 "가상 번들": `nature-core` + `nature-bash` + `nature-todo` + `nature-agent`가 기본으로 설치된 상태로 시작.

---

## 7. Registry & dispatch

```python
class PackRegistry:
    _packs: dict[str, Pack]
    _capabilities: dict[str, Capability]     # name -> cap
    _tools: dict[str, Tool]                  # name -> tool
    _interventions: dict[str, Intervention]  # id -> intervention

    # Pre-built trigger index for dispatch
    _by_tool_pre: dict[str, list[Intervention]]
    _by_tool_post: dict[str, list[Intervention]]
    _by_event: dict[EventType, list[Intervention]]
    _by_turn: dict[TurnPhase, list[Intervention]]
    _by_frame: dict[FramePhase, list[Intervention]]
    _by_condition: list[Intervention]        # 매 tick마다 predicate 평가

    # Event type registration (pack-provided events)
    _custom_event_types: dict[str, EventType]
```

### 7.1 Trigger index

Install 시 registry는 각 intervention의 `trigger`를 보고 위의 dict/list에 넣는다. dispatch 시에는 O(1) lookup으로 반응할 intervention 목록을 찾는다.

예: `post tool call Edit`이 일어나면:
1. `_by_tool_post["Edit"]`로 listener 리스트 획득
2. `_by_tool_post[None]` (wildcard)도 추가
3. 각 intervention의 `where` filter 평가
4. 통과한 것들에 대해 `action(ctx)` 호출
5. 반환된 Effect 리스트를 effect_applier에 전달

### 7.2 Effect application order

여러 intervention이 같은 트리거에 반응할 때, Effect 적용 순서는:

1. **Gate를 먼저 평가** (단일 패스, phase 개념 없음). 하나라도 `Block`을 반환하면 그 지점에서 멈추고 액션을 거부. 후속 intervention은 실행되지 않음.
2. **ModifyToolInput** Gate가 있으면 먼저 적용. 같은 action에 여러 modify가 있으면 선언 순서.
3. **PRIMARY phase Listener 전부 실행** → effects 수집. 이 phase 내 listener들은 서로의 결과를 못 본다 (선언 순서로만 정렬).
4. **POST_EFFECT phase Listener 실행** → `ctx.primary_effects`에 PRIMARY effect 리스트가 주입됨. POST_EFFECT는 PRIMARY가 만든 effect를 보고 추가 effect 산출. (§4.5 참고)
5. **PRIMARY + POST_EFFECT effects 일괄 적용**:
   - `EmitEvent` → event store에 append. **이 emit이 같은 dispatch 안에서 다른 listener를 깨우지 않음** — 다음 dispatch cycle에서만 보임.
   - `UpdateFrameField` → frame 상태 mutation
   - `AppendFooter/Instructions/InjectUserMessage` → context composer 큐에 적재
   - `SwapModel` → frame.model 교체
   - `ModifyToolResult` → 다음 단계의 tool result 블록 수정
6. **Contributor는 별도 플로우** — `ContextComposer`가 footer/instructions/prompt 빌드 시 `registry.run_contributors(trigger, ctx)`를 호출하고, 반환된 Effect 중 `AppendFooter` / `AppendInstructions`를 연결한다. **이 호출이 해당 컨텍스트 섹션의 유일한 소스**다. ContextComposer는 기존 footer rule 모듈을 직접 import하지 않는다; 그 rule들은 Contributor Intervention으로 포팅되어 registry에 등록된 상태. 병렬 경로 없음.

**Cycle 방지**: 2-phase model로 구조적으로 cycle 불가능. depth limit 같은 안전장치 불필요. PRIMARY → POST_EFFECT → 끝, 그 너머는 다음 dispatch cycle에서 처리.

### 7.3 Error handling

- Intervention action이 예외를 던지면: registry가 잡고 `ERROR` 이벤트 emit, 해당 Effect 리스트는 빈 리스트 취급. 턴은 계속 진행.
- 같은 intervention이 연속 N회(예: 3회) 예외를 던지면: default_enabled와 상관없이 자동 disable하고 세션 로그에 기록. 수동 재활성화 필요.
- 의도적 중단은 `Block` Effect로 표현. 예외 메커니즘과 분리.

---

## 8. Events in this model

Phase 1에서 도입한 EventType / 카테고리 / reconstruct 인프라를 **그대로** 재사용한다. 이 모델이 덮어쓰는 건 하나도 없다.

**이벤트의 새 역할**:

- **Wire protocol**: Capability 사이의 loose-coupled 통신 매개. `edit_guards.loop_detector`는 `edit_guards.fuzzy_suggest`를 import하지 않아도, `EDIT_MISS` 이벤트로 상호작용한다.
- **Listener trigger**: `OnEvent(EDIT_MISS)`로 구독. 다른 Pack이 발행한 이벤트도 구독 가능 (의존성 선언은 권장).
- **Effect의 결과물**: `EmitEvent` Effect가 actual event store append로 치환됨.

**STATE_TRANSITION vs TRACE 분류 재해석**:

- **STATE_TRANSITION 이벤트를 emit하는 Listener**는 frame state를 바꾸는 intervention (ledger write, budget increment, model swap).
- **TRACE 이벤트를 emit하는 Listener**는 순수 관찰자 (warning, blocked, loop detected).

Phase 1에서 이 분류를 미리 정해둔 건 우연이 아니라 이 모델과 정확히 같은 원리의 두 이름이었음.

---

## 9. Per-agent scoping

v2 §5의 `allowed_hints` 요청이 이 모델에선 자연스럽게 2축 whitelist가 된다:

```python
@dataclass
class FrameAgentConfig:
    description: str
    model: str | None
    instructions: str
    allowed_tools: list[str] | None            # (A) Tool whitelist — 기존 그대로
    allowed_interventions: list[str] | None    # (B) Intervention whitelist — NEW
```

**해석**:
- `None` = 전체 허용 (default enabled 따름).
- `[]` = 전부 차단.
- 구체 리스트 = 나열된 것만 허용.

**프로파일 디폴트** (v2 §5.3의 재해석):

| 프로파일 | `allowed_interventions` 디폴트 |
|---|---|
| `default` (Sonnet/Haiku) | `None` (전체) |
| `small-local` (qwen 류) | `["edit_guards.*", "budget.warn", "ledger.*"]` |
| `classifier` (Judge) | `["synthesis_gate"]` |
| `patcher` | `["edit_guards.*", "ledger.*"]` |

Intervention id의 glob 패턴(`edit_guards.*`)은 registry가 install 시 전개해서 구체 id 리스트로 normalize.

**대시보드**: 기존 `allowed_tools` 체크박스 패널 옆에 `allowed_interventions` 체크박스 패널 추가. `PATCH /api/config/agents/{name}`로 hot-reload.

---

## 10. Python type sketch

전체 타입을 한 파일에 모아본 초안. 실제 구현 시 `nature/packs/types.py`에 분리할 것.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Protocol

from nature.events.types import Event, EventType
from nature.frame.frame import Frame


# ── enums ─────────────────────────────────────────────────────────────

class ToolPhase(str, Enum):
    PRE = "pre"
    POST = "post"

class LLMPhase(str, Enum):
    PRE = "pre"
    POST = "post"

class TurnPhase(str, Enum):
    BEFORE_LLM = "before_llm"
    AFTER_LLM = "after_llm"

class FramePhase(str, Enum):
    OPENED = "opened"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    ERRORED = "errored"
    CLOSED = "closed"

InterventionKind = Literal["gate", "listener", "contributor"]


class InterventionPhase(int, Enum):
    PRIMARY = 0       # default — 트리거에 직접 반응
    POST_EFFECT = 1   # PRIMARY가 만든 effect 리스트를 보고 추가 반응 (§4.5)


# ── triggers ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OnToolCall:
    tool_name: str | None = None
    phase: ToolPhase = ToolPhase.POST
    where: Callable[["ToolCallInfo"], bool] | None = None

@dataclass(frozen=True)
class OnLLMCall:
    phase: LLMPhase

@dataclass(frozen=True)
class OnEvent:
    event_type: EventType

@dataclass(frozen=True)
class OnTurn:
    phase: TurnPhase

@dataclass(frozen=True)
class OnFrame:
    phase: FramePhase

@dataclass(frozen=True)
class OnCondition:
    predicate: Callable[["FrameSnapshot"], bool]

TriggerSpec = OnToolCall | OnLLMCall | OnEvent | OnTurn | OnFrame | OnCondition


# ── effects ───────────────────────────────────────────────────────────

@dataclass
class Block:
    reason: str
    trace_event: EventType | None = None

@dataclass
class ModifyToolInput:
    patch: dict[str, Any]

@dataclass
class ModifyToolResult:
    output: str | None = None
    is_error: bool | None = None
    append_hint: str | None = None

@dataclass
class AppendFooter:
    text: str
    source_id: str

@dataclass
class AppendInstructions:
    text: str
    source_id: str

@dataclass
class InjectUserMessage:
    text: str
    source_id: str
    ttl: int = 1

@dataclass
class SwapModel:
    new_model: str
    reason: str = ""

@dataclass
class UpdateFrameField:
    path: str
    value: Any
    mode: Literal["set", "append", "merge"] = "set"

@dataclass
class EmitEvent:
    event_type: EventType
    payload: Any

Effect = (
    Block | ModifyToolInput | ModifyToolResult
    | AppendFooter | AppendInstructions | InjectUserMessage
    | SwapModel | UpdateFrameField | EmitEvent
)


# ── context ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolCallInfo:
    tool_name: str
    tool_use_id: str
    tool_input: dict[str, Any]
    phase: ToolPhase
    result_output: str | None = None
    result_is_error: bool | None = None

@dataclass
class InterventionContext:
    session_id: str
    now: float
    registry: "PackRegistry"
    frame: Frame | None = None              # deepcopy at dispatch — read-only 약속
    event: Event | None = None
    tool_call: ToolCallInfo | None = None
    # POST_EFFECT phase에서만 채워짐 (PRIMARY phase는 빈 리스트).
    primary_effects: list[Effect] = field(default_factory=list)
    # compose-time dispatch 전용 — Frame이 없는 경우 (M1 design doc §13 참고).
    body: "ContextBody | None" = None
    header: "ContextHeader | None" = None
    self_actor: str = ""


# ── Tool / Intervention / Capability / Pack ──────────────────────────

class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]
    async def execute(
        self, tool_input: dict[str, Any], ctx: "ToolExecutionContext"
    ) -> "ToolResult": ...

InterventionAction = Callable[
    [InterventionContext], Awaitable[list[Effect]] | list[Effect]
]

@dataclass
class Intervention:
    id: str
    kind: InterventionKind
    trigger: TriggerSpec
    action: InterventionAction
    phase: InterventionPhase = InterventionPhase.PRIMARY  # listener-only (§4.5)
    description: str = ""
    default_enabled: bool = True

@dataclass
class Capability:
    name: str
    description: str
    tools: list[Tool] = field(default_factory=list)
    interventions: list[Intervention] = field(default_factory=list)
    event_types: list[EventType] = field(default_factory=list)

@dataclass
class PackMeta:
    name: str
    version: str
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    provides_events: list[str] = field(default_factory=list)

class Pack(Protocol):
    meta: PackMeta
    capabilities: list[Capability]
    def on_install(self, registry: "PackRegistry") -> None: ...
    def on_uninstall(self, registry: "PackRegistry") -> None: ...
```

위 타입은 **초안**이다. 구현 과정에서 필드 추가/정리 예상.

---

## 11. Migration strategy

### 11.1 Legacy shim — 레이어 통일 원칙

기존 코드와의 관계는 **"래핑"이 아니라 "포팅"**이다. 두 경로가 병존하지 않고 하나로 수렴한다.

**Tool (A side)**: 기존 `Tool` 클래스 구현은 내부 로직 그대로 유지. `legacy_shim`이 기존 인스턴스들을 묶어 가상 Capability `nature.legacy_tools`로 등록한다. dispatch hook이 없으면 동작 변화 0.

**Contributor (B side — footer/instructions)**: 기존 `context/footer/rules/*.py` 모듈들은 **Contributor Intervention으로 포팅**한다 (래핑 아님). 포팅 이후:
- `ContextComposer`는 기존 rule 모듈을 **더 이상 직접 import하지 않는다**.
- Footer/instructions 빌드는 오직 `registry.run_contributors(trigger, ctx)` 한 경로를 통한다.
- 새 Contributor(예: `edit_guards.reread_hint`)와 기존 Contributor(예: `synthesis_nudge`)가 **같은 메커니즘으로 스케줄링**된다. 두 종류로 나뉜 코드 경로가 없다.

**Gate / Listener (B side — reactive)**: 기존 Bash 11개 safety check 같은 reactive 가드는 Gate Intervention으로 포팅된다. 마찬가지로 legacy 경로 제거.

**마이그레이션 1단계의 구체 작업**:

1. `nature/packs/types.py` 신설 (§10의 타입)
2. `nature/packs/registry.py` 신설 (PackRegistry + dispatch)
3. `nature/packs/effects.py` 신설 (Effect 적용 함수들 — `_apply_block`, `_apply_emit_event`, `_apply_append_footer`, ...)
4. `nature/packs/legacy_shim.py` 신설 — 기존 tool 인스턴스 묶음 등록 + 기존 footer rule들을 Contributor Intervention으로 포팅
5. `FrameManager._execute_single_tool` 지점에 registry dispatch hook 추가 (pre/post tool call)
6. `ContextComposer`의 footer/instructions 빌드 지점에 `registry.run_contributors()` 호출 추가 + 기존 rule import 제거

**검증**: 이 시점에 (아직 새 Pack이 하나도 없는 상태에서) 기존 테스트가 모두 통과해야 한다. 레이어 통일이 올바로 됐다면 "레지스트리를 거치는 경로"와 "현재 동작"이 동일 결과.

### 11.2 Phased migration order

| 단계 | 작업 | 기존 코드 영향 |
|---|---|---|
| M1 | types.py + registry.py + effects.py + legacy_shim (footer rule 포팅 포함) | 0 (신규 파일만) |
| M2 | frame/manager.py + ContextComposer에 dispatch hook 삽입. 기존 footer rule 직접 import 제거 | 최소 |
| M3 | edit_guards Pack 신규 작성 (Phase 2.1/2.2/2.3) | 0 — Edit 본체 건드리지 않음 |
| M4 | Phase 2 검증: `solo-qwen-coder-32b × t1` 재실행 | 0 |
| M5 | Phase 3 budget Pack 신규 작성 | 0 |
| M6 | Phase 4 parser 통합 (별도 계층) | 0 |
| M7 | Phase 5 `allowed_interventions` 필드 추가 + 대시보드 UI | 최소 (FrameAgentConfig 확장) |
| ... | 기존 footer rules → Contributor Intervention으로 순차 이동 | 점진 |
| ... | 기존 Bash 11개 safety check → `bash_exec` Capability의 Gate들로 추출 | 점진 |
| 장기 | `~/.nature/packs/` 스캔 메커니즘 | Phase 2~5 완료 이후 |

핵심: **M1~M6는 기존 파일을 거의 건드리지 않는다**. Phase 2~3의 새 기능이 온전히 새 구조 위에서 돌아가는 것을 먼저 증명한 뒤에야 기존 코드 마이그레이션에 손을 댄다.

---

## 12. Pilot: Phase 2 — edit_guards Pack

Phase 2의 세 기능을 이 모델의 첫 번째 실 구현체로 삼는다.

### 12.1 파일 구조

```
nature/
├── packs/
│   ├── __init__.py
│   ├── types.py              # §10의 타입
│   ├── registry.py           # PackRegistry + dispatch
│   ├── effects.py            # Effect 적용 함수들
│   ├── legacy_shim.py        # 기존 tools/footer 래핑
│   └── builtin/
│       └── edit_guards/
│           ├── __init__.py
│           ├── pack.py              # Pack 객체 export
│           ├── fuzzy_suggest.py     # Intervention (2.1)
│           ├── loop_detector.py     # Intervention (2.2)
│           └── reread_hint.py       # Intervention (2.3)
└── frame/
    └── manager.py            # dispatch hook 2줄 추가 (pre/post tool call)
```

### 12.2 `edit_guards.fuzzy_suggest` (Phase 2.1)

```python
from nature.packs.types import (
    Intervention, OnToolCall, ToolPhase,
    ModifyToolResult, EmitEvent, InterventionContext,
)
from nature.events.types import EventType
from nature.events.payloads import EditMissPayload

async def _fuzzy_suggest_action(ctx: InterventionContext):
    tc = ctx.tool_call
    if tc is None or not tc.result_is_error:
        return []
    # "old_string not found" 감지 → 파일 열어서 fuzzy match → hint 보강
    old_string = tc.tool_input.get("old_string", "")
    file_path = tc.tool_input.get("file_path", "")
    match = _find_closest(file_path, old_string)   # difflib.get_close_matches
    if match is None:
        return [EmitEvent(EventType.EDIT_MISS,
                          EditMissPayload(file=file_path))]
    return [
        ModifyToolResult(append_hint=_format_hint(match)),
        EmitEvent(
            EventType.EDIT_MISS,
            EditMissPayload(
                file=file_path,
                fuzzy_match=match.text,
                lineno=match.lineno,
            ),
        ),
    ]

fuzzy_suggest = Intervention(
    id="edit_guards.fuzzy_suggest",
    kind="listener",
    trigger=OnToolCall(
        tool_name="Edit",
        phase=ToolPhase.POST,
        where=lambda tc: tc.result_is_error is True,
    ),
    action=_fuzzy_suggest_action,
    description="When Edit fails with old_string not found, append closest match.",
)
```

### 12.3 `edit_guards.loop_detector` (Phase 2.2)

`fuzzy_suggest`와 동일한 트리거에 붙되, **POST_EFFECT phase**로 선언한다. PRIMARY가 끝난 뒤 `ctx.primary_effects`에서 EDIT_MISS emit이 있었는지 확인하고 카운팅. 같은 phase 내 cross-reference가 아니라 명시적 phase 분리이므로 깔끔.

```python
async def _loop_detector_action(ctx: InterventionContext):
    # PRIMARY가 emit한 EDIT_MISS effect를 보고 frame.ledger의 누적 카운트와 비교
    edit_misses = [
        e for e in ctx.primary_effects
        if isinstance(e, EmitEvent) and e.event_type == EventType.EDIT_MISS
    ]
    if not edit_misses:
        return []
    # 누적 동일-입력 hash 수를 frame.budget_counts['edit_loop'] 같은 곳에 쌓고,
    # 임계치(예: 3)를 넘으면 LOOP_DETECTED emit.
    same_hash_streak = _count_same_hash(ctx.frame, edit_misses)
    if same_hash_streak >= 3:
        return [
            EmitEvent(
                event_type=EventType.LOOP_DETECTED,
                payload=LoopDetectedPayload(
                    tool="Edit",
                    input_hash=edit_misses[-1].payload.fuzzy_match or "",
                    attempts=same_hash_streak,
                ),
            ),
            UpdateFrameField(path="budget_counts.edit_loop_blocked", value=1, mode="set"),
        ]
    return []

loop_detector = Intervention(
    id="edit_guards.loop_detector",
    kind="listener",
    trigger=OnToolCall(
        tool_name="Edit",
        phase=ToolPhase.POST,
        where=lambda tc: tc.result_is_error is True,
    ),
    phase=InterventionPhase.POST_EFFECT,   # ← 핵심
    action=_loop_detector_action,
    description="Count consecutive same-input Edit failures after fuzzy_suggest emits EDIT_MISS.",
)

loop_block = Intervention(
    id="edit_guards.loop_block",
    kind="gate",
    trigger=OnToolCall(tool_name="Edit", phase=ToolPhase.PRE),
    action=_check_loop_block,   # frame.budget_counts['edit_loop_blocked']를 보고 Block 결정
    description="Refuse Edit calls after the loop detector flagged the frame.",
)
```

세 intervention의 dispatch 흐름 (Edit 실패 한 번):

1. PRE: `loop_block` 평가 — frame.budget_counts['edit_loop_blocked'] 확인. 0이면 통과.
2. Edit 실행 → 실패 → POST 트리거 발화.
3. PRIMARY phase 일괄 실행: `fuzzy_suggest` (EDIT_MISS emit + ModifyToolResult), `reread_hint` (InjectUserMessage).
4. POST_EFFECT phase 실행: `loop_detector` — `ctx.primary_effects`에서 EDIT_MISS effect 발견 → 카운팅 → 임계치 도달 시 LOOP_DETECTED emit + UpdateFrameField로 차단 플래그 set.
5. 모든 effect 일괄 적용: 이벤트 emit, frame field 업데이트, tool result 수정, user 메시지 inject.
6. 다음 턴에 사용자가 Edit을 다시 호출하면 step 1의 `loop_block`이 차단 플래그를 발견하고 Block.

### 12.4 `edit_guards.reread_hint` (Phase 2.3)

```python
async def _reread_hint_action(ctx: InterventionContext):
    tc = ctx.tool_call
    if tc is None or not tc.result_is_error:
        return []
    return [
        InjectUserMessage(
            text="The last Edit failed. If you haven't re-read the file since, do it before retrying.",
            source_id="edit_guards.reread_hint",
            ttl=1,
        ),
    ]

reread_hint = Intervention(
    id="edit_guards.reread_hint",
    kind="listener",
    trigger=OnToolCall(tool_name="Edit", phase=ToolPhase.POST,
                       where=lambda tc: tc.result_is_error is True),
    action=_reread_hint_action,
)
```

### 12.5 Pack export

```python
from nature.packs.types import Pack, PackMeta, Capability
from nature.events.types import EventType
from .fuzzy_suggest import fuzzy_suggest
from .loop_detector import loop_detector, loop_block
from .reread_hint import reread_hint

edit_guards_capability = Capability(
    name="edit_guards",
    description="Fuzzy match, loop detection, and re-read hint for Edit tool.",
    tools=[],  # no new tool — augments existing Edit
    interventions=[fuzzy_suggest, loop_detector, loop_block, reread_hint],
    event_types=[EventType.EDIT_MISS, EventType.LOOP_DETECTED, EventType.LOOP_BLOCKED],
)

pack: Pack = _build_pack(
    meta=PackMeta(
        name="nature-edit-guards",
        version="0.1.0",
        description="Phase 2 Edit feedback improvements",
        depends_on=[],
        provides_events=["edit.miss", "loop.detected", "loop.blocked"],
    ),
    capabilities=[edit_guards_capability],
)
```

### 12.6 검증 경로

- `tests/test_packs_types.py` — 타입 라운드트립, Effect 생성/직렬화
- `tests/test_packs_registry.py` — install/uninstall, 트리거 인덱스 정확성
- `tests/test_edit_guards_fuzzy.py` — fuzzy suggest 단위 테스트 (mock frame + mock file)
- `tests/test_edit_guards_loop.py` — 3회 동일 실패 → LOOP_DETECTED, 4회째 → block
- `tests/test_edit_guards_reread.py` — inject 결과 확인
- **통합 테스트**: `.claude/skills/nature-eval`로 `solo-qwen-coder-32b × t1` 재실행. Phase 2 이전 10회 실패 → Phase 2 이후 fuzzy 힌트 받아 1~2회 내 성공하는지 측정.

Phase 2 exit criterion: qwen이 Edit miss 한 번에서 복구해서 t1 PASS.

---

## 13. Open questions / resolved decisions

### Resolved

- **Intervention id namespace**: dot path (`edit_guards.fuzzy_suggest`).
- **Effect application order**: declaration order. `priority` 필드는 충돌 사례가 생길 때 추가.
- **`InterventionContext.frame`**: 항상 Frame 전체 (deepcopy). 성능 개선은 측정 후.
- **Contributor와 ContextComposer 관계**: ContextComposer는 footer/instructions/prompt 빌드 시 `registry.run_contributors()` 한 경로만 호출. 기존 footer rule은 Contributor Intervention으로 포팅되어 그 경로로 흘러들어옴. 병렬 경로 없음.
- **Intervention 간 상호 의존 (Q4)**: 동일 레이어 listener끼리 cross-reference 금지. 대신 **2-phase listener model**:
  - `phase=PRIMARY` (default): 트리거에 직접 반응. 다른 listener의 결과를 못 본다.
  - `phase=POST_EFFECT`: PRIMARY listener들의 effect 리스트를 `ctx.primary_effects`로 받아서 그걸 보고 반응.
  - 같은 dispatch 안에서 더 이상의 phase 없음 (depth limit 자체가 불필요 — 구조적으로 cycle 불가능).
  - 이벤트 cascade(Listener A emit한 event를 Listener B가 OnEvent로 받기)는 **다음 dispatch cycle에서** 처리. 같은 턴에 영향 주려면 POST_EFFECT phase로 명시.
- **Replay에서 intervention 재실행 (Q7)**: `reconstruct()`는 intervention 재실행 안 함. 하지만 forward execution(resume 후 다음 턴)에서는 contributor/listener 가 새 trigger마다 정상 실행됨. footer 같은 ephemeral output은 매 compose()에서 새로 계산. **footer가 reconstruct되지 않아도 무방한 이유**: footer가 LLM에게 미친 영향은 그 turn의 assistant 응답에 이미 박혀있고, 그 응답은 `MESSAGE_APPENDED` state 이벤트로 영구 저장되어 reconstruct로 복원되기 때문. footer는 ephemeral compute, downstream effect는 immutable state — 두 레이어 분리.

### Still open (M2/M3 진행에 비블로킹)

- **Frame deepcopy 성능 재검토 시점**: M3 edit_guards 출하 후 측정. dispatch가 turn time의 >5%면 lazy-snapshot으로 전환.
- **compose 경로 frame 미노출**: 현재 contributor는 `body`/`header`/`self_actor`만 받음. 실제로 frame state(ledger 등)를 contributor가 필요로 하는 사례가 등장하면 llm_agent signature 확장 검토.
- **Time-travel 디스플레이의 footer 재현**: M1 기본은 라이브 실행 시 emit된 `HINT_INJECTED` trace 이벤트로 재현. 룰 변경 후 과거 시점을 현재 룰로 다시 보고 싶은 경우 contributor 재실행 옵션 추가 — 필요해질 때.
- **Pack discovery (dir scan vs entry_points)**: M3–M6은 명시 하드코딩 리스트. 진짜 third-party 등장할 때 entry_points 추가.
- **Built-in Pack 위치**: `nature/packs/builtin/<pack_name>/`. M3에서 처음 적용.

---

## 14. Non-goals

명시적으로 **이 문서의 범위 밖**:

- **이벤트 저장소 교체**: 기존 `FileEventStore` / `reconstruct` 그대로 사용.
- **Core frame/session 모델 변경**: Frame, Session, AgentRole 시그니처는 건드리지 않음 (field 추가는 허용).
- **Plugin marketplace / registry / 원격 설치**: 이건 local install 이후 단계.
- **비동기/백그라운드 intervention**: intervention은 턴 루프 안에서만 실행. background task / 데몬 intervention은 out of scope.
- **Cross-session pack state**: pack이 자체 state를 세션 간 유지하려면 event store에 frame-scoped로 기록하거나, 별도 저장소를 명시적으로 선언해야 한다. 암묵적 전역 state 금지.
- **동적 Python 코드 로딩 보안**: Pack이 임의 Python을 실행할 수 있으므로 trust boundary는 사용자 책임. 샌드박싱은 별개 이슈.

---

## 15. 관련 문서

- `ARCHITECTURE.md` — 시스템 전반 (Pack/Host/Agent/Preset/Frame 분해, 현재 구현 상태 포함).
- `EXPERIMENTATION.md` — 실험 매트릭스 실행 및 eval-run 분석 가이드.
- Phase 1 구현 산출물 (`nature/events/types.py` 외): 이 모델의 이벤트 레이어. 변경 없이 재사용.

---

## 16. 다음 단계 (implementation plan)

이 문서가 리뷰 + stable 되면:

1. **M1**: `nature/packs/types.py` + `registry.py` + `effects.py` + `legacy_shim.py` 골격 작성. 테스트 포함.
2. **M2**: `frame/manager.py`에 dispatch hook 2줄 삽입. 기존 테스트 통과 확인.
3. **M3**: `nature/packs/builtin/edit_guards/` Pack 작성. 단위 테스트 + 통합 테스트.
4. **M4**: nature-eval로 Phase 2 pilot 검증 (`solo-qwen-coder-32b × t1` 전/후 비교).
5. **M5**: 결과 검증 후 다음 Pack(budget)으로 진행.

각 단계는 독립 커밋 + 리뷰 가능.

---

## 17. Future: Scatter-Gather Read (auto-delegate pattern)

**Status**: design sketch. 의존성: v2-P6 (Memory Ledger MVP) 이후 착수.

### 17.1 문제

한 agent가 큰 파일 또는 여러 파일을 Read하면 context가 비대해지고 판단력이 저하된다. 2026-04-15 벤치마크 `current × t5`에서 agent가 36번 Read한 뒤 Edit을 한 번도 못 한 건 context 과부하가 한 원인이다. `reads_budget`(Phase 3.1)은 Read 수를 캡하지만, 캡에 도달하면 그냥 **Block** — 대안이 없다. 실제로 agent가 많은 파일을 이해해야 하는 task에서는 Block만으로 부족하다.

### 17.2 핵심 아이디어: Block → auto-delegate

`reads_budget`의 확장. 한도 초과 시 Block 대신, framework가 자동으로 **reader sub-agent를 spawn**하여 파일을 읽고 **정해진 규격으로 요약**해서 parent에게 돌려준다. Parent agent는 raw file content 대신 structured summary만 받는다.

```
orchestrator: "파일 A, B, C, D, E를 이해해야 함"
  │
  ├─ reader_A(file_A) ─→ structured summary ─→ ledger ──┐
  ├─ reader_B(file_B) ─→ structured summary ─→ ledger  ──┤ 병렬
  └─ reader_C(file_C) ─→ structured summary ─→ ledger ──┘
                                                          │
                                              aggregator ─┘
                                              ledger 요약 취합
                                                  │
                                              orchestrator
                                              합성된 이해로 작업 진행
```

**Map-Reduce / Scatter-Gather 패턴**: map 단계(각 reader가 파일별 요약 생성)와 reduce 단계(aggregator가 합성)를 framework가 오케스트레이션.

### 17.3 구성 요소

#### (a) 새 Effect type: `AutoDelegate`

`reads_budget` Gate가 한도 초과 시 반환. manager가 이 effect를 보면 Read 실행 대신 sub-agent를 spawn.

```python
@dataclass
class AutoDelegate:
    """Block + 자동 위임. Gate 전용.

    manager는 이 effect를 보면:
    1. Read를 직접 실행하지 않는다.
    2. reader_role 템플릿으로 child frame을 열고 prompt를 넘긴다.
    3. child가 resolve하면 bubble message를 parent의 tool_result 로 돌려준다.
    """
    role_template: str          # "file_reader"
    prompt: str                 # "Read {file_path} and produce a summary per §17.4"
    parallel_group: str | None  # 같은 group이면 gather로 묶어서 병렬 실행
```

#### (b) Reader role template

전용 role. Read만 허용. 최소한의 instructions.

```python
AgentRole(
    name="file_reader",
    description="Reads a single file and produces a structured summary.",
    instructions="""
You are a focused file-reading agent. You receive a file path.
1. Read the file with the Read tool.
2. Produce a JSON summary in the schema below.
3. Resolve immediately with the summary as your bubble message.

Do NOT edit, do NOT call other tools, do NOT speculate.
""",
    allowed_tools=["Read"],
    model=None,  # parent와 동일 모델 상속
)
```

#### (c) Structured summary format

reader가 출력해야 하는 규격. prompt에 embedded.

```json
{
  "file": "/path/to/file.py",
  "lines": 342,
  "purpose": "한 문장으로 이 파일이 하는 일",
  "key_symbols": [
    {"name": "AreaManager", "kind": "class", "line": 77, "description": "frame execution loop"},
    {"name": "_execute_single_tool", "kind": "method", "line": 772, "description": "single tool dispatch"}
  ],
  "imports": ["nature.events", "nature.context.composer"],
  "relevant_sections": [
    {"range": "772-834", "why": "tool dispatch hooks — 수정 대상"},
    {"range": "266-334", "why": "main run loop — 이해 필요"}
  ],
  "notes": "특이사항이나 주의점"
}
```

#### (d) Aggregator

reader들의 요약을 취합. 두 가지 전략:

**전략 A — parent가 직접 취합**: reader들의 bubble message(요약 JSON)가 tool_result로 parent에게 돌아옴. parent context에 요약만 들어오므로 원본 파일 대비 context가 1/10~1/20. parent가 직접 종합.

**전략 B — 별도 aggregator agent**: 요약들을 ledger에 기록하고, aggregator role이 ledger를 읽어서 종합 보고서 작성. parent는 보고서만 보고 작업 결정. 더 깊은 분석이 필요한 경우.

M1 착수 시 전략 A부터 — 더 단순하고 기존 Agent delegation으로 구현 가능.

### 17.4 기존 인프라와의 연결

| 기존 | 역할 |
|---|---|
| `reads_budget.gate` (Phase 3.1) | 한도 초과 감지 지점. Block → AutoDelegate 전환 |
| Agent tool (기존) | child frame spawn 인프라 |
| `LEDGER_FILE_CONFIRMED` (Phase 1) | reader가 요약 작성 시 emit하는 이벤트 |
| Memory Ledger (v2-P6) | 요약 저장소 + sub-frame간 공유 |
| `allowed_tools` per-agent | reader role에 Read만 허용 |
| `allowed_interventions` (Phase 5) | reader에겐 reread_hint 같은 guard 불필요 → 끔 |
| Parallel delegation (기존) | 여러 reader를 asyncio.gather로 병렬 실행 |

### 17.5 구현 순서

```
Phase 5 (per-agent config)      ← reader role에 맞는 intervention set 지정
    ↓
v2-P6 (Memory Ledger MVP)      ← 요약 저장 인프라
    ↓
★ Scatter-Gather Read Pack:
  1. AutoDelegate effect type 추가 (types.py)
  2. manager._execute_single_tool에서 AutoDelegate 처리 → child frame spawn
  3. file_reader role template 등록 (config/profiles 또는 Pack의 tool로 등록)
  4. reads_budget.gate 확장: limit 초과 시 Block 대신 AutoDelegate 반환
  5. summary 규격 finalize + reader instructions에 embed
  6. 검증: nature-eval t3 (large refactor) 같은 multi-file task로 전/후 비교
```

### 17.6 열린 질문

- **reader의 모델**: parent와 같은 모델? 아니면 더 싼 모델(Haiku)? 파일 읽고 요약하기는 저렴한 작업이니 싼 모델이 비용 효율적일 수 있음.
- **요약 품질 검증**: reader가 엉터리 요약을 내놓으면? 별도 validator? 아니면 요약 schema validation만으로 충분?
- **요약 크기 vs 원본 크기**: 100줄 파일의 요약이 원본보다 길어질 수 있음. 작은 파일은 직접 Read가 나을 수 있다 → threshold 필요 (예: 200줄 이하는 직접 Read, 이상은 scatter).
- **parallel group 관리**: 한 턴에 여러 Read가 동시에 발생했을 때, 이걸 개별 AutoDelegate로 보내나, 아니면 batch AutoDelegate로 묶나?
- **캐싱**: 같은 파일을 여러 agent가 읽으려 하면 ledger에 이미 있는 요약을 재사용할지. ledger hit이면 reader spawn 안 함.
