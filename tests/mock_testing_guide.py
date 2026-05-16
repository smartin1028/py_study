"""
Mock Testing Guide — Python vs Java 비교
========================================
pyTool 프로젝트를 위한 mock 테스트 가이드.
모든 예제에 Python(unittest.mock) ↔ Java(Mockito + JUnit 5) 비교 주석을 포함한다.

## Mock이 필요한 이유

1. **외부 의존성 격리** — API, DB, 파일시스템에 의존하는 코드를 독립적으로 검증한다.
2. **결정적(deterministic) 테스트** — 네트워크 지연, DB 상태, 시간 등 비결정적 요소를 제거한다.
3. **빠른 피드백** — 실제 I/O 없이 수 밀리초 안에 완료된다. (Java도 동일)
4. **에지 케이스 검증** — 타임아웃, 500 에러, DB 장애 등 실제로 재현하기 어려운 시나리오를 시뮬레이션한다.
5. **호출 검증** — "이 함수가 올바른 인자로 호출되었는가"를 검증할 수 있다.

## Mockito vs unittest.mock 빠른 비교표

| 개념              | Java (Mockito)                              | Python (unittest.mock)                    |
|-------------------|---------------------------------------------|-------------------------------------------|
| Mock 객체 생성     | `@Mock` / `mock(Foo.class)`                 | `Mock()` / `Mock(spec=Foo)`               |
| Stub 반환값 설정   | `when(x.foo()).thenReturn("A")`             | `x.foo.return_value = "A"`                |
| 예외 발생          | `when(x.foo()).thenThrow(new Xxx())`        | `x.foo.side_effect = Xxx()`               |
| 호출 검증          | `verify(x).foo(arg)`                        | `x.foo.assert_called_once_with(arg)`      |
| 호출 없음 검증     | `verify(x, never()).foo()`                  | `x.foo.assert_not_called()`               |
| 인자 Matcher       | `anyString()`, `eq(3)`                      | `ANY`, `mock.ANY` 등                      |
| DI 주입            | `@InjectMocks` + `MockitoExtension`         | 수동 주입 (생성자에 직접 전달)              |
| 모듈/static 교체   | `MockedStatic<Foo>` (Mockito 3.4+)          | `@patch("module.Foo")`                    |
| 순서 검증          | `InOrder`                                   | `assert_has_calls([...], any_order=False)` |
| Fake (가짜 객체)   | `@Spy` / 직접 구현                           | 직접 구현 (인메모리 dict 등)               |
| 비동기             | `CompletableFuture` / `Mono` stub           | `AsyncMock()`                             |

## Mock 할당 순서: stub → mock → fake → real

| 기법   | 사용 시점                              | Python                          | Java                                      |
|--------|---------------------------------------|----------------------------------|-------------------------------------------|
| stub   | 단순히 값만 제공하면 될 때              | `return_value = ...`            | `when(x.foo()).thenReturn(...)`           |
| mock   | 호출 여부·횟수·인자 검증이 필요할 때    | `assert_called_once_with(...)` | `verify(x).foo(arg)`                      |
| fake   | 상태를 가진 의존성 (DB, 캐시 등)        | 인메모리 dict/list 구현           | `@Spy` 또는 인메모리 구현                   |
| real   | 의존성이 순수 함수에 가까울 때          | mock 불필요                      | mock 불필요                                |
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

# ============================================================================
# 1. API 요청 예시
# ============================================================================


class UserClient:
    """외부 API로 사용자 정보를 조회하는 클라이언트."""

    def __init__(self, base_url: str, session: "requests.Session"):
        # Java: @Autowired 또는 생성자 주입. Mockito에서는 @InjectMocks 로 주입.
        self._base_url = base_url
        self._session = session

    def fetch_user(self, user_id: int) -> dict:
        resp = self._session.get(f"{self._base_url}/users/{user_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()


# --- 1.1 stub: 성공 응답 ---------------------------------------------------
# Java 비교:
#   @Mock RestTemplate restTemplate;
#   @InjectMocks UserClient client;
#
#   @Test void fetchUser_returnsParsedJson() {
#       var mockResp = new ResponseEntity<>("{\"id\":1,\"name\":\"Alice\"}", HttpStatus.OK);
#       when(restTemplate.getForEntity(anyString(), eq(Map.class))).thenReturn(mockResp);
#       // 또는 MockHttpServer(WireMock)를 써서 실제 HTTP를 흉내 내기도 함
#       Map<String, Object> user = client.fetchUser(1);
#       assertEquals("Alice", user.get("name"));
#   }

def test_fetch_user_returns_parsed_json():
    # Given — Python은 Mock() 하나로 모든 속성/메서드를 동적으로 생성 가능.
    # Java는 Mockito.mock(Foo.class)처럼 클래스 단위로만 mock 생성.
    mock_session = Mock()
    # Python: 메서드 체이닝마다 .return_value 로 연결해야 한다.
    # Mockito: when(mock.method()).thenReturn(val) — 체이닝 시 자연스럽게 기술.
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = {
        "id": 1,
        "name": "Alice",
        "email": "alice@example.com",
    }
    # Python: 의존성은 수동 주입 (Mockito의 @InjectMocks 같은 자동 DI 없음)
    client = UserClient(base_url="https://api.example.com", session=mock_session)

    # When
    user = client.fetch_user(user_id=1)

    # Then
    # Python: assert 문으로 직접 검증 (JUnit의 assertEquals, assertThat 에 해당)
    assert user["name"] == "Alice"
    assert user["email"] == "alice@example.com"


# --- 1.2 stub / mock: HTTP 에러 응답 ---------------------------------------
# Java 비교:
#   @Test void fetchUser_throwsOnHttpError() {
#       when(restTemplate.getForEntity(anyString(), eq(Map.class)))
#           .thenThrow(new HttpClientErrorException(HttpStatus.NOT_FOUND));
#       assertThrows(HttpClientErrorException.class, () -> client.fetchUser(999));
#   }
#   // thenThrow()가 Python의 side_effect + raise 에 대응

def test_fetch_user_raises_on_http_error():
    import requests

    # Given
    mock_session = Mock()
    mock_response = Mock()
    # Python: side_effect 에 예외 객체를 할당하면 호출 시 예외가 발생한다.
    # Java: when(x.foo()).thenThrow(new Xxx()) 와 동일.
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    mock_session.get.return_value = mock_response
    client = UserClient(base_url="https://api.example.com", session=mock_session)

    # When / Then
    # Python: pytest.raises 컨텍스트 매니저로 예외 검증.
    # Java: assertThrows(Xxx.class, () -> ...) 와 동일.
    with pytest.raises(requests.HTTPError):
        client.fetch_user(user_id=999)


# --- 1.3 mock: 호출 인자 검증 -----------------------------------------------
# Java 비교:
#   @Mock ApiClient apiClient;
#   @InjectMocks NotificationService service;
#
#   @Test void sendWelcome_postsCorrectPayload() {
#       when(apiClient.post(eq("/notifications"), any())).thenReturn(202);
#       boolean result = service.sendWelcome(42, "alice@example.com");
#
#       assertTrue(result);
#       // Java: verify()로 호출 검증. ArgumentCaptor 로 인자 캡처 가능.
#       verify(apiClient).post(eq("/notifications"), argThat(payload ->
#           payload.get("to").equals("alice@example.com") &&
#           payload.get("subject").equals("Welcome!")
#       ));
#   }


class NotificationService:
    def __init__(self, api_client: "APIClient"):
        self._client = api_client

    def send_welcome(self, user_id: int, email: str) -> bool:
        payload = {
            "to": email,
            "subject": "Welcome!",
            "body": f"User {user_id} has been registered.",
        }
        resp = self._client.post("/notifications", json=payload)
        return resp.status_code == 202


def test_send_welcome_posts_correct_payload():
    # Given
    # Python: spec=["post"] 로 허용할 속성을 제한 가능.
    # Java: Mockito.mock(Foo.class) 는 항상 해당 타입의 메서드만 허용하므로
    #       spec 제한이 기본값. Python은 기본적으로 어떤 속성도 접근 가능.
    mock_client = Mock(spec=["post"])
    mock_client.post.return_value.status_code = 202
    service = NotificationService(api_client=mock_client)

    # When
    result = service.send_welcome(user_id=42, email="alice@example.com")

    # Then
    assert result is True
    # Python: assert_called_once_with() 로 인자까지 정확히 검증.
    # Java: verify(apiClient).post(eq("/notifications"), any()) 와 유사하나
    #       Python은 실제 인자값을 직접 비교한다.
    mock_client.post.assert_called_once_with(
        "/notifications",
        json={
            "to": "alice@example.com",
            "subject": "Welcome!",
            "body": "User 42 has been registered.",
        },
    )


# --- 1.4 side_effect: 여러 응답 시퀀스 ---------------------------------------
# Java 비교:
#   when(restTemplate.getForEntity(anyString(), eq(Map.class)))
#       .thenThrow(new HttpClientErrorException(HttpStatus.SERVICE_UNAVAILABLE))
#       .thenReturn(new ResponseEntity<>(Map.of("status", "ok"), HttpStatus.OK));
#   // Mockito는 thenThrow().thenReturn() 체이닝으로 시퀀스 정의.
#   // Python은 list로 side_effect 에 전달 — 더 직관적.


class RetryableClient:
    """첫 요청 실패 시 최대 max_retries 회 재시도."""

    def __init__(self, session: "requests.Session", max_retries: int = 3):
        self._session = session
        self._max_retries = max_retries

    def fetch(self, path: str) -> dict:
        for _ in range(self._max_retries):
            resp = self._session.get(f"https://api.example.com{path}")
            if resp.status_code == 200:
                return resp.json()
        raise ConnectionError("All retries exhausted")


def test_retry_on_first_failure():
    # Given — 첫 호출 실패, 두 번째 성공
    mock_session = Mock()
    # Python: side_effect 에 list 를 전달하면 호출마다 순서대로 값을 반환.
    # Java: when(...).thenThrow(...).thenReturn(...) 으로 동일.
    # Python 방식은 list 인덱스 기반이라 시퀀스 길이를 초과하면 StopIteration.
    mock_session.get.side_effect = [
        Mock(status_code=503, raise_for_status=Mock(side_effect=ConnectionError)),
        Mock(status_code=200, json=Mock(return_value={"status": "ok"})),
    ]
    client = RetryableClient(session=mock_session, max_retries=3)

    # When
    result = client.fetch("/health")

    # Then
    assert result == {"status": "ok"}
    # Python: call_count 로 총 호출 횟수 검증.
    # Java: verify(restTemplate, times(2)).getForEntity(...) 와 동일.
    assert mock_session.get.call_count == 2


# --- 1.5 patch 데코레이터: 모듈 수준 의존성 교체 ----------------------------
# Java 비교:
#   // Mockito 3.4+ 에서 MockedStatic<T> 로 static 메서드 mock.
#   try (var mocked = mockStatic(UserService.class)) {
#       mocked.when(() -> UserService.createUser("Bob"))
#             .thenReturn(new User(100, "Bob"));
#       // ...
#   }
#   // Python의 @patch 는 더 간결하지만, 문자열 경로 오타에 취약.


# src/user_service.py 에 있다고 가정:
# from requests import post
#
# def create_user(name: str) -> dict:
#     resp = post("https://api.example.com/users", json={"name": name}, timeout=10)
#     resp.raise_for_status()
#     return resp.json()


@patch("src.user_service.post")
def test_create_user_calls_api_with_correct_body(mock_post):
    # Python: @patch("모듈경로.대상") — 정의 위치가 아닌 '사용 위치'의 모듈 경로.
    # Java: mockStatic(ClassName.class) — 클래스 자체를 mock.
    # Python의 patch는 데코레이터 인자가 mock 객체로 주입된다.
    # Mockito는 try-with-resources 블록 안에서만 static mock 유효.

    # Given
    mock_post.return_value.status_code = 201
    mock_post.return_value.json.return_value = {"id": 100, "name": "Bob"}
    from src.user_service import create_user

    # When
    result = create_user("Bob")

    # Then
    assert result == {"id": 100, "name": "Bob"}
    mock_post.assert_called_once_with(
        "https://api.example.com/users",
        json={"name": "Bob"},
        timeout=10,
    )


# --- 1.6 patch.dict: 환경 변수 교체 ----------------------------------------
# Java 비교:
#   // Java에서 환경 변수 mock은 번거롭다.
#   // System Stubs 라이브러리 (system-stubs-junit5) 사용:
#   @ExtendWith(SystemStubsExtension.class)
#   class ApiClientTest {
#       @SystemStub EnvironmentVariables env =
#           new EnvironmentVariables("API_KEY", "test-secret-key");
#       // ...
#   }
#   // Python은 @patch.dict 한 줄로 해결 — 훨씬 간단.


class APIClient:
    def __init__(self):
        # 생성자에서 환경 변수 읽기 — 테스트 시 mock 필요.
        # Java: System.getenv("API_KEY") 를 생성자에서 읽는 패턴과 동일.
        self._key = __import__("os").environ["API_KEY"]

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}"}


@patch.dict(__import__("os").environ, {"API_KEY": "test-secret-key"})
def test_headers_includes_api_key():
    # Given — @patch.dict 로 os.environ 을 테스트 기간 동안만 교체.
    # Java: SystemStubs 또는 EnvironmentVariableRunner 필요.
    client = APIClient()

    # When
    headers = client.headers()

    # Then
    assert headers == {"Authorization": "Bearer test-secret-key"}


# ============================================================================
# 2. DB 요청 예시
# ============================================================================

from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    # Python: dataclass(frozen=True) = Java의 record (Java 14+).
    #   Java: record User(int id, String name, String email) {}
    id: int
    name: str
    email: str


class UserRepository:
    # Java: @Repository — Spring Data JPA 흔히 extends JpaRepository<User, Long>
    #   기본 CRUD는 자동 생성되므로, 커스텀 쿼리 메서드만 테스트 작성.

    def __init__(self, db_session: "Session"):
        # Python: SQLAlchemy Session 주입.
        # Java: EntityManager (JPA) 또는 JdbcTemplate (Spring JDBC) 주입.
        self._db = db_session

    def find_by_id(self, user_id: int) -> User | None:
        # ORM 메서드 체이닝: query().filter().first()
        # Java JPA: entityManager.find(User.class, userId) — 단일 호출.
        # Java Criteria API: cb.createQuery().where(...) — Python과 유사한 체이닝.
        return self._db.query(User).filter(User.id == user_id).first()


# --- 2.1 stub: 메서드 체이닝 mock -------------------------------------------
# Java 비교:
#   @Mock EntityManager em;
#   @InjectMocks UserRepository repo;
#
#   @Test void findById_returnsUserWhenExists() {
#       var user = new User(1, "Alice", "alice@example.com");
#       when(em.find(User.class, 1L)).thenReturn(user);
#       // JPA는 단일 find() 호출이므로 체이닝 mock 불필요.
#       // MyBatis: when(sqlSession.selectOne("findById", 1)).thenReturn(user);
#   }

def test_find_by_id_returns_user_when_exists():
    # Given
    mock_db = Mock()
    # Python: 메서드 체이닝마다 .return_value 를 연결해야 한다.
    #   query() → Mock → .return_value → filter() → Mock → .return_value → first()
    # Java Mockito: 체이닝 mock 이 필요하면 deep stub 사용.
    #   when(mock.query(any()).filter(any()).first()).thenReturn(user); (Mockito 1.10+)
    #   또는 @Mock(answer = RETURNS_DEEP_STUBS) 선언.
    mock_db.query.return_value.filter.return_value.first.return_value = User(
        id=1, name="Alice", email="alice@example.com"
    )
    repo = UserRepository(db_session=mock_db)

    # When
    user = repo.find_by_id(user_id=1)

    # Then
    assert user == User(id=1, name="Alice", email="alice@example.com")


def test_find_by_id_returns_none_when_missing():
    # Java: when(em.find(User.class, 999L)).thenReturn(null);
    # Given
    mock_db = Mock()
    mock_db.query.return_value.filter.return_value.first.return_value = None
    repo = UserRepository(db_session=mock_db)

    # When
    user = repo.find_by_id(user_id=999)

    # Then
    # Python: is None 으로 명시적 비교 (is not None 원칙).
    # Java: assertNull(user);
    assert user is None


# --- 2.2 DB 예외 시뮬레이션 -------------------------------------------------
# Java 비교:
#   @Test void findById_wrapsDbError() {
#       when(em.find(User.class, 1L))
#           .thenThrow(new PersistenceException("Connection refused"));
#       assertThrows(PersistenceException.class, () -> repo.findById(1L));
#   }


class DatabaseError(Exception):
    """커스텀 DB 예외 — Java의 PersistenceException / DataAccessException 에 대응."""


def test_find_by_id_wraps_db_error():
    # Given
    mock_db = Mock()
    # Python: side_effect 에 예외 할당.
    # Java: thenThrow(new Xxx()) 와 동일.
    mock_db.query.side_effect = DatabaseError("Connection refused")
    repo = UserRepository(db_session=mock_db)

    # When / Then
    with pytest.raises(DatabaseError, match="Connection refused"):
        repo.find_by_id(1)


# --- 2.3 Fake: 상태를 가진 인메모리 저장소 ----------------------------------
# Java 비교:
#   // Java는 주로 H2 인메모리 DB로 Fake 구현 대신 실제 DB를 띄움 (@DataJpaTest).
#   // 순수 Fake 구현이 필요하면 ConcurrentHashMap 을 쓴다.
#   class FakeUserRepository implements UserRepository {
#       private final Map<Long, User> store = new ConcurrentHashMap<>();
#       @Override public User findById(Long id) { return store.get(id); }
#       @Override public void save(User u) { store.put(u.getId(), u); }
#   }
#   // Python은 동적 타이핑 덕분에 인터페이스 구현 없이도 Fake 객체 사용 가능.


class FakeUserRepository:
    """인메모리 UserRepository — 통합 테스트에서 실제 DB 대체.

    Python: ABC를 구현하지 않아도 덕 타이핑으로 UserRepository 자리에 주입 가능.
    Java:   반드시 UserRepository 인터페이스를 implements 해야 함.
    """

    def __init__(self):
        # Python: dict[int, User] 로 제네릭 타입 힌트.
        # Java:   Map<Long, User> store = new HashMap<>();
        self._store: dict[int, User] = {}

    def save(self, user: User) -> None:
        self._store[user.id] = user

    def find_by_id(self, user_id: int) -> User | None:
        # Python: dict.get() 은 키 없으면 None 반환.
        # Java:   store.get(userId) — 동일하게 null 반환.
        return self._store.get(user_id)

    def delete(self, user_id: int) -> None:
        # Python: dict.pop(key, None) — 키 없어도 예외 없음.
        # Java:   store.remove(userId) — 동일.
        self._store.pop(user_id, None)


def test_fake_repository_persists_and_retrieves():
    # Fake 객체는 mock이 아니라 실제 동작하는 객체 — 상태 검증이 가능하다.
    # Java에서도 동일: Fake 객체 생성 → save() → find() → assert.
    # Given
    repo = FakeUserRepository()
    user = User(id=1, name="Alice", email="alice@example.com")

    # When
    repo.save(user)
    found = repo.find_by_id(1)

    # Then
    assert found == user


def test_fake_repository_delete_removes_user():
    # Given
    repo = FakeUserRepository()
    repo.save(User(id=1, name="Alice", email="alice@example.com"))

    # When
    repo.delete(1)

    # Then
    assert repo.find_by_id(1) is None


# ============================================================================
# 3. 트랜잭션 서비스 예시
# ============================================================================


@dataclass(frozen=True)
class Account:
    id: int
    balance: int


class TransferService:
    """계좌 이체 서비스.

    Java: @Service + @Transactional — Spring의 선언적 트랜잭션.
    Python: 별도 트랜잭션 데코레이터 없이 수동 관리하거나
            SQLAlchemy의 session.begin() 컨텍스트 매니저 사용.
    """

    def __init__(self, account_repo: "AccountRepository", notifier: "Notifier"):
        # Java: @Autowired 로 주입. Mockito 테스트에선 @InjectMocks.
        # Python: 생성자에서 수동 주입 — 더 명시적이지만 DI 컨테이너 없음.
        self._repo = account_repo
        self._notifier = notifier

    def transfer(self, from_id: int, to_id: int, amount: int) -> None:
        sender = self._repo.find_by_id(from_id)
        receiver = self._repo.find_by_id(to_id)

        if sender is None or receiver is None:
            raise ValueError("Account not found")

        if sender.balance < amount:
            raise ValueError("Insufficient funds")

        # Java: @Transactional 이 메서드 전체를 감싸므로 예외 발생 시 자동 롤백.
        # Python: 명시적 rollback 또는 context manager 필요.
        self._repo.update_balance(from_id, sender.balance - amount)
        self._repo.update_balance(to_id, receiver.balance + amount)
        self._notifier.send(to_id, f"Received {amount} from {from_id}")


# --- 3.1 mock: 호출 인자 + 순서 검증 ---------------------------------------
# Java 비교:
#   @Test void transfer_updatesBothBalances() {
#       when(repo.findById(1L)).thenReturn(new Account(1L, 1000));
#       when(repo.findById(2L)).thenReturn(new Account(2L, 500));
#
#       service.transfer(1L, 2L, 300);
#
#       // Java: ArgumentCaptor 나 eq()로 인자 검증.
#       verify(repo).updateBalance(eq(1L), eq(700));
#       verify(repo).updateBalance(eq(2L), eq(800));
#       verify(notifier).send(eq(2L), eq("Received 300 from 1"));
#   }

def test_transfer_updates_both_balances():
    # Given
    mock_repo = Mock()
    # Python: find_by_id 가 두 번 호출되므로 side_effect 리스트로 순차 반환.
    # Java:   when().thenReturn() 을 각각 선언 — 체이닝: .thenReturn(a).thenReturn(b)
    mock_repo.find_by_id.side_effect = [
        Account(id=1, balance=1000),
        Account(id=2, balance=500),
    ]
    mock_notifier = Mock()
    service = TransferService(account_repo=mock_repo, notifier=mock_notifier)

    # When
    service.transfer(from_id=1, to_id=2, amount=300)

    # Then
    # Python: assert_any_call() — 여러 호출 중 하나라도 해당 인자로 호출되었는지 검증.
    # Java:   verify(repo).updateBalance(eq(1L), eq(700)) 와 동일 (순서 무관).
    mock_repo.update_balance.assert_any_call(1, 700)
    mock_repo.update_balance.assert_any_call(2, 800)
    # Python: assert_called_once_with() — 정확히 1회 + 인자 일치.
    # Java:   verify(notifier, times(1)).send(2L, "Received 300 from 1");
    mock_notifier.send.assert_called_once_with(2, "Received 300 from 1")


def test_transfer_asserts_call_order():
    """assert_has_calls 로 호출 순서까지 검증한다.

    Java 비교:
        InOrder inOrder = inOrder(repo);
        inOrder.verify(repo).findById(1L);
        inOrder.verify(repo).findById(2L);
        inOrder.verify(repo).updateBalance(1L, 700);
        inOrder.verify(repo).updateBalance(2L, 800);
        // Python: assert_has_calls([...], any_order=False) — list로 순서 명시.
    """
    # Given
    mock_repo = Mock()
    mock_repo.find_by_id.side_effect = [
        Account(id=1, balance=1000),
        Account(id=2, balance=500),
    ]
    mock_notifier = Mock()
    service = TransferService(account_repo=mock_repo, notifier=mock_notifier)

    # When
    service.transfer(from_id=1, to_id=2, amount=300)

    # Then — any_order=False 로 순서 검증.
    # Java: InOrder.verify() 시퀀스와 동일. 실패 시 "Calls not found" 메시지.
    mock_repo.assert_has_calls(
        [
            mock_repo.find_by_id(1),
            mock_repo.find_by_id(2),
            mock_repo.update_balance(1, 700),
            mock_repo.update_balance(2, 800),
        ],
        any_order=False,
    )


def test_transfer_raises_when_insufficient_funds():
    # Java: assertThrows(IllegalArgumentException.class, () -> service.transfer(...));
    # Given
    mock_repo = Mock()
    mock_repo.find_by_id.return_value = Account(id=1, balance=100)
    mock_notifier = Mock()
    service = TransferService(account_repo=mock_repo, notifier=mock_notifier)

    # When / Then
    with pytest.raises(ValueError, match="Insufficient funds"):
        service.transfer(from_id=1, to_id=2, amount=500)

    # Then — 실패 시 부수 효과가 없었는지 검증.
    # Python: assert_not_called() — 한 번도 호출되지 않았는지.
    # Java:   verify(repo, never()).updateBalance(any(), any());
    mock_repo.update_balance.assert_not_called()
    mock_notifier.send.assert_not_called()


# --- 3.2 주문 서비스: 결제 실패 시 롤백 ------------------------------------
# Java 비교:
#   @Test void placeOrder_cancelsWhenPaymentFails() {
#       when(orderRepo.create(any(), any(), anyDouble()))
#           .thenReturn(new Order(500L, "pending"));
#       when(payment.charge(anyLong(), anyDouble()))
#           .thenThrow(new PaymentException("Card declined"));
#
#       assertThrows(PaymentException.class,
#           () -> service.placeOrder(1L, List.of(new Item(5000, 1))));
#
#       verify(orderRepo).cancel(500L);
#       verify(orderRepo, never()).confirm(anyLong(), anyString());
#   }
#   // @Transactional 테스트에서는 실제 롤백 확인을 위해
#   // @BeforeEach/@AfterEach 로 트랜잭션 시작/롤백 처리.


class PaymentError(Exception):
    pass


class OrderService:
    def __init__(self, order_repo: "OrderRepository", payment_gateway: "PaymentGateway"):
        self._order_repo = order_repo
        self._payment = payment_gateway

    def place_order(self, user_id: int, items: list[dict]) -> dict:
        total = sum(item["price"] * item["qty"] for item in items)
        order = self._order_repo.create(user_id=user_id, items=items, total=total)
        try:
            charge_id = self._payment.charge(user_id=user_id, amount=total)
        except PaymentError:
            # 예외 발생 시 주문 취소 (롤백). Java: @Transactional 이 자동 처리.
            self._order_repo.cancel(order["id"])
            raise
        self._order_repo.confirm(order["id"], charge_id=charge_id)
        return order


def test_place_order_cancels_when_payment_fails():
    """결제 실패 시 취소 + confirm 미호출 검증.

    Java Mockito: verify(repo, never()).confirm(...) 으로 호출 없음 검증.
    Python:       assert_not_called() — Mock 객체의 모든 메서드에 내장.
    """
    # Given
    mock_order_repo = Mock()
    mock_order_repo.create.return_value = {"id": 500, "status": "pending"}
    mock_payment = Mock()
    # Python: side_effect=PaymentError(...) — 호출 시 예외 발생.
    # Java:   when(payment.charge(...)).thenThrow(new PaymentError("Card declined"));
    mock_payment.charge.side_effect = PaymentError("Card declined")
    service = OrderService(order_repo=mock_order_repo, payment_gateway=mock_payment)

    # When / Then
    with pytest.raises(PaymentError, match="Card declined"):
        service.place_order(user_id=1, items=[{"price": 5000, "qty": 1}])

    # Then — 결제 실패 시 주문 취소 + confirm 미호출.
    # Java: verify(orderRepo).cancel(500L);
    mock_order_repo.create.assert_called_once()
    mock_payment.charge.assert_called_once_with(user_id=1, amount=5000)
    mock_order_repo.cancel.assert_called_once_with(500)
    # Python: assert_not_called() — Java의 verify(x, never()).method() 와 동일.
    mock_order_repo.confirm.assert_not_called()


def test_place_order_confirms_when_payment_succeeds():
    """결제 성공 시 confirm 호출 + cancel 미호출 검증."""
    # Given
    mock_order_repo = Mock()
    mock_order_repo.create.return_value = {"id": 500, "status": "pending"}
    mock_payment = Mock()
    mock_payment.charge.return_value = "ch_abc123"
    service = OrderService(order_repo=mock_order_repo, payment_gateway=mock_payment)

    # When
    result = service.place_order(user_id=1, items=[{"price": 5000, "qty": 2}])

    # Then
    assert result == {"id": 500, "status": "pending"}
    mock_order_repo.confirm.assert_called_once_with(500, charge_id="ch_abc123")
    mock_order_repo.cancel.assert_not_called()


# ============================================================================
# 4. 파일 I/O 예시
# ============================================================================
# Java 비교:
#   // JUnit 5: @TempDir Path tmpDir — Python의 tmp_path fixture 와 동일.
#   @Test void saveAndLoadCsv(@TempDir Path tmpDir) {
#       Path file = tmpDir.resolve("data.csv");
#       // ... Files.writeString(file, csvContent);
#       // ... String loaded = Files.readString(file);
#   }
#   // Java는 try-with-resources + BufferedReader/Writer 필요.
#   // Python은 df.to_csv() / pd.read_csv() 한 줄 — pandas 덕분에 간결.


def test_save_and_load_csv(tmp_path):
    """tmp_path fixture — 실제 파일시스템 사용 (mock 불필요).

    Python: pytest 내장 tmp_path (pathlib.Path) — 테스트 후 자동 삭제.
    Java:   JUnit 5 @TempDir Path tmpDir — 동일한 역할.
    """
    import pandas as pd

    # Given
    file_path = tmp_path / "data.csv"
    df = pd.DataFrame({"name": ["Alice", "Bob"], "score": [95, 87]})

    # When
    df.to_csv(file_path, index=False)
    loaded = pd.read_csv(file_path)

    # Then
    pd.testing.assert_frame_equal(loaded, df)


# --- 4.1 patch("builtins.open") ----------------------------------------------
# Java 비교:
#   // Java에선 builtins.open 같은 전역 함수를 mock 할 수 없다.
#   // 대신 파일 읽기를 추상화한 인터페이스를 만들고 mock:
#   interface FileReader { String read(String path); }
#   // 또는 jimfs (in-memory filesystem) 라이브러리 사용.


def load_config(path: str) -> dict:
    import json

    with open(path) as f:
        return json.load(f)


@patch("builtins.open", new_callable=mock_open, read_data='{"debug": true, "port": 8080}')
def test_load_config_parses_json(mock_file):
    # Python: builtins.open 자체를 patch — open() 호출을 가로챔.
    #   new_callable=mock_open: 파일 읽기 전용 mock 제공.
    #   read_data: open() 으로 읽었을 때 반환할 문자열.
    # Java:   이 방식이 불가능하므로 FileReader 인터페이스로 추상화.

    # When
    config = load_config("/etc/app/config.json")

    # Then
    assert config == {"debug": True, "port": 8080}


def test_load_config_file_not_found():
    # When / Then
    # Python: 컨텍스트 매니저로 patch — 함수 내에서만 유효.
    # Java:   try-with-resources mockStatic(...) 과 유사.
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent.json")


# ============================================================================
# 5. 이메일/알림 발송 예시
# ============================================================================
# Java 비교:
#   // Spring Mail: JavaMailSender.send() — 실제 SMTP 대신 GreenMail(임베디드)이나 mock.
#   // 템플릿 엔진: Thymeleaf — mock 하거나 실제 렌더링.
#   @Mock JavaMailSender mailer;
#   @Mock TemplateEngine templateEngine;
#   @InjectMocks WelcomeEmailService service;


class WelcomeEmailService:
    def __init__(self, mailer: "Mailer", template_engine: "TemplateEngine"):
        self._mailer = mailer
        self._templates = template_engine

    def send_welcome(self, user: User) -> bool:
        body = self._templates.render("welcome.html", name=user.name)
        return self._mailer.send(to=user.email, subject="Welcome", body=body)


def test_send_welcome_renders_template_and_sends():
    # Given
    mock_mailer = Mock()
    mock_mailer.send.return_value = True
    mock_templates = Mock()
    mock_templates.render.return_value = "<h1>Hello, Alice!</h1>"
    service = WelcomeEmailService(mailer=mock_mailer, template_engine=mock_templates)
    user = User(id=1, name="Alice", email="alice@example.com")

    # When
    sent = service.send_welcome(user)

    # Then
    assert sent is True
    # Java: verify(templateEngine).process("welcome.html", ctx);
    mock_templates.render.assert_called_once_with("welcome.html", name="Alice")
    mock_mailer.send.assert_called_once_with(
        to="alice@example.com",
        subject="Welcome",
        body="<h1>Hello, Alice!</h1>",
    )


def test_send_welcome_returns_false_when_mailer_fails():
    # Java: when(mailer.send(any())).thenReturn(false);
    # Given
    mock_mailer = Mock()
    mock_mailer.send.return_value = False
    mock_templates = Mock()
    service = WelcomeEmailService(mailer=mock_mailer, template_engine=mock_templates)

    # When
    sent = service.send_welcome(User(id=1, name="Bob", email="bob@example.com"))

    # Then
    assert sent is False
    # Java: verify(mailer).send(any()); — 발송 자체는 호출되었음을 확인 가능.
    mock_mailer.send.assert_called_once()


# ============================================================================
# 6. 시간 의존성 mock
# ============================================================================
# Java 비교:
#   // Java 8+ Clock 추상화 (권장):
#   public boolean isTokenExpired(Instant issuedAt, Clock clock) {
#       return Instant.now(clock).isAfter(issuedAt.plusSeconds(3600));
#   }
#   @Test void expired_afterTtl() {
#       Clock fixedClock = Clock.fixed(Instant.parse("2026-05-17T12:00:00Z"), UTC);
#       assertTrue(service.isTokenExpired(issuedAt, fixedClock));
#   }
#   // Clock 안 쓰고 LocalDateTime.now() 직접 호출 시 mockStatic 필요.
#   // Python: @patch("module.datetime") — now()의 반환값만 교체.

from datetime import datetime, timedelta


def is_token_expired(issued_at: datetime, ttl_seconds: int = 3600) -> bool:
    # Python: datetime.now() — 모듈 참조를 mock.
    # Java Best Practice: Clock 객체를 주입받아 Instant.now(clock).
    #   Clock을 쓰면 mock 필요 없이 Clock.fixed() 로 제어 가능.
    return datetime.now() > issued_at + timedelta(seconds=ttl_seconds)


@patch("src.token_service.datetime")
def test_is_token_expired_returns_true_after_ttl(mock_datetime):
    # Given — 현재 시간을 2026-05-17 12:00:00 로 고정.
    # Java: Clock.fixed(Instant.parse("2026-05-17T12:00:00Z"), ZoneOffset.UTC)
    mock_datetime.now.return_value = datetime(2026, 5, 17, 12, 0, 0)
    issued = datetime(2026, 5, 17, 10, 0, 0)  # 2시간 전 발급 → TTL 1시간 초과

    # When
    expired = is_token_expired(issued_at=issued, ttl_seconds=3600)

    # Then
    assert expired is True


@patch("src.token_service.datetime")
def test_is_token_expired_returns_false_within_ttl(mock_datetime):
    # Given
    mock_datetime.now.return_value = datetime(2026, 5, 17, 12, 0, 0)
    issued = datetime(2026, 5, 17, 11, 30, 0)  # 30분 전 발급 → TTL 1시간 이내

    # When
    expired = is_token_expired(issued_at=issued, ttl_seconds=3600)

    # Then
    assert expired is False


# ============================================================================
# 7. 비동기 mock (AsyncMock)
# ============================================================================
# Java 비교:
#   // Java 비동기는 주로 CompletableFuture 나 Reactor(Mono/Flux) 사용.
#   @Test void fetchUserAsync_returnsParsedJson() {
#       when(asyncClient.fetchUser(1L))
#           .thenReturn(CompletableFuture.completedFuture(Map.of("id", 1, "name", "Alice")));
#       // 또는 WebClient(Spring WebFlux) mock:
#       // when(webClient.get().uri(...).retrieve().bodyToMono(Map.class))
#       //     .thenReturn(Mono.just(Map.of("id", 1, "name", "Alice")));
#   }
#   // Python의 AsyncMock은 coroutine 자체를 mock — async/await 패턴에 특화.


class AsyncUserClient:
    """비동기 HTTP 클라이언트 — aiohttp 사용.

    Java: WebClient (Spring WebFlux) 또는 AsyncHttpClient 에 대응.
    """

    def __init__(self, session: "aiohttp.ClientSession"):
        self._session = session

    async def fetch_user(self, user_id: int) -> dict:
        # Python: async with — 비동기 context manager.
        # Java: WebClient.get().uri(...).retrieve().bodyToMono(...)
        async with self._session.get(
            f"https://api.example.com/users/{user_id}"
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


@pytest.mark.asyncio
# Python: @pytest.mark.asyncio — 코루틴 테스트 마커 필수.
# Java:   @Test + CompletableFuture.get() 또는 StepVerifier (Reactor).
async def test_fetch_user_async_returns_parsed_json():
    # Given
    # Python: MagicMock — __aenter__ 같은 매직 메서드를 자동 지원.
    # Java:   Mockito.mock(WebClient.class) — 별도 매직 메서드 지원.
    mock_session = MagicMock()
    mock_resp = AsyncMock()  # Python 3.8+: 비동기 메서드 전용 Mock
    # Python: await 호출은 AsyncMock 이 자동으로 awaitable 반환.
    # Java:   CompletableFuture.completedFuture(result) 로 감싸야 함.
    mock_resp.json.return_value = {"id": 1, "name": "Alice"}
    # Python: __aenter__ 로 async with 진입점 mock.
    # Java:   WebClient.ResponseSpec mock 은 retrieve() → bodyToMono() 체인.
    mock_session.get.return_value.__aenter__.return_value = mock_resp
    client = AsyncUserClient(session=mock_session)

    # When
    user = await client.fetch_user(user_id=1)

    # Then
    assert user == {"id": 1, "name": "Alice"}


@pytest.mark.asyncio
async def test_fetch_user_async_raises_on_404():
    import aiohttp

    # Given
    mock_session = MagicMock()
    mock_resp = AsyncMock()
    # Python: AsyncMock 도 side_effect 로 예외 발생 가능.
    # Java:   when(...).thenReturn(Mono.error(new HttpClientErrorException(404)))
    mock_resp.raise_for_status.side_effect = aiohttp.ClientResponseError(
        status=404, message="Not Found", headers={}, request_info=None
    )
    mock_session.get.return_value.__aenter__.return_value = mock_resp
    client = AsyncUserClient(session=mock_session)

    # When / Then
    # Java: StepVerifier.create(mono).expectError(ClientResponseError.class).verify();
    with pytest.raises(aiohttp.ClientResponseError):
        await client.fetch_user(user_id=999)


# --- 7.1 context manager mock 간소화 ---------------------------------------
# Java 비교:
#   // Redis (Spring Data Redis): RedisTemplate<String, String>
#   @Mock RedisTemplate<String, String> redisTemplate;
#   @Mock ValueOperations<String, String> valueOps;
#   @BeforeEach void setUp() {
#       when(redisTemplate.opsForValue()).thenReturn(valueOps);
#   }
#   @Test void setWithTtl_callsRedisWithCorrectArgs() {
#       service.setWithTtl("user:1:token", "abc123", 7200);
#       verify(valueOps).set("user:1:token", "abc123", Duration.ofSeconds(7200));
#   }


class AsyncTokenStore:
    def __init__(self, redis: "Redis"):
        self._redis = redis

    async def set_with_ttl(self, key: str, value: str, ttl: int = 3600) -> None:
        # Python: aioredis — await self._redis.set(key, value, ex=ttl)
        # Java:   redisTemplate.opsForValue().set(key, value, Duration.ofSeconds(ttl))
        await self._redis.set(key, value, ex=ttl)


@pytest.mark.asyncio
async def test_set_with_ttl_calls_redis_with_correct_args():
    # Given
    # Python: AsyncMock — await 호출 자동 처리. context manager 없이 간단.
    # Java:   @Mock + when().thenReturn() — 비동기도 동일한 Mockito 패턴.
    mock_redis = AsyncMock()
    store = AsyncTokenStore(redis=mock_redis)

    # When
    await store.set_with_ttl("user:1:token", "abc123", ttl=7200)

    # Then
    # Python: AsyncMock 도 assert_called_once_with() 사용 가능.
    mock_redis.set.assert_called_once_with("user:1:token", "abc123", ex=7200)


# ============================================================================
# 8. spec= 사용법 — 인터페이스 계약 검증
# ============================================================================
# Java 비교:
#   // Mockito는 타입 기반이므로 spec= 이 항상 적용된 상태.
#   MyService mock = mock(MyService.class);
#   mock.nonexistentMethod();  // 컴파일 에러 (타입에 없는 메서드)
#   // Python은 동적 타이핑이므로 spec= 으로 명시적 제한 필요.
#   // Mockito의 strict stubbing (@ExtendWith(MockitoExtension.class))과 유사:
#   //   선언되지 않은 stub 호출 시 UnnecessaryStubbingException.


class PaymentService:
    """Java: interface PaymentService { String charge(int amount, String currency); }"""
    def charge(self, amount: int, currency: str) -> str: ...
    def refund(self, charge_id: str) -> bool: ...


def test_spec_prevents_calling_nonexistent_method():
    """spec= 을 지정하면 존재하지 않는 메서드 호출 시 AttributeError 발생.

    Java: 타입 시스템이 컴파일 타임에 방지하므로 별도 spec 불필요.
    Python: spec= 으로 Java와 동등한 안전성 확보.
    """
    # Given
    # Python: Mock(spec=PaymentService) — PaymentService 에 정의된 속성만 허용.
    # Java:   mock(PaymentService.class) — 타입 시스템이 보장.
    mock_payment = Mock(spec=PaymentService)
    mock_payment.charge.return_value = "ch_123"

    # When
    result = mock_payment.charge(amount=1000, currency="KRW")

    # Then
    assert result == "ch_123"
    mock_payment.charge.assert_called_once_with(amount=1000, currency="KRW")

    # mock_payment.nonexistent()  # Python: AttributeError 발생 → 즉시 버그 감지.
    # Java: 애초에 컴파일되지 않음.


# ============================================================================
# 9. MagicMock — __iter__, __getitem__ 등 매직 메서드 필요할 때
# ============================================================================
# Java 비교:
#   // Mockito.mock(List.class) — 기본적으로 빈 List 처럼 동작.
#   List<String> mockList = mock(List.class);
#   when(mockList.size()).thenReturn(3);
#   when(mockList.get(0)).thenReturn("Alice");
#   // Python: MagicMock() — __iter__, __len__, __getitem__ 등을 자동 지원.
#   //         Mock() — 기본만 지원. 이터레이션이 필요하면 MagicMock 사용.


def test_magicmock_supports_iteration():
    """MagicMock은 __iter__, __getitem__, __len__ 등 매직 메서드를 기본 지원한다.

    Java: Mockito.mock(List.class) 는 List 인터페이스 기반이므로
          size(), get(), iterator() 등 컬렉션 메서드가 기본 제공.
    Python: Mock()은 __iter__ 지원 안 함 → list(mock.find_all()) 시 TypeError.
            MagicMock()은 지원 → list() 가능.
    """
    # Given
    # Python: MagicMock — 매직 메서드 자동 구현.
    # Java:   mock(List.class) — 인터페이스 메서드 자동 제공.
    mock_repo = MagicMock()
    mock_repo.find_all.return_value = [
        User(id=1, name="Alice", email="alice@example.com"),
        User(id=2, name="Bob", email="bob@example.com"),
    ]

    # When
    users = list(mock_repo.find_all())  # __iter__ 필요 → Mock() 이면 TypeError

    # Then
    assert len(users) == 2


# ============================================================================
# 10. 종합 예제: 인증 + DB + 이메일 통합 서비스
# ============================================================================
# Java 비교:
#   // Spring 통합 테스트: @SpringBootTest + @MockBean
#   @SpringBootTest
#   class RegistrationServiceTest {
#       @MockBean UserRepository userRepo;
#       @MockBean PasswordEncoder passwordEncoder;
#       @MockBean JavaMailSender mailer;
#       @Autowired RegistrationService service;
#
#       @Test void register_createsUserAndSendsEmail() {
#           when(userRepo.findByEmail("alice@example.com")).thenReturn(Optional.empty());
#           when(passwordEncoder.encode("s3cret")).thenReturn("hashed_pw_abc");
#           when(userRepo.save(any())).thenReturn(new User(1L, "Alice", "alice@example.com"));
#
#           User user = service.register("Alice", "alice@example.com", "s3cret");
#
#           assertEquals(1L, user.getId());
#           verify(userRepo).save(argThat(u ->
#               u.getName().equals("Alice") && u.getEmail().equals("alice@example.com")));
#           verify(mailer).send(any(SimpleMailMessage.class));
#       }
#   }


class RegistrationService:
    """회원가입 서비스 — 여러 의존성을 조합하는 전형적인 서비스 레이어.

    Python: 순수 Python 객체로 DI 구현 — 생성자 주입.
    Java:   @Service + @Autowired 생성자 주입 — Spring 컨테이너가 관리.
    """

    def __init__(
        self,
        user_repo: UserRepository,
        password_hasher: "PasswordHasher",
        mailer: "Mailer",
    ):
        self._user_repo = user_repo
        self._hasher = password_hasher
        self._mailer = mailer

    def register(self, name: str, email: str, password: str) -> User:
        existing = self._user_repo.find_by_email(email)
        if existing is not None:
            raise ValueError("Email already registered")

        hashed = self._hasher.hash(password)
        user = self._user_repo.create(name=name, email=email, password_hash=hashed)
        self._mailer.send(to=email, subject="Welcome", body=f"Hello, {name}!")
        return user


def test_register_creates_user_and_sends_welcome_email():
    """정상 흐름: 신규 사용자 등록 → 비밀번호 해싱 → 저장 → 이메일 발송.

    Java: @Test + Mockito.verify() — 동일한 Given-When-Then 구조.
    차이점: Python은 assert + assert_called_once_with 조합.
           Java는 assertEquals + verify 조합.
    """
    # Given — 모든 의존성을 Mock() 으로 생성 후 수동 주입.
    # Java: @Mock + @InjectMocks 또는 생성자에 직접 전달.
    mock_repo = Mock()
    mock_repo.find_by_email.return_value = None  # 미존재 사용자
    mock_repo.create.return_value = User(id=1, name="Alice", email="alice@example.com")
    mock_hasher = Mock()
    mock_hasher.hash.return_value = "hashed_pw_abc"
    mock_mailer = Mock()
    service = RegistrationService(
        user_repo=mock_repo, password_hasher=mock_hasher, mailer=mock_mailer
    )

    # When
    user = service.register(name="Alice", email="alice@example.com", password="s3cret")

    # Then
    # Python: assert + mock.assert_* 조합으로 상태 + 호출 모두 검증.
    # Java:   assertEquals + verify 조합.
    assert user.id == 1
    # 중복 확인 호출 검증
    mock_repo.find_by_email.assert_called_once_with("alice@example.com")
    # 비밀번호 해싱 호출 검증
    mock_hasher.hash.assert_called_once_with("s3cret")
    # 저장소 생성 호출 검증
    mock_repo.create.assert_called_once_with(
        name="Alice", email="alice@example.com", password_hash="hashed_pw_abc"
    )
    # 이메일 발송 호출 검증
    mock_mailer.send.assert_called_once_with(
        to="alice@example.com", subject="Welcome", body="Hello, Alice!"
    )


def test_register_raises_when_email_exists():
    """에지 케이스: 중복 이메일 → 예외 발생 + 부수 효과 없음.

    Java: assertThrows + verify(x, never()) 조합.
    Python: pytest.raises + assert_not_called() 조합.
    """
    # Given — 이미 존재하는 사용자 반환.
    mock_repo = Mock()
    mock_repo.find_by_email.return_value = User(
        id=1, name="Existing", email="alice@example.com"
    )
    mock_hasher = Mock()
    mock_mailer = Mock()
    service = RegistrationService(
        user_repo=mock_repo, password_hasher=mock_hasher, mailer=mock_mailer
    )

    # When / Then
    with pytest.raises(ValueError, match="Email already registered"):
        service.register(name="Alice", email="alice@example.com", password="s3cret")

    # Then — 중복 시 어떤 부수 효과도 발생하지 않았는지 검증.
    # Java: verify(passwordEncoder, never()).encode(anyString());
    #       verify(userRepo, never()).save(any());
    #       verify(mailer, never()).send(any());
    mock_hasher.hash.assert_not_called()
    mock_repo.create.assert_not_called()
    mock_mailer.send.assert_not_called()


# ============================================================================
# 부록: 주요 패턴 요약 (Python ↔ Java 1:1 대응)
# ============================================================================
#
# [Mock 생성]
#   Python: mock = Mock() / Mock(spec=Foo)
#   Java:   Foo mock = mock(Foo.class)
#
# [반환값 설정]
#   Python: mock.method.return_value = 42
#   Java:   when(mock.method()).thenReturn(42)
#
# [예외 발생]
#   Python: mock.method.side_effect = ValueError("msg")
#   Java:   when(mock.method()).thenThrow(new IllegalArgumentException("msg"))
#
# [호출 검증]
#   Python: mock.method.assert_called_once_with(a, b)
#   Java:   verify(mock).method(a, b)  // times(1) 기본
#
# [호출 횟수 검증]
#   Python: assert mock.method.call_count == 3
#   Java:   verify(mock, times(3)).method()
#
# [호출 없음 검증]
#   Python: mock.method.assert_not_called()
#   Java:   verify(mock, never()).method()
#
# [호출 순서 검증]
#   Python: mock.assert_has_calls([call.x(), call.y()], any_order=False)
#   Java:   InOrder inOrder = inOrder(mock); inOrder.verify(mock).x(); inOrder.verify(mock).y();
#
# [static/module mock]
#   Python: @patch("module.func")
#   Java:   try (var ms = mockStatic(Clazz.class)) { ms.when(() -> Clazz.func()).thenReturn(x); }
#
# [환경 변수 mock]
#   Python: @patch.dict(os.environ, {"KEY": "val"})
#   Java:   SystemStubs 또는 @ExtendWith(SystemStubsExtension.class)
#
# [비동기 mock]
#   Python: AsyncMock() — await 자동 지원
#   Java:   CompletableFuture.completedFuture(result) / Mono.just(result)
#
# [Fake 객체]
#   Python: class FakeRepo: (dict 로 상태 관리) — 덕 타이핑으로 인터페이스 불필요
#   Java:   class FakeRepo implements Repository { Map store = new HashMap<>(); }
