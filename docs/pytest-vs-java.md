# pytest vs Java 테스트 어노테이션 비교

## @pytest.fixture

### 개념
`@pytest.fixture`는 JUnit의 `@BeforeEach` + `@AfterEach` + 의존성 주입(DI)을 합친 것과 유사하다.
가장 큰 차이는 **fixture는 값을 반환하고, 테스트가 그 값을 인자로 주입받는다**는 점이다.

### Java 비교

```java
// Java (JUnit 5)
class UserServiceTest {

    private UserRepository repo;
    private UserService service;

    @BeforeEach
    void setUp() {
        repo = mock(UserRepository.class);
        service = new UserService(repo);   // 수동 DI
    }

    @Test
    void test_find_user() {
        when(repo.findById(1)).thenReturn(new User("Alice"));
        User user = service.findUser(1);
        assertEquals("Alice", user.name());
    }
}
```

```python
# Python (pytest)
import pytest

@pytest.fixture
def service():
    repo = Mock(spec=UserRepository)
    repo.find_by_id.return_value = User(name="Alice")
    return UserService(repo)   # fixture가 반환한 값이 테스트 인자로 주입됨

def test_find_user(service):
    user = service.find_user(1)
    assert user.name == "Alice"
```

### fixture의 핵심 포인트

| 특징 | Java | pytest |
|------|------|--------|
| **setup 방식** | `@BeforeEach` 메서드에서 필드 할당 | fixture 함수 반환값을 인자로 주입 |
| **값 전달** | 클래스 필드(가변 상태) | 함수 인자(불변, 스레드 안전) |
| **의존성 해결** | DI 프레임워크 or 수동 | 프레임워크가 fixture 이름으로 자동 매칭 |
| **teardown** | `@AfterEach` | `yield` 이후 코드, 또는 `tmp_path` |
| **재사용** | 상속 or `@Nested` 클래스 | `conftest.py`에 정의하면 자동 공유 |

### fixture로 DB 세션 관리 비교

```java
// Java
class RepositoryTest {
    private Session session;

    @BeforeEach
    void openSession() {
        session = HibernateUtil.getSessionFactory().openSession();
        session.beginTransaction();
    }

    @AfterEach
    void closeSession() {
        session.getTransaction().rollback();
        session.close();
    }
}
```

```python
# Python
@pytest.fixture
def session():
    s = Session()
    s.begin()
    yield s          # <-- 여기까지가 @BeforeEach
    s.rollback()     # <-- 여기부터가 @AfterEach
    s.close()
```

---

## 테스트 어노테이션 전체 비교

### 테스트 정의

| Java (JUnit 5) | pytest | 설명 |
|----------------|--------|------|
| `@Test` | `def test_*():` | 테스트 함수 (pytest는 어노테이션 불필요) |
| `@DisplayName("...")` | 함수명 + docstring | 테스트 설명 |
| `@Disabled` | `@pytest.mark.skip` | 테스트 비활성화 |
| `@Tag("slow")` | `@pytest.mark.slow` | 태그/마크로 분류 |

### 생명주기

| Java (JUnit 5) | pytest | 설명 |
|----------------|--------|------|
| `@BeforeAll` | `setup_module()` / `setup_class()` | 클래스 전체 1회 실행 |
| `@BeforeEach` | `setup_method()` / fixture `scope="function"` | 각 테스트 전 실행 |
| `@AfterEach` | `teardown_method()` / fixture `yield` 이후 | 각 테스트 후 실행 |
| `@AfterAll` | `teardown_module()` / `teardown_class()` | 클래스 전체 후 1회 실행 |

### 테스트 더블 (Mock)

| Java (Mockito) | pytest (unittest.mock) | 설명 |
|----------------|------------------------|------|
| `mock(Class.class)` | `Mock()` | mock 객체 생성 |
| `when(x.foo()).thenReturn(v)` | `x.foo.return_value = v` | 반환값 설정 |
| `verify(x).foo(arg)` | `x.foo.assert_called_once_with(arg)` | 호출 검증 |
| `spy(realObj)` | `Mock(wraps=real_obj)` | 부분 mock |
| `@Mock` / `@ExtendWith(MockitoExtension.class)` | `mocker` fixture (`pytest-mock`) | 자동 mock 주입 |
| `ArgumentCaptor` | `call_args` / `call_args_list` | 호출 인자 캡처 |

### 파라미터화 테스트

```java
// Java
@ParameterizedTest
@CsvSource({"1, 2, 3", "4, 5, 9"})
void test_add(int a, int b, int expected) {
    assertEquals(expected, a + b);
}
```

```python
# Python
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (4, 5, 9),
])
def test_add(a, b, expected):
    assert expected == a + b
```

### 예외 검증

```java
// Java
@Test
void test_divide_by_zero() {
    assertThrows(ArithmeticException.class, () -> 1 / 0);
}
```

```python
# Python
def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        1 / 0
```

---

## 요약

pytest는 어노테이션 대신 **네이밍 컨벤션**과 **의존성 주입**으로 동작한다.
- `@pytest.fixture` = `@BeforeEach` + DI, 가장 큰 차이
- `@pytest.mark.parametrize` = `@ParameterizedTest` + `@CsvSource`
- Mock 설정은 `return_value` / `side_effect`로, `when().thenReturn()`에 대응
