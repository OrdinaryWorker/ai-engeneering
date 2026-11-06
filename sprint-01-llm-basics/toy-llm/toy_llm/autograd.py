"""Автоматическое дифференцирование — нужно только для ОБУЧЕНИЯ.

Чтобы подобрать веса (урок 01, раздел 2), надо знать, в какую сторону сдвинуть каждое
число, чтобы ошибка уменьшилась. Это направление называется градиентом.

Value — число, которое запоминает, из каких чисел и какой операцией оно получено.
Из таких чисел строится граф вычислений: от весов до итоговой ошибки. Метод backward
проходит этот граф от ошибки назад и для каждого веса считает градиент.

Это то, что делает PyTorch, только по одному числу за раз и в тысячи раз медленнее.
При ИНФЕРЕНСЕ градиенты не нужны: движок просто читает поле .data.
"""
import math


class Value:
    """Число с историей происхождения: значение, градиент и ссылки на «родителей».

    data         — само значение;
    grad         — градиент: насколько изменится итоговая ошибка, если чуть увеличить data;
    _parents     — числа, из которых получено это число;
    _local_grads — производная этого числа по каждому из родителей (правило цепочки).
    """

    __slots__ = ("data", "grad", "_parents", "_local_grads")

    def __init__(self, data, parents=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self._parents = parents
        self._local_grads = local_grads

    def __add__(self, other):
        """a + b. Производная по каждому слагаемому равна 1."""
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        """a × b. Производная по a равна b, по b равна a. Умножение на обычное число тоже можно."""
        if isinstance(other, Value):
            return Value(self.data * other.data, (self, other), (other.data, self.data))
        return Value(self.data * other, (self,), (other,))

    def __pow__(self, k):
        """a в степени k (k — обычное число). Производная: k · a^(k−1)."""
        return Value(self.data**k, (self,), (k * self.data ** (k - 1),))

    def exp(self):
        """e в степени a. Производная равна самому результату."""
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        """Натуральный логарифм. Производная: 1 / a."""
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        """ReLU: max(0, a) — нелинейность в полносвязной сети. Производная 1 при a > 0, иначе 0."""
        return Value(max(0.0, self.data), (self,), (float(self.data > 0),))

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        """−a."""
        return self * -1.0

    def __sub__(self, other):
        """a − b."""
        return self + (-other)

    def __truediv__(self, other):
        """a / b, через умножение на b в степени −1."""
        return self * other**-1

    def backward(self):
        """Считает градиенты всех чисел, от которых зависит это число (обычно это ошибка).

        1. Обходим граф и выстраиваем числа в порядке «сначала родители, потом дети».
        2. Градиент самой ошибки по себе равен 1.
        3. Идём в обратном порядке и раздаём градиент родителям по правилу цепочки:
           градиент родителя += локальная производная × градиент ребёнка.
        """
        order, seen = [], set()

        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    visit(p)
                order.append(v)

        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            for parent, local in zip(v._parents, v._local_grads):
                parent.grad += local * v.grad


def vsum(xs):
    """Сумма списка Value одним узлом графа.

    Результат тот же, что у цепочки a + b + c + …, но узел один, а не тысяча.
    Это сильно ускоряет обучение на чистом Python.
    """
    return Value(sum(x.data for x in xs), tuple(xs), (1.0,) * len(xs))


def dot(xs, ws):
    """Скалярное произведение двух векторов одним узлом графа: Σ xᵢ · wᵢ.

    Это главная операция всей модели: умножение вектора на матрицу — это
    скалярные произведения вектора со столбцами, сравнение запроса с ключом — тоже.
    Производная по xᵢ равна wᵢ, по wᵢ равна xᵢ.
    """
    data = sum(x.data * w.data for x, w in zip(xs, ws))
    return Value(data, tuple(xs) + tuple(ws), tuple(w.data for w in ws) + tuple(x.data for x in xs))
