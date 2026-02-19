import math

#1
def machine_precision_base10():
    m = 0
    prev_u = None
    while True:
        m += 1
        u = 10.0 ** (-m)
        if 1.0 + u == 1.0:
            return m - 1, prev_u
        prev_u = u

m, u = machine_precision_base10()
print(f"[1] m = {m}, u = {u:.20g}")
print(f"    1.0 + u     = {(1.0 + u):.20g}")
print(f"    1.0 + u/10  = {(1.0 + u/10):.20g}")

#2.1
x = 1.0
y = u / 10.0
z = u / 10.0

left_add  = (x + y) + z
right_add = x + (y + z)

print("\n[2.1] Neasociativitate +c")
print(f"x = {x:.20g}")
print(f"y = {y:.20g}")
print(f"z = {z:.20g}")
print(f"(x +c y) +c z  = {left_add:.20g}")
print(f"x +c (y +c z)  = {right_add:.20g}")
print("Diferite?", left_add != right_add)

#2.2
def find_nonassoc_mul():
    y = 1.0

    while math.isfinite(y) and math.isfinite(y * y):
        y *= 2.0

    if not math.isfinite(y):
        raise RuntimeError("Nu am găsit un y finit care să facă overflow la y*y (neobișnuit).")

    z = 1.0 / y

    while z == 0.0:
        y /= 2.0
        z = 1.0 / y

    x = y
    return x, y, z

x, y, z = find_nonassoc_mul()

left  = (x * y) * z
right = x * (y * z)

print("\n[2.2] Neasociativitate ×c")
print("x =", x)
print("y =", y)
print("z =", z)

print("(x*y)*z  =", left)
print("x*(y*z)  =", right)
print("Diferite?", left != right)

print("\nDetalii:")
print("x*y este inf?", math.isinf(x * y))
print("y*z =", y * z)
print("x*(y*z) este finit?", math.isfinite(right))


#3
import math
import random
import time
from statistics import mean, median

PI = math.pi
HALF_PI = PI / 2.0
QUARTER_PI = PI / 4.0

def reduce_to_minus_halfpi_halfpi(x: float) -> float:
    return math.remainder(x, PI)

def is_near_halfpi(x: float, tol: float = 1e-15) -> bool:
    return abs(abs(x) - HALF_PI) <= tol

# -------------------------
# TAN prin fracții continue (Lentz modificat)
# -------------------------
def my_tan_cf(x: float, eps: float = 1e-12, mic: float = 1e-12, max_iter: int = 10000) -> float:
    xr = reduce_to_minus_halfpi_halfpi(x)

    if is_near_halfpi(xr):
        return math.copysign(math.inf, xr)

    sign = 1.0
    if xr < 0.0:
        sign = -1.0
        xr = -xr

    if xr == 0.0:
        return 0.0

    x2 = xr * xr

    b0 = 1.0
    f = b0
    if f == 0.0:
        f = mic

    C = f
    D = 0.0

    for j in range(1, max_iter + 1):
        a_j = -x2
        b_j = 2.0 * j + 1.0

        D = b_j + a_j * D
        if D == 0.0:
            D = mic
        D = 1.0 / D

        C = b_j + a_j / C
        if C == 0.0:
            C = mic

        delta = C * D
        f *= delta

        if abs(delta - 1.0) < eps:
            break

    F = f
    return sign * (xr / F)

# -------------------------
# TAN prin polinom (Maclaurin până la x^9)
# -------------------------
def poly_tan_small(x: float) -> float:
    c1 = 0.33333333333333333      # 1/3
    c2 = 0.133333333333333333     # 2/15
    c3 = 0.053968253968254        # 17/315
    c4 = 0.0218694885361552       # 62/2835

    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    x7 = x5 * x2
    x9 = x7 * x2

    return x + c1 * x3 + c2 * x5 + c3 * x7 + c4 * x9

def my_tan_poly(x: float) -> float:
    xr = reduce_to_minus_halfpi_halfpi(x)

    if is_near_halfpi(xr):
        return math.copysign(math.inf, xr)

    if xr < 0.0:
        return -my_tan_poly(-xr)

    if xr <= QUARTER_PI:
        return poly_tan_small(xr)

    t = poly_tan_small(HALF_PI - xr)
    if t == 0.0:
        return math.inf
    return 1.0 / t


def percentile(values, p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    i = int(math.floor(k))
    j = int(math.ceil(k))
    if i == j:
        return s[i]
    return s[i] * (j - k) + s[j] * (k - i)

def compare_tans():
    print("\n[3] Comparare metode tan(x)")
    random.seed(42)

    n = 10_000

    margin = 1e-12
    lo = -HALF_PI + margin
    hi =  HALF_PI - margin

    xs = [random.uniform(lo, hi) for _ in range(n)]

    t0 = time.perf_counter()
    tan_true = [math.tan(x) for x in xs]
    t_true = time.perf_counter() - t0

    # CF
    eps = 1e-12
    t1 = time.perf_counter()
    tan_cf = [my_tan_cf(x, eps=eps) for x in xs]
    t_cf = time.perf_counter() - t1

    # Poly
    t2 = time.perf_counter()
    tan_poly = [my_tan_poly(x) for x in xs]
    t_poly = time.perf_counter() - t2

    # erori absolute
    err_cf = [abs(a - b) for a, b in zip(tan_true, tan_cf)]
    err_poly = [abs(a - b) for a, b in zip(tan_true, tan_poly)]

    def report(name, errs, t_method):
        print(f"\n--- {name} ---")
        print(f"Timp total (10.000 eval): {t_method:.6f} sec")
        print(f"Eroare medie:   {mean(errs):.6e}")
        print(f"Eroare mediană: {median(errs):.6e}")
        print(f"Eroare p95:     {percentile(errs, 95):.6e}")
        print(f"Eroare maximă:  {max(errs):.6e}")

        # top 5 cele mai mari erori
        top = sorted(range(len(errs)), key=lambda i: errs[i], reverse=True)[:5]
        print("Top 5 erori (x, tan(x), my_tan(x), abs_err):")
        for i in top:
            print(f"  x={xs[i]: .15f} | tan={tan_true[i]: .15e} | my={ (tan_cf[i] if name=='CF (Lentz)' else tan_poly[i]): .15e} | err={errs[i]:.6e}")

    print(f"Interval x: ({lo}, {hi})")
    print(f"Timp math.tan pentru 10.000: {t_true:.6f} sec (doar ca referință)")

    report("CF (Lentz)", err_cf, t_cf)
    report("Polinom (Maclaurin)", err_poly, t_poly)


compare_tans()
