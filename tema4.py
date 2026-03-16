import math

max_counts = []
d0 = []
b = []
d1 = []
d2 = []

p = []
q = []

eps = 0

def load_vector(file_obj):
    return [float(line) for line in file_obj if line.strip()]

def pct1():
    for i in range(1, 6):
        with (
            open(f"tema4/d0_{i}.txt", "r", encoding="utf-8", errors="replace") as fd0,
            open(f"tema4/b_{i}.txt", "r", encoding="utf-8", errors="replace") as fb,
            open(f"tema4/d1_{i}.txt", "r", encoding="utf-8", errors="replace") as fd1,
            open(f"tema4/d2_{i}.txt", "r", encoding="utf-8", errors="replace") as fd2,
            ):
            d0.append(load_vector(fd0))
            b.append(load_vector(fb))
            d1.append(load_vector(fd1))
            d2.append(load_vector(fd2))

            if(len(d0[i-1]) == len(b[i-1])):
                print(f"Vectorul d0_{i} are aceeasi lungime ca vectorul b_{i}: {len(d0[i-1])} == {len(b[i-1])}")
            else:
                print(f"Vectorul d0_{i} NU are aceeasi lungime ca vectorul b_{i}: {len(d0[i-1])} != {len(b[i-1])}")


def pct2():
    for i in range(1, 6):
        print(f"Pentru setul {i}:")
        print(f"Len D1_{i} = {len(d1[i-1])} cu p = {len(d0[i-1]) - len(d1[i-1])}")
        print(f"Len D2_{i} = {len(d2[i-1])} cu q = {len(d0[i-1]) - len(d2[i-1])}")

        p.append(len(d0[i-1]) - len(d1[i-1]))
        q.append(len(d0[i-1]) - len(d2[i-1]))


def gauss_seidel_sparse(d0, d1, d2, b, eps, kmax=10000):
    n = len(d0)

    if len(b) != n:
        raise ValueError("d0 si b trebuie sa aiba aceeasi dimensiune.")

    if len(d1) >= n or len(d2) >= n:
        raise ValueError("Dimensiunile lui d1 si d2 sunt invalide.")

    p = n - len(d1)
    q = n - len(d2)

    if p <= 0 or q <= 0:
        raise ValueError("Valorile p si q trebuie sa fie strict pozitive.")

    for i in range(n):
        if abs(d0[i]) <= eps:
            raise ValueError(
                "Exista un element nul pe diagonala principala. "
                "Metoda Gauss-Seidel nu poate fi aplicata."
            )

    x = [0.0] * n

    for _ in range(kmax):
        delta = 0.0

        for i in range(n):
            old_xi = x[i]
            s = b[i]

            # A[i][i-p] = d1[i-p]
            if i - p >= 0:
                s -= d1[i - p] * x[i - p]

            # A[i][i-q] = d2[i-q]
            if i - q >= 0:
                s -= d2[i - q] * x[i - q]

            # A[i][i+p] = d1[i]
            if i + p < n:
                s -= d1[i] * x[i + p]

            # A[i][i+q] = d2[i]
            if i + q < n:
                s -= d2[i] * x[i + q]

            x[i] = s / d0[i]
            delta = max(delta, abs(x[i] - old_xi))

        if delta < eps:
            return x

        if delta > 1e10 or not math.isfinite(delta):
            raise RuntimeError("Metoda diverge.")

    raise RuntimeError("Metoda nu a convergat in numarul maxim de iteratii.")
        
def compute_y_sparse(d0, d1, d2, x):
    n = len(d0)

    if len(x) != n:
        raise ValueError("x si d0 trebuie sa aiba aceeasi dimensiune.")

    p = n - len(d1)
    q = n - len(d2)

    y = [0.0] * n

    for i in range(n):
        s = d0[i] * x[i]

        # A[i][i-p] = d1[i-p]
        if i - p >= 0:
            s += d1[i - p] * x[i - p]

        # A[i][i+p] = d1[i]
        if i + p < n:
            s += d1[i] * x[i + p]

        # A[i][i-q] = d2[i-q]
        if i - q >= 0:
            s += d2[i - q] * x[i - q]

        # A[i][i+q] = d2[i]
        if i + q < n:
            s += d2[i] * x[i + q]

        y[i] = s

    return y

def main():
    try:
        eps = float(input("eps (e.g. 1e-9) = ").strip())
    except Exception:
        print("Invalid input. Example: n=200, eps=1e-9")
        return
    
    if eps <= 0:
        print("eps must be positive.")
        return
    
    pct1()
    pct2()

    try:
        for i in range(5):
            print(f"Rezolvarea pentru setul {i+1}...")
            x = gauss_seidel_sparse(d0[i], d1[i], d2[i], b[i], eps)
            print(f"Solutia pentru setul {i+1}: {x[:10]}... (primele 10 elemente)")

            y = compute_y_sparse(d0[i], d1[i], d2[i], x)
            norma = max(abs(y[j] - b[i][j]) for j in range(len(y)))
            print(f"Norma ||y - b|| pentru setul {i+1}: {norma}\n")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()