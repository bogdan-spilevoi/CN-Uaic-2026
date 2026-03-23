import numpy as np


def read_matrix_from_input() -> np.ndarray:
    """
    Citește o matrice din input:
    prima linie: p n
    următoarele p linii: câte n numere reale
    """
    p, n = map(int, input("Dimensiuni p n = ").split())
    data = []
    print(f"Introdu {p} linii a câte {n} valori:")
    for _ in range(p):
        row = list(map(float, input().split()))
        if len(row) != n:
            raise ValueError("Număr greșit de coloane pe linie.")
        data.append(row)
    return np.array(data, dtype=float)


# ============================================================
# 1) METODA JACOBI PENTRU MATRICE SIMETRICE (p = n)
# ============================================================

def is_symmetric(A: np.ndarray, tol: float = 1e-12) -> bool:
    return np.allclose(A, A.T, atol=tol)


def offdiag_max_index(A: np.ndarray):
    """
    Alege (p, q) ca fiind poziția elementului nediagonal cu modul maxim,
    căutat doar în partea strict inferior triunghiulară.
    """
    n = A.shape[0]
    p, q = 1, 0
    max_val = abs(A[p, q]) if n > 1 else 0.0

    for i in range(1, n):
        for j in range(i):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                p, q = i, j

    return p, q, max_val


def jacobi_eigen(A: np.ndarray, eps: float = 1e-10, kmax: int = 10_000):
    """
    Aproximare valori/vectori proprii pentru matrice simetrică folosind
    metoda Jacobi, ghidată de pseudocodul din cerință.

    Returnează:
      eigenvalues  - valorile proprii aproximative
      U            - vectorii proprii pe coloane
      Afinal       - matricea finală aproape diagonală
      iterations   - numărul de iterații
      residual     - ||A_init * U - U * Lambda||_F
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Jacobi cere matrice pătratică.")
    if not is_symmetric(A):
        raise ValueError("Jacobi din cerință cere matrice simetrică (A = A^T).")

    A = A.astype(float).copy()
    A_init = A.copy()
    n = A.shape[0]
    U = np.eye(n)

    p, q, max_off = offdiag_max_index(A)
    k = 0

    while max_off > eps and k <= kmax:
        apq = A[p, q]
        app = A[p, p]
        aqq = A[q, q]

        # Cazul apq = 0 => matrice diagonală
        if abs(apq) <= eps:
            break

        # Formulele (3) și (4) din cerință
        alpha = (app - aqq) / (2.0 * apq)
        sign_alpha = 1.0 if alpha >= 0 else -1.0
        t = -alpha + sign_alpha * np.sqrt(alpha * alpha + 1.0)
        c = 1.0 / np.sqrt(1.0 + t * t)
        s = t / np.sqrt(1.0 + t * t)

        # Salvăm coloanele/elementele vechi necesare pentru actualizare
        old_col_p = A[:, p].copy()
        old_col_q = A[:, q].copy()
        old_app = app
        old_aqq = aqq
        old_apq = apq

        # Formula (5): actualizare directă a lui A
        for j in range(n):
            if j != p and j != q:
                new_apj = c * old_col_p[j] + s * old_col_q[j]
                new_aqj = -s * old_col_p[j] + c * old_col_q[j]
                A[p, j] = A[j, p] = new_apj
                A[q, j] = A[j, q] = new_aqj

        A[p, p] = old_app + t * old_apq
        A[q, q] = old_aqq - t * old_apq
        A[p, q] = A[q, p] = 0.0

        # Formula (7): actualizare directă a lui U
        old_u_p = U[:, p].copy()
        old_u_q = U[:, q].copy()
        U[:, p] = c * old_u_p + s * old_u_q
        U[:, q] = -s * old_u_p + c * old_u_q

        p, q, max_off = offdiag_max_index(A)
        k += 1

    eigenvalues = np.diag(A).copy()
    Lambda = np.diag(eigenvalues)
    residual = np.linalg.norm(A_init @ U - U @ Lambda, ord='fro')

    return eigenvalues, U, A, k, residual


# ============================================================
# 2) ȘIRUL DE MATRICE CU CHOLESKY
# ============================================================

def cholesky_iteration(A: np.ndarray, eps: float = 1e-10, kmax: int = 1000):
    """
    Construiește șirul:
      A^(0) = A
      A^(k) = L_k L_k^T
      A^(k+1) = L_k^T L_k
    până când ||A^(k) - A^(k-1)|| < eps sau k > kmax.

    Observație: Cholesky cere matrice simetrică pozitiv definită.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Iterația Cholesky cere matrice pătratică.")
    if not is_symmetric(A):
        raise ValueError("Iterația Cholesky cere matrice simetrică.")

    A_prev = A.astype(float).copy()
    k = 0

    while k < kmax:
        try:
            L = np.linalg.cholesky(A_prev)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Cholesky a eșuat. Matricea trebuie să fie simetrică pozitiv definită."
            ) from exc

        A_next = L.T @ L
        diff = np.linalg.norm(A_next - A_prev, ord='fro')

        if diff < eps:
            return A_next, k + 1, diff

        A_prev = A_next
        k += 1

    final_diff = np.linalg.norm(A_prev - (np.linalg.cholesky(A_prev).T @ np.linalg.cholesky(A_prev)), ord='fro')
    return A_prev, k, final_diff


# ============================================================
# 3) SVD PENTRU p > n
# ============================================================

def numerical_rank_from_svd(singular_values: np.ndarray, eps: float = 1e-10) -> int:
    return int(np.sum(singular_values > eps))


def condition_number_from_svd(singular_values: np.ndarray, eps: float = 1e-10) -> float:
    positive = singular_values[singular_values > eps]
    if positive.size == 0:
        return np.inf
    return float(np.max(positive) / np.min(positive))


def moore_penrose_pinv_from_svd(A: np.ndarray, eps: float = 1e-10):
    """
    A = U S V^T, cu full_matrices=True pentru a respecta forma teoretică.
    A^I = V S^I U^T
    """
    p, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    V = Vt.T

    S_inv = np.zeros((n, p), dtype=float)
    for i, sigma in enumerate(s):
        if sigma > eps:
            S_inv[i, i] = 1.0 / sigma

    A_I = V @ S_inv @ U.T
    return s, U, V, A_I


def least_squares_pinv(A: np.ndarray):
    """
    A^J = (A^T A)^(-1) A^T
    Formula cere ca A^T A să fie inversabilă.
    """
    AtA = A.T @ A
    return np.linalg.inv(AtA) @ A.T


def svd_requirements(A: np.ndarray, eps: float = 1e-10):
    """
    Rezolvă cerințele pentru cazul p > n.
    """
    p, n = A.shape
    if not (p > n):
        raise ValueError("Această secțiune se aplică doar pentru p > n.")

    singular_values, U, V, A_I = moore_penrose_pinv_from_svd(A, eps=eps)
    rank_num = numerical_rank_from_svd(singular_values, eps=eps)
    cond_num = condition_number_from_svd(singular_values, eps=eps)

    A_J = None
    diff_norm_1 = None
    try:
        A_J = least_squares_pinv(A)
        diff_norm_1 = np.linalg.norm(A_I - A_J, ord=1)
    except np.linalg.LinAlgError:
        # Dacă A^T A nu e inversabilă, formula directă nu se poate aplica.
        pass

    return {
        "singular_values": singular_values,
        "rank": rank_num,
        "condition_number": cond_num,
        "A_I": A_I,
        "A_J": A_J,
        "norm1_A_I_minus_A_J": diff_norm_1,
    }


# ============================================================
# AFIȘARE ȘI DEMO
# ============================================================

def print_matrix(name: str, M: np.ndarray):
    print(f"\n{name} =")
    print(np.array2string(M, precision=10, suppress_small=True))


def solve_theme(A: np.ndarray, eps: float = 1e-10, kmax: int = 1000):
    p, n = A.shape
    print_matrix("A", A)
    print(f"\nDimensiune: p={p}, n={n}, eps={eps}, kmax={kmax}")

    if p == n:
        print("\n====================")
        print("CAZUL p = n")
        print("====================")

        if is_symmetric(A):
            print("\n1) Metoda Jacobi pentru valori/vectori proprii")
            eigvals, U, Afinal, steps, residual = jacobi_eigen(A, eps=eps, kmax=kmax)
            Lambda = np.diag(eigvals)

            print_matrix("A_final (aprox. diagonală)", Afinal)
            print_matrix("U (vectorii proprii pe coloane)", U)
            print_matrix("Lambda", Lambda)
            print(f"\nNumăr iterații Jacobi: {steps}")
            print(f"Norma ||A_init * U - U * Lambda||_F = {residual:.12e}")
            print("Valorile proprii aproximative sunt elementele de pe diagonala lui A_final.")

            print("\n2) Șirul de matrice folosind Cholesky")
            try:
                A_last, k_used, last_diff = cholesky_iteration(A, eps=eps, kmax=kmax)
                print_matrix("Ultima matrice din șir", A_last)
                print(f"Număr iterații Cholesky: {k_used}")
                print(f"Ultima diferență ||A^(k) - A^(k-1)||_F ≈ {last_diff:.12e}")
                print("Observație: matricea tinde spre o formă aproape diagonală, iar pe diagonală apar informații despre valorile proprii.")
            except ValueError as e:
                print(f"Cholesky nu poate fi aplicat: {e}")
        else:
            print("Pentru p = n, partea Jacobi cere matrice simetrică. Matricea introdusă nu este simetrică.")

    elif p > n:
        print("\n====================")
        print("CAZUL p > n")
        print("====================")
        res = svd_requirements(A, eps=eps)

        print("\nValorile singulare:")
        print(np.array2string(res["singular_values"], precision=10, suppress_small=True))
        print(f"\nRang(A) = {res['rank']}")
        print(f"Numărul de condiționare = {res['condition_number']:.12e}")
        print_matrix("Pseudoinversa Moore-Penrose A_I", res["A_I"])

        if res["A_J"] is not None:
            print_matrix("Pseudo-inversa least squares A_J", res["A_J"])
            print(f"Norma ||A_I - A_J||_1 = {res['norm1_A_I_minus_A_J']:.12e}")
        else:
            print("A_J = (A^T A)^(-1) A^T nu poate fi calculată deoarece A^T A nu este inversabilă.")
    else:
        print("Cazul p < n nu este cerut în această temă.")


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    print("Alege modul de lucru:")
    print("1 - introducere matrice de la tastatură")
    print("2 - exemple predefinite")
    choice = input("Opțiune = ").strip()

    eps = float(input("eps = ").strip())
    kmax = int(input("kmax = ").strip())

    if choice == "1":
        A = read_matrix_from_input()
        solve_theme(A, eps=eps, kmax=kmax)
    else:
        print("\nExemplul 1: matrice simetrică 3x3")
        A1 = np.array([
            [1, 1, 2],
            [1, 1, 2],
            [2, 2, 2]
        ], dtype=float)
        solve_theme(A1, eps=eps, kmax=kmax)

        print("\n\nExemplul 2: matrice p > n")
        A2 = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ], dtype=float)
        solve_theme(A2, eps=eps, kmax=kmax)
