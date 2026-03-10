import numpy as np


def sign_pdf(x: float) -> float:
    """semn(x) din pdf: +1 daca x >= 0, -1 daca x < 0"""
    return 1.0 if x >= 0 else -1.0


def generate_random_invertible_matrix(n: int, low: float = -10.0, high: float = 10.0,
                                      eps_det: float = 1e-10, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)

    while True:
        A = rng.uniform(low, high, size=(n, n))
        if abs(np.linalg.det(A)) > eps_det:
            return A


def generate_random_vector(n: int, low: float = -10.0, high: float = 10.0,
                           seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=n)


def compute_b(A: np.ndarray, s: np.ndarray) -> np.ndarray:
    """b = A * s"""
    return A @ s


def solve_upper_triangular(R: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Rezolva Rx = y prin substitutie inversa."""
    n = R.shape[0]
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) <= eps:
            raise ValueError(f"Matrice singulara sau aproape singulara: R[{i},{i}] ~ 0")
        s = np.dot(R[i, i + 1:], x[i + 1:])
        x[i] = (y[i] - s) / R[i, i]

    return x


def householder_qr(A: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Factorizare Householder conform ideii din pdf.
    Returneaza:
        Qt = Q^T
        R
        singular (True/False)
    """
    A_work = A.astype(float).copy()
    n = A_work.shape[0]
    Qt = np.eye(n, dtype=float)

    singular = False

    for r in range(n - 1):
        # sigma = sum_{i=r}^{n-1} a_{i,r}^2
        sigma = np.dot(A_work[r:, r], A_work[r:, r])

        if sigma <= eps:
            singular = True
            break

        k = -sign_pdf(A_work[r, r]) * np.sqrt(sigma)
        beta = sigma - k * A_work[r, r]

        if abs(beta) <= eps:
            singular = True
            break

        u = np.zeros(n, dtype=float)
        u[r] = A_work[r, r] - k
        u[r + 1:] = A_work[r + 1:, r]

        # transformarea coloanelor j = r+1,...,n-1
        for j in range(r + 1, n):
            gamma = np.dot(A_work[r:, j], u[r:]) / beta
            A_work[r:, j] -= gamma * u[r:]

        # coloana r devine triunghiulara superior
        A_work[r, r] = k
        A_work[r + 1:, r] = 0.0

        # actualizare Q^T
        for j in range(n):
            gamma = np.dot(Qt[r:, j], u[r:]) / beta
            Qt[r:, j] -= gamma * u[r:]

    R = A_work

    # verificare singularitate pe diagonala lui R
    for i in range(n):
        if abs(R[i, i]) <= eps:
            singular = True
            break

    return Qt, R, singular


def solve_with_householder(A: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rezolva Ax=b folosind QR Householder implementat manual.
    Returneaza:
        x, Qt, R
    """
    Qt, R, singular = householder_qr(A, eps)
    if singular:
        raise ValueError("Matricea este singulara sau aproape singulara (Householder).")

    y = Qt @ b
    x = solve_upper_triangular(R, y, eps)
    return x, Qt, R


def solve_with_library_qr(A: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rezolva Ax=b folosind QR din biblioteca numpy.
    Returneaza:
        x, Q, R
    """
    Q, R = np.linalg.qr(A)

    # verificare singularitate
    for i in range(R.shape[0]):
        if abs(R[i, i]) <= eps:
            raise ValueError("Matricea este singulara sau aproape singulara (QR biblioteca).")

    y = Q.T @ b
    x = solve_upper_triangular(R, y, eps)
    return x, Q, R


def inverse_from_householder(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Calculeaza inversa matricei A folosind QR Householder:
        A = Q R
        A^{-1} = [x_1 x_2 ... x_n], unde R x_j = Q^T e_j
    """
    n = A.shape[0]
    Qt, R, singular = householder_qr(A, eps)

    if singular:
        raise ValueError("Matricea este singulara; inversa nu poate fi calculata.")

    A_inv = np.zeros((n, n), dtype=float)

    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0

        y = Qt @ e_j
        x_j = solve_upper_triangular(R, y, eps)
        A_inv[:, j] = x_j

    return A_inv


def vector_norm_2(x: np.ndarray) -> float:
    return np.linalg.norm(x, ord=2)


def matrix_norm(A: np.ndarray) -> float:
    """
    Pentru diferenta dintre inverse folosim norma Frobenius.
    Daca profesorul cere alta norma, aici se poate schimba usor.
    """
    return np.linalg.norm(A)


def print_matrix(name: str, M: np.ndarray) -> None:
    print(f"{name} =")
    print(M)
    print()


def main():
    # -------------------------
    # Parametri de intrare
    # -------------------------
    n = int(input("Introduceti n = "))
    eps = float(input("Introduceti eps = "))
    seed = 42

    # -------------------------
    # Initializare random
    # -------------------------
    A_init = generate_random_invertible_matrix(n, seed=seed)
    s = generate_random_vector(n, seed=seed + 1)

    b_init = compute_b(A_init, s)

    # -------------------------
    # Rezolvare cu Householder
    # -------------------------
    x_householder, Qt_h, R_h = solve_with_householder(A_init, b_init, eps)

    # -------------------------
    # Rezolvare cu QR biblioteca
    # -------------------------
    x_qr, Q_lib, R_lib = solve_with_library_qr(A_init, b_init, eps)

    # -------------------------
    # Diferenta dintre solutii
    # -------------------------
    diff_x = vector_norm_2(x_qr - x_householder)

    # -------------------------
    # Erori cerute
    # -------------------------
    err1 = vector_norm_2(A_init @ x_householder - b_init)
    err2 = vector_norm_2(A_init @ x_qr - b_init)

    norm_s = vector_norm_2(s)
    if norm_s <= eps:
        raise ValueError("Vectorul s are norma prea mica; nu se poate calcula eroarea relativa.")

    err3 = vector_norm_2(x_householder - s) / norm_s
    err4 = vector_norm_2(x_qr - s) / norm_s

    # -------------------------
    # Inversa cu Householder
    # -------------------------
    A_inv_householder = inverse_from_householder(A_init, eps)

    # Inversa din biblioteca
    A_inv_lib = np.linalg.inv(A_init)

    # Comparatie inverse
    inv_diff_norm = matrix_norm(A_inv_householder - A_inv_lib)

    # -------------------------
    # Afisare rezultate
    # -------------------------
    np.set_printoptions(precision=6, suppress=True)

    print("=== DATE DE INTRARE ===")
    print(f"n = {n}")
    print(f"eps = {eps}\n")

    print_matrix("A_init", A_init)
    print_matrix("s", s)
    print_matrix("b_init = A_init @ s", b_init)

    print("=== FACTORIZARE HOUSEHOLDER ===")
    print_matrix("Q^T (Householder)", Qt_h)
    print_matrix("R (Householder)", R_h)

    print("=== SOLUTII ===")
    print_matrix("x_householder", x_householder)
    print_matrix("x_qr_biblioteca", x_qr)

    print("||x_QR - x_Householder||_2 =", diff_x)
    print()

    print("=== ERORI CERUTE ===")
    print("1) ||A_init * x_householder - b_init||_2 =", err1)
    print("2) ||A_init * x_qr - b_init||_2 =", err2)
    print("3) ||x_householder - s||_2 / ||s||_2 =", err3)
    print("4) ||x_qr - s||_2 / ||s||_2 =", err4)
    print()

    print("=== INVERSE ===")
    print_matrix("A_inv_householder", A_inv_householder)
    print_matrix("A_inv_biblioteca", A_inv_lib)
    print("||A_inv_householder - A_inv_biblioteca|| =", inv_diff_norm)
    print()

    print("=== VERIFICARE PRAG 1e-6 ===")
    print("err1 < 1e-6 :", err1 < 1e-6)
    print("err2 < 1e-6 :", err2 < 1e-6)
    print("err3 < 1e-6 :", err3 < 1e-6)
    print("err4 < 1e-6 :", err4 < 1e-6)


if __name__ == "__main__":
    main()