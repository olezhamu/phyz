import numpy as np
import matplotlib.pyplot as plt

# Физические параметры
g = 9.81
h0 = 2.00
xr = 4.60
yr = 3.05
Ls = 4.975
Hs = 3.42
hs = 1.05
k = 0.82
eps_x = 0.10
eps_y = 0.03
dt = 0.01
max_time = 5.0

# Дополнительный параметр для низких траекторий (расстояние от левого края допуска)
MIN_DIST_FROM_LEFT_EDGE = 0.12   # 12 см
LOW_ANGLE_THRESHOLD = 35.0       # градусы, ниже которого траектория считается "низкой"

# Реальный радиус кольца
R_RING_REAL = 0.225  # диаметр 0.45 м

# Функция интерполяции для нахождения x при заданном y

def find_x_at_y(x_arr, y_arr, y_target):
    """Возвращает x, соответствующий первому пересечению y_target снизу вверх.
       Если пересечения нет, возвращает None."""
    for i in range(len(y_arr) - 1):
        y1, y2 = y_arr[i], y_arr[i+1]
        if y1 <= y_target <= y2 or y2 <= y_target <= y1:
            if y2 != y1:
                frac = (y_target - y1) / (y2 - y1)
                x_at = x_arr[i] + frac * (x_arr[i+1] - x_arr[i])
                return x_at
    return None


# Моделирование одной траектории

def simulate_trajectory(v0, alpha_deg, store_traj=True, check_low_trajectory=False):
    """
    Возвращает:
        outcome : str   - "Прямое попадание", "Попадание после отскока" или "Промах"
        traj     : dict - {'x': list, 'y': list} если store_traj=True, иначе None
    """
    alpha = np.radians(alpha_deg)
    vx = v0 * np.cos(alpha)
    vy = v0 * np.sin(alpha)
    x = 0.0
    y = h0
    t = 0.0

    if store_traj:
        traj = {'x': [x], 'y': [y]}
    else:
        traj = None

    bounced = False
    hit = False
    outcome = "Промах"

    while t <= max_time and not hit:
        # --- Проверка попадания в кольцо ---
        if abs(x - xr) <= eps_x and abs(y - yr) <= eps_y:
            hit = True
            outcome = "Попадание после отскока" if bounced else "Прямое попадание"
            break

        # --- Шаг Эйлера ---
        x_next = x + vx * dt
        y_next = y + vy * dt

        # --- Обработка столкновения со щитом (только один раз) ---
        if not bounced and x < Ls and x_next >= Ls and vx > 0:
            t_hit = (Ls - x) / vx
            y_hit = y + vy * t_hit

            if (Hs - hs/2) <= y_hit <= (Hs + hs/2):
                bounced = True
                vx = -k * vx
                x = Ls
                y = y_hit
                t = t + t_hit
                if store_traj:
                    traj['x'].append(x)
                    traj['y'].append(y)
                continue

        # --- Обычное обновление ---
        x = x_next
        y = y_next
        vy = vy - g * dt
        t = t + dt

        if y <= 0 or x < -2 or x > Ls + 3:
            break

        if store_traj:
            traj['x'].append(x)
            traj['y'].append(y)

    # --- Дополнительная проверка для низких прямых траекторий ---
    if (hit and outcome == "Прямое попадание" and check_low_trajectory
            and alpha_deg < LOW_ANGLE_THRESHOLD and traj is not None):
        x_at_yr = find_x_at_y(traj['x'], traj['y'], yr)
        if x_at_yr is not None:
            left_edge = xr - eps_x
            if x_at_yr - left_edge <= MIN_DIST_FROM_LEFT_EDGE:
                outcome = "Промах (низкая траектория: мяч задел бы переднюю дужку)"
                hit = False

    return outcome, traj


# Поиск интервалов углов

def get_continuous_intervals(angle_list, gap=0.3):
    if not angle_list:
        return []
    intervals = []
    start = angle_list[0]
    for i in range(1, len(angle_list)):
        if angle_list[i] - angle_list[i-1] > gap:
            intervals.append((start, angle_list[i-1]))
            start = angle_list[i]
    intervals.append((start, angle_list[-1]))
    return intervals

def scan_angles(v0, angle_step=0.2, min_angle=5.0, max_angle=85.0):
    angles = np.arange(min_angle, max_angle + angle_step/2, angle_step)
    direct = []
    bank = []
    print(f"Сканирование углов для v0 = {v0:.2f} м/с (с проверкой низких траекторий)...")
    for a in angles:
        outcome, _ = simulate_trajectory(v0, a, store_traj=False, check_low_trajectory=True)
        if outcome == "Прямое попадание":
            direct.append(a)
        elif outcome == "Попадание после отскока":
            bank.append(a)
    return get_continuous_intervals(direct), get_continuous_intervals(bank)


# Аналитическая нижняя граница (левый край кольца)

def alpha_min_direct(v0, g=9.81, h0=2.0, xr=4.60, yr=3.05, eps_x=0.10):
    x0 = xr - eps_x
    y0 = yr
    A = g * x0**2 / (2 * v0**2)
    D = x0**2 - 4 * A * (y0 - h0 + A)
    if D < 0:
        return None
    u = (x0 - np.sqrt(D)) / (2 * A)
    alpha = np.degrees(np.arctan(u))
    return max(0.0, min(alpha, 90.0))


# Построение графика (с отдельными линиями для кольца и зоны допуска)

def plot_trajectory(traj, v0, alpha_deg, outcome):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(traj['x'], traj['y'], 'b-', lw=2, label='Траектория мяча')

    # Реальное кольцо (красная линия, длина 0.45 м)
    ring_left = xr - R_RING_REAL
    ring_right = xr + R_RING_REAL
    ax.plot([ring_left, ring_right], [yr, yr], 'r-', lw=3, label='Реальное кольцо (0.45 м)')

    # Зона допуска попадания (зелёная линия, длина 2*eps_x)
    tolerance_left = xr - eps_x
    tolerance_right = xr + eps_x
    ax.plot([tolerance_left, tolerance_right], [yr, yr], 'g-', lw=2, label=f'Зона попадания (±{eps_x:.2f} м)')
    # Вертикальные границы зоны допуска (зелёные пунктиры)
    ax.plot([tolerance_left, tolerance_left], [yr - eps_y, yr + eps_y], 'g--', lw=1)
    ax.plot([tolerance_right, tolerance_right], [yr - eps_y, yr + eps_y], 'g--', lw=1)

    # Щит
    y_board_bottom = Hs - hs/2
    y_board_top = Hs + hs/2
    ax.fill_betweenx([y_board_bottom, y_board_top], Ls, Ls + 0.05, color='gray', alpha=0.5, label='Щит')
    ax.axvline(Ls, color='k', linestyle='--', lw=1)
    ax.plot(0, h0, 'ko', markersize=8, label='Бросок')

    ax.set_xlabel('x, м')
    ax.set_ylabel('y, м')
    ax.set_title(f'v0 = {v0:.2f} м/с, α = {alpha_deg:.1f}° — {outcome}')
    ax.legend()
    ax.grid(True, linestyle=':')
    ax.axis('equal')
    ax.set_xlim(-0.5, Ls + 1)
    ax.set_ylim(0, max(max(traj['y']), yr + 0.5))
    plt.show()


# Основное меню

def main():
    print("Моделирование движения баскетбольного мяча (материальная точка)")
    print("1 — вычислить траекторию по заданной скорости и углу")
    print("2 — найти диапазоны углов попадания для заданной скорости")
    choice = input("Ваш выбор (1 или 2): ")

    if choice == '1':
        v0 = float(input("Начальная скорость v0 (м/с): "))
        alpha = float(input("Угол броска (градусы): "))
        outcome, traj = simulate_trajectory(v0, alpha, store_traj=True, check_low_trajectory=True)
        print(f"Результат: {outcome}")
        if traj and len(traj['x']) > 1:
            plot_trajectory(traj, v0, alpha, outcome)
        else:
            print("Не удалось построить траекторию.")

    elif choice == '2':
        v0 = float(input("Начальная скорость v0 (м/с): "))
        direct_intervals, bank_intervals = scan_angles(v0, angle_step=0.2)

        print("\n=== Результаты сканирования ===")
        print(f"Скорость: {v0:.2f} м/с")
        alpha_min_theor = alpha_min_direct(v0)
        if alpha_min_theor is not None:
            print(f"\nАналитическая нижняя граница (левый край допуска): {alpha_min_theor:.2f}°")
        print("\nПрямые попадания (без щита):")
        if direct_intervals:
            for a, b in direct_intervals:
                if abs(a - b) < 0.3:
                    print(f"  {a:.1f}°")
                else:
                    print(f"  от {a:.1f}° до {b:.1f}°")
        else:
            print("  не найдены")
        print("\nПопадания после отскока от щита:")
        if bank_intervals:
            for a, b in bank_intervals:
                if abs(a - b) < 0.3:
                    print(f"  {a:.1f}°")
                else:
                    print(f"  от {a:.1f}° до {b:.1f}°")
        else:
            print("  не найдены")

    else:
        print("Неверный выбор.")

if __name__ == "__main__":
    main()