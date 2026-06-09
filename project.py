import numpy as np
import matplotlib.pyplot as plt

g = 9.81
h0 = 2.0
x_ring = 4.6
y_ring = 3.05
R_ring = 0.225
R_ball = 0.12
H_board = 3.42
h_board = 1.04
Y_board_bottom = H_board - h_board/2
Y_board_top = H_board + h_board/2
X_board = x_ring + 0.375
X_rim_front = x_ring - R_ring
X_rim_back = x_ring + R_ring
Y_rim = y_ring
eps_x = 0.1
eps_y = 0.03
k = 0.82

def simulate_trajectory(v0, angle_deg, max_time=10.0):
    alpha = np.radians(angle_deg)
    dt = 0.001

    x = 0.0
    y = h0
    vx = v0 * np.cos(alpha)
    vy = v0 * np.sin(alpha)

    trajectory = {'x': [x], 'y': [y]}

    t = 0.0
    hit_board = False
    score_type = "Промах"
    scored = False

    while not scored and t < max_time:
        x_old, y_old = x, y

        vy -= g * dt
        x += vx * dt
        y += vy * dt
        t += dt

        yc_board = max(Y_board_bottom, min(y, Y_board_top))

        dx = x - X_board
        dy = y - yc_board
        dist_board = np.sqrt(dx ** 2 + dy ** 2)

        if dist_board <= R_ball:
            hit_board = True
            nx = dx / dist_board if dist_board > 0 else -1.0
            ny = dy / dist_board if dist_board > 0 else 0.0

            x = X_board + nx * R_ball * 1.001
            y = yc_board + ny * R_ball * 1.001

            v_dot_n = vx * nx + vy * ny
            if v_dot_n < 0:
                vx = vx - (1 + k) * v_dot_n * nx
                vy = vy - (1 + k) * v_dot_n * ny

        for rx, ry in [(X_rim_front, Y_rim), (X_rim_back, Y_rim)]:
            dx_r = x - rx
            dy_r = y - ry
            dist_rim = np.sqrt(dx_r ** 2 + dy_r ** 2)

            if dist_rim <= R_ball:
                nx = dx_r / dist_rim
                ny = dy_r / dist_rim

                x = rx + nx * R_ball * 1.001
                y = ry + ny * R_ball * 1.001

                v_dot_n = vx * nx + vy * ny
                if v_dot_n < 0:
                    vx = vx - (1 + k) * v_dot_n * nx
                    vy = vy - (1 + k) * v_dot_n * ny

        if y_old > Y_rim >= y:
            if abs(x - x_ring) <= eps_x and abs(y - y_ring) <= eps_y:
                if not scored:
                    scored = True
                    score_type = "От щита" if hit_board else "Чистое попадание"

        trajectory['x'].append(round(x, 3))
        trajectory['y'].append(round(y, 3))

        if y <= R_ball and vy < 0:
            break

    return score_type, trajectory


def get_ranges(angle_list):
    if not angle_list:
        return []

    ranges = []
    start = angle_list[0]

    for i in range(1, len(angle_list)):
        if angle_list[i] - angle_list[i - 1] > 0.15:
            ranges.append((round(start, 1), round(angle_list[i - 1], 1)))
            start = angle_list[i]

    ranges.append((round(start, 1), round(angle_list[-1], 1)))
    return ranges


def scan_angles(v0: float, angle_step: float = 0.1,
                min_angle: float = 10.0, max_angle: float = 85.0):
    if v0 < 7.0:
        print('Скорость слишком низкая')
        return None

    if v0 > 15.0:
        print('Скорость слишком высокая')
        return None

    angles = np.arange(min_angle, max_angle, angle_step)
    direct_angles = []
    bank_angles = []
    print('Сканирование углов, пожалуйста, подождите...')
    for ang in angles:
        result = simulate_trajectory(v0, ang)
        if result[0] == "Чистое попадание":
            direct_angles.append(ang)
        elif result[0] == "От щита":
            bank_angles.append(ang)

    direct_ranges = get_ranges(direct_angles)
    bank_ranges = get_ranges(bank_angles)

    return direct_ranges, bank_ranges


print('Какой сценарий вы хотите выбрать? Первый - симуляция броска, второй - нахождение углов')

if input() == '1':
    vel, ang = map(float, input('Введите скорость и угол(через пробел): ').split())
    res = simulate_trajectory(vel, ang)
    print(f"Результат: {res[0]}")
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(res[1]['x'], res[1]['y'], color='blue', lw=2)

    ax.axhline(0, color='green', lw=2, label='Пол')
    ax.plot([X_board, X_board], [Y_board_bottom, Y_board_top], color='black', lw=4, label='Щит')

    ax.plot([X_rim_front, X_rim_back], [Y_rim, Y_rim], color='orange', lw=3, label='Кольцо')
    ax.scatter([X_rim_front, X_rim_back], [Y_rim, Y_rim], color='red', s=50, zorder=5, label='Дужки')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, X_board + 1)
    ax.set_ylim(0, max(np.max(res[1]['y']) * 1.1, Y_board_top + 0.5))
    ax.set_xlabel("Расстояние X (м)")
    ax.set_ylabel("Высота Y (м)")
    ax.set_title("Честная векторная симуляция баскетбольного броска")
    ax.legend()
    ax.grid(True, linestyle='--')
    plt.show()
else:
    vel = float(input('Введите скорость: '))
    angs = scan_angles(vel)
    print('Чистые:')
    for start, end in angs[0]:
        if start == end:
            print("Угол: ", start)
        else:
            print(f'Угол от {start} до {end}')

    print('От щита:')
    for start, end in angs[1]:
        if start == end:
            print("Угол: ", start)
        else:
            print(f'Угол от {start} до {end}')
