import cv2
import pygame
import mediapipe as mp

# pygame 初始化
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# 飛船
ship_x, ship_y = WIDTH // 2, HEIGHT - 80

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        index_finger = hand.landmark[8]  # 食指尖

        # 映射到遊戲座標
        ship_x = int(index_finger.x * WIDTH)
        ship_y = int(index_finger.y * HEIGHT)

    # 畫面
    screen.fill((0, 0, 0))
    pygame.draw.polygon(screen, (0, 255, 255),
        [(ship_x, ship_y),
         (ship_x - 20, ship_y + 40),
         (ship_x + 20, ship_y + 40)]
    )

    pygame.display.update()
    clock.tick(60)

cap.release()
pygame.quit()
