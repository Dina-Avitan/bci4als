import pygame
from Button import *

def main():
    pygame.init()

    clock = pygame.time.Clock()


    screen = pygame.display.set_mode((700, 500))
    screen.fill((223,207,229))
    # path = 'C:\\Users\\ellah\\PycharmProjects\\Nitzanim\\Nitzagram\\Images\\mountain.jpg'
    # img = pygame.image.load(path)
    # img = pygame.transform.scale(img, (300, 100))

    running = True

    # Buttons
    button_offline = Button((100,100),80,200)
    button_offline.draw_butten(screen,'Offline',text_size=35)
    button_online = Button((100, 300), 80, 200)
    button_online.draw_butten(screen,'Online',text_size=35)
    button_right_left = Button((350,100), 80, 200)
    button_right_left.draw_butten(screen, 'right, left, idle')
    button_tongue_hands = Button((350, 300), 80, 200)
    button_tongue_hands.draw_butten(screen, 'tongue, hands, idle')
    type = 0
    path = 0
    button_run = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            #     check for click on screen
            if event.type == pygame.MOUSEBUTTONDOWN:
                click_pose = event.pos # the position of the event

                if mouse_in_button(button_offline,click_pose):
                    type = set_experiment_type(1)

                if mouse_in_button(button_online,click_pose):
                    type= set_experiment_type(0)

                if mouse_in_button(button_right_left,click_pose):
                    name = 'right_left_idle'
                    path= set_folder_path(name)

                if mouse_in_button(button_tongue_hands,click_pose):
                    name = 'tongue_hands_idle'
                    path = set_folder_path(name)
                if button_run:
                    if mouse_in_button(button_run,click_pose):
                        return type, path

        if type and path:
            button_run = Button((300, 400), 50, 100)
            button_run.draw_butten(screen, 'Run!', text_size=30)

        #screen.blit(img, (100, 150))
        pygame.display.update()
        clock.tick(60)

    pygame.quit()




def set_experiment_type(type):
    if type == 1:
        return 'offline'
    if type == 0:
        return 'online'

def set_folder_path(name):
    path = r'C:\Users\pc\Desktop\bci4als\recordings'
    if name == "right_left_idle":
        path = path+'\\'+'avi_'+name
        return path

def mouse_in_button(button, mouse_pos):

    if button.pos[0] + button.width > mouse_pos[0] > button.pos[0] and \
            button.pos[1]< mouse_pos[1] < button.pos[1] + button.height:
        return True

main()
