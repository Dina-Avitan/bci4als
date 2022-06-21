import pygame
from Button import *

def main():
    pygame.init()

    clock = pygame.time.Clock()

    screen_size = (795, 445)
    screen = pygame.display.set_mode(screen_size)
    #screen.fill((223,207,229))
    img_path = r'..\images\GUI_background.png'
    img = pygame.image.load(img_path)
    img = pygame.transform.scale(img, screen_size)
    screen.blit(img, (0,0))
    pygame.display.update()

    running = True

    # Buttons
    button_offline = Button((130,105),100,210)
    #button_offline.draw_butten_rect(screen,'Offline',text_size=35)
    button_online = Button((130, 270), 100, 210)
    #button_online.draw_butten_rect(screen,'Online',text_size=35)
    button_right_left = Button((470,105), 100, 210)
    #button_right_left.draw_butten_rect(screen, 'right, left, idle')
    button_tongue_hands = Button((470, 270), 100, 210)
    #button_tongue_hands.draw_butten_rect(screen, 'tongue, hands, idle')
    type = None
    path = ""
    button_run_existence = False
    gui_keys = None
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            #     check for click on screen
            if event.type == pygame.MOUSEBUTTONDOWN:
                click_pose = event.pos # the position of the event

                if button_run_existence and mouse_in_button(button_run,click_pose):
                    print("run clicked")
                    button_run.push_button()
                    return type, path, gui_keys

                if mouse_in_button(button_offline,click_pose):
                    type = set_experiment_type(1)
                    button_offline.push_button()
                    if button_offline.pushed:
                        button_offline.draw_line(screen,width=5)
                        if button_online.pushed:
                            button_online.push_button()
                            button_online.draw_line(screen, color=(255, 255, 255), width=5)
                    else:
                        button_offline.draw_line(screen, color=(255,255,255), width=5)

                if mouse_in_button(button_online,click_pose):
                    type= set_experiment_type(0)
                    button_online.push_button()
                    if button_online.pushed:
                        button_online.draw_line(screen, width=5)
                        if button_offline.pushed:
                            button_offline.push_button()
                            button_offline.draw_line(screen, color=(255, 255, 255), width=5)
                    else:
                        button_online.draw_line(screen,color=(255,255,255),width=5)

                if mouse_in_button(button_right_left,click_pose):
                    name = 'right_left_idle'
                    path= set_folder_path(name)
                    gui_keys = (0,1,2)
                    button_right_left.push_button()
                    if button_right_left.pushed:
                        button_right_left.draw_line(screen, width=5)
                        if button_tongue_hands.pushed:
                            button_tongue_hands.push_button()
                            button_tongue_hands.draw_line(screen, color=(255, 255, 255), width=5)
                    else:
                        button_right_left.draw_line(screen,color=(255,255,255),width=5)

                if mouse_in_button(button_tongue_hands,click_pose):
                    name = 'tongue_hands_idle'
                    path = set_folder_path(name)
                    gui_keys = (2,3,4)
                    button_tongue_hands.push_button()
                    if button_tongue_hands.pushed:
                        button_tongue_hands.draw_line(screen, width=5)
                        if button_right_left.pushed:
                            button_right_left.push_button()
                            button_right_left.draw_line(screen, color=(255, 255, 255), width=5)
                    else:
                        button_tongue_hands.draw_line(screen,color=(255,255,255),width=5)

        if type is not None and path!="":
            button_run = Button((377, 375), 50, 50)
            #button_run.draw_butten_rect(screen, 'Run!', text_size=20)
            button_run.draw_butten_circle(screen, 'Run!', text_size=20)
            button_run_existence = True

        # screen.blit(img, (0,0))
        # pygame.display.update()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def set_experiment_type(type):
    if type == 1:
        return 'offline'
    if type == 0:
        return 'online'

def set_folder_path(name):
    path = r'..\recordings'
    if name == "right_left_idle":
        path = path+'\\'+'avi_'+name
        return path

def mouse_in_button(button, mouse_pos):
    if button.pos[0] + button.width > mouse_pos[0] > button.pos[0] and \
            button.pos[1]< mouse_pos[1] < button.pos[1] + button.height:
        return True

