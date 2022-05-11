import pygame
from Button import *

def main():
    pygame.init()

    clock = pygame.time.Clock()


    screen = pygame.display.set_mode((700, 900))
    screen.fill((140,150,170))
    # path = 'C:\\Users\\ellah\\PycharmProjects\\Nitzanim\\Nitzagram\\Images\\mountain.jpg'
    # img = pygame.image.load(path)
    # img = pygame.transform.scale(img, (300, 100))

    running = True
    button_type = Button((100,100),100,200)
    button_type.draw_butten(screen)
    button_2 = Button((300, 300), 100, 200)
    button_2.draw_butten(screen)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            #     check for click on screen
            if event.type == pygame.MOUSEBUTTONDOWN:
                click_pose = event.pos # the position of the event
                if mouse_in_button(button_type,click_pose):
                    type = set_experiment_type(1)
                    #running = False
                    return type

                if mouse_in_button(button_2,click_pose):
                    type= set_experiment_type(0)
                    #running = False
                    return type


        #screen.blit(img, (100, 150))
        pygame.display.update()
        clock.tick(60)

    pygame.quit()




def set_experiment_type(type):
    if type == 1:
        return 'offline'
    if type == 0:
        return 'online'

def mouse_in_button(button, mouse_pos):

    if button.pos[0] + button.width > mouse_pos[0] > button.pos[0] and \
            button.pos[1]< mouse_pos[1] < button.pos[1] + button.height:
        return True

main()
