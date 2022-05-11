import pygame

class Button():
    def __init__(self,origin:tuple,height, width):
        self.pos = origin
        self.height = height
        self.width = width

    def draw_butten(self,screen,text,color = (244,129,247),text_size = 20):
        pygame.draw.rect(screen, color,pygame.Rect(self.pos[0], self.pos[1],self.width,self.height))
        font_name="Ariel"
        font = pygame.font.SysFont(font_name, text_size)
        screen.blit(font.render(text,True,(7,7,7)),(self.pos[0]+(self.width/3),self.pos[1]+(self.height/3)))
