import pygame

class Button():
    def __init__(self,origin:tuple,height, width):
        self.pos = origin
        self.height = height
        self.width = width

    def draw_butten(self,screen,color = (22,44,80)):
        pygame.draw.rect(screen, color,pygame.Rect(self.pos[0], self.pos[1],self.width,self.height))
