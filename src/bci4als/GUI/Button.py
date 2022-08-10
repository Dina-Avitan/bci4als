import pygame

class Button():
    def __init__(self,origin:tuple,height, width):
        self.pos = origin
        self.height = height
        self.width = width
        self.pushed = False

    def draw_butten_rect(self,screen,text,color = (244,129,247),text_size = 20):
        pygame.draw.rect(screen, color,pygame.Rect(self.pos[0], self.pos[1],self.width,self.height))
        font_name="Ariel"
        font = pygame.font.SysFont(font_name, text_size)
        screen.blit(font.render(text,True,(7,7,7)),(self.pos[0]+(self.width/3),self.pos[1]+(self.height/3)))

    def draw_butten_circle(self,screen,text,color = (255,255,255),text_size = 15,radius=30):
        origin = (self.pos[0]+self.width/2,self.pos[1]+self.width/2)
        pygame.draw.circle(screen, color,origin,radius=radius)
        font_name="Ariel"
        font = pygame.font.SysFont(font_name, text_size)
        screen.blit(font.render(text,True,(7,7,7)),(origin[0]-(radius/3.5),origin[1]-(radius/7)))

    def draw_line(self, screen, color=(255, 0, 0), width=5):
        beg = [self.pos[0]+10,self.pos[1]+self.height-5]
        end =  [self.pos[0]+self.width-10,self.pos[1]+self.height-5]
        pygame.draw.line(screen,color,beg,end,width)

    def push_button(self):
        if self.pushed:
            self.pushed = False
        else:
            self.pushed = True

