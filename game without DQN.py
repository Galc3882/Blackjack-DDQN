import pygame
import random
import time

pygame.init()

hight = 720
width = 1280
screen = pygame.display.set_mode((width, hight))
pygame.display.set_caption('Blackjack')

font = pygame.font.Font(None, 46)
creamcolor = [255, 253, 208]

cardpack = []
dealercards = []
splitdealercard = []
playercards = []
lastpacksize = 0

running = True
start = True

global lastkeypressed
lastkeypressed = None
firstmove = False
score = 0
lastscore = 0

debugvar = True

isstand = False
cansplit = False
issplit = 0
splitcard = 0

isdouble = 0
candouble = False

def showtext(text, x, y, color = []):
    screen.blit(font.render(str(text), True, color), (x, y))
    pygame.display.update()

def clearscreen():
    screen.fill((0, 0, 0))
    pygame.display.update()

def drawcardpack():
    numofpacks = 8
    while numofpacks > 0:
        i = 1
        while i < 14:
            cardpack.extend([i, i, i, i])
            i += 1
        numofpacks -= 1

def initgame():
    playercards.clear()
    dealercards.clear()
    playercards.append(getnextcard())
    dealercards.append(getnextcard())
    playercards.append(getnextcard())
    dealercards.append(getnextcard())

def isnatural(cardlist):
    return max(totalcards(cardlist)) == 21

def playerlose(score):
    showtext('player lost', width*0.8, hight*0.05, creamcolor)
    start, running = asknewgame()
    return start, running, score - 1

def tie():
    showtext('tie', width*0.8, hight*0.05, creamcolor)
    start, running = asknewgame()
    return start, running

def asknewgame():
    global lastkeypressed
    lastkeypressed = None
    running = True
    time.sleep(0.5)
    showtext('restart? [Y]', width*0.45, hight*0.9, creamcolor)
    while (lastkeypressed != pygame.K_y and running == True):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                lastkeypressed = event.key
            if event.type == pygame.QUIT:
                running = False
                break
    clearscreen()
    start = True
    return start, running

def getnextcard():
    if len(cardpack) == 0:
        drawcardpack()
    return cardpack.pop(random.randrange(0, len(cardpack), 1))

def totalcards(cardlist):
    k = 0
    for i in cardlist:
        if i > 10:
            k += 10
        else:
            k += i
    if cardlist.count(1) > 0 and k + 10 < 22:
        return [k, k + 10]
    return [k, 0]

def showcard(card):
    if card > 10:
        if card == 11:
            return 'J'
        elif card == 12:
            return 'Q'
        elif card == 13:
            return 'K'
    elif card == 1:
        return 'A'
    return card
        
# draws the text at the start of the game
def drawstarttext():
    drawplayercards(creamcolor)
    drawscore()

# draws first dealer cards at start
def drawfirstdealercard():
    showtext('dealers cards: ' + str(showcard(dealercards[0])), width*0.3, hight*0.35, creamcolor)

# draws player cards
def drawplayercards(color):
    showtext('players cards: ' + str(showcard(playercards[0])) + ', ' + str(showcard(playercards[1])), width*0.3, hight*0.65, color)
    showtext('sum: ' + str(totalcards(playercards)[0]), width*0.3, hight*0.7, color)
    if totalcards(playercards)[1] != 0:
        showtext(', ' + str(totalcards(playercards)[1]), width*0.4, hight*0.7, color)

def drawnextplayercard():
    showtext(", " + str(showcard(playercards[len(playercards) - 1])), width * (0.54 + 0.035 * (len(playercards) - 3)), hight*0.65, creamcolor)
    showtext('sum: ' + str(totalcards(playercards)[0]), width*0.3, hight*0.7, creamcolor)
    if totalcards(playercards)[1] != 0:
        showtext(', ' + str(totalcards(playercards)[1]), width*0.4, hight*0.7, creamcolor)

def drawnextdealercard():
    showtext(", " + str(showcard(dealercards[len(dealercards) - 1])), width * (0.54 + 0.035 * (len(dealercards) - 3)), hight*0.35, creamcolor)
    showtext('sum: ' + str(totalcards(dealercards)[0]), width*0.3, hight*0.4, creamcolor)
    if totalcards(dealercards)[1] != 0:
        showtext(', ' + str(totalcards(dealercards)[1]), width*0.4, hight*0.4, creamcolor)

def drawscore():
    showtext('score: ' + str(score), width*0.05, hight*0.05, creamcolor)

def asknextmove(running, isstand, cansplit, issplit, candouble, isdouble):
    if isdouble == 2:
        hit()
        return running, isstand, False, issplit, False, isdouble - 1
    elif isdouble == 1:
        return running, True, False, issplit, False, isdouble
    elif issplit == 1:
        candouble = True
    
    global lastkeypressed
    lastkeypressed = -1
    time.sleep(0.5)
    splitkey = pygame.K_d
    doublekey = pygame.K_d

    if candouble == True and cansplit == True:
        showtext('stand: [A] split: [S] double down: [W] hit: [D]', width*0.2, hight*0.9, creamcolor)
        splitkey = pygame.K_s
        doublekey = pygame.K_w
    elif candouble == True:
        showtext('stand: [A] double down: [W] hit: [D]', width*0.3, hight*0.9, creamcolor)
        doublekey = pygame.K_w
    else:
        showtext('stand: [A] hit: [D]', width*0.4, hight*0.9, creamcolor)

    while (lastkeypressed != pygame.K_a and lastkeypressed != pygame.K_d and lastkeypressed != splitkey and lastkeypressed!= doublekey and running == True):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                lastkeypressed = event.key
            if event.type == pygame.QUIT:
                running = False

    if candouble == True and cansplit == True:
        showtext('stand: [A] split: [S] double down: [W] hit: [D]', width*0.2, hight*0.9, (0, 0, 0))
        if lastkeypressed == pygame.K_s:
            issplit = 2
        elif lastkeypressed == pygame.K_w:
            isdouble = 2
    elif candouble == True:
        showtext('stand: [A] double down: [W] hit: [D]', width*0.3, hight*0.9, (0, 0, 0))
        if lastkeypressed == pygame.K_w:
            isdouble = 2
    else:
        showtext('stand: [A] hit: [D]', width*0.4, hight*0.9, (0, 0, 0))

    if lastkeypressed == pygame.K_a:
        isstand = True
    elif lastkeypressed == pygame.K_d:
        hit()
    return running, isstand, False, issplit, False, isdouble

def hit():
    showtext('sum: ' + str(totalcards(playercards)[0]), width*0.3, hight*0.7, (0, 0, 0))
    if totalcards(playercards)[1] != 0:
        showtext(', ' + str(totalcards(playercards)[1]), width*0.4, hight*0.7, (0, 0, 0))
    playercards.append(getnextcard())
    drawnextplayercard()

def dealermove(start, running, score):
    drawnextdealercard()
    while totalcards(dealercards)[1] < 17 and totalcards(dealercards)[1] > 0:
        showtext('sum: ' + str(totalcards(dealercards)[0]), width*0.3, hight*0.4, (0, 0, 0))
        if totalcards(dealercards)[1] != 0:
            showtext(', ' + str(totalcards(dealercards)[1]), width*0.4, hight*0.4, (0, 0, 0))
        dealercards.append(getnextcard())
        time.sleep(0.5)
        drawnextdealercard()
    if not(totalcards(dealercards)[1] > 0):
        while totalcards(dealercards)[0] < 17 and totalcards(dealercards)[1] < 17:
            showtext('sum: ' + str(totalcards(dealercards)[0]), width*0.3, hight*0.4, (0, 0, 0))
            if totalcards(dealercards)[1] != 0:
                showtext(', ' + str(totalcards(dealercards)[1]), width*0.4, hight*0.4, (0, 0, 0))
            dealercards.append(getnextcard())
            time.sleep(0.5)
            drawnextdealercard()
        if totalcards(dealercards)[0] > 21:
            start, running, score = dealerlose(score)
    return start, running, score
    
def dealerlose(score):
    showtext('player win', width*0.8, hight*0.05, creamcolor)
    start, running = asknewgame()
    return start, running, score + 1

def checkbust(start, running, score, isdouble):
    if totalcards(playercards)[0] > 21:
        start, running, score = playerlose(score)
        if isdouble == 1:
            isdouble = 0
            score -= 1
    return start, running, score, isdouble

def showpacksize(lastpacksize):
    showtext('deck size: ' + str(lastpacksize), width*0.05, hight*0.95, (0, 0, 0))
    showtext('deck size: ' + str(len(cardpack)), width*0.05, hight*0.95, creamcolor)
    return len(cardpack)

def split(cardpack, playercards, dealercards, splitcard, splitdealercard, issplit):
    drawplayercards((0, 0, 0))
    if issplit == 2:
        splitcard = playercards[1]
        splitdealercard = dealercards
        playercards.pop(1)
        playercards.append(getnextcard())
    else:
        cardpack.append(playercards.pop(1))
        cardpack.append(playercards.pop(0))
        cardpack.append(dealercards.pop(1))
        cardpack.append(dealercards.pop(0))
        dealercards = splitdealercard
        playercards.append(splitcard)
        playercards.append(getnextcard())
        splitdealercard = []
        splitcard = 0
    drawplayercards(creamcolor)
    return cardpack, playercards, dealercards, splitcard, splitdealercard, issplit - 1

def checknatural(dealercards, playercards, score, start, running):
    isinsured = False
    if dealercards[0] == 1 and not(isnatural(playercards)):
        isinsured, running = insurance(running)
        if isinsured:
            if not(isnatural(dealercards)):
                score -= 0.5
                showtext('insurance lost', width*0.1, hight*0.5, creamcolor)
            else:
                score -= 1
                drawnextdealercard()
                showtext('insurance won', width*0.1, hight*0.5, creamcolor)
                start, running, score = dealerlose(score)
    if  running == True:
        if isnatural(dealercards) and isinsured == False:
            time.sleep(0.5)
            drawnextdealercard()
            time.sleep(0.5)
            if isnatural(playercards):
                start, running = tie()
            else:
                start, running, score = playerlose(score)
        elif isnatural(playercards):
            drawnextdealercard()
            start, running, score = dealerlose(score)
            score += 0.5
    return start, running, score

def insurance(running):
    global lastkeypressed
    lastkeypressed = -1
    showtext('insurance? [A-yes/D-no]', width*0.1, hight*0.5, creamcolor)
    while (lastkeypressed != pygame.K_a and lastkeypressed != pygame.K_d and running == True):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                lastkeypressed = event.key
            if event.type == pygame.QUIT:
                running = False
                break
    showtext('insurance? [A-yes/D-no]', width*0.1, hight*0.5, (0, 0, 0))
    if lastkeypressed == pygame.K_a:
        return True, running
    return False, running

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if start == True and running == True:
        start = False
        isstand = False
        firstmove = True
        candouble = True
        lastscore = score

        initgame()
        while totalcards(playercards)[0] > 21:
            initgame()
        
        #if debugvar == True:
            #dealercards = [10, 1]
            #dealercards[0] = 1
            #playercards = [6, 5]
            #playercards[1] = playercards[0]
            #debugvar = False
        
        drawstarttext()
        
        # draws this if didnt lose at the start
        drawfirstdealercard()

        # check split
        if (playercards[0] == playercards[1] or (max(totalcards(playercards)) == 20 and playercards[0] > 9)) and issplit == 0:
            cansplit = True
        elif issplit == 1 and firstmove == True:
            cardpack, playercards, dealercards, splitcard, splitdealercard, issplit = split(cardpack, playercards, dealercards, splitcard, splitdealercard, issplit)

        # check natural
        start, running, score = checknatural(dealercards, playercards, score, start, running)
        if running == False:
            break
                    
        
    lastpacksize = showpacksize(lastpacksize)

    if running == True and start == False:
        if max(totalcards(playercards)) != 21:
            running, isstand, cansplit, issplit, candouble, isdouble = asknextmove(running, isstand, cansplit, issplit, candouble, isdouble)
            if issplit == 2 and firstmove == True:
                cardpack, playercards, dealercards, splitcard, splitdealercard, issplit = split(cardpack, playercards, dealercards, splitcard, splitdealercard, issplit)
            start, running, score, isdouble = checkbust(start, running, score, isdouble)
            if len(playercards) > 5 and start == False:
                start, running, score = dealerlose(score)
        else:
            isstand = True

        firstmove = False

        if isstand == True and start == False:
            isstand = False
            start, running, score = dealermove(start, running, score)
            if start == False and running == True:
                if max(totalcards(dealercards)) > max(totalcards(playercards)):
                    start, running, score = playerlose(score)
                elif max(totalcards(dealercards)) == max(totalcards(playercards)):
                    start, running = tie()
                else:
                    start, running, score = dealerlose(score)
            if isdouble == 1:
                isdouble = 0
                score += score - lastscore



    # If the final sum is higher than the sum of the dealer, the player gets a play-off of 1:1 of his initial stake.
    # If the players combination is Blackjack, the play-off is 3:2 of the initial stake

