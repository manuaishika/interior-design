"""Stand-in room pictures: the SAME room (same walls, window, bed, spot) in three finishes."""
from PIL import Image, ImageDraw, ImageFilter
W, H = 1200, 900

def room(p, name):
    im = Image.new("RGB", (W, H), p["wall"]); d = ImageDraw.Draw(im)
    # back wall gradient
    for y in range(0, 600):
        t = y / 600; c = tuple(int(p["wall"][i]*(1-0.10*t)) for i in range(3)); d.line([(0,y),(W,y)], fill=c)
    # ceiling strip
    d.polygon([(0,0),(W,0),(W-110,70),(110,70)], fill=p["ceil"])
    d.polygon([(0,0),(110,70),(110,600),(0,700)], fill=p["side"])
    d.polygon([(W,0),(W-110,70),(W-110,600),(W,700)], fill=p["side"])
    # floor
    d.polygon([(0,700),(110,600),(W-110,600),(W,700),(W,H),(0,H)], fill=p["floor"])
    for i in range(1, 9):
        y = 600 + i*(H-600)//9
        d.line([(0 if y>700 else 110-(110*(y-600)/100), y),(W,y)], fill=p["floor2"], width=2)
    # window
    d.rectangle([760,150,1030,430], fill=p["frame"]); d.rectangle([778,168,1012,412], fill=p["sky"])
    d.line([(895,168),(895,412)], fill=p["frame"], width=10); d.line([(778,290),(1012,290)], fill=p["frame"], width=10)
    # curtains
    if p["curtain"]:
        d.rectangle([730,130,790,470], fill=p["curtain"]); d.rectangle([1000,130,1060,470], fill=p["curtain"])
    # door on left wall
    d.rectangle([150,220,300,600], fill=p["door"]); d.ellipse([275,410,290,425], fill=p["knob"])
    # rug
    d.polygon([(330,760),(870,760),(960,860),(240,860)], fill=p["rug"])
    # bed
    d.rectangle([400,440,900,470], fill=p["head"])                         # headboard
    d.rectangle([400,470,900,690], fill=p["bed"]); d.rectangle([400,600,900,700], fill=p["bed2"])
    d.rectangle([390,690,910,730], fill=p["base"])
    for x0 in (430, 640):
        d.rounded_rectangle([x0,470,x0+200,540], 16, fill=p["pillow"])
    d.rectangle([400,560,900,600], fill=p["throw"])
    # bedside + lamp
    d.rectangle([930,600,1030,730], fill=p["base"]); d.rectangle([944,620,1016,660], fill=p["bed2"])
    d.rectangle([975,540,985,600], fill=p["knob"]); d.polygon([(945,540),(1015,540),(1000,490),(960,490)], fill=p["shade"])
    # wardrobe on right wall, always there
    d.rectangle([1060,180,W-110+40,640], fill=p["ward"]) if False else None
    # wall art
    if p["art"]:
        d.rectangle([540,230,760,380], fill=p["frame"]); d.rectangle([552,242,748,368], fill=p["art"])
        d.ellipse([610,270,690,350], fill=p["art2"])
    # plant
    if p["plant"]:
        d.rectangle([180,620,230,700], fill=p["pot"])
        for a in [(150,520,215,640),(190,500,255,630),(215,540,285,650)]:
            d.ellipse(a, fill=p["plant"])
    # clutter
    for (x,y,w,h,c) in p["clutter"]:
        d.rectangle([x,y,x+w,y+h], fill=c)
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    im.save(name, quality=92)

before = dict(wall=(214,205,190), ceil=(232,228,220), side=(198,188,172), floor=(150,120,92), floor2=(138,110,84),
    frame=(120,104,88), sky=(190,214,232), curtain=(176,158,140), door=(140,108,78), knob=(200,180,120),
    rug=(120,104,90), head=(104,80,60), bed=(186,170,150), bed2=(170,152,132), base=(112,86,64), pillow=(214,204,188),
    throw=(150,120,100), shade=(226,212,180), art=None, art2=None, plant=None, pot=(120,90,70),
    clutter=[(430,650,90,30,(90,110,150)),(800,640,60,50,(180,90,80)),(300,700,120,28,(70,70,70)),(560,720,70,24,(210,190,90))])
after = dict(wall=(226,222,210), ceil=(240,238,230), side=(212,208,196), floor=(196,168,132), floor2=(184,156,120),
    frame=(70,58,46), sky=(206,226,238), curtain=(234,228,214), door=(201,175,140), knob=(60,50,40),
    rug=(206,196,178), head=(120,96,70), bed=(238,234,224), bed2=(226,220,208), base=(150,120,88), pillow=(250,248,242),
    throw=(150,168,140), shade=(245,235,210), art=(236,228,212), art2=(178,120,90), plant=(84,120,78), pot=(160,120,90), clutter=[])
edited = dict(after, wall=(170,190,160), side=(150,172,142), ceil=(228,232,222))
room(before, "before.jpg"); room(after, "after.jpg"); room(edited, "edited.jpg")
