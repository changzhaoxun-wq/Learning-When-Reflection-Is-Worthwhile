"""Math-only paper figures. Tensor colors are schematic, not measured activations."""

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
INK = "#202C34"
GRAY = "#70808B"
LINE = "#D6DFE4"
BLUE = "#24789F"
TEAL = "#138778"
RED = "#BC514F"
GOLD = "#AB7826"
PALE = "#F2F5F7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "dejavusans"})


def canvas(height):
    fig, ax = plt.subplots(figsize=(14, height))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 14), ylim=(0, height))
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size=14, color=INK, ha="center", weight="normal", **kwargs):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
                   fontweight=weight, linespacing=1.35, **kwargs)


def arrow(ax, start, end, color=GRAY, dashed=False, curve=0, lw=1.7):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                 linewidth=lw, color=color, linestyle="--" if dashed else "-",
                 connectionstyle=f"arc3,rad={curve}", shrinkA=2, shrinkB=2))


def route(ax, pts, color=GRAY, dashed=False):
    ax.plot(*zip(*pts[:-1]), color=color, lw=1.6, ls="--" if dashed else "-")
    arrow(ax, pts[-2], pts[-1], color=color, dashed=dashed)


def section(ax, y, letter, title, right):
    text(ax, .42, y, letter, 17, BLUE, ha="left", weight="bold")
    text(ax, .9, y, title, 17, ha="left", weight="bold")
    text(ax, 13.5, y, right, 12, GRAY, ha="right")


def model(ax, x, y, name, color=BLUE, scale=1):
    w, h = .92 * scale, .62 * scale
    for k in (2, 1, 0):
        dx, dy = -.065*k*scale, .065*k*scale
        ax.add_patch(Polygon([(x-w/2+dx,y-h/2+dy), (x+w/2+dx,y-h/2+dy),
                             (x+w/2+dx,y+h/2+dy), (x-w/2+dx,y+h/2+dy)],
                             fc="white", ec=color, lw=1.3))
    text(ax, x, y, name, 16*scale, color)


def answer(ax, x, y, s, color=BLUE):
    ax.add_patch(Circle((x,y), .22, fc="white", ec=color, lw=1.6))
    text(ax,x,y,s,14,color)


def sheet(ax, x, y, title, detail):
    ax.add_patch(Polygon([(x,y),(x+1,y),(x+1,y+.98),(x+.81,y+1.17),(x,y+1.17)],
                         fc="white",ec=BLUE,lw=1.5))
    ax.plot([x+.81,x+.81,x+1],[y+1.17,y+.98,y+.98],c=BLUE,lw=1)
    text(ax,x+.48,y+.8,title,16,BLUE)
    for k,w in enumerate((.67,.5,.62)):
        ax.plot([x+.16,x+.16+w],[y+.51-.14*k]*2,color=LINE,lw=2)
    text(ax,x+.5,y-.25,detail,12,GRAY)


def save(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for ax in fig.axes:
        for item in ax.texts:
            box = item.get_window_extent(renderer)
            if not fig.bbox.contains(box.x0,box.y0) or not fig.bbox.contains(box.x1,box.y1):
                outside.append(item.get_text())
    if outside:
        raise ValueError(f"Text exceeds canvas: {outside}")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT/f"{stem}.{ext}", dpi=240, facecolor="white")
    plt.close(fig)
    target = PROJECT/"paper/figures"
    target.mkdir(exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        shutil.copy2(OUT/f"{stem}.{ext}", target/f"{stem}.{ext}")


def framework():
    fig, ax = canvas(8.9)
    text(ax,.45,8.52,"Adaptive reflection for mathematical reasoning",22,ha="left",weight="bold")
    section(ax,7.92,"A", "Offline candidate supervision", "Gold answers are used only here")
    sheet(ax,.55,5.83,r"$q$","Training question")
    arrow(ax,(1.65,6.48),(2.22,6.48),BLUE)
    model(ax,2.8,6.48,r"$M_B$")
    text(ax,2.8,5.81,"Frozen Base",12,GRAY)
    arrow(ax,(3.32,6.48),(3.75,6.48),BLUE)
    answer(ax,4.05,6.48,r"$y_0$")
    route(ax,[(4.33,6.48),(4.65,6.48),(4.65,7.14),(8.55,7.14)],BLUE)
    text(ax,5.08,7.42,"D",13,BLUE,weight="bold")
    arrow(ax,(4.33,6.48),(5.65,6.48),TEAL)
    model(ax,6.2,6.48,r"$M_R$",TEAL,.86)
    arrow(ax,(6.66,6.48),(7.1,6.48),TEAL)
    answer(ax,7.39,6.48,r"$y_1$",TEAL)
    arrow(ax,(7.65,6.48),(8.55,6.48),TEAL)
    text(ax,5.08,6.74,"R1",13,TEAL,weight="bold")
    route(ax,[(7.39,6.22),(7.39,5.64),(7.64,5.64)],GOLD)
    model(ax,8.17,5.64,r"$M_R$",GOLD,.73)
    arrow(ax,(8.57,5.64),(8.85,5.64),GOLD)
    text(ax,6.55,5.65,"R2",13,GOLD,weight="bold")
    text(ax,6.38,5.16,"Each revision sees the previous answer",12,GRAY)
    # An outcome ledger, not fabricated numerical results.
    xcols=(9.13,9.79,10.49)
    for x,s in zip(xcols,(r"$a$",r"$C_a$",r"$T_a$")):
        text(ax,x,7.46,s,12,weight="bold")
    ax.plot([8.75,10.88],[7.23]*2,c=INK,lw=1)
    for y,a,c,t,col in [(7.0,"D",r"$C_0$",r"$T_0$",BLUE),
                        (6.44,"R1",r"$C_1$",r"$T_1$",TEAL),
                        (5.88,"R2",r"$C_2$",r"$T_2$",GOLD)]:
        for x,s in zip(xcols,(a,c,t)):
            text(ax,x,y,s,14,col)
    ax.plot([8.75,10.88],[5.53]*2,c=LINE,lw=1)
    text(ax,9.8,5.18,r"$C_a=\mathrm{match}(y_a,g)$",12,GRAY)
    arrow(ax,(10.99,6.46),(11.48,6.46),RED,dashed=True)
    text(ax,12.48,7.14,"Accuracy first",15,RED,weight="bold")
    text(ax,12.48,6.66,r"$a^*=\arg\max_a(C_a,-T_a)$",14)
    text(ax,12.48,6.05,"Gate target: D / reflect\nRound target: R1 / R2",13)
    text(ax,12.48,5.33,"Supervise the two policy heads",12,RED)
    ax.plot([.45,13.55],[4.83]*2,c=LINE,lw=1.2)
    section(ax,4.45,"B","Online budget allocation", "No gold answer or candidate score at input")
    sheet(ax,.55,2.28,r"$q$","New question")
    arrow(ax,(1.65,2.9),(2.22,2.9),BLUE)
    model(ax,2.8,2.9,r"$M_B$")
    text(ax,2.8,2.23,"Frozen Base",12,GRAY)
    arrow(ax,(3.32,2.9),(3.77,2.9),BLUE)
    answer(ax,4.05,2.9,r"$y_0$")
    text(ax,4.2,2.24,r"State $(q,y_0)$",13)
    arrow(ax,(4.32,2.9),(4.98,2.9),BLUE)
    for k in range(4):
        ax.add_patch(Rectangle((5.14,2.42+.19*k),.75,.14,fc=PALE,ec=GRAY,lw=.8))
    text(ax,5.5,3.65,"Frozen encoder",13,GRAY)
    text(ax,5.5,2.05,"Hidden vectors",12,GRAY)
    arrow(ax,(5.98,2.9),(6.42,2.9),TEAL)
    ax.add_patch(Circle((6.82,2.9),.32,ec=RED,fc="white",lw=1.8))
    text(ax,6.82,2.9,"G",17,RED,weight="bold")
    text(ax,6.82,2.18,"Gate",13,RED)
    # Direct bypass is visibly separate from the two paid paths.
    route(ax,[(6.82,3.26),(6.82,3.77),(12.97,3.77)],BLUE)
    text(ax,9.45,3.98,"D: skip reflection",13,BLUE)
    answer(ax,13.27,3.77,r"$y_0$",BLUE)
    arrow(ax,(7.17,2.9),(7.72,2.9),RED)
    ax.add_patch(Circle((8.08,2.9),.32,ec=RED,fc="white",lw=1.8))
    text(ax,8.08,2.9,"R",17,RED,weight="bold")
    text(ax,8.61,2.21,"Round",13,RED)
    arrow(ax,(8.45,2.9),(9.62,2.9),TEAL)
    model(ax,10.18,2.9,r"$M_R$",TEAL,.86)
    arrow(ax,(10.66,2.9),(12.97,2.9),TEAL)
    text(ax,11.81,3.16,"R1",13,TEAL,weight="bold")
    answer(ax,13.27,2.9,r"$y_1$",TEAL)
    route(ax,[(8.08,2.54),(8.08,1.38),(9.62,1.38)],GOLD)
    model(ax,10.18,1.38,r"$M_R$",GOLD,.86)
    arrow(ax,(10.68,1.38),(11.26,1.38),GOLD)
    model(ax,11.8,1.38,r"$M_R$",GOLD,.86)
    arrow(ax,(12.3,1.38),(12.97,1.38),GOLD)
    answer(ax,13.27,1.38,r"$y_2$",GOLD)
    text(ax,11.02,1.98,"R2: two revisions",13,GOLD)
    text(ax,3.64,1.3,"Both heads see only the initial state.\nThe number of rounds is chosen before generation.",13,GRAY)
    ax.plot([.45,13.55],[.7]*2,c=LINE,lw=1)
    text(ax,.55,.35,r"Path cost: $T=\sum_k(T_k^{\rm input}+T_k^{\rm output})$",14,ha="left")
    text(ax,13.45,.35,"Count retries; reject unresolved truncation",13,RED,ha="right")
    save(fig,"adaptive_reflection_framework")


def tensor(ax,x,y,color):
    for i in range(6):
        for j in range(4):
            ax.add_patch(Rectangle((x+j*.14,y+i*.14),.12,.12,fc=color,
                                  alpha=.2+.12*((i+j)%5),ec="none"))


def vector(ax,x,y,color):
    for k in range(5):
        ax.add_patch(Rectangle((x+k*.13,y),.11,.16,fc=color,alpha=.35+.13*(k%4),ec="none"))


def head(ax,y,name):
    text(ax,11.72,y+1.0,name,16,RED,weight="bold")
    xs=(10.35,11.28,12.2,13.13)
    coords=[]
    for x,n in zip(xs,(5,4,3,2)):
        ys=np.linspace(y-.5,y+.5,n)
        coords.append([(x,float(v)) for v in ys])
    for a,b in zip(coords,coords[1:]):
        for xy1 in a:
            for xy2 in b:
                ax.plot([xy1[0],xy2[0]],[xy1[1],xy2[1]],c=RED,alpha=.2,lw=.8)
    for col in coords:
        for x,v in col:
            ax.add_patch(Circle((x,v),.075,fc="white",ec=RED,lw=1.15))
    for x,s in zip(xs,("4096","64","16","2")):
        text(ax,x,y-.84,s,12,RED)
    for x,s in ((11.28,"ReLU"),(12.2,"ReLU"),(13.13,"Softmax")):
        text(ax,x,y-1.15,s,11,GRAY)


def network():
    fig, ax=canvas(9.5)
    text(ax,.45,9.1,"Inside the hierarchical policy selector",22,ha="left",weight="bold")
    text(ax,.48,8.6,"Text representation",16,BLUE,ha="left",weight="bold")
    text(ax,10.03,8.6,"Trainable action heads",16,RED,ha="left",weight="bold")
    ax.plot([9.77,9.77],[2.13,8.27],color=LINE,lw=1.5,ls="--")
    text(ax,1.5,7.68,"Question +\ninitial answer",16,BLUE,weight="bold")
    for row,(s,col) in enumerate(((r"$q_1$",BLUE),(r"$q_2$",BLUE),("...",BLUE),(r"$y_{0,1}$",TEAL),("...",TEAL))):
        y=6.91-row*.43
        ax.add_patch(Rectangle((.77,y-.14),1.32,.29,fc=col,alpha=.1,ec=col,lw=1))
        text(ax,1.43,y,s,14,col)
    text(ax,1.47,4.42,"Tokenize + mask\nL <= 1024",13,GRAY)
    text(ax,1.5,3.59,"No gold answer\nNo hand-built scores",13,GRAY)
    route(ax,[(2.17,5.97),(2.53,5.97),(2.53,3.42),(2.88,3.42)],BLUE)
    layers=[(3.16,"Embedding",False),(3.79,"Block 1",False),(4.42,"...",False),
            (5.05,"Block 26",False),(5.68,"Block 27",True),(6.31,"Block 28",True)]
    for y,name,selected in layers:
        c=TEAL if selected else GRAY
        ax.add_patch(Polygon([(2.94,y),(4.48,y),(4.59,y+.13),(4.59,y+.52),(3.05,y+.52),(2.94,y+.39)],
                             fc="#E6F3F0" if selected else PALE,ec=c,lw=1.15))
        text(ax,3.76,y+.26,name,14,c)
    for (y,_,_), (next_y,_,_) in zip(layers,layers[1:]):
        arrow(ax,(3.76,y+.53),(3.76,next_y-.01),GRAY,lw=1)
    text(ax,3.75,7.4,"Qwen3-0.6B",16,BLUE,weight="bold")
    text(ax,3.75,7.04,"28 blocks; width 1024",12,GRAY)
    text(ax,3.8,2.64,"ALL encoder weights frozen",13,GRAY,weight="bold")
    text(ax,3.8,2.23,"Last layers are read, not fine-tuned",12,GRAY)
    arrow(ax,(4.65,6.57),(5.19,6.57),TEAL)
    route(ax,[(4.65,5.94),(4.89,5.94),(4.89,4.59),(5.19,4.59)],TEAL)
    tensor(ax,5.27,6.17,TEAL)
    tensor(ax,5.27,4.18,BLUE)
    text(ax,5.54,7.37,r"$H^{(-1)}$",15,TEAL)
    text(ax,5.54,7.15,"After final norm",10.5,GRAY)
    text(ax,5.54,5.38,r"$H^{(-2)}$",15,BLUE)
    text(ax,5.55,3.8,r"$L\times1024$",12,GRAY)
    for y,c in ((6.57,TEAL),(4.58,BLUE)):
        arrow(ax,(5.89,y),(6.52,y+.3),c)
        arrow(ax,(5.89,y),(6.52,y-.32),c)
        vector(ax,6.66,y+.22,c)
        vector(ax,6.66,y-.42,c)
        text(ax,6.9,y+.64,"Masked mean",11.5,c)
        text(ax,6.9,y-.7,"Last valid token",11.5,c)
    text(ax,6.36,3.29,"1024 dimensions each",12,GRAY)
    ax.plot([7.47,7.68,7.68,7.47],[6.98,6.98,4.1,4.1],color=GRAY,lw=1.3)
    arrow(ax,(7.7,5.54),(8.16,5.54),GRAY)
    # Four pooled vectors are concatenated in layer order, not four trained layers.
    for i,(s,c) in enumerate(((r"$\mu_{-2}$",BLUE),(r"$v_{-2}$",BLUE),(r"$\mu_{-1}$",TEAL),(r"$v_{-1}$",TEAL))):
        y=6.84-i*.66
        ax.add_patch(Rectangle((8.3,y-.28),.72,.54,fc=c,alpha=.16,ec=c,lw=1))
        text(ax,8.66,y,s,14,c)
    text(ax,8.67,7.55,"Concat",15,weight="bold")
    text(ax,8.67,3.87,r"$z\in\mathbb{R}^{4096}$",16)
    text(ax,8.67,2.84,"Separate train-fitted\nscalers for G and R",12,GRAY)
    route(ax,[(9.06,5.85),(9.41,5.85),(9.41,6.91),(10.12,6.91)],TEAL)
    route(ax,[(9.06,5.22),(9.41,5.22),(9.41,3.97),(10.12,3.97)],TEAL)
    head(ax,6.91,"Gate: D / reflect")
    head(ax,3.97,"Round: R1 / R2")
    text(ax,11.75,2.2,"Same initial text; independent parameters",12,GRAY)
    ax.plot([.45,13.55],[1.84]*2,c=LINE,lw=1.2)
    text(ax,.55,1.43,"Supervision",14,RED,ha="left",weight="bold")
    text(ax,2.28,1.43,r"$g^*=\mathbb{1}[a^*\ne D]$",15,ha="left")
    text(ax,5.48,1.43,r"$r^*=\arg\max_{a\in\{R1,R2\}}(C_a,-T_a)$",14,ha="left")
    text(ax,13.48,1.43,"Cross-entropy per head",13,RED,ha="right")
    text(ax,.55,.87,"Gradient update",14,RED,ha="left",weight="bold")
    arrow(ax,(13.2,.87),(10.05,.87),RED,dashed=True,lw=2)
    ax.plot([9.98,9.98],[.69,1.05],c=RED,lw=2)
    text(ax,6.12,.87,"No backward pass into the encoder",13,GRAY)
    text(ax,.55,.32,"Inference: Gate threshold first; Round chooses the budget only when reflection is selected.",13,ha="left")
    save(fig,"adaptive_reflection_policy_network")


if __name__ == "__main__":
    framework()
    network()
    (OUT/"figure_provenance.json").write_text(json.dumps({
        "scope":"mathematics only", "encoder":"Qwen3-0.6B-Base", "blocks":28,
        "encoder_training":"frozen/no_grad", "selected_layers":[-2,-1],
        "pooling":"mean_last", "embedding_dim":4096, "heads":[4096,64,16,2],
        "sources":["src/transformer_policy.py","src/neural_policy.py","scripts/run_hierarchical_multiseed.sh"],
        "tensor_colors":"schematic, not measured activations", "experimental_results_shown":False,
    },indent=2)+"\n")
