"""Three uncluttered model figures: recurrence, one MPT layer, logits head."""
from draw_explanatory_figures import canvas,text,box,arrow,save,BLUE,TEAL,GRAY,INK

def route(ax,points,color=GRAY,dashed=False):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,linewidth=1.1,linestyle='--' if dashed else '-')
    arrow(ax,points[-2],points[-1],color,dashed)

def overview():
    fig,ax=canvas(8.6)
    text(ax,9,8.12,'01  循环整体：逐帧更新隐变量',13,weight='bold')
    text(ax,9,7.4,r'零初始化 $V_{-1}=C_{-1}=0$；13 帧共用同一套 MPT 参数',10.5)
    centers=[2.5,6.9,11.1,15.4]
    for x,t in zip(centers,['0','1',None,'12']):
        if t is None:
            text(ax,x,5.9,r'$\cdots$',19);text(ax,x,4.3,r'$\cdots$',19);continue
        box(ax,x,5.9,2.7,.85,r'$x_{'+t+r'},\ p$',size=13)
        arrow(ax,(x,5.4),(x,4.95))
        box(ax,x,4.3,2.9,1.1,'MPT'+ '\n'+r'$\mathcal{T}_{\theta}$',size=12)
    arrow(ax,(4.05,4.3),(5.35,4.3),BLUE)
    arrow(ax,(8.45,4.3),(10.35,4.3),BLUE)
    arrow(ax,(11.85,4.3),(13.85,4.3),BLUE)
    text(ax,9,3.3,r'横向传递的只有隐变量 $(V,C)$，不是上一轮的 bit 或 logits',10.5,color=BLUE)
    box(ax,6.1,1.95,9.0,1.0,r'保存各帧隐变量 $C_0,C_1,\ldots,C_{12}$',size=11)
    box(ax,14.6,1.95,5.2,1.0,r'最终隐变量 $V_{12},C_{12}$',size=11)
    route(ax,[(2.5,3.7),(1.0,3.7),(1.0,1.95),(1.45,1.95)],TEAL)
    route(ax,[(15.4,3.7),(17.5,3.7),(17.5,1.95),(17.3,1.95)],TEAL)
    text(ax,9,.65,'这些隐变量在读完序列后送入 logits 输出头；CS8 不参与时间循环。',10,color=GRAY)
    save(fig,'model_01_recurrence')

def layer():
    fig,ax=canvas(11.2)
    text(ax,9,10.7,'02  单帧 MPT 细节：一层中的两次交叉注意力',13,weight='bold')
    text(ax,9,9.9,'先更新隐变量 V，再用更新后的 V 更新隐变量 C',11)
    for y,label in [(7.75,r'$V$'),(3.9,r'$C$')]:box(ax,1.7,y,2.0,1.0,label,size=16)
    box(ax,6.9,7.75,4.4,1.8,'掩码交叉注意力\nQ 来自 LN(V)\nK、Value 来自 C',size=10.5)
    box(ax,12.0,7.75,3.7,1.8,'残差相加\nLN → FFN\n再做残差相加',size=10.5)
    box(ax,16.3,7.75,2.0,1.0,r'$V^{+}$',size=16)
    arrow(ax,(2.8,7.75),(4.6,7.75),BLUE);arrow(ax,(9.2,7.75),(10.0,7.75),BLUE);arrow(ax,(13.95,7.75),(15.2,7.75),BLUE)
    route(ax,[(2.8,3.9),(3.7,3.9),(3.7,6.55),(6.9,6.55),(6.9,6.8)],TEAL)
    box(ax,6.9,3.9,4.4,1.8,'掩码交叉注意力\nQ 来自 LN(C)\nK、Value 来自新的 V',size=10.5)
    box(ax,12.0,3.9,3.7,1.8,'残差相加\nLN → FFN\n再做残差相加',size=10.5)
    box(ax,16.3,3.9,2.0,1.0,r'$C^{+}$',size=16)
    arrow(ax,(2.8,3.9),(4.6,3.9),BLUE);arrow(ax,(9.2,3.9),(10.0,3.9),BLUE);arrow(ax,(13.95,3.9),(15.2,3.9),BLUE)
    route(ax,[(16.3,7.15),(16.3,5.8),(6.9,5.8),(6.9,4.85)],TEAL)
    text(ax,11.2,6.1,'更新后的隐变量 V',10,color=TEAL)
    text(ax,9,2.25,'Q、K、Value 分别经过线性投影；4 头注意力，隐变量维度为 64。',10)
    text(ax,9,1.45,'同一层的两个方向复用参数；每帧执行两层，第 1 层后及第 2 层后均做 LayerNorm。',9.5)
    text(ax,9,.65,'掩码固定取自校验矩阵；当前帧嵌入与 p 条件在进入 MPT 前注入隐变量。',9.5,color=GRAY)
    save(fig,'model_02_mpt_layer')

def readout():
    fig,ax=canvas(12.5)
    text(ax,9,12.0,'03  从隐变量到故障 logits：以第 i 个故障为例',13,weight='bold')
    box(ax,3.2,10.75,5.2,1.0,r'$C_0,C_1,\ldots,C_{12}$',size=13)
    box(ax,9.0,10.75,5.0,1.0,r'$V_{12},C_{12}$',size=13)
    box(ax,14.8,10.75,4.9,1.0,'故障 i 的其他输入',size=11)
    box(ax,3.2,8.45,5.2,2.0,'Linear（64 → 16）\n按故障 i 关联的位置平均\n得到 16 维',color=TEAL,fill='#ECF6F1',size=10.5)
    box(ax,9.0,8.45,5.0,2.0,'V、C 分别求均值后拼接\nLinear（128 → 16）\n得到 16 维',color=TEAL,fill='#ECF6F1',size=10.5)
    box(ax,14.8,8.45,4.9,2.0,'故障嵌入：16 维\n检测事件计数：1 维\n先验 logit / 10：1 维',color=TEAL,fill='#ECF6F1',size=10.5)
    for x in [3.2,9,14.8]:arrow(ax,(x,10.15),(x,9.55),TEAL)
    box(ax,9,5.8,14.5,1.0,'拼接：16 + 16 + 16 + 1 + 1 = 50 维',size=12)
    for x in [3.2,9,14.8]:arrow(ax,(x,7.35),(x,6.4),TEAL)
    box(ax,9,3.7,11.5,1.15,'Linear（50 → 32）→ GELU → Linear（32 → 1）\n输出修正量 '+r'$r_{\theta,i}$',size=11)
    arrow(ax,(9,5.2),(9,4.4),BLUE)
    box(ax,9,1.5,11.5,1.15,r'$z_i=\log\frac{q_i}{1-q_i}+r_{\theta,i}$',size=16)
    arrow(ax,(9,3.0),(9,2.2),BLUE)
    text(ax,9,.35,'所有故障 i 并行计算；输出 '+r'$z$'+' 后再接 CS8。训练与推理均不使用辅助逻辑头。',9.5,color=GRAY)
    save(fig,'model_03_logits_head')

if __name__=='__main__':overview();layer();readout();print('THREE_MODEL_PANELS_CREATED')
