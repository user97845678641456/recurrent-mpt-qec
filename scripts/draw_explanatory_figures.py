"""Scientific vector figures tied to the released BB matrices and model."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle,Rectangle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report/figures'
FONT=Path('C:/Windows/Fonts/msyh.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,'mathtext.fontset':'stix','font.size':10})
BLUE='#236B9B';ORANGE='#B96623';TEAL='#287F75';INK='#233647';GRAY='#657581';LIGHT='#EDF4F8'

def canvas(h=10.5):
    fig,ax=plt.subplots(figsize=(7.2,h*.4));ax.set(xlim=(0,18),ylim=(0,h));ax.axis('off');fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98);return fig,ax

def text(ax,x,y,s,size=10,color=INK,ha='center',weight='normal'):
    return ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,fontweight=weight,linespacing=1.5)

def box(ax,x,y,w,h,s,color=BLUE,fill=LIGHT,size=10,dashed=False):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.08,rounding_size=.12',linewidth=1,edgecolor=color,facecolor=fill,linestyle='--' if dashed else '-'))
    text(ax,x,y,s,size,color)

def arrow(ax,a,b,color=GRAY,dashed=False,rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,linewidth=1.1,color=color,linestyle='--' if dashed else '-',connectionstyle=f'arc3,rad={rad}',shrinkA=3,shrinkB=3))

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for suffix in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{suffix}',dpi=220,facecolor='white')
    plt.close(fig)

def bb():
    a=np.load(ROOT/'assets/bb72_r6/code_matrices.npz');hx=a['Hx'];hz=a['Hz'];assert not (hx@hz.T%2).any()
    support=np.flatnonzero(hx[0]);assert np.array_equal(support,[1,2,18,39,42,48])
    fig,ax=canvas(11.3)
    text(ax,9,10.85,'BB72：一个实际的六体校验如何连接数据比特',12,weight='bold')
    text(ax,9,10.18,'72 个数据比特 = L、R 两个 6 × 6 索引块；这里只展开 '+r'$H_X$'+' 的第 0 行',9,color=GRAY)
    coords={}
    for block,x0 in [(0,1.7),(1,12.0)]:
        text(ax,x0+2.1,9.55,'L 块（0–35）' if block==0 else 'R 块（36–71）',10,weight='bold')
        for r in range(6):
            for c in range(6):
                q=block*36+r*6+c;x=x0+c*.84;y=8.9-r*.84;coords[q]=(x,y)
                ax.add_patch(Circle((x,y),.095 if q not in support else .15,facecolor='#DCE3E7' if q not in support else BLUE,edgecolor='white',linewidth=.5,zorder=3))
                if q in support:text(ax,x+.25,y+.2,str(q),8,color=BLUE,ha='left')
        ax.add_patch(Rectangle((x0-.35,4.3),4.9,5.0,fill=False,edgecolor='#CDD8DF',linewidth=.8))
    box(ax,9,6.6,1.7,1.0,r'$S^X_0$',color=BLUE,size=13)
    for q in support:
        x,y=coords[q];ax.plot([8.05 if q<36 else 9.95,x],[6.6,y],color=BLUE,alpha=.65,linewidth=1,zorder=1)
    text(ax,9,5.45,'X 型稳定子',9,color=BLUE)
    text(ax,9,3.6,r'$S^X_0=X_1X_2X_{18}X_{39}X_{42}X_{48}$',13,color=BLUE)
    text(ax,2.4,2.7,'同一连接的 Tanner 表示',9,ha='left',weight='bold')
    ax.add_patch(Rectangle((8.78,2.15),.44,.44,facecolor=BLUE));text(ax,9,2.95,'校验节点',8,color=BLUE)
    for x,q in zip(np.linspace(3,15,6),support):
        ax.plot([9,x],[2.15,1.13],color=BLUE,linewidth=1)
        ax.add_patch(Circle((x,1),.16,facecolor='white',edgecolor=BLUE,linewidth=1.2));text(ax,x,.5,r'$q_{'+str(q)+'}$',10)
    text(ax,9,.05,'连线来自实际 '+r'$H_X$'+' 矩阵；索引块用于说明连接，不代表芯片位置或 CNOT 布线。',8,color=GRAY)
    save(fig,'bb72_check_structure')
    return dict(code='bb72_r6',matrix='Hx',row=0,qubit_indices=support.tolist(),row_weight=int(hx[0].sum()),Hx_HzT_zero=True)

def css():
    fig,ax=canvas(10.2)
    text(ax,9,9.7,'CSS 码：用两类稳定子分别检查两类 Pauli 错误',12,weight='bold')
    box(ax,9,8.55,7.5,1.15,r'$E_{\rm phys}=X^{e_X}Z^{e_Z}$'+'\n物理错误的二进制表示：X、Z 两个分量',color=INK,fill='#F4F6F8',size=10)
    arrow(ax,(6.5,7.9),(4.3,6.85),BLUE);arrow(ax,(11.5,7.9),(13.6,6.85),ORANGE)
    box(ax,4.4,6.1,6.8,1.3,'Z 型稳定子'+ '\n'+r'$H_Z\;\longrightarrow\;s_Z=H_Ze_X$',color=BLUE,size=12)
    box(ax,13.4,6.1,6.8,1.3,'X 型稳定子'+ '\n'+r'$H_X\;\longrightarrow\;s_X=H_Xe_Z$',color=ORANGE,fill='#FCF1E8',size=12)
    text(ax,4.4,4.9,'X 与 Z 反对易，因此 Z 校验检测 X 分量',9,color=BLUE)
    text(ax,13.4,4.9,'Z 与 X 反对易，因此 X 校验检测 Z 分量',9,color=ORANGE)
    text(ax,9,4.05,'Y 错误同时具有 X、Z 分量，两类校验都可能响应。',10)
    box(ax,9,2.85,14.8,1.15,r'$H_XH_Z^{\mathsf{T}}=0$'+'\n两组稳定子彼此对易，可以共同定义编码子空间',color=TEAL,fill='#ECF6F1',size=11)
    text(ax,9,1.55,'syndrome 是校验结果（+1 / −1 的二进制编码），不直接测出逻辑量子态。',9)
    text(ax,9,.87,'满足全部校验仍可能存在逻辑错误：恢复后还要检查逻辑作用。',9,weight='bold')
    text(ax,9,.18,'忽略整体相位；此图 '+r'$e_X,e_Z$'+' 是物理错误分量，报告中的 E 则是电路故障坐标。',8,color=GRAY)
    save(fig,'css_checks_basics')

def syndrome():
    fig,ax=canvas(11.0)
    text(ax,9,10.55,'多轮测量：从相邻读出的差分提取检测事件',12,weight='bold')
    text(ax,9,9.85,r'$d_t=m_t\oplus m_{t-1}$'+'：相邻读出的差分就是检测事件',11)
    xs=np.linspace(6.7,16.6,7)
    for x,t in zip(xs,range(7)):text(ax,x,9.1,r'$t='+str(t)+'$',10,weight='bold')
    def row(y,label,values,highlight=False):
        text(ax,.6,y,label,9,ha='left')
        for j,(x,v) in enumerate(zip(xs,values)):
            if v==1 and highlight:ax.add_patch(FancyBboxPatch((x-.32,y-.23),.64,.46,boxstyle='round,pad=.02',facecolor='#FBEBD9',edgecolor=ORANGE,linewidth=.7))
            text(ax,x,y,'—' if v is None else str(v),11,color=ORANGE if v==1 and highlight else INK)
    text(ax,.6,8.25,'A. 与该校验反对易的数据错误在 t=3 前出现，之后未再发生错误',10,ha='left',color=BLUE,weight='bold')
    row(7.5,'真实校验值 '+r'$s_t$',[0,0,0,1,1,1,1])
    row(6.75,'测得的读出 '+r'$m_t$',[0,0,0,1,1,1,1])
    row(6.0,'检测事件 '+r'$d_t$',[None,0,0,1,0,0,0],True)
    text(ax,11.6,5.3,'读出持续翻转，但差分只在变化的时刻出现。',9,color=BLUE)
    ax.plot([.6,17.4],[4.85,4.85],color='#D3DEE4',linewidth=1)
    text(ax,.6,4.25,'B. 数据没有出错，只在 t=3 把该校验的读出记反',10,ha='left',color=ORANGE,weight='bold')
    row(3.5,'真实校验值 '+r'$s_t$',[0,0,0,0,0,0,0])
    row(2.75,'测得的读出 '+r'$m_t$',[0,0,0,1,0,0,0])
    row(2.0,'检测事件 '+r'$d_t$',[None,0,0,1,1,0,0],True)
    text(ax,11.6,1.27,'单次读出错误会在相邻两个差分位置留下痕迹。',9,color=ORANGE)
    text(ax,9,.4,'仅展示一个校验的理想化时间轨迹；实际电路故障可影响多个校验，需结合时空记录推断。',8,color=GRAY)
    save(fig,'syndrome_time_history')

def parity_circuit():
    fig,ax=canvas(8.7)
    text(ax,9,8.25,'syndrome 怎么测？用辅助比特提取奇偶性',12,weight='bold')
    text(ax,9,7.45,'示例：测量 '+r'$Z_1Z_2$'+'，只询问两个数据比特的奇偶，不逐个读出它们。',10)
    for y,label in [(6.0,'数据比特 '+r'$q_1$'),(4.5,'数据比特 '+r'$q_2$'),(3.0,'辅助比特 '+r'$|0\rangle$')]:
        text(ax,.7,y,label,10,ha='left');ax.plot([4.3,15.2],[y,y],color=INK,linewidth=1.1)
    for x,y in [(7,6),(10,4.5)]:
        ax.plot([x,x],[3,y],color=BLUE,linewidth=1.2)
        ax.add_patch(Circle((x,y),.10,facecolor=BLUE))
        ax.add_patch(Circle((x,3),.20,facecolor='white',edgecolor=BLUE,linewidth=1.1));ax.plot([x-.20,x+.20],[3,3],color=BLUE);ax.plot([x,x],[2.8,3.2],color=BLUE)
    box(ax,13.3,3,1.3,.75,r'$M_Z$',size=12)
    text(ax,16.1,3,r'$m\in\{0,1\}$',11)
    text(ax,8.5,2.1,r'$m=0:\ +1$'+' 特征值（偶校验）    '+r'$m=1:\ -1$'+' 特征值（奇校验）',10)
    text(ax,9,1.3,'X 型校验可用 '+r'$|+\rangle$'+' 辅助比特作控制，逐一 CNOT 后做 X 基测量。',9)
    text(ax,9,.45,'这是权重 2 的理想教学电路；BB 的六体校验按支持集推广，报告使用原公开实现的完整含噪调度。',8,color=GRAY)
    save(fig,'stabilizer_parity_measurement')

def main():
    record=bb();css();parity_circuit();syndrome()
    (OUT/'figure_notes.json').write_text(json.dumps(dict(BB=record,CSS='Conceptual physical Pauli representation; not DEM coordinates',syndrome='Single-check conceptual traces, not sampled experimental data',model='96-dimensional, four-head recurrent MPT with shared-fault edges and mixed-score absolute-LLR CS8'),indent=2))
    print('FOUR_VECTOR_FIGURES_CREATED',OUT)

if __name__=='__main__':main()
