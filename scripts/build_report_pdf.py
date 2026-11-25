"""Build a Chinese XeLaTeX technical report from the maintained Markdown."""
from pathlib import Path
import re,subprocess,shutil

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'report'
OUT=ROOT/'output/pdf'
BUILD=ROOT/'tmp/pdfs/build'

SPECIAL={c:'\\'+c for c in '&%$#_{}'}
SPECIAL.update({'\\':r'\textbackslash{}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'})
UNICODE={'∈':r'\(\in\)','F₂ᴺ':r'\(\mathbb F_2^N\)','φ':r'\(\phi\)','θ':r'\(\theta\)','Δ':r'\(\Delta\)','λᵢ':r'\(\lambda_i\)','qᵢ':r'\(q_i\)','zᵢ':r'\(z_i\)','dᵦ':r'\(d_b\)','Σ':r'\(\Sigma\)','∑':r'\(\sum\)','⁻⁴':r'\(^{-4}\)','⁻⁵':r'\(^{-5}\)','⁶':r'\(^{6}\)','ₚ':r'\(_p\)','ᵢ':r'\(_i\)','⊕':r'\(\oplus\)','−':'-','—':'--','–':'--','≥':r'\(\ge\)','≤':r'\(\le\)','→':r'\(\to\)','×':r'\(\times\)','±':r'\(\pm\)','∩':r'\(\cap\)','ₜ':r'\(_t\)'}

def escape(text):
    tokens=[]
    for literal,latex in [(r'\hat E',r'\(\hat E\)'),('H_P',r'\(H_P\)'),('L_P',r'\(L_P\)')]:
        if literal in text:
            text=text.replace(literal,'@@U'+str(len(tokens))+'@@');tokens.append(latex)
    for k,v in sorted(UNICODE.items(),key=lambda v:-len(v[0])):
        text=text.replace(k,'@@U'+str(len(tokens))+'@@') if k in text else text
        tokens.append(v)
    text=''.join(SPECIAL.get(c,c) for c in text)
    for i,v in enumerate(tokens):text=text.replace('@@U'+str(i)+'@@',v)
    return text

def inline(text):
    pattern=re.compile(r'\\\(.*?\\\)|\[([^\]]+)\]\(([^)]+)\)|\*\*(.+?)\*\*|`([^`]+)`')
    parts=[];start=0
    for m in pattern.finditer(text):
        parts.append(escape(text[start:m.start()]))
        if m.group(0).startswith(r'\('):parts.append(m.group(0))
        elif m.group(1) is not None:parts.append(r'\href{'+m.group(2).replace('%',r'\%').replace('#',r'\#')+'}{'+inline(m.group(1))+'}')
        elif m.group(3) is not None:parts.append(r'\textbf{'+inline(m.group(3))+'}')
        else:parts.append(r'\texttt{'+escape(m.group(4))+'}')
        start=m.end()
    parts.append(escape(text[start:]));return ''.join(parts)

def convert(source):
    lines=source.splitlines();out=[];i=0
    # Update prose to incorporate the completed sum-product baseline directly,
    # rather than printing a conversational "baseline update" notice.
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# ') or line.startswith('作者：') or line.startswith('项目代码：') or line.startswith('**基线更新说明'):i+=1;continue
        if line.startswith('## 实验材料索引'):break
        if line==r'\[':
            block=[line];i+=1
            while i<len(lines):
                block.append(lines[i])
                if lines[i].strip()==r'\]':break
                i+=1
            out.append('\n'.join(block));i+=1;continue
        if line.startswith('## '):
            heading=re.sub(r'^\d+\s+','',line[3:]);command=r'\section*' if heading in ['摘要','参考文献与代码来源'] else r'\section'
            out.append(r'\FloatBarrier'+(r'\Needspace{70mm}' if heading=='结论' else '')+command+'{'+inline(heading)+'}');i+=1;continue
        if line.startswith('### '):
            heading=re.sub(r'^\d+\.\d+\s+','',line[4:]);out.append(r'\subsection{'+inline(heading)+'}');i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\-]+',c) for c in cells):rows.append(cells)
                i+=1
            n=len(rows[0]);colwidth=f'{(160-3.2*n)/n:.2f}mm'
            columns='@{}'+''.join(r'>{\centering\arraybackslash}p{'+colwidth+'}' for _ in range(n))+'@{}'
            head=' & '.join(r'\textbf{'+inline(v)+'}' for v in rows[0])+r' \\'
            placement='H' if any('BP+CS8' in v for v in rows[0]) else 'htbp'
            out.append(r'\begin{table}['+placement+r']\centering\small\setlength{\tabcolsep}{1.6mm}\renewcommand{\arraystretch}{1.25}\begin{tabular}{'+columns+'}\n'+r'\toprule'+'\n'+head+'\n'+r'\midrule'+'\n'+'\n'.join(' & '.join(inline(v) for v in row)+r' \\' for row in rows[1:])+'\n'+r'\bottomrule\end{tabular}\end{table}')
            continue
        image=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if image:
            # Use vector figures where available.
            path=REPORT/image.group(2);vector=path.with_suffix('.pdf')
            path=vector if vector.exists() else path
            placement='H' if path.name.startswith('model_') else 'htbp'
            out.append(r'\begin{figure}['+placement+r']\centering\includegraphics[width=\linewidth,height=.32\textheight,keepaspectratio]{'+path.relative_to(REPORT).as_posix()+r'}\caption{'+inline(image.group(1))+'}'+r'\end{figure}')
            i+=1;continue
        if re.match(r'^\d+\. ',line):
            out.append(r'\noindent '+inline(line)+r'\par\smallskip');i+=1;continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||!\[|\\\[)',lines[i].strip()):para.append(lines[i].strip());i+=1
        text=' '.join(para)
        out.append(inline(text)+'\n');
    return '\n\n'.join(out)

PREAMBLE=r'''\documentclass[UTF8,a4paper,zihao=-4,fontset=fandol]{ctexart}
\usepackage[left=25mm,right=25mm,top=23mm,bottom=24mm]{geometry}
\usepackage{amsmath,amssymb,bm,graphicx,booktabs,longtable,array}
\usepackage{placeins,float,fancyhdr,xcolor,hyperref,microtype,needspace}
\definecolor{accent}{HTML}{194D68}
\hypersetup{colorlinks=true,linkcolor=accent,urlcolor=accent,pdftitle={循环MPT与CS8融合的BB码量子纠错解码技术报告},pdfauthor={钟路安}}
\setlength{\parindent}{2em}\setlength{\parskip}{0.2em}
\linespread{1.11}\setlength{\emergencystretch}{3em}
\ctexset{section={format=\Large\bfseries\color{accent}},subsection={format=\large\bfseries}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small 循环MPT与CS8融合量子纠错解码}\fancyhead[R]{\small 技术报告}\fancyfoot[C]{\thepage}
\setlength{\headheight}{15pt}\renewcommand{\headrulewidth}{0.3pt}
\allowdisplaybreaks[1]
\clubpenalty=10000\widowpenalty=10000
\begin{document}
\begin{center}
{\LARGE\bfseries 面向多轮含噪测量的循环交叉注意力\\[0.25em]与有序统计融合量子纠错解码}\par
\vspace{0.7em}{\large 钟路安}\par
西安交通大学数学与统计学院\par
\vspace{0.35em}{\small AI for Science 项目技术报告\quad 2026年11月26日}
\par\vspace{0.4em}{\small 项目代码：\url{https://github.com/user97845678641456/recurrent-mpt-qec}}
\end{center}
\vspace{0.5em}
'''

def main():
    OUT.mkdir(parents=True,exist_ok=True);BUILD.mkdir(parents=True,exist_ok=True)
    text=(REPORT/'technical_report.md').read_text(encoding='utf-8-sig')
    source=PREAMBLE+convert(text)+'\n\\end{document}\n'
    # The full report is distributed alongside its vector figure assets.
    tex=REPORT/'technical_report.tex';tex.write_text(source,encoding='utf-8')
    for _ in range(2):
        proc=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory='+str(BUILD),str(tex)],cwd=REPORT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (BUILD/'console.log').write_text(proc.stdout,encoding='utf-8')
        if proc.returncode:print(proc.stdout[-8000:]);raise SystemExit(proc.returncode)
    final=OUT/'recurrent_mpt_qec_technical_report.pdf';shutil.copy2(BUILD/'technical_report.pdf',final)
    print('PDF_CREATED',final)

if __name__=='__main__':main()
