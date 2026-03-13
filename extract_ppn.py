import os
import re
import shutil
import glob

def extract_lines(filepath, start_line, end_line):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    return ''.join(lines[start_line-1:end_line])

def main():
    out_dir = r"F:\Black Mesa Research Facility\Main Facility\Physics department\SCT Theory\theory\derivations"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "PPN1_literature.tex")

    will_dir = r"C:\Users\youre\AppData\Local\Temp\arxiv_sources\1403.7377"
    edholm_dir = r"C:\Users\youre\AppData\Local\Temp\arxiv_sources\1604.01989"
    biswas_dir = r"C:\Users\youre\AppData\Local\Temp\arxiv_sources\0508194"

    # Copy all images and sty files
    for d in [will_dir, edholm_dir, biswas_dir]:
        for ext in ['**/*.png', '**/*.PNG', '**/*.eps', '**/*.pdf', '**/*.jpg', '**/*.sty', '**/*.bst']:
            for img in glob.glob(os.path.join(d, ext), recursive=True):
                if os.path.isfile(img):
                    shutil.copy(img, out_dir)
                    
    will_tex = os.path.join(will_dir, "article.tex")
    will_bbl = os.path.join(will_dir, "article.bbl")
    edholm_tex = os.path.join(edholm_dir, "Universality-finalarxiv.tex")
    biswas_tex = os.path.join(biswas_dir, "curvaturef.tex")

    will_text = extract_lines(will_tex, 1823, 4119)
    will_text = will_text.replace('PLOTS/', '')
    will_text = re.sub(r'\\includegraphics', r'% \\includegraphics', will_text)
    edholm_text = extract_lines(edholm_tex, 218, 468)
    
    with open(biswas_tex, 'r', encoding='utf-8', errors='ignore') as f:
        biswas_lines = f.readlines()
    biswas_text1 = ''.join(biswas_lines[746:816])
    biswas_text2 = ''.join(biswas_lines[1805:1890])

    # Replace conflicting macros in Edholm text
    edholm_text = re.sub(r'\\da\b', r'\\delta', edholm_text)
    edholm_text = re.sub(r'\\e\b', r'\\overline', edholm_text)
    
    # Replace conflicting macros in Biswas text
    biswas_text1 = re.sub(r'\\da\b', r'^{\\dagger}', biswas_text1)
    biswas_text2 = re.sub(r'\\da\b', r'^{\\dagger}', biswas_text2)
    biswas_text1 = re.sub(r'\\e\b', r'\\epsilon', biswas_text1)
    biswas_text2 = re.sub(r'\\e\b', r'\\epsilon', biswas_text2)
    biswas_text1 = re.sub(r'\\s\b', r'\\sigma', biswas_text1)
    biswas_text2 = re.sub(r'\\s\b', r'\\sigma', biswas_text2)

    preamble = r"""\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,amsfonts,color}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\usepackage{epubtk}
\providecommand{\cl}[1]{{\centerline{#1}}}
\providecommand{\nat}{Nature}
\providecommand{\apjl}{Astrophys. J. Lett.}
\providecommand{\apj}{Astrophys. J.}
\providecommand{\mnras}{Mon. Not. R. Astron. Soc.}
\providecommand{\prd}{Phys. Rev. D}
\providecommand{\pra}{Phys. Rev. A}
\providecommand{\prc}{Phys. Rev. C}
\providecommand{\na}{New Astron.}
\providecommand{\aap}{Astron. Astrophys.}
\DeclareMathOperator{\diag}{diag}
\newcounter{boxcounter}
\setcounter{boxcounter}{0}
\newenvironment{mybox}[1]
{\refstepcounter{boxcounter}\vspace{2 em}\noindent\hrulefill\begin{center}{\bf Box~\theboxcounter. #1}\end{center}\hrulefill}
{\hrulefill\vspace{1 em}}
\newenvironment{widetext}{}{}

\providecommand{\lsim}{\mbox{\raisebox{-.6ex}{~$\stackrel{<}{\sim}$~}}}
\providecommand{\gsim}{\mbox{\raisebox{-.6ex}{~$\stackrel{>}{\sim}$~}}}
\providecommand{\col}{\textcolor{blue}}
\providecommand{\ba}{\begin{eqnarray}}
\providecommand{\ea}{\end{eqnarray}}
\providecommand{\be}{\begin{equation}}
\providecommand{\ee}{\end{equation}}
\providecommand{\bi}{\begin{itemize}}
\providecommand{\ei}{\end{itemize}}
\providecommand{\al}{\alpha}
\providecommand{\bt}{\beta}
\providecommand{\ga}{\gamma}
\providecommand{\ta}{\theta}
\providecommand{\va}{\vartheta}
\providecommand{\vi}{\varphi}
\providecommand{\la}{\lambda}
\providecommand{\ka}{\kappa}
\providecommand{\za}{\zeta}
\providecommand{\sa}{\sigma}
\providecommand{\en}{\epsilon}
\providecommand{\oa}{\omega}
\providecommand{\Ga}{\Gamma}
\providecommand{\Ta}{\Theta}
\providecommand{\Da}{\Delta}
\providecommand{\Oa}{\Omega}
\providecommand{\La}{\Lambda}
\providecommand{\cE}{{\cal E}}
\providecommand{\cH}{{\cal H}}
\providecommand{\cF}{{\cal F}}
\providecommand{\cP}{{\cal P}}
\providecommand{\cQ}{{\cal Q}}
\providecommand{\cR}{{\cal R}}
\providecommand{\cO}{{\cal O}}
\providecommand{\cW}{{\cal W}}
\providecommand{\cK}{{\cal K}}
\providecommand{\cN}{{\cal N}}
\providecommand{\cL}{{\cal L}}
\providecommand{\cS}{{\cal S}}
\providecommand{\cT}{{\cal T}}
\providecommand{\+}{^{\dagger}}
\providecommand{\w}{\widetilde}
\providecommand{\x}{\star}
\providecommand{\st}{\stackrel}
\providecommand{\s}{\st{\textvisiblespace}}
\providecommand{\mstar}{M_{\star}}
\providecommand{\hp}{h^{\perp}}
\providecommand{\Ap}{A^{\perp}}
\providecommand{\n}{\nabla}
\providecommand{\W}{\wedge}
\providecommand{\ra}{\rightarrow}
\providecommand{\Ra}{\Rightarrow}
\providecommand{\im}{\Longleftrightarrow}
\providecommand{\LF}{\left(}
\providecommand{\RF}{\right)}
\providecommand{\LT}{\left[}
\providecommand{\RT}{\right]}
\providecommand{\Ld}{\left.}
\providecommand{\Rd}{\right.}
\providecommand{\ah}{\widehat{a}}
\providecommand{\bh}{\widehat{b}}
\providecommand{\ch}{\widehat{c}}
\providecommand{\dht}{\widehat{d}}
\providecommand{\eh}{\widehat{e}}
\providecommand{\gh}{\widehat{g}}
\providecommand{\ph}{\widehat{p}}
\providecommand{\qh}{\widehat{q}}
\providecommand{\mh}{\widehat{m}}
\providecommand{\nh}{\widehat{n}}
\providecommand{\Dh}{\widehat{D}}
\providecommand{\Mw}{\w{M}}
\providecommand{\rw}{\w{r}}
\providecommand{\kw}{\w{k}}
\providecommand{\cw}{\w{c}}
\providecommand{\Gw}{\w{G}}
\providecommand{\hb}{\overline{h}}
\providecommand{\gb}{\overline{g}}
\providecommand{\as}{\s{a}}
\providecommand{\2}{\frac{1}{2}}
\providecommand{\3}{\frac{1}{3}}
\providecommand{\4}{\frac{1}{4}}
\providecommand{\6}{\frac{1}{6}}
\providecommand{\8}{\frac{1}{8}}
\providecommand{\stwo}{\sqrt{2}}
\providecommand{\sthree}{\sqrt{3}}
\providecommand{\mx}{\mbox}
\providecommand{\mt}{\mathtt}
\providecommand{\mand}{\mx{ and }}
\providecommand{\for}{\mx{ for }}
\providecommand{\where}{\mx{ where }}
\providecommand{\with}{\mx{ with }}
\providecommand{\eff}{\mt{eff}}
\providecommand{\tot}{\mt{tot}}
\providecommand{\mtR}{\mt{R}}
\providecommand{\mtS}{\mt{S}}
\providecommand{\mtW}{\mt{W}}
\providecommand{\mtC}{\mt{C}}
\providecommand{\ie}{{\it i.e.\ }}
\providecommand{\hs}{\hspace{5mm}}
\providecommand{\vs}{\vspace{0mm}\\}
\providecommand{\non}{\nonumber\\}
\providecommand{\Dc}{\mathcal{D}}
\providecommand{\Fc}{\mathcal{F}}
\providecommand{\Gc}{\mathcal{G}}
\providecommand{\Lc}{\mathcal{L}}
\providecommand{\Mc}{\mathcal{M}}
\providecommand{\pd}{\partial}
\providecommand{\cpd}{\nabla}
\providecommand{\D}{\nabla}
\providecommand{\Rc}{\mathcal{R}}
\providecommand{\hco}{\widehat{\mathcal{O}}}
\providecommand{\p}{\partial}

\providecommand{\beqy}{\begin{eqnarray}}
\providecommand{\eeqy}{\end{eqnarray}}
\providecommand{\ov}{\overline}
\providecommand{\mb}{\mbox}
\providecommand{\dt}{\mathtt{d}}
\providecommand{\bb}{\beta}
\providecommand{\te}{\theta}
\providecommand{\Te}{\Theta}
\providecommand{\de}{\delta}
\providecommand{\De}{\Delta}
\providecommand{\et}{\tilde{e}}
\providecommand{\ze}{\zeta}
\providecommand{\om}{\omega}
\providecommand{\Om}{\Omega}
\providecommand{\hn}{\widehat{\nabla}}
\providecommand{\hph}{\widehat{\phi}}
\providecommand{\ddh}{\widehat{d}}
\providecommand{\stu}{\st{\textvisiblespace}}
\providecommand{\au}{\stu{a}}
\providecommand{\bu}{\stu{b}}
\providecommand{\cu}{\stu{c}}
\providecommand{\du}{\stu{d}}
\providecommand{\eu}{\stu{e}}
\providecommand{\mmu}{\stu{m}}
\providecommand{\nnu}{\stu{n}}
\providecommand{\pu}{\stu{p}}
\providecommand{\Du}{\stu{D}}
\providecommand{\sto}{\st{\circ}}
\providecommand{\az}{\st{c}{a}}
\providecommand{\bz}{\st{c}{b}}
\providecommand{\cz}{\st{c}{c}}
\providecommand{\dz}{\st{c}{d}}
\providecommand{\Dz}{\st{c}{D}}
\providecommand{\ez}{\st{c}{e}}
\providecommand{\fz}{\st{c}{f}}
\providecommand{\nz}{\st{c}{n}}
\providecommand{\mz}{\st{c}{m}}
\providecommand{\tb}{\overline{\theta}}
\providecommand{\ti}{\widetilde}
\providecommand{\sqw}{\sqrt{w\over 2}\ }
\providecommand{\Delt}{\p^{\star}}
\providecommand{\hepth}[1]{{\tt hep-th/#1}}
\providecommand{\hepph}[1]{{\tt hep-ph/#1}}
\providecommand{\grqc}[1]{{\tt gr-qc/#1}}
\providecommand{\astroph}[1]{{\tt astro-ph/#1}}
\providecommand{\newjournal}[5]{#1 \textbf{#3}, #5 (#4)}
\providecommand{\jhep}[3]{JHEP \textbf{#2}, #3}

\title{Comprehensive Literature Review: PPN Formalism for Nonlocal Gravity}
\author{David Alfyorov}
\date{\today}

\begin{document}
\maketitle

\tableofcontents
\newpage

\section{Introduction}
This document aggregates the complete Parameterized Post-Newtonian (PPN) formalism framework, metric potentials, and derivations for nonlocal gravity theories from key literature. The primary sources are Will (2014) for the standard 10-parameter PPN framework, Edholm et al. (2016) for the universality of the Newtonian potential in ghost-free gravity, and Biswas et al. (2006) for the bouncing cosmology and Newtonian limit in string-inspired nonlocal gravity.

\section{Standard PPN Framework (extracted from Will 2014)}
"""

    mid1 = r"""
\newpage
\section{Newtonian Potential in Ghost-Free Gravity (extracted from Edholm et al. 2016)}
"""

    mid2 = r"""
\newpage
\section{Newtonian Limit in Nonlocal Gravity (extracted from Biswas et al. 2006)}
"""

    end = r"""
\newpage
\section{Comparison and Gaps}
\begin{itemize}
\item \textbf{Will (2014)} provides the definitive framework for the 10 PPN parameters ($\gamma, \beta, \xi, \alpha_1, \alpha_2, \alpha_3, \zeta_1, \zeta_2, \zeta_3, \zeta_4$). It covers standard metric theories (scalar-tensor, vector-tensor, etc.) but does not explicitly address infinite-derivative (nonlocal) gravity.
\item \textbf{Edholm et al. (2016)} proves that for a wide class of infinite-derivative, ghost-free gravity theories, the Newtonian potential exactly matches GR in the infrared (large distances) but is regularized at the origin. However, they mainly focus on the linear regime and the $\Phi$ potential, not the full 10-parameter nonlinear PPN expansion.
\item \textbf{Biswas et al. (2006)} computes the Newtonian limit and shows asymptotic freedom at short distances. They show how the propagator maps to the modified Newtonian potential. 
\item \textbf{Gap:} A complete nonlinear PPN analysis (extracting $\beta$ and other nonlinear parameters) for covariant infinite-derivative gravity is not fully synthesized in these works. SCT Theory must compute the full PPN parameters, especially focusing on $\gamma$ and $\beta$, by expanding the full nonlocal field equations to $\mathcal{O}(c^{-4})$.
\end{itemize}

"""

    with open(will_bbl, 'r', encoding='utf-8', errors='ignore') as f:
        bbl_text = f.read()

    doc = preamble + will_text + mid1 + edholm_text + mid2 + biswas_text1 + "\n\n" + biswas_text2 + end + bbl_text + r"\end{document}"

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(doc)
    
    print(f"Written {out_file}")

if __name__ == "__main__":
    main()
