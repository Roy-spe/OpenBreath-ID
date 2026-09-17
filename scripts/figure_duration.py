"""Two complementary views of saved duration results; no statistical reruns."""
import csv
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
CSV=ROOT/'results/table_duration_eer.csv'
TEX=ROOT/'figures/figure_duration_readable.tex'
DURATIONS=[30,60,120,180,240,300]
SCHEDULES=[('fixed_enroll',r'300 / \textit{d}','fixed enrollment','blue'),
           ('fixed_probe',r'\textit{d} / 60','fixed probe','orange'),
           ('matched',r'\textit{d} / \textit{d}','matched durations','green')]
HEIGHT=620
COLOR_LIMIT=2.1

def rows():
    with CSV.open(encoding='utf-8',newline='') as stream: data=list(csv.DictReader(stream))
    assert len(data)==15 and len({r['condition'] for r in data})==15
    return {(int(r['enrollment_seconds']),int(r['probe_seconds'])):r for r in data}

def pair(schedule,d):
    return (300,d) if schedule=='fixed_enroll' else ((d,60) if schedule=='fixed_probe' else (d,d))

def xcoord(d):return 120+(d-30)*568/270
def ycoord(eer):return 420-(eer-30)*290/12
def delta(row):return 100*(float(row['parameter_matched_stacked_mean_eer'])-float(row['bie_mean_eer']))
def cell_color(value):
    assert abs(value)<=COLOR_LIMIT
    return ('heatblue' if value>=0 else 'heatorange')+f'!{100*abs(value)/COLOR_LIMIT:.4f}!white'

PREAMBLE=r'''% Generated from saved duration CSV; presentation-only change.
% 18 schedule positions map to 15 unique conditions; duplicates are not independent evidence.
\documentclass[tikz,border=0pt]{standalone}
\usepackage{iftex}
\ifXeTeX
  \usepackage{newtxtext}
\else
  \renewcommand{\rmdefault}{ptm}
\fi
\definecolor{ink}{HTML}{18212B}
\definecolor{muted}{HTML}{55616E}
\definecolor{axisgray}{HTML}{76818C}
\definecolor{gridgray}{HTML}{E0E5EB}
\definecolor{bluecolor}{HTML}{0072B2}
\definecolor{orangecolor}{HTML}{D55E00}
\definecolor{greencolor}{HTML}{00805A}
\definecolor{heatblue}{HTML}{2166AC}
\definecolor{heatorange}{HTML}{B35806}
\definecolor{badgefill}{HTML}{E8F2FB}
\tikzset{
 textbase/.style={anchor=base,font=\fontsize{9.24bp}{11bp}\selectfont,text=ink},
 heading/.style={textbase,anchor=base west,font=\fontsize{10.08bp}{12bp}\selectfont\bfseries},
 badge/.style={heading,anchor=base,text=bluecolor},
 note/.style={textbase,text=muted},
 axis/.style={draw=axisgray,line width=0.448bp},
 grid/.style={draw=gridgray,line width=0.336bp},
 blue/.style={draw=bluecolor,line width=1.12bp},
 orange/.style={draw=orangecolor,line width=1.12bp,dash pattern=on 3.92bp off 2.24bp},
 green/.style={draw=greencolor,line width=1.12bp,dash pattern=on 1.12bp off 1.68bp},
 bluepoint/.style={draw=bluecolor,line width=0.896bp,fill=white},
 orangepoint/.style={draw=orangecolor,line width=0.896bp,fill=white},
 greenpoint/.style={draw=greencolor,line width=0.896bp,fill=white},
 primary/.style={draw=ink,line width=1.12bp}
}
\begin{document}
\begin{tikzpicture}[x=0.28bp,y=-0.28bp,every node/.style={inner sep=0pt,outer sep=0pt}]
\path[use as bounding box] (0,0) rectangle (1800,620);
\clip (0,0) rectangle (1800,620);
\fill[white] (0,0) rectangle (1800,620);
% y-range=30,42; x-range=30,300; linear axes
% Heatmap: delta=stacked-bie; symmetric color limits=-2.1,+2.1; categorical duration columns
\fill[badgefill,rounded corners=2.24bp] (76,14) rectangle (119,57);
\node[badge] at (97.5,47) {A};
\node[heading] at (133,47) {BIE duration trends};
\fill[badgefill,rounded corners=2.24bp] (804,14) rectangle (847,57);
\node[badge] at (825.5,47) {B};
\node[heading] at (862,47) {EER reduction vs. stacked CNN};
\node[note] at (400,93) {Enrollment / probe schedules};
\node[note] at (1372,93) {Varied duration \textit{d} (s)};
\node[textbase,rotate=90] at (35,278) {BIE EER (\%)};
\draw[axis] (100,130) -- (100,420) -- (710,420);'''

def marker(style,x,y):
    if style=='blue':return rf'\draw[bluepoint] ({x-8:.3f},{y-8:.3f}) rectangle ({x+8:.3f},{y+8:.3f});'
    if style=='orange':return rf'\draw[orangepoint] ({x:.3f},{y:.3f}) circle[radius=8];'
    return rf'\draw[greenpoint] ({x:.3f},{y-10:.3f}) -- ({x-9:.3f},{y+7:.3f}) -- ({x+9:.3f},{y+7:.3f}) -- cycle;'

def generate():
    data=rows();lines=[PREAMBLE]
    for tick in [30,34,38,42]:
        y=ycoord(tick)
        if tick!=30:lines.append(rf'\draw[grid] (100,{y:.3f}) -- (710,{y:.3f});')
        lines.append(rf'\node[textbase,anchor=base east] at (84,{y+10:.3f}) {{{tick}}};')
    for d in DURATIONS:
        x=xcoord(d)
        lines += [rf'\draw[axis] ({x:.3f},420) -- ({x:.3f},427);',
                  rf'\node[textbase] at ({x:.3f},458) {{{d}}};']
    lines.append(r'\node[textbase] at (400,500) {Varied duration \textit{d} (s)};')
    for index,(schedule,label,description,style) in enumerate(SCHEDULES):
        points=[(xcoord(d),ycoord(100*float(data[pair(schedule,d)]['bie_mean_eer']))) for d in DURATIONS]
        lines.append(r'\draw['+style+'] '+' -- '.join(f'({x:.3f},{y:.3f})' for x,y in points)+';')
        for d,(x,y) in zip(DURATIONS,points):
            row=data[pair(schedule,d)]
            lines += [f'% point {schedule} {row["condition"]} {d} {100*float(row["bie_mean_eer"]):.9f}',marker(style,x,y)]
        ly=530+33*index
        lines += [rf'\draw[{style}] (142,{ly}) -- (203,{ly});',marker(style,172,ly),
                  rf'\node[textbase,anchor=base west] at (226,{ly+10}) {{{label}: {description}}};']
        cy=151+84*index
        lines.append(rf'\node[textbase,anchor=base east] at (971,{cy+51}) {{{label}}};')
        for j,d in enumerate(DURATIONS):
            row=data[pair(schedule,d)];value=delta(row);primary=schedule=='fixed_enroll' and d in [30,60]
            x=1000+124*j
            if index==0:lines.append(rf'\node[textbase] at ({x+62},132) {{{d}}};')
            lines += [f'% cell {schedule} {row["condition"]} {d} {value:+.9f} primary={str(primary).lower()}',
                      rf'\fill[{cell_color(value)}] ({x},{cy}) rectangle ({x+124},{cy+84});',
                      rf'\draw[white,line width=0.56bp] ({x},{cy}) rectangle ({x+124},{cy+84});',
                      rf'\node[textbase,text={"white" if abs(value)>1.20 else "ink"}] at ({x+62},{cy+52}) {{{value:+.2f}}};']
            if primary:lines.append(rf'\draw[primary] ({x+4},{cy+4}) rectangle ({x+120},{cy+80});')
    lines += [r'\node[textbase] at (1372,449) {Stacked CNN minus BIE (pp)};']
    for k in range(100):
        value=-COLOR_LIMIT+(k+0.5)*2*COLOR_LIMIT/100
        x=1095+k*5.54
        lines.append(rf'\fill[{cell_color(value)}] ({x:.3f},470) rectangle ({x+5.56:.3f},493);')
    for label,x in [('-2.1',1095),('0',1372),('+2.1',1649)]:
        lines += [rf'\draw[axis] ({x},493) -- ({x},500);',rf'\node[textbase] at ({x},535) {{{label}}};']
    lines += [r'\node[note] at (1372,570) {Positive favors BIE};',
              r'\draw[primary] (951,581) rectangle (978,604);',
              r'\node[note,anchor=base west] at (997,603) {Primary endpoints; other cells descriptive};',
              r'\end{tikzpicture}',r'\end{document}']
    return '\n'.join(lines)+'\n'

def audit(source):
    data=rows();seen=set();point_count=cell_count=primary_count=0
    for schedule,condition,d,value,draw in re.findall(r'% point (\w+) (\w+) (\d+) ([\d.]+)\n([^\n]+)',source):
        d=int(d);row=data[pair(schedule,d)];assert row['condition']==condition
        expected=100*float(row['bie_mean_eer']);assert abs(float(value)-expected)<1e-8
        style=next(s[3] for s in SCHEDULES if s[0]==schedule)
        assert draw==marker(style,xcoord(d),ycoord(expected))
        point_count+=1;seen.add(condition)
    for schedule,condition,d,value,primary,fill,edge,label in re.findall(r'% cell (\w+) (\w+) (\d+) ([+-][\d.]+) primary=(true|false)\n([^\n]+)\n([^\n]+)\n([^\n]+)',source):
        d=int(d);row=data[pair(schedule,d)];assert row['condition']==condition
        expected=delta(row);assert abs(float(value)-expected)<1e-8
        assert '['+cell_color(expected)+']' in fill and '{'+f'{expected:+.2f}'+'}' in label
        assert primary==str(schedule=='fixed_enroll' and d in [30,60]).lower()
        cell_count+=1;primary_count+=primary=='true';seen.add(condition)
    assert point_count==cell_count==18 and len(seen)==15 and primary_count==2
    assert source==generate(),'Unexpected data, geometry, sign, color scale, label or primary outline change'
    return dict(saved_conditions=15,plotted_points=point_count,heatmap_cells=cell_count,primary_cells=primary_count,
        curves=3,shared_y_range=[30,42],durations=DURATIONS,linear_axes=True,heatmap_columns='categorical',
        delta_definition='stacked_minus_bie_percentage_points',color_limits=[-2.1,2.1],
        range_pp=[min(delta(r) for r in data.values()),max(delta(r) for r in data.values())],
        repeated_schedule_positions=3,significance_claims=False,new_statistics=False,native_times_text=True,
        source=CSV.relative_to(ROOT).as_posix())

if __name__=='__main__':print(generate(),end='')
