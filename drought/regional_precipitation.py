"""Regional annual precipitation comparisons for Figure S11."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .style import save_figure


def plot_regional_precipitation(ctx):
    with ctx.path('regional_precipitation', 'regional_precipitation.csv').open(encoding='utf-8-sig',newline='') as stream:
        allrows = list(csv.DictReader(stream))
    required = {'region','year','GHCD','GPCC','CRU'}
    if not allrows or not required.issubset(allrows[0]):
        raise ValueError('Required columns: region, year, GHCD, GPCC, CRU')
    if set(row['region'] for row in allrows) != {'China','Africa','Africa_no_support'}:
        raise ValueError('Expected China, Africa, Africa_no_support regions')
    allrows.sort(key=lambda row: (row['region'], int(row['year'])))
    china=[r for r in allrows if r['region']=='China']
    africa_all=[r for r in allrows if r['region']=='Africa']
    africa=[r for r in allrows if r['region']=='Africa_no_support']
    recent=[r for r in china if int(r['year'])>=2015]
    panels=[('a',china,'GPCC','China, 1973–2024'),('b',recent,'GPCC','China, 2015–2024'),('c',africa_all,'CRU','Africa, 1973–2024'),('d',africa,'CRU','Africa: subset without CRU station support')]
    for label, rows, reference, title in panels:
        years = [int(row['year']) for row in rows]
        if years != list(range(2015 if label == 'b' else 1973, 2025)):
            raise ValueError(f'Unexpected or duplicate years in panel {label}')
        for product in ['GHCD', reference]:
            values = np.array([float(row[product]) for row in rows])
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Annual precipitation must be finite and nonnegative')
    if ctx.check_inputs:
        print('S11: annual precipitation inputs checked for four panels.')
        return
    source=[]
    plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':7.5,
       'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
       'axes.linewidth':.6,'legend.frameon':False,'xtick.labelsize':6,'ytick.labelsize':6})
    colors={'GHCD':'#0072B2','GPCC':'#CC79A7','CRU':'#009E73'}
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.8))
    fig.subplots_adjust(left=.105,right=.98,bottom=.10,top=.85,wspace=.30,hspace=.48)
    for ax,(label,rr,reference,title) in zip(axs.flat,panels):
        years=np.array([int(r['year']) for r in rr])
        for product in ['GHCD',reference]:
            key=product
            values=np.array([float(r[key]) for r in rr]);assert np.isfinite(values).all()
            ax.plot(years,values,color=colors[product],lw=1.25,label=product)
            for y,v in zip(years,values):source.append({'panel':label,'year':int(y),'product':product,'precipitation_mm_year':float(v),'weighting':'grid-cell area normalized within fixed region; Forcing mask2 == 0'})
        ax.set_title(title,pad=10);ax.set_ylabel('Precipitation (mm/year)');ax.set_xlabel('Year');ax.grid(False)
        ax.set_xlim(years[0],years[-1]);ax.set_xticks([2015,2018,2021,2024] if label=='b' else [1980,1990,2000,2010,2020])
        ax.text(-.15,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=9)
        if label in 'ab':ax.set_ylim(600,1250);ax.set_yticks([600,800,1000,1200])
        if label=='d':
            ax.axvspan(2015,2024,color='.92',zorder=0)
            ax.text(.02,.04,'No CRU station support, 2015–2024',transform=ax.transAxes,fontsize=6,va='bottom')
    handles=[Line2D([0],[0],color=colors[n],lw=1.5,label=n) for n in ['GHCD','GPCC','CRU']]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.99),ncol=3,fontsize=7)
    save_figure(fig,ctx)
