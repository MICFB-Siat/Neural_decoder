import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.runtime import TEMPLATES,output,style,save_figure
import argparse
import numpy as np
import nibabel as nib
import cortex
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.cm import ScalarMappable

def main():
    parser=argparse.ArgumentParser();parser.add_argument('panel',choices=['B','D']);args=parser.parse_args();out=output(args.panel)
    db=TEMPLATES/'pycortex_db'
    for folder in ['transforms','cache']:(db/'fsLR32k'/folder).mkdir(exist_ok=True)
    cortex.options.config.set('basic','filestore',str(db));cortex.db.filestore=str(db)
    mapping=np.load(TEMPLATES/'vertex_mapping.npz')['full_index']
    def full(values):
        result=np.full(64984,np.nan)
        result[mapping]=values
        return result
    z=np.load(out/'group.npz');arrays=[]
    if args.panel=='B':
        for key in ['high','low']:
            v=z[key].copy();v[v<.1]=np.nan;arrays.append(full(v))
        cmap=LinearSegmentedColormap.from_list('encoding',['#000000','#B00000','#FF0000','#FF8C00','#FFFF00'])
        maximum=.6;label='Pearson correlation';titles=['Cognitive-inference based encoding','Stimulus-response based encoding']
    else:
        for j in [1,0]:
            v=z['mean'][:,j].copy();v[~((v>.01)&(z['q'][:,j]<.05))]=np.nan;arrays.append(full(v))
        cmap=LinearSegmentedColormap.from_list('representational_geometry',plt.get_cmap('Reds')(np.linspace(.25,.85,256)),N=1024)
        maximum=float(z['mean'][:,:2].max());label='Spearman ρ';titles=['High-level','Low-level']
    colormaps.register(cmap);style();images=[]
    for values in arrays:
        vertex=cortex.Vertex(values,'fsLR32k',cmap=cmap.name,vmin=0,vmax=maximum)
        f=cortex.quickflat.make_figure(vertex,height=1400,dpi=300,with_curvature=True,
            curvature_brightness=.72 if args.panel=='D' else .55,curvature_contrast=.16 if args.panel=='D' else .35,
            curvature_threshold=args.panel=='D',with_rois=False,with_sulci=False,with_labels=False,
            with_borders=False,with_colorbar=False,sampler='nearest',nanmean=False)
        f.canvas.draw();images.append(np.asarray(f.canvas.buffer_rgba()).copy());plt.close(f)
    fig,axes=plt.subplots(2,1,figsize=(7.2,6.1));fig.subplots_adjust(left=.025,right=.87,top=.95,bottom=.035,hspace=.20)
    for ax,image,title in zip(axes,images,titles):ax.imshow(image);ax.axis('off');ax.set_title(title,fontsize=11,pad=8)
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,maximum),cmap=cmap),cax=fig.add_axes([.9,.30,.022,.40]));cb.set_label(label)
    save_figure(fig,out/('panel_'+args.panel))

if __name__=='__main__':main()
