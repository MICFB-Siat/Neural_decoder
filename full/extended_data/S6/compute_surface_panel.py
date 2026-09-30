
from pathlib import Path
import argparse,sys,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parent
CODE=ROOT/'_source/zhf_exp/zhf_exp_fmri_eigen/code'
sys.path.insert(0,str(CODE))
import eigenmode_common as ec

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/A');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 matrices={};indices={};eigenvalues={}
 for name,surface,mask,kind in [('template',ec.TEMPLATE_VTK,ec.TEMPLATE_MASK,'vtk'),('sub-22',ec.SUB22_SURF,ec.SUB22_MASK,'gifti'),('sub-24',ec.SUB24_SURF,ec.SUB24_MASK,'gifti')]:
  coords,faces=ec.load_vtk_surface(surface) if kind=='vtk' else ec.load_gifti_surface(surface)
  selected=ec.load_text_mask(mask) if kind=='vtk' else ec.load_gifti_mask(mask)
  c,f,idx=ec.mask_surface(coords,faces,selected)
  values,modes=ec.compute_lb_modes_with_mode0(c,f,5)
  full=ec.expand_to_full(modes,idx,len(coords));matrices[name]=full;indices[name]=idx;eigenvalues[name]=values.tolist()
  np.savez_compressed(a.output/(name+'_fresh_modes.npz'),eigenvalues=values,modes=full,keep_indices=idx)
  print(name,values,flush=True)
 spec=importlib.util.spec_from_file_location('render_surface_modes',CODE/'05_render_template_sub22_sub24_mode0_4.py');render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
 template=matrices['template'];rows=[]
 for name,surface in [('template',ec.TEMPLATE_RENDER_SURF),('sub-22',ec.SUB22_SURF),('sub-24',ec.SUB24_SURF)]:
  modes=matrices[name] if name=='template' else ec.align_modes_to_reference(matrices[name],template,indices['template'])
  modes=ec.force_mode0_blue(modes)
  rows.append(render.render_row(name,surface,modes))
 render.build_grid(rows,a.output/'S6A.png')
 (a.output/'provenance.json').write_text(json.dumps(dict(computation='Fresh first five Laplace-Beltrami eigenmodes from three surface meshes; no saved modes read',n_modes=5,eigenvalues=eigenvalues),indent=2))
if __name__=='__main__':main()
