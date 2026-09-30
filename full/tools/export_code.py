




from pathlib import Path
import argparse,tarfile,csv,os
ROOT=Path(__file__).resolve().parents[1]
RESOURCE_SUFFIXES={'.pt','.pth','.ckpt','.safetensors','.bin','.h5','.hdf5','.npy','.npz','.mat','.pkl','.pickle','.gii','.vtk','.nii','.gz','.zip','.png','.pdf','.svg','.tiff','.tif','.jpg','.jpeg'}
SKIP={'__pycache__','.git','.pytest_cache','outputs','output','retained_results'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.resolve().is_relative_to(ROOT):raise ValueError('Archive destination must be outside the release directory')
 resources=[];added=0
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(a.output,'w:gz',dereference=False) as archive:
  for parent,dirs,files in os.walk(ROOT,followlinks=False):
   relparent=Path(parent).relative_to(ROOT)


   archive.add(parent,arcname=str(Path(ROOT.name)/relparent),recursive=False)
   for name in list(dirs):
    target=Path(parent)/name
    if name not in SKIP and target.is_symlink() and target.resolve().is_relative_to(ROOT):
     archive.add(target,arcname=str(Path(ROOT.name)/target.relative_to(ROOT)),recursive=False)
   dirs[:]=[d for d in dirs if d not in SKIP and not (Path(parent)/d).is_symlink()]
   for name in sorted(files):
    path=Path(parent)/name;rel=path.relative_to(ROOT)
    if path.is_symlink() and not path.resolve().is_relative_to(ROOT):continue
    if name.endswith('.strace'):continue
    if path.suffix.lower() in RESOURCE_SUFFIXES:
     resources.append(dict(path=str(rel),bytes=path.stat().st_size,kind='numeric_or_visual_resource'));continue
    archive.add(path,arcname=str(Path(ROOT.name)/rel),recursive=False);added+=1
 with a.output.with_suffix('.resources.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['path','bytes','kind']);w.writeheader();w.writerows(resources)
 print('Code files:',added,'separate resources:',len(resources),'archive:',a.output)
if __name__=='__main__':main()
