
import argparse,json
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--h5',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
with np.load(a.split) as z:rows=z['test_rows'].astype(int);stim=z['test_stim_ids']
with h5py.File(a.h5) as f:
 if not np.array_equal(np.array([f['stim_id'][i] for i in rows]),stim):raise ValueError('Test row and stimulus identity mismatch')
 _,h,w,c=f['labels'].shape
 if c!=3:raise ValueError('Expected RGB reference images')
 images=np.lib.format.open_memmap(a.output,mode='w+',dtype=np.float32,shape=(len(rows),3,h,w))
 for j,i in enumerate(rows):images[j]=f['labels'][i].transpose(2,0,1).astype(np.float32)/255.
 images.flush()
a.output.with_suffix('.json').write_text(json.dumps(dict(h5=str(a.h5),split=str(a.split),rows=rows.tolist(),stimulus_ids=stim.tolist(),normalization='uint8 RGB / 255, NCHW'),indent=2))
print('Exported',len(rows),'matched reference images')
