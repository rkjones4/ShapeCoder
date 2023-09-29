import sys
import dill
import data
import prog_utils as pu
import matplotlib.pyplot as plt
import numpy as np
import os
import utils

MODE = '3D'

assert MODE == '3D'

sys.path.append('../s3d_lang')
import partnet_grammar as pg
import executor as ex            

# View results of a shapecoder run, by passing in last roun path, number of shapes to visualize for each abstraction, and where to save results
def main(sc_load_path, num, save_path):

    try:
        print("Trying to load pkl files")
        L = dill.load(open(f'{sc_load_path}/lib.pkl', 'rb'))        
        D = dill.load(open(f'{sc_load_path}/data.pkl', 'rb'))
        print("Success")
    except Exception as e:
        print("Failed to load pkl files, defaulting to text for 3D domain, this may fail")
        import library as lib
        domain = 's3d'        
        L = lib.makeLibrary(
            ex, domain,
        )
        D = utils.vis_load_lib_and_data(sc_load_path, L)
        
    vis_data = []
    
    for d in list(D):
        text = d.inf_prog
        for k,v in d.inf_vmap.items():
            text = text.replace(str(k), str(v))

        sub_progs = []        

        if 'Union' not in text:
            vis_data.append((text, [text], [p for _,p in d.prims]))
            continue

        o_text = text
        
        q = [text]
        
        while len(q) > 0:
            t = q.pop(0)
            fn, params = pu.split_func_expr(t)

            if fn != 'Union':
                sub_progs.append(t)
                continue
            
            if 'Union' not in params[0]:
                sub_progs.append(params[0])
            elif 'Union' in params[0]:
                q.append(params[0])

            if 'Union' not in params[1]:
                sub_progs.append(params[1])
                
            elif 'Union' in params[1]:
                q.append(params[1])

        vis_data.append((o_text, sub_progs, [p for _,p in d.prims]))

    os.system(f'mkdir {save_path}')

    for fn in L.abs_map.keys():
        print(f"Abs {fn}")
        saw = 0
        
        for i, (o_text, sub_progs, gt_prims) in enumerate(vis_data):

            saw += make_vis(
                L,
                f'{save_path}/vis_{fn}_{i}',
                o_text,
                sub_progs,
                gt_prims,
                fn,
            )

            if saw >= num:
                break

def getColor(cnt):
    L = ['r','b','g','y','orange','lightblue','lightgreen']
    return L[cnt]


def make_vis(
    L,
    name,
    text,
    sub_progs,
    gt_prims,
    match_name,
):    
    abs_prims = []
    colors = []

    cnt = 0

    info = [f'Abs {match_name} = {L.abs_map[match_name].exp_fn(L.abs_map[match_name].param_names)}']
    
    for sp in sub_progs:

        L.executor.run(sp)
        _prims = L.executor.getPrimitives(2)
        abs_prims += _prims

        if match_name is not None and match_name in sp:
            colors += [getColor(cnt) for _ in _prims]
            info.append(f'Color({getColor(cnt)}) = {sp}')
            cnt += 1
        else:
            colors += ['black' for _ in _prims]

    if cnt == 0:        
        return 0

    plt.close()
    
    fig, ax = plt.subplots(1, 2, figsize=(12,9),
                subplot_kw={'projection': '3d'}
    )

    extent = 1.0
    
    for i in range(2):
    
        ax[i].get_yaxis().set_visible(False)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_zaxis().set_visible(False)
        
        ax[i].set_xlim(-extent, extent)
        ax[i].set_ylim(extent, -extent)
        ax[i].set_zlim(-extent, extent)
            
        ax[i].axis('off')
            
        ax[i].set_proj_type('persp')
        ax[i].set_box_aspect(aspect = (1,1,1))
    
    import vis
    from scipy.spatial.transform import Rotation as R

    def do_split(txt, amt):    
        split_txt = ''
        while len(txt) > 0:
            split_txt += f'{txt[:amt]}\n'
            txt = txt[amt:]
        return split_txt
    
    ax[0].set_title('\n'.join([do_split(i, 84) for i in info]), fontsize=10)

    ax[0].text(
        -1.0, 0.0, -3.0, do_split(text, 64), wrap=True, horizontalalignment='left', fontsize=8, color='black')
    
    class makeCube:
        def __init__(self, cdata):

            W, H, D, X, Y, Z, XA, YA, ZA = cdata
            self.X = X
            self.Y = Y
            self.Z = Z
            self.W = W
            self.H = H
            self.D = D

            TM = R.from_euler('xyz', [XA, YA, ZA], degrees=False).as_matrix()

            RIGHTF = np.array([1.0, 0., 0.0])
            TOPF = np.array([0.0, 1.0, 0.0])
            FRONTF = np.array([0.0, 0., 1.0])
                        
            self.Xnorm = TM @ RIGHTF
            self.Ynorm = TM @ TOPF
            self.Znorm = TM @ FRONTF
            
        def getFaceNormals(self):
            return self.Xnorm, self.Ynorm, self.Znorm
                                            
    for g,c in zip(abs_prims, colors):        
        cube = makeCube(g)            
        vis.draw_box(ax[0], cube, c)

    for g in gt_prims:        
        cube = makeCube(g)            
        vis.draw_box(ax[1], cube, 'black')            
            
    if name is not None:
        plt.savefig(f'{name}.png')
        plt.close()
        plt.clf()
    else:
        plt.show()
            
    return 1
        
if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
