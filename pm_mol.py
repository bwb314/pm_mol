# PyMOL molecule class, useful for custom manipulations to molecules where 
# selections can be made with the PyMOL interface.
import numpy as np
from pymol import cmd 

symbols = ["X", "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG",
    "AL", "SI", "P", "S", "CL", "AR", "K", "CA", "SC", "TI", "V", "CR", "MN", "FE", "CO",
    "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR", "NB",
    "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "I", "XE", "CS",
    "BA", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM",
    "YB", "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI",
    "PO", "AT", "RN", "FR", "RA", "AC", "TH", "PA", "U", "NP", "PU", "AM", "CM", "BK",
    "CF", "ES", "FM", "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS", "RG",
    "UUB", "UUT", "UUQ", "UUP", "UUH", "UUS", "UUO"]


mass = [
    0., 1.00782503207, 4.00260325415, 7.016004548, 9.012182201, 11.009305406,
    12, 14.00307400478, 15.99491461956, 18.998403224, 19.99244017542,
    22.98976928087, 23.985041699, 26.981538627, 27.97692653246, 30.973761629,
    31.972070999, 34.968852682, 39.96238312251, 38.963706679, 39.962590983,
    44.955911909, 47.947946281, 50.943959507, 51.940507472, 54.938045141,
    55.934937475, 58.933195048, 57.935342907, 62.929597474, 63.929142222,
    68.925573587, 73.921177767, 74.921596478, 79.916521271, 78.918337087,
    85.910610729, 84.911789737, 87.905612124, 88.905848295, 89.904704416,
    92.906378058, 97.905408169, 98.906254747, 101.904349312, 102.905504292,
    105.903485715, 106.90509682, 113.90335854, 114.903878484, 119.902194676,
    120.903815686, 129.906224399, 126.904472681, 131.904153457, 132.905451932,
    137.905247237, 138.906353267, 139.905438706, 140.907652769, 141.907723297,
    144.912749023, 151.919732425, 152.921230339, 157.924103912, 158.925346757,
    163.929174751, 164.93032207, 165.930293061, 168.93421325, 173.938862089,
    174.940771819, 179.946549953, 180.947995763, 183.950931188, 186.955753109,
    191.96148069, 192.96292643, 194.964791134, 196.966568662, 201.970643011,
    204.974427541, 207.976652071, 208.980398734, 208.982430435, 210.987496271,
    222.017577738, 222.01755173, 228.031070292, 227.027752127, 232.038055325,
    231.03588399, 238.050788247, 237.048173444, 242.058742611, 243.06138108,
    247.07035354, 247.07030708, 251.079586788, 252.082978512, 257.095104724,
    258.098431319, 255.093241131, 260.105504, 263.112547, 255.107398, 259.114500,
    262.122892, 263.128558, 265.136151, 281.162061, 272.153615, 283.171792, 283.176451,
    285.183698, 287.191186, 292.199786, 291.206564, 293.214670]

el2mass = dict(zip(symbols,mass))



class pm_mol():

    def __init__(self, mol): 
    
        if type(mol) is str and mol.endswith('.xyz'):
            self.fil = mol
            with open(mol,'r') as ofil:
                n = int(next(ofil).split('\n')[0])
                next(ofil)
                els = []
                geom = np.array([]).reshape(0,3)
                for l in range(n):
                    lin = next(ofil)
                    ls = lin.split()
                    els.append(ls[0])
                    coords = [float(x) for x in ls[1:]]
                    geom = np.vstack([geom,coords])
                mol = [els,geom]
        self.mol = mol
        self.frags = []

    def print_out(self):
    
        els = self.mol[0]
        print(len(els))
        print("In Angstrom")
        coords = self.mol[1]
        for i in range(len(els)): 
            el = els[i]
            c = coords[i]
            print("%2s %7.3f %7.3f %7.3f" % (el,c[0],c[1],c[2]))
        print("")
    
    def copy(self):
        return copy.deepcopy(self)

    def write_out(self,fil_name):
    
        with open(fil_name,'w') as fil:
            els = self.mol[0]
            fil.write("%d\n" % (len(els)))
            fil.write("In Angstrom\n")
            coords = self.mol[1]
            for i in range(len(els)): 
                el = els[i]
                c = coords[i]
                fil.write("%2s %7.3f %7.3f %7.3f\n" % (el,c[0],c[1],c[2]))

    def bfs(self):
    
         els = self.mol[0]
         coords = self.mol[1]
         lc = len(coords)
         bonds = []
         #find all bonded atoms
         for i in range(1,lc):
             for j in range(i):
                 if not bound(els[i],coords[i],els[j],coords[j]): continue
                 bonds.append(set([i,j]))
         #Construct subsets such that each pair has a null intersection (BFS)
         minds = []
         while bonds != []:
             mol = bonds[0]
             bonds.remove(bonds[0])
             intersect = True
             while intersect: 
                 intersect = False
                 remove = []
                 for i in bonds:
                     if i & mol == set([]): continue
                     for j in i: mol.add(j)
                     intersect = True
                     remove.append(i)
                 for i in remove: bonds.remove(i)
             minds.append(mol)
         #build mol out of separated inds
         mols = []
         for mol in minds:
             mels = []
             mcoords = []
             for i in mol:
                 mels.append(els[i])
                 mcoords.append(coords[i])
             mols.append(pm_mol([mels,mcoords]))
         self.frags = mols
        
 
    def cut(self, frags):
        # remove atoms of other frags
        # bfs
        # find frag with desired atoms
        # extract and set equal to frag
        
        for frag in frags.keys():
            border_atoms = frags[frag]
            
            other_borders = []
            for frag2 in frags.keys():
                if frag == frag2: continue
                for border_atom in frags[frag2]: 
                    other_borders.append(border_atom)           
 
            copy_mol = self.copy() 
            
            for border in other_borders:
                copy_mol.mol[0][border] = 'XXX'
            
            new_copy_els = []
            new_copy_coords = []
            for i in range(len(copy_mol.mol[0])):
                if copy_mol.mol[0][i] == 'XXX': continue
                new_copy_els.append(copy_mol.mol[0][i]) 
                new_copy_coords.append(copy_mol.mol[1][i]) 
            
            new_copy = pm_mol([new_copy_els, new_copy_coords])
            new_copy.bfs()
            for border in border_atoms:
                target_coords = self.mol[1][border]
                found_frag = False
                for new_frag in new_copy.frags:
                    for coords in new_frag.mol[1]:
                        if not np.array_equal(target_coords, coords): continue
                        found_frag = new_frag
                        break
                    if found_frag != False: break
                if found_frag != False: break
            for atom_coords in found_frag.mol[1]:
                 for ind in range(len(self.mol[1])):
                    if not np.array_equal(self.mol[1][ind], atom_coords): continue
                    if ind not in frags[frag]: frags[frag].append(ind)
                    break         
       
        self.color_frags(frags)
        self.write_frags(frags)


    # Need to resolve naming of `frags`, already exists as deprecated member data from crystal class
    def write_frags(self, frags):

        with open('fA.dat', 'w') as fA:
            for frag in frags:
                classification = frag.split('_')[-1].upper()
                if classification != 'A': continue
                fA.write(frag+' ')
                for atom in frags[frag]: 
                    fA.write(str(atom+1)+' ')
                fA.write('\n')

        with open('fB.dat', 'w') as fB:
            for frag in frags:
                classification = frag.split('_')[-1].upper()
                if classification != 'B': continue
                fB.write(frag+' ')
                for atom in frags[frag]: 
                    fB.write(str(atom+1)+' ')
                fB.write('\n')
                

    #color fragments based on fragmentation from bfs
    def color_frags(self, frags):

        for frag in frags:
            selection = ''
            for atom in frags[frag]:
                selection += 'rank '+str(atom)+' '
            cmd.select("("+frag+")", selection)
            g = random.uniform(0, 1)           
            classification = frag.split('_')[-1].upper()
            if classification == 'C':
                cmd.color("grey", "("+frag+")")
                continue
            elif classification == 'A':
                r = random.uniform(0.5, 1)           
                b = random.uniform(0, 0.5)           
            elif classification == 'B':
                r = random.uniform(0, 0.5)           
                b = random.uniform(0.5, 1) 
            color = [r, g, b]
            cmd.set_color("color_"+frag, color) 
            cmd.color("color_"+frag, "("+frag+")")
    
    #def com(self):
        
 

# Read main geometry
def read_original_geometry():

    fil_name = cmd.get_names("all")[0]+'.xyz'
    geometry = pm_mol(fil_name)
    return geometry

def com_dister():
    
    # Make it pretty 
    cmd.show("sticks", "all")

    # Initialize pm molecule object
    total_molecule = read_original_geometry()

    # Take in user border atoms
    frag_names = cmd.get_names("all")[1:]
    frags = {}
    for name in frag_names:
        frags[name] = []
        stored.list=[]
        cmd.iterate("("+name+")","stored.list.append((name,rank))")
        for atom in stored.list: 
            frags[name].append(atom[1])

  
    for f in frags:
        for atom in frags[f]:
            name = total_molecule.mol[0][atom]
            coords = total_molecule.mol[1][atom]
    # Simple pass, find distance between COM of two fragments
    # ASSUMES only two fragments have been selected

cmd.extend("com_dister",com_dister)
    
