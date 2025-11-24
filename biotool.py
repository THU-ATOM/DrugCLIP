import copy
import io
import os
import warnings

import numpy as np
from Bio.PDB import Chain, Model, PDBParser, Structure, is_aa
from Bio.PDB.Atom import DisorderedAtom
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Residue import DisorderedResidue, Residue
from Bio.PDB.StructureBuilder import PDBConstructionWarning
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolfiles import MolToPDBBlock

warnings.filterwarnings(action="ignore", category=PDBConstructionWarning)


def res_is_connected(residue1, residue2):
    ca1 = residue1["CA"]
    ca2 = residue2["CA"]
    distance = ca1 - ca2
    if abs(distance - 3.8) <= 0.2:
        return 1
    else:
        return 0


def merge_chains(pdbstruct, af2=False, tgt=None, renumber=False):
    break_point = []
    # p = PDBParser()
    model = pdbstruct[0]
    tmp_chain = Chain.Chain("A")
    rid = 0
    last_res = None
    if not renumber and len(model) > 1:
        raise ValueError(
            "Multiple chains detected, please renumber the residues"
        )
    for chain in model:
        # break_point.append(rid)
        for res in chain:
            current_rid = rid if renumber else res.id[1]
            try:
                res.detach_parent()
                if not (is_aa(res, standard=True)):
                    continue
                if af2 and res["CA"].bfactor < 50:
                    continue
                if res.is_disordered():
                    if isinstance(res, DisorderedResidue):
                        res = res.selected_child
                        res.id = (res.id[0], current_rid, res.id[2])
                    else:
                        new_res = Residue(res.id, res.resname, res.segid)
                        for atom in res:
                            if isinstance(atom, DisorderedAtom):
                                atom.selected_child.disordered_flag = 0
                                new_res.add(atom.selected_child.copy())
                            else:
                                new_res.add(atom)
                        res = new_res
                        res.id = (res.id[0], current_rid, res.id[2])
                else:
                    res.id = (res.id[0], current_rid, res.id[2])

                if last_res is None or not res_is_connected(res, last_res):
                    break_point.append(current_rid)
                last_res = copy.deepcopy(res)
                tmp_chain.add(res.copy())
                rid += 1
            except Exception:
                pass
    tmp_structure = Structure.Structure(pdbstruct.id)
    tmp_model = Model.Model(0)
    tmp_structure.add(tmp_model)
    tmp_model.add(tmp_chain)
    if tgt is not None:
        io = PDBIO()
        io.set_structure(tmp_structure)
        io.save(os.path.join(tgt, pdbstruct.id + ".pdb"))
    return tmp_structure, break_point


def read_mol2_ligand(path, return_pdbstring=False):
    """
    mol = Chem.MolFromMol2File(path, removeHs=True, sanitize=False)
    if mol is None:
        print("cannot read mol2", path)
    coords = mol.GetConformer().GetPositions()
    atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
    return {'coord': np.array(coords), 'atom_type': atom_types, 'mol': mol, 'smi': Chem.MolToSmiles(mol)}
    """
    # try:
    #     mol2_df = PandasMol2().read_mol2(path ,columns={0:('atom_id', int), 1:('atom_name', str), 2:('x', float), 3:('y', float), 4:('z', float), 5:('atom_type', str), 6:('subst_id', int), 7:('residue_name', str), 8:('useless1', float), 9:('useless2', str)})

    #     coords = mol2_df.df[['x', 'y', 'z']]
    # except Exception:
    mol = Chem.MolFromMol2File(path, sanitize=False)
    try:
        mw = Descriptors.MolWt(mol)
        mol_noH = Chem.RemoveHs(mol)
    except Exception:
        mw = 0
        mol_noH = mol
    coords = mol_noH.GetConformer().GetPositions()
    if return_pdbstring:
        pdb_string = MolToPDBBlock(mol, flavor=2)
        return np.array(coords), mw, pdb_string
    else:
        return np.array(coords), mw


def get_binding_pockets(original_pdb, lig_coord, thres=10, rm_sidchain=True):
    chain = original_pdb[0]["A"]  # only deal with A chain
    tmp_chain = Chain.Chain("A")
    resid = set()
    for res in chain:
        res_coord = np.array(
            [i.get_coord() for i in res.get_atoms() if i.element != "H"]
        )
        dist = np.linalg.norm(
            res_coord[:, None, :] - lig_coord[None, :, :], axis=-1
        ).min()
        if dist <= thres:
            tmp_chain.add(res.copy())
            resid.add(str(res.id[1]))
    tmp_structure = Structure.Structure(original_pdb.id)
    tmp_model = Model.Model(0)
    tmp_structure.add(tmp_model)
    tmp_model.add(tmp_chain)
    if rm_sidchain:
        for res in tmp_chain:
            remove_atom_ids = []
            for atom in res.get_atoms():
                if atom.name not in ["C", "CA", "CB", "N", "O"]:
                    remove_atom_ids.append(
                        atom.id
                    )  # should not iter over a modified collection
            for atom_id in remove_atom_ids:
                res.detach_child(atom_id)
    return tmp_structure, resid


def pqr_parser(filename, return_score=False):
    with open(filename, "r") as f:
        data = f.readlines()
    coord = []
    for line in data:
        if "Pocket Score" in line:
            score = float(line.split()[-1])
        elif "Real volume" in line:
            volume = float(line.split()[-1])
        elif line[:4] == "ATOM":
            coord.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    coord = np.array(coord)
    if return_score:
        return coord, score, volume
    else:
        return coord


def pdb_parser(filename):
    with open(filename, "r") as f:
        data = f.read()
    return pdbstr2coord(data)


def pdbstr2coord(pdb_str):
    lines = pdb_str.split("\n")
    coord = []
    for line in lines:
        if line[:4] == "ATOM" or line[:6] == "HETATM":
            coord.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    return np.array(coord)


def save_pdb(pdb_struct, output):
    io = PDBIO()
    io.set_structure(pdb_struct)
    io.save(output)


def pdb2dict(biopy_structure, pocket_name):
    recpt = list(biopy_structure.get_atoms())
    pocket_atom_type = [x.element for x in recpt]
    pocket_coord = [x.coord for x in recpt]
    return {
        "pocket": pocket_name,
        "pocket_atoms": pocket_atom_type,
        "pocket_coordinates": pocket_coord,
    }


def read_sdf(sdfile):
    # read a sdf file with rdkit and return all coord array in a list
    suppl = Chem.SDMolSupplier(sdfile)
    mol_coord = []
    for mol in suppl:
        if mol is not None:
            # remove H atoms
            mol = Chem.RemoveHs(mol)
            mol_coord.append(mol.GetConformer(0).GetPositions())
    return np.array(mol_coord)


def sdfstr2coord(sdf_str):
    sdf_bytes = sdf_str.encode("utf-8")
    sdf_io = io.BytesIO(sdf_bytes)
    suppl = Chem.ForwardSDMolSupplier(sdf_io)
    mol_coord = []
    for mol in suppl:
        if mol is not None:
            mol = Chem.RemoveHs(mol)
            mol_coord.append(mol.GetConformer(0).GetPositions())
    return np.array(mol_coord)


def sdfstr2pdbstr(sdf_str):
    sdf_bytes = sdf_str.encode("utf-8")
    sdf_io = io.BytesIO(sdf_bytes)
    suppl = Chem.ForwardSDMolSupplier(sdf_io)
    pdb_str = ""
    for mol in suppl:
        if mol is not None:
            pdb_str += MolToPDBBlock(mol, flavor=2)
    return pdb_str


def ligand2coord(file_name):
    if file_name[-4:] == ".sdf":
        return read_sdf(file_name)
    elif file_name[-5:] == ".mol2":
        return read_mol2_ligand(file_name)[0]
    elif file_name[-4:] == ".pqr":
        return pqr_parser(file_name)
    elif file_name[-4:] == ".pdb":
        return pdb_parser(file_name)
    else:
        raise ValueError("File format not supported")


def ligandstr2coord(ligand, ligand_fmt):
    if ligand_fmt == "pdb":
        return pdbstr2coord(ligand)
    elif ligand_fmt == "sdf":
        return sdfstr2coord(ligand)
    elif ligand_fmt == "path":
        return ligand2coord(ligand)
    else:
        raise ValueError(f"ligand format {ligand_fmt} not supported")


def process_one_pair(
    pdb_file,
    ligand_file,
    pocket_name,
    renumber=False,
    dist_thres=10,
    rm_sidechain=True,
    return_dict=False,
    save_pdb_dir=None,
):
    lig_coord = ligand2coord(ligand_file)
    p = PDBParser()
    pdb_struct = p.get_structure("", pdb_file)
    tmp_structure, break_point = merge_chains(pdb_struct, renumber=renumber)
    pocket_struct, resid = get_binding_pockets(
        tmp_structure, lig_coord, dist_thres, rm_sidechain
    )
    if save_pdb_dir is not None:
        io = PDBIO()
        io.set_structure(pocket_struct)
        io.save(os.path.join(save_pdb_dir, pocket_name + ".pdb"))
    if return_dict:
        recpt = list(pocket_struct.get_atoms())
        pocket_atom_type = [x.element for x in recpt if x.element != "H"]
        pocket_coord = [x.coord for x in recpt if x.element != "H"]
        return {
            "pocket": pocket_name,
            "pocket_atoms": pocket_atom_type,
            "pocket_coordinates": pocket_coord,
        }


def pdbstr2struct(protein, protein_fmt="pdb"):
    parser = PDBParser()
    if protein_fmt == "pdb":
        pdb_io = io.StringIO(protein)
        struct = parser.get_structure("", pdb_io)
    elif protein_fmt == "path":
        struct = parser.get_structure("", protein)
    else:
        raise ValueError(f"protein format {protein_fmt} not supported")
    return struct


def extract_pocket(
    protein: str,
    ligand: str,
    protein_fmt: str = "pdb",
    ligand_fmt: str = "pdb",
    size: float = 6.0,
    save_path: str = None,
) -> str:
    ligand_coord = ligandstr2coord(ligand, ligand_fmt)
    struct = pdbstr2struct(protein, protein_fmt)
    tmp_structure, break_point = merge_chains(struct, renumber=True)
    pocket_struct, resid = get_binding_pockets(
        tmp_structure, ligand_coord, size, True
    )
    pocket_io = PDBIO()
    pocket_io.set_structure(pocket_struct)

    if save_path is not None:
        pocket_io.save(save_path)

    pocket_str = io.StringIO()
    pocket_io.save(pocket_str)
    return pocket_str.getvalue()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract pocket from a pdb file"
    )
    parser.add_argument("pdb_file", type=str, help="The receptor file")
    parser.add_argument("ligand_file", type=str, help="The ligand file")
    parser.add_argument("pocket_name", type=str, help="The pocket name")
    parser.add_argument(
        "--renumber", action="store_true", help="Renumber the chains"
    )
    parser.add_argument(
        "--dist_thres", type=float, default=10, help="The distance threshold"
    )
    parser.add_argument(
        "--rm_sidechain", action="store_true", help="Remove side chains"
    )
    parser.add_argument(
        "--save_pdb_dir", type=str, help="Save the pocket pdb file"
    )
    args = parser.parse_args()
    process_one_pair(
        args.pdb_file,
        args.ligand_file,
        args.pocket_name,
        args.renumber,
        args.dist_thres,
        args.rm_sidechain,
        save_pdb_dir=args.save_pdb_dir,
    )