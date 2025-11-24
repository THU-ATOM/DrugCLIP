import json
import argparse
from pathlib import Path
from tqdm import tqdm

import io
import logging
import pickle
from pathlib import Path

import lmdb
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

from biotool import extract_pocket, pdbstr2struct
from docker_ import (
    #encode_mols,
    #encode_pocket,
    load_mol_embedding,
    load_pocket_embedding,
)
from tqdm import tqdm

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union

import numpy as np

import subprocess
import os


def encode_pocket(workdir: str, checkpoint_dir: str, gpu_id: int = 0):
    """
    调用 encode_pocket.sh 生成 pocket embedding
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 脚本路径固定到 /app
    script_path = "/workspace/encode_pocket.sh"
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"{script_path} not found in container!")
    
    cmd = f"bash {script_path} {workdir}"
    subprocess.run(cmd, shell=True, check=True, env=env)


def encode_mols(workdir: str, checkpoint_dir: str, gpu_id: int = 0):
    """
    调用 encode_mols.sh 生成 ligand embedding
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 脚本路径固定到 /app
    script_path = "/workspace/encode_mols.sh"
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"{script_path} not found in container!")
    
    cmd = f"bash {script_path} {workdir}"
    subprocess.run(cmd, shell=True, check=True, env=env)

class RerankMethod(ABC):
    """Baseline method should inherit from this class."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the method."""
        raise NotImplementedError

    @abstractmethod
    def process(
        self, raw_data: Iterator[Dict[str, Any]], workdir: Path
    ) -> Union[Iterator, List, Any]:
        """Process raw data.

        The input data is an iterator of dictionaries, each dictionary contains the
        following keys:
            - "protein_name": name of the protein
            - "protein_pdb": protein in PDB format
            - "probe_ligand": probe ligand in PDB format
            - "ligands": list of docked molecules in SDF format
                - "name": name of the ligand
                - "sdf": ligand in SDF format
                - "label": label of the ligand
        Intermediate files should be saved in `workdir` and cleaned up if are
        not used in `compute_score`. Processed data can also be saved in `workdir`
        and read in `compute_score`.

        The return value could be any format, and will be passed to `compute_score`.
        Possible ways:
            - Return an iterator of processed data
            - Return processed data in memory as a list or dictionary
            - Save processed data in `workdir` and return None
        """
        raise NotImplementedError

    @abstractmethod
    def compute_score(self, processed_data, workdir: Path) -> List[List[float]]:
        """Compute score based on processed data.

        The return value should be a list of lists, each list contains the
        predicted scores for each molecule in the corresponding assay. The order
        should be the same as the input data.
        """
        raise NotImplementedError

    def normalized_score(self, score: List[List[float]]) -> List[List[float]]:
        """robust z-score normalization of the score."""
        for target_idx, target_scores in enumerate(score):
            median = np.median(target_scores)
            mad = np.median(np.abs(target_scores - median))
            score[target_idx] = (target_scores - median) / (mad if mad else 1)
        return score

    def on_process_start(self):
        """Hook function called before processing data."""

    def on_process_end(self):
        """Hook function called after processing data."""

    def on_score_start(self):
        """Hook function called before computing score."""

    def on_score_end(self):
        """Hook function called after computing score."""

    def safe_predict(self, model, sample, task_name: str = "affinity_ranking"):
        """
        通用预测包装函数：
        自动将各种输入(sample)转为 List[BasicInput]，避免 'tuple' object has no attribute 'task' 错误。
        """
        from airdd.task import TaskRegistry

        input_cls = TaskRegistry.get_input_class(task_name)

        # === Debug 信息 ===
        print(f"[DEBUG] safe_predict() 收到类型: {type(sample)}")

        # --- case 1: sample 已经是 BasicInput
        if hasattr(sample, "task"):
            inputs_list = [sample]

        # --- case 2: sample 是 dict
        elif isinstance(sample, dict):
            sample = dict(sample)
            sample.setdefault("task", task_name)
            inputs_list = [input_cls(**sample)]

        # --- case 3: sample 是 tuple 或 list
        elif isinstance(sample, (tuple, list)):
            # 如果是 (“input”, AffinityRankingInput(...))
            if len(sample) == 2 and sample[0] == "input":
                return self.safe_predict(model, sample[1], task_name)

            # 如果里面已经是 BasicInput
            if all(hasattr(x, "task") for x in sample):
                inputs_list = list(sample)
            elif len(sample) == 1 and isinstance(sample[0], dict):
                sample[0].setdefault("task", task_name)
                inputs_list = [input_cls(**sample[0])]
            else:
                raise ValueError(f"无法解析 sample tuple: {sample}")

        else:
            raise ValueError(f"Unsupported input type: {type(sample)} | {sample}")

        # 确保 task 存在
        for inp in inputs_list:
            if not getattr(inp, "task", None):
                inp.task = task_name

        # 调用真正的模型预测
        return model.predict(inputs_list)

logger = logging.getLogger(__name__)


def write_lmdb(data, lmdb_path, start_idx=0, force_recreate=True):
    if force_recreate and Path(lmdb_path).exists():
        lmdb_path.unlink()
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=1099511627776,
    )
    with env.begin(write=True) as txn:
        for d in data:
            txn.put(str(start_idx).encode("ascii"), pickle.dumps(d))
            start_idx += 1

    return start_idx


GLOBAL_MAX_POCKET_ATOMS = 511


def pocket2dict(pocket_name, pocket_pdb):
    biopy_chain = pdbstr2struct(pocket_pdb)
    recpt = list(biopy_chain.get_atoms())
    pocket_atom_type = [x.element for x in recpt if x.element != "H"]
    pocket_coord = [x.coord for x in recpt if x.element != "H"]
    if len(pocket_atom_type) > GLOBAL_MAX_POCKET_ATOMS:
        logger.warning(
            f"Pocket {pocket_name} has more than"
            f" {GLOBAL_MAX_POCKET_ATOMS} atoms."
        )
    return {
        "pocket": pocket_name,
        "pocket_atoms": pocket_atom_type,
        "pocket_coordinates": pocket_coord,
        #"coords": pocket_coord
    }


LFC = rdMolStandardize.LargestFragmentChooser()
remover = SaltRemover.SaltRemover()


def desalt(mol):
    mol = remover.StripMol(mol)
    mol = LFC.choose(mol)
    return mol


def gen_conformation(mol, num_conf=1, num_worker=1):
    try:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        if len(Chem.GetMolFrags(mol)) > 1:
            mol = desalt(mol)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_conf,
            numThreads=num_worker,
            pruneRmsThresh=1,
            maxAttempts=10000,
            useRandomCoords=False,
        )
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker)
        except Exception:
            pass
        mol = Chem.RemoveHs(mol)
    except Exception:
        logger.error("cannot gen conf", Chem.MolToSmiles(mol))
        return None
    if mol.GetNumConformers() == 0:
        logger.error("cannot gen conf", Chem.MolToSmiles(mol))
        return None
    return mol


def ligand2dict(ligand_name, ligand_sdf, pocket_name, keep_original_conf=False):
    try:
        sdf_bytes = ligand_sdf.encode("utf-8")
        sdf_io = io.BytesIO(sdf_bytes)
        mol = [x for x in Chem.ForwardSDMolSupplier(sdf_io) if x is not None][0]
        if mol.GetNumConformers() == 0:
            logger.warning(f"mol {ligand_name} has no conf, plan to generate")
            keep_original_conf = False
        if not keep_original_conf:
            mol = gen_conformation(mol)
        coords = mol.GetConformer().GetPositions()
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        return {
            "name": ligand_name,
            "smiles": Chem.MolToSmiles(mol),
            "coordinates": coords,
            "atoms": atoms,
            "pocket": pocket_name,
        }
    except Exception:
        logger.error(f"Failed to process ligand {ligand_name}")
        return None
    
class DrugCLIPReranker(RerankMethod):
    POCKET_SIZE = 6

    def __init__(self, checkpoint_dir: str, gpu_id: int = 0):
        self.checkpoint_dir = checkpoint_dir
        self.gpu_id = gpu_id

    @property
    def name(self):
        return "DrugCLIP"

    

    def process(self, raw_data, workdir):
        """
        处理原始数据，生成 lmdb + embedding
        """
        logger.info("Processing data...")
        pockets = []
        ligands = []
        ligand_indices = []
        miss_indices = []
        ligand_idx = 0
        processed_data = []
        

        for sample in tqdm(raw_data, ncols=80):
            print("==== SAMPLE KEYS ====", sample.keys())

            # 先把 sample 保存到 processed_data
            processed_data.append(sample)

            protein_name = sample.get("protein_name", sample.get("name", "UNKNOWN"))

            # 1️⃣ 使用已有 pocket_pdb
            pocket_pdb = sample.get("pocket_pdb", None)

            # 2️⃣ 如果没有 pocket_pdb，则调用 extract_pocket
            if not pocket_pdb:
                logger.info(f"No precomputed pocket for {protein_name}, extracting...")
                probe_ligand = sample.get("probe_ligand", None)

                if not probe_ligand and "ligands" in sample and len(sample["ligands"]) > 0:
                    # 使用第一个 ligand 的 sdf
                    probe_ligand = sample["ligands"][0]["sdf"]

                pocket_pdb = extract_pocket(
                    sample["protein_pdb"],
                    probe_ligand,
                    size=self.POCKET_SIZE
                )
            else:
                logger.info(f"Using precomputed pocket for {protein_name}")

           
            print(f"protein_name: {protein_name}")
            print(f"pocket_pdb: {pocket_pdb}")


            pocket_pdb= "/data"

            print(f"docker-corrected pocket_pdb: {pocket_pdb}")
          
            with open(pocket_pdb, "r") as f:
                pdb_str = f.read()

            pocket_dict = pocket2dict(protein_name, pdb_str)

            
            pockets.append(pocket_dict)

          

            for ligand in sample.get("ligands", []):
            
                    ligand_dict = ligand2dict(
                        ligand["name"], ligand["sdf"], protein_name
                    )
                    if ligand_dict is not None:
                        ligands.append(ligand_dict)
                        ligand_indices.append(ligand_idx)
                    else:
                        miss_indices.append(ligand_idx)
                    ligand_idx += 1

        # 写入 lmdb
        self.pocket_lmdb = workdir / "pocket.lmdb"
        write_lmdb(pockets, self.pocket_lmdb, force_recreate=True)

        self.ligand_lmdb = workdir / "mol.lmdb"
        write_lmdb(ligands, self.ligand_lmdb, force_recreate=True)

        # 编码 embedding
        encode_pocket(str(workdir), self.checkpoint_dir, self.gpu_id)
        encode_mols(str(workdir), self.checkpoint_dir, self.gpu_id)


        pocket_embedding = load_pocket_embedding(workdir)
        ligand_embedding = load_mol_embedding(workdir)

        return {
            "pockets": pocket_embedding,
            "ligands": ligand_embedding,
            "raw_data": processed_data,
            "ligand_indices": np.array(ligand_indices, dtype=int),
            "miss_indices": np.array(miss_indices, dtype=int),
        }

  
    def fill_miss_ligand_embedding(self, ligand_embedding, ligand_indices, miss_indices):
        # 保证行数一致
        if len(ligand_embedding) != len(ligand_indices):
            ligand_embedding = ligand_embedding[:len(ligand_indices)]
        
        total_ligands = len(ligand_indices) + len(miss_indices)
        new_ligand_embedding = np.zeros((total_ligands, ligand_embedding.shape[1]))
        
        # 填充已知 ligand
        new_ligand_embedding[ligand_indices] = ligand_embedding
        
        # 填充缺失 ligand
        mean_embedding = np.mean(ligand_embedding, axis=0)
        new_ligand_embedding[miss_indices] = mean_embedding
        
        return new_ligand_embedding


    def compute_score(self, processed_data, workdir):
        pocket_embedding = processed_data["pockets"]
        ligand_embedding = self.fill_miss_ligand_embedding(
            processed_data["ligands"],
            processed_data["ligand_indices"],
            processed_data["miss_indices"],
        )
        logger.info(
            f"Pocket embedding shape: {pocket_embedding.shape}, "
            f"Ligand embedding shape: {ligand_embedding.shape}"
        )
        assert pocket_embedding.shape[1] == ligand_embedding.shape[1], (
            f"Pocket embedding shape {pocket_embedding.shape} "
            f"does not match ligand embedding shape {ligand_embedding.shape}"
        )
        total_ligands = sum(
            [len(sample["ligands"]) for sample in processed_data["raw_data"]]
        )
        assert total_ligands == ligand_embedding.shape[0], (
            f"Total ligands {total_ligands} does not match "
            f"ligand embedding shape {ligand_embedding.shape[0]}"
        )

        result = []
        ligand_offset = 0
        for pocket_embedding, sample in zip(
            pocket_embedding,
            processed_data["raw_data"],
        ):
            n_ligands = len(sample["ligands"])
            ligands_embedding = ligand_embedding[
                ligand_offset : ligand_offset + n_ligands
            ]
            scores = pocket_embedding[None, :] @ ligands_embedding.T
            result.append(scores.flatten().tolist())
        return result


def load_input_json(input_file):
    """格式必须与 model.py preprocess 的 raw_data 一致"""
    with open(input_file, "r") as f:
        data = json.load(f)
    return data



def save_output_json(output_dir, raw_data, scores):
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample, score_list in zip(raw_data, scores):
        # sample["ligands"] 应该是一个 list，每个元素有 name
        ligands = sample.get("ligands", [])

        # 构造 {ligand_name: score}
        ligand_scores_dict = {}
        for lig, score in zip(ligands, score_list):
            ligand_scores_dict[lig["name"]] = score

        out = {
            "task": "affinity_ranking",
            "dataset": sample.get("dataset", None),   # 如果没有 dataset 可以为 None
            "model": sample.get("model", "DrugCLIP"), # 可按需修改
            "name": sample["name"],
            "ligand_scores": ligand_scores_dict,
        }

        outfile = output_dir / f"{sample['name']}.json"
        with open(outfile, "w") as f:
            json.dump(out, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="input.json")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output)
    checkpoint = Path(args.checkpoint)

    # 1) 读取输入
    raw_data = load_input_json(input_file)

    # 2) 初始化 DrugCLIP Reranker
    reranker = DrugCLIPReranker(checkpoint_dir=str(checkpoint), gpu_id=args.gpu)

    # 3) 创建工作目录（与外层 docker 的 /data 映射一致）
    rundir = input_file.parent
    workdir = rundir / "workdir"
    workdir.mkdir(exist_ok=True)

    # 4) preprocess（生成 lmdb + embedding）
    processed_data = reranker.process(raw_data, workdir)

    # 5) compute affinity score
    scores = reranker.compute_score(processed_data, workdir)

    # 6) 保存结果
    save_output_json(output_dir, raw_data, scores)

    print("✓ 完成 DrugCLIP affinity ranking.")
    print(f"结果已输出到: {output_dir}")


if __name__ == "__main__":
    main()
